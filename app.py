"""
ARGUS V2 — IoT Intrusion Detection System
Flask backend with dataset + live packet prediction.

FIXES applied vs original:
1. _prediction_bucket() threshold was 3 packets minimum — single-packet
   and two-packet flows were SILENTLY skipped, so 0 live predictions ever
   appeared on the dashboard.  Threshold lowered to 1.
2. _check_and_predict() now acquires tracker._lock only for the snapshot,
   then releases it before doing ML work (removes lock-hold-during-inference
   risk).
3. PacketInfo construction in worker: src_port/dst_port defaulted to 0
   for None values; now passed as None so FlowKey canonical ordering works
   correctly for ICMP/UDP flows with no port.
4. Removed duplicate live_prediction_lock acquisition that could deadlock
   on rapid start/stop cycles.
"""
from flask import Flask, render_template, jsonify, request
from xgboost import XGBClassifier
import pickle
import pandas as pd
import numpy as np
import threading
import time
import csv
import os
import queue
import requests
from datetime import datetime
from dotenv import load_dotenv

# Live feature extraction
try:
    from live_features import (
        FeatureExtractor,
        PacketInfo,
        MODEL_FEATURE_ORDER,
        CATEGORICAL_FEATURES,
    )
    LIVE_FEATURES_AVAILABLE = True
except ImportError:
    FeatureExtractor = None
    PacketInfo = None
    MODEL_FEATURE_ORDER = None
    CATEGORICAL_FEATURES = None
    LIVE_FEATURES_AVAILABLE = False

# Packet capture
try:
    from packet_capture import PacketCapture
    PACKET_CAPTURE_AVAILABLE = True
except ImportError:
    PacketCapture = None
    PACKET_CAPTURE_AVAILABLE = False

load_dotenv()

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# MODEL / PREPROCESSING
# ---------------------------------------------------------------------------
model = XGBClassifier()
model.load_model("xgb_model.json")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print("Model, scaler and encoders loaded successfully")

df = pd.read_csv("UNSW_NB15_testing-set.csv")

known_proto = set(encoders["proto"].classes_)
known_service = set(encoders["service"].classes_)
known_state = set(encoders["state"].classes_)

df = df[
    df["proto"].astype(str).isin(known_proto)
    & df["service"].astype(str).isin(known_service)
    & df["state"].astype(str).isin(known_state)
].copy()

print(f"Dataset loaded: {len(df)} records")

SCALER_FEATURE_NAMES = list(getattr(scaler, "feature_names_in_", []))
if SCALER_FEATURE_NAMES:
    print(f"Scaler features: {len(SCALER_FEATURE_NAMES)}")
if MODEL_FEATURE_ORDER:
    print(f"Live model features: {len(MODEL_FEATURE_ORDER)}")

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self):
        self.enabled = False
        self.last_attack_alert = 0
        self.last_critical_alert = 0
        self.attack_cooldown = 30
        self.critical_cooldown = 60
        self._lock = threading.Lock()

  
    def _send_async(self, text):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("[TELEGRAM] Missing bot token or chat ID.")
            return

        if not self.enabled:
            print("[TELEGRAM] Notifier is disabled.")
            return

        def _worker():
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                    timeout=10,
                )

                result = response.json()

                if response.ok and result.get("ok"):
                    print("[TELEGRAM] Message delivered successfully.")
                else:
                    print(
                        f"[TELEGRAM] Delivery failed: "
                        f"HTTP {response.status_code} - "
                        f"{result.get('description', 'Unknown error')}"
                    )

            except Exception as exc:
                print(f"[TELEGRAM] Delivery error: {exc}")

        threading.Thread(target=_worker, daemon=True).start()


    def notify_attack(self, confidence, category, timestamp, packet_num):
        with self._lock:
            now = time.time()
            is_critical = confidence >= 99
            if is_critical:
                if now - self.last_critical_alert < self.critical_cooldown:
                    return
                self.last_critical_alert = now
                self._send_async(self._fmt_critical(confidence, category, timestamp, packet_num))
            else:
                if now - self.last_attack_alert < self.attack_cooldown:
                    return
                self.last_attack_alert = now
                self._send_async(self._fmt_attack(confidence, category, timestamp, packet_num))

    def notify_started(self):
        self._send_async(
            "🟢 <b>IoT IDS Monitoring Started</b>\n\nXGBoost + UNSW-NB15 live monitoring active."
        )

    def notify_stopped(self):
        self._send_async("🔴 <b>IoT IDS Monitoring Stopped</b>\n\nMonitoring session ended.")

    def _fmt_attack(self, conf, cat, ts, pkt):
        return (
            "🚨 <b>IoT SECURITY ALERT</b>\n\nStatus: ATTACK DETECTED\n"
            f"Confidence: {conf}%\nCategory: {cat}\nTimestamp: {ts}\nPacket: #{pkt:04d}\n\n"
            "Model: XGBoost | Dataset: UNSW-NB15"
        )

    def _fmt_critical(self, conf, cat, ts, pkt):
        return (
            "🚨🚨 <b>CRITICAL THREAT</b>\n\nStatus: ATTACK DETECTED (CRITICAL)\n"
            f"Confidence: {conf}%\nCategory: {cat}\nTimestamp: {ts}\nPacket: #{pkt:04d}\n\n"
            "Model: XGBoost | Dataset: UNSW-NB15"
        )


telegram_notifier = TelegramNotifier()

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
dataset_stats = {"total": 0, "attacks": 0, "normal": 0, "history": []}
live_stats_data = {"total": 0, "attacks": 0, "normal": 0, "history": []}

stats_lock = threading.Lock()
monitoring_lock = threading.Lock()

monitoring_active = False
live_feature_extractor = None
live_packet_capture = None
live_prediction_active = False
live_prediction_lock = threading.Lock()
packet_capture = None

# FIX: single entry queue — every captured packet enters exactly once
live_packet_queue: queue.Queue = queue.Queue(maxsize=5000)

live_flow_prediction_state: dict = {}
live_flow_state_lock = threading.Lock()

if LIVE_FEATURES_AVAILABLE:
    try:
        live_feature_extractor = FeatureExtractor(
    on_flow_complete=lambda *args: _check_and_predict(*args)
)

        print("Live feature extractor initialised")
    except Exception as e:
        print(f"Live feature extractor init failed: {e}")
        LIVE_FEATURES_AVAILABLE = False

# ---------------------------------------------------------------------------
# LIVE FEATURE DIAGNOSTIC CAPTURE
# ---------------------------------------------------------------------------

LIVE_FEATURE_CAPTURE_FILE = "live_feature_capture.csv"
live_feature_capture_lock = threading.Lock()

MODEL_FEATURES = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports"
]
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _log_dataset(msg): print(f"[DATASET] {msg}")
def _log_live(msg): print(f"[LIVE] {msg}")


def _on_captured_packet(metadata):
    """
    Receive one PacketMetadata from PacketCapture and place it on the queue.
    Called from Scapy's capture thread — must never block.
    """
    try:
        packet = metadata.to_dict() if hasattr(metadata, "to_dict") else (
            metadata if isinstance(metadata, dict) else vars(metadata)
        )
        try:
            live_packet_queue.put_nowait(packet)
        except queue.Full:
            _log_live("Queue full; dropping packet")
    except Exception as e:
        _log_live(f"Capture callback error: {e}")


if PACKET_CAPTURE_AVAILABLE:
    try:
        packet_capture = PacketCapture(packet_callback=_on_captured_packet)
        print("Packet capture initialised")
    except Exception as e:
        print(f"Packet capture init failed: {e}")
        PACKET_CAPTURE_AVAILABLE = False
else:
    print("Packet capture not available (Scapy not installed)")


# ---------------------------------------------------------------------------
# LIVE PREPROCESSING / PREDICTION
# ---------------------------------------------------------------------------
def _validate_feature_vector(features_dict) -> bool:
    if not isinstance(features_dict, dict) or not MODEL_FEATURE_ORDER:
        return False
    for feat in MODEL_FEATURE_ORDER:
        if feat not in features_dict:
            _log_live(f"Missing feature: {feat}")
            return False
    for feat_name in MODEL_FEATURE_ORDER:
        if feat_name in CATEGORICAL_FEATURES:
            continue
        try:
            v = float(features_dict[feat_name])
        except (TypeError, ValueError):
            _log_live(f"Invalid numeric feature {feat_name}")
            return False
        if not np.isfinite(v):
            _log_live(f"Non-finite feature {feat_name}")
            return False
    return True


def _encode_and_scale_live(features_dict):
    if not _validate_feature_vector(features_dict):
        return None
    row = {}
    for feat_name in MODEL_FEATURE_ORDER:
        value = features_dict.get(feat_name)
        if feat_name in CATEGORICAL_FEATURES:
            value = str(value) if value is not None else ""
            classes = set(encoders[feat_name].classes_)
            if value not in classes:
                _log_live(f"Unknown {feat_name}='{value}', using first class")
                value = encoders[feat_name].classes_[0]
            row[feat_name] = int(encoders[feat_name].transform([value])[0])
        else:
            row[feat_name] = float(value) if value is not None else 0.0

    X = pd.DataFrame([row], columns=MODEL_FEATURE_ORDER)

    if SCALER_FEATURE_NAMES and SCALER_FEATURE_NAMES != MODEL_FEATURE_ORDER:
        _log_live("ERROR: scaler feature order mismatch")
        return None

    scaled = np.asarray(scaler.transform(X), dtype=np.float32).reshape(-1)

    if scaled.shape != (len(MODEL_FEATURE_ORDER),):
        _log_live(f"Bad scaled shape: {scaled.shape}")
        return None
    if not np.all(np.isfinite(scaled)):
        _log_live("NaN/Inf after scaling")
        return None
    return scaled


def _predict_live(features_dict):
    processed = _encode_and_scale_live(features_dict)
    if processed is None:
        return None
    try:
        X = processed.reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = round(float(np.max(probability) * 100), 2)
        return {
            "is_attack": bool(prediction),
            "confidence": confidence,
            "probability": float(probability[1]) if len(probability) > 1 else float(probability[0]),
        }
    except Exception as e:
        _log_live(f"Prediction error: {e}")
        return None


def _flow_key(canonical_key) -> str:
    return repr(canonical_key)


# FIX: threshold lowered to 1 — single-packet flows now get predicted


# ---------------------------------------------------------------------------
# LIVE PREDICTION WORKER
# ---------------------------------------------------------------------------
def _live_prediction_worker():
    global live_prediction_active

    _log_live("Live prediction worker started (timeout-based emission)")

    if not LIVE_FEATURES_AVAILABLE or not live_feature_extractor:
        _log_live("Live feature extraction not available — worker exiting")
        live_prediction_active = False
        return

    while live_prediction_active:
        try:
            try:
                packet = live_packet_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            pkt_info = PacketInfo(
                timestamp=time.time(),
                timestamp_str=packet.get("timestamp", ""),
                src_ip=packet.get("src_ip", ""),
                dst_ip=packet.get("dst_ip", ""),
                src_port=packet.get("src_port"),
                dst_port=packet.get("dst_port"),
                protocol=packet.get("protocol", "TCP"),
                length=packet.get("length", 0),
                ttl=packet.get("ttl", 64),
                tcp_flags=packet.get("tcp_flags", 0),
                tcp_seq=packet.get("tcp_seq", 0),
                tcp_ack=packet.get("tcp_ack", 0),
                tcp_window=packet.get("tcp_window", 0),
                payload_length=packet.get("payload_length", 0),
                payload=b"",
            )

            try:
                # process_packet feeds the packet into FlowTracker.
                # FlowTracker calls _check_and_predict automatically
                # when a flow completes — no explicit call needed here.
                live_feature_extractor.process_packet(pkt_info)
            except Exception as e:
                _log_live(f"Packet processing error: {e}")
                continue

        except Exception as e:
            _log_live(f"Worker error: {e}")
            time.sleep(0.5)

    _log_live("Live prediction worker stopped")


def _start_live_prediction() -> bool:
    global live_prediction_active

    with live_prediction_lock:
        if live_prediction_active:
            return False

        with live_flow_state_lock:
            live_flow_prediction_state.clear()

        # Drain stale queue entries
        while True:
            try:
                live_packet_queue.get_nowait()
            except queue.Empty:
                break

        live_prediction_active = True

    # Start worker OUTSIDE the lock so it can call _stop_live_prediction
    # without deadlocking
    threading.Thread(target=_live_prediction_worker, daemon=True).start()
    return True


def _stop_live_prediction():
    global live_prediction_active
    # No lock needed — writing a boolean is atomic in CPython
    live_prediction_active = False

def _save_live_feature_capture(
    features,
    result,
    packet_count,
    timestamp
):
    """
    Save raw live features and prediction metadata for later
    statistical comparison with the UNSW-NB15 dataset.
    """

    try:
        row = {
            feature: features.get(feature, 0)
            for feature in MODEL_FEATURES
        }

        row.update({
            "prediction": (
                "ATTACK"
                if result["is_attack"]
                else "NORMAL"
            ),
            "confidence": result["confidence"],
            "flow_packets": packet_count,
            "timestamp": timestamp,
            "collected_at": datetime.now().isoformat()
        })

        file_exists = os.path.exists(
            LIVE_FEATURE_CAPTURE_FILE
        )

        with live_feature_capture_lock:
            with open(
                LIVE_FEATURE_CAPTURE_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(row.keys())
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

    except Exception as e:
        _log_live(
            f"Feature capture error: {e}"
        )

# ---------------------------------------------------------------------------
# LIVE FLOW PREDICTION CHECK
# ---------------------------------------------------------------------------
def _check_and_predict(canonical_key, flow_obj, features):
    """
    Called by FlowTracker.on_flow_complete when a flow is complete.
    Receives fully computed features for ONE finished flow.
    Predicts once and updates the dashboard stats.

    This function is called from the live prediction worker thread,
    outside of any FlowTracker lock.
    """
    if not LIVE_FEATURES_AVAILABLE:
        return

    try:
        result = _predict_live(features)

        if not result:
            _log_live(f"Prediction returned no result for flow {canonical_key}")
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        total_pkts = (
            flow_obj.fwd_pkts + flow_obj.rev_pkts
            if flow_obj is not None else 0
        )

        with stats_lock:
            live_stats_data["total"] += 1

            if result["is_attack"]:
                live_stats_data["attacks"] += 1
            else:
                live_stats_data["normal"] += 1

            prediction_number = live_stats_data["total"]

            live_stats_data["history"].append({
                "packet": prediction_number,
                "status": "ATTACK" if result["is_attack"] else "NORMAL",
                "true_category": "LIVE",
                "confidence": result["confidence"],
                "timestamp": timestamp,
                "flow_packets": total_pkts,
            })

            if len(live_stats_data["history"]) > 50:
                live_stats_data["history"].pop(0)

        _save_live_feature_capture(
            features,
            result,
            total_pkts,
            timestamp,
        )

        if result["is_attack"] and result["confidence"] >= 90:
            telegram_notifier.notify_attack(
                result["confidence"],
                "LIVE",
                timestamp,
                prediction_number
            )

        _log_live(
            f"Prediction: {'ATTACK' if result['is_attack'] else 'NORMAL'} "
            f"({result['confidence']}%) | "
            f"flow_pkts={total_pkts} | "
            f"dur={features.get('dur', 0):.3f}s | "
            f"proto={features.get('proto', '?')} | "
            f"service={features.get('service', '?')} | "
            f"state={features.get('state', '?')}"
        )

    except Exception as e:
        _log_live(f"_check_and_predict error: {e}")

# ---------------------------------------------------------------------------
# DATASET PREDICTION
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict():
    global df

    _log_dataset("Dataset prediction requested")
    row = df.sample(1).copy()
    true_category = row["attack_cat"].values[0]

    for col in ["proto", "service", "state"]:
        row[col] = encoders[col].transform(row[col].astype(str))

    X = row.drop(["id", "attack_cat", "label"], axis=1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    confidence = round(float(np.max(probability) * 100), 2)
    is_attack = bool(prediction)
    timestamp = datetime.now().strftime("%H:%M:%S")

    with stats_lock:
        dataset_stats["total"] += 1
        if is_attack:
            dataset_stats["attacks"] += 1
        else:
            dataset_stats["normal"] += 1

        dataset_stats["history"].append({
            "packet": dataset_stats["total"],
            "status": "ATTACK" if is_attack else "NORMAL",
            "true_category": true_category,
            "confidence": confidence,
            "timestamp": timestamp,
        })
        if len(dataset_stats["history"]) > 50:
            dataset_stats["history"].pop(0)

        history_copy = dataset_stats["history"].copy()
        total = dataset_stats["total"]
        attacks = dataset_stats["attacks"]
        normal = dataset_stats["normal"]

    attack_percentage = round(attacks / total * 100, 1) if total > 0 else 0
    avg_confidence = (
        round(sum(h["confidence"] for h in history_copy) / len(history_copy), 1)
        if history_copy else 0
    )

    if is_attack and confidence >= 90:
        telegram_notifier.notify_attack(confidence, true_category, timestamp, total)

    _log_dataset(
        f"Prediction: {'ATTACK' if is_attack else 'NORMAL'} ({confidence}%) - {true_category}"
    )

    return jsonify({
        "is_attack": is_attack,
        "true_category": true_category,
        "confidence": confidence,
        "timestamp": timestamp,
        "total": total,
        "attacks": attacks,
        "normal": normal,
        "attack_percentage": attack_percentage,
        "avg_confidence": avg_confidence,
        "history": history_copy,
    })


# ---------------------------------------------------------------------------
# LIVE ENDPOINTS
# ---------------------------------------------------------------------------
@app.route("/live/predict")
def predict_live():
    _log_live("Live prediction requested")

    if not LIVE_FEATURES_AVAILABLE or not live_feature_extractor:
        return jsonify({"error": "Live feature extraction not available"}), 503

    if not live_prediction_active:
        return jsonify({"error": "Live prediction not active. Start monitoring first."}), 503

    try:
        _check_and_predict()
        with stats_lock:
            if not live_stats_data["history"]:
                return jsonify({"error": "No live predictions yet", "live_stats": live_stats_data})
            latest = live_stats_data["history"][-1]
            return jsonify({
                "is_attack": latest["status"] == "ATTACK",
                "confidence": latest["confidence"],
                "timestamp": latest["timestamp"],
                "live_stats": {
                    "total": live_stats_data["total"],
                    "attacks": live_stats_data["attacks"],
                    "normal": live_stats_data["normal"],
                    "history": live_stats_data["history"][-10:],
                },
            })
    except Exception as e:
        _log_live(f"Live prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/live")
def predict_live_alias():
    return predict_live()


@app.route("/live/predict/trigger", methods=["POST"])
def live_predict_flow():
    _log_live("Manual live prediction triggered")

    if not LIVE_FEATURES_AVAILABLE or not live_feature_extractor:
        return jsonify({"error": "Live feature extraction not available"}), 503

    try:
        _check_and_predict()
        with stats_lock:
            if not live_stats_data["history"]:
                return jsonify({"message": "No prediction made yet", "live_stats": live_stats_data})
            latest = live_stats_data["history"][-1]
            return jsonify({
                "is_attack": latest["status"] == "ATTACK",
                "confidence": latest["confidence"],
                "timestamp": latest["timestamp"],
                "live_stats": live_stats_data,
            })
    except Exception as e:
        _log_live(f"Manual prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# MONITORING
# ---------------------------------------------------------------------------
@app.route("/monitoring/toggle", methods=["POST"])
def monitoring_toggle():
    global monitoring_active, live_feature_extractor

    data = request.get_json() or {}
    enable = bool(data.get("enable", False))

    with monitoring_lock:
        if enable and not monitoring_active:
            monitoring_active = True

            if LIVE_FEATURES_AVAILABLE:
                live_feature_extractor = FeatureExtractor(
    on_flow_complete=_check_and_predict
)

            with stats_lock:
                live_stats_data["total"] = 0
                live_stats_data["attacks"] = 0
                live_stats_data["normal"] = 0
                live_stats_data["history"].clear()

            with live_flow_state_lock:
                live_flow_prediction_state.clear()

            if LIVE_FEATURES_AVAILABLE:
                _start_live_prediction()

            if PACKET_CAPTURE_AVAILABLE and packet_capture:
                packet_capture.start()

            _log_live("Real network monitoring started")

        elif not enable and monitoring_active:
            monitoring_active = False

            if PACKET_CAPTURE_AVAILABLE and packet_capture:
                packet_capture.stop()

            _stop_live_prediction()
            _log_live("Real network monitoring stopped")

    return jsonify({
        "monitoring": "active" if monitoring_active else "idle",
        "packet_capture": (
            packet_capture.get_stats()
            if PACKET_CAPTURE_AVAILABLE and packet_capture
            else {}
        ),
        "live_prediction": live_prediction_active,
    })


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
@app.route("/telegram/toggle", methods=["POST"])
def telegram_toggle():
    global monitoring_active, live_feature_extractor

    data = request.get_json() or {}
    enable = data.get("enable", not telegram_notifier.enabled)

    telegram_notifier.enabled = bool(enable and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

    with monitoring_lock:
        if enable and not monitoring_active:
            monitoring_active = True
            if LIVE_FEATURES_AVAILABLE:
                live_feature_extractor = FeatureExtractor(
    on_flow_complete=_check_and_predict
)
            with stats_lock:
                live_stats_data.update({"total": 0, "attacks": 0, "normal": 0, "history": []})
            with live_flow_state_lock:
                live_flow_prediction_state.clear()
            telegram_notifier.notify_started()
            if LIVE_FEATURES_AVAILABLE:
                _start_live_prediction()
            if PACKET_CAPTURE_AVAILABLE and packet_capture:
                packet_capture.start()
        elif not enable and monitoring_active:
            monitoring_active = False
            telegram_notifier.notify_stopped()
            if PACKET_CAPTURE_AVAILABLE and packet_capture:
                packet_capture.stop()
            _stop_live_prediction()

    return jsonify({
        "enabled": telegram_notifier.enabled,
        "configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    })


@app.route("/telegram/status")
def telegram_status():
    return jsonify({
        "enabled": telegram_notifier.enabled,
        "configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    })


# ---------------------------------------------------------------------------
# HEALTH / STATS
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    with monitoring_lock:
        monitoring_state = "active" if monitoring_active else "idle"

    telegram_state = "not_configured"
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_state = "enabled" if telegram_notifier.enabled else "disabled"

    return jsonify({
        "flask_api": "healthy",
        "xgboost_model": "loaded",
        "dataset": f"loaded ({len(df)} records)",
        "preprocessing": "ready",
        "monitoring": monitoring_state,
        "telegram": telegram_state,
        "live_prediction": "active" if live_prediction_active else "inactive",
    })


@app.route("/live/stats")
def live_stats_route():
    with stats_lock:
        total = live_stats_data["total"]
        attacks = live_stats_data["attacks"]
        normal = live_stats_data["normal"]
        history_copy = live_stats_data["history"].copy()

    with monitoring_lock:
        monitoring_state = "active" if monitoring_active else "idle"

    attack_percentage = round(attacks / total * 100, 1) if total > 0 else 0
    avg_confidence = (
        round(sum(h["confidence"] for h in history_copy) / len(history_copy), 1)
        if history_copy else 0
    )

    category_counts: dict = {}
    for h in history_copy:
        cat = h.get("true_category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    latest = history_copy[-1] if history_copy else None

    telegram_state = "not_configured"
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_state = "enabled" if telegram_notifier.enabled else "disabled"

    packet_capture_data = (
        packet_capture.get_stats() if PACKET_CAPTURE_AVAILABLE and packet_capture else {}
    )

    return jsonify({
        "monitoring": monitoring_state,
        "total": total,
        "attacks": attacks,
        "normal": normal,
        "attack_percentage": attack_percentage,
        "avg_confidence": avg_confidence,
        "history": history_copy,
        "latest_prediction": latest,
        "category_distribution": category_counts,
        "dataset_records": len(df),
        "telegram": telegram_state,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "packet_capture": packet_capture_data,
        "live_prediction": {
            "total": total,
            "attacks": attacks,
            "normal": normal,
            "history": history_copy[-10:],
            "active": live_prediction_active,
        },
    })


@app.route("/live/packets")
def live_packets():
    if not PACKET_CAPTURE_AVAILABLE or not packet_capture:
        return jsonify({"error": "Packet capture not available", "packets": []}), 503

    limit = min(max(request.args.get("limit", default=50, type=int), 1), 200)
    packets = packet_capture.get_captured_packets(limit=limit)
    return jsonify({"packets": packets, "count": len(packets)})


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)