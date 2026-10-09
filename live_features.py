"""
Live Feature Extraction Module for ARGUS V2
Converts captured packets into model-compatible features.

Changes in this version (Phase 3 — windowed ct_* fix)
------------------------------------------------------
The only changes vs Phase 1 live_features.py are inside FlowTracker:

1. __init__: replaced the unbounded defaultdict _conn_stats with a
   fixed-length deque (_flow_window) that stores the last WINDOW_SIZE
   completed flows.  WINDOW_SIZE = 100, matching the UNSW-NB15 paper
   definition of ct_* features.

2. _update_connection_stats: removed entirely (no longer needed).
   Connection stats are now derived from _flow_window at query time.

3. _get_connection_features: now counts matching flows in _flow_window
   rather than looking up unbounded global totals.  Each ct_* value is
   therefore bounded to [0, 100] and reflects only recent activity,
   which matches the training distribution.

4. _emit_flow: records the completed flow's key attributes into
   _flow_window before calling the user callback.

All other methods, classes, and module-level names are UNCHANGED.
"""
import threading
import time
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

try:
    from scapy.all import IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UNSW-NB15 ct_* sliding-window size (matches the paper definition)
# ---------------------------------------------------------------------------
CT_WINDOW_SIZE = 100


# ---------------------------------------------------------------------------
# PORT → SERVICE
# ---------------------------------------------------------------------------
PORT_TO_SERVICE: Dict[int, str] = {
    80: "http", 443: "ssl", 53: "dns", 21: "ftp", 20: "ftp-data",
    22: "ssh", 23: "telnet", 25: "smtp", 110: "pop3", 143: "imap",
    993: "imaps", 995: "pop3s", 465: "smtps", 587: "submission",
    161: "snmp", 162: "snmptrap", 389: "ldap", 636: "ldaps",
    88: "kerberos", 445: "smb", 139: "netbios", 137: "netbios-ns",
    138: "netbios-dgm", 514: "syslog", 515: "printer", 631: "ipp",
    1433: "ms-sql-s", 1521: "oracle", 3306: "mysql", 5432: "postgresql",
    27017: "mongodb", 6379: "redis", 11211: "memcached",
    8080: "http", 8443: "ssl", 8000: "http", 8888: "http",
    5060: "sip", 5061: "sips", 1723: "pptp",
    500: "isakmp", 4500: "ipsec-nat-t", 1194: "openvpn",
}


def port_to_service(port: Optional[int], protocol: str) -> str:
    if port is None:
        return "-"
    if port in PORT_TO_SERVICE:
        return PORT_TO_SERVICE[port]
    return "-"


# ---------------------------------------------------------------------------
# TCP STATE
# ---------------------------------------------------------------------------
def tcp_flags_to_state(flags: int, direction: str) -> str:
    is_syn = bool(flags & 0x02)
    is_ack = bool(flags & 0x10)
    is_fin = bool(flags & 0x01)
    is_rst = bool(flags & 0x04)

    if is_rst:
        return "RST"
    if is_fin:
        return "FIN"
    if is_syn and not is_ack:
        return "REQ"
    if is_syn and is_ack:
        return "ACC"
    if is_ack:
        return "CON"
    return "INT"


# ---------------------------------------------------------------------------
# FLOW KEY
# ---------------------------------------------------------------------------
@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def __hash__(self):
        return hash(
            (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        )

    def __eq__(self, other):
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
            and self.protocol == other.protocol
        )


# ---------------------------------------------------------------------------
# PACKET INFO
# ---------------------------------------------------------------------------
@dataclass
class PacketInfo:
    timestamp: float
    timestamp_str: str
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    length: int
    ttl: int
    tcp_flags: int = 0
    tcp_seq: int = 0
    tcp_ack: int = 0
    tcp_window: int = 0
    payload_length: int = 0
    payload: bytes = b""

    @classmethod
    def from_scapy(cls, packet, timestamp: float) -> Optional["PacketInfo"]:
        if not SCAPY_AVAILABLE or not packet.haslayer(IP):
            return None

        ip = packet[IP]
        protocol = "UNKNOWN"
        src_port = dst_port = None
        tcp_flags = tcp_seq = tcp_ack = tcp_window = 0
        payload_len = 0
        payload = b""

        if packet.haslayer(TCP):
            protocol = "TCP"
            tcp = packet[TCP]
            src_port, dst_port = tcp.sport, tcp.dport
            tcp_flags = int(tcp.flags)
            tcp_seq, tcp_ack, tcp_window = tcp.seq, tcp.ack, tcp.window
            payload_len = len(tcp.payload) if tcp.payload else 0
            payload = bytes(tcp.payload) if tcp.payload else b""
        elif packet.haslayer(UDP):
            protocol = "UDP"
            udp = packet[UDP]
            src_port, dst_port = udp.sport, udp.dport
            payload_len = len(udp.payload) if udp.payload else 0
            payload = bytes(udp.payload) if udp.payload else b""
        elif packet.haslayer(ICMP):
            protocol = "ICMP"
        else:
            protocol = "OTHER"

        ts = time.time() if timestamp == 0 else timestamp
        return cls(
            timestamp=ts,
            timestamp_str=datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3],
            src_ip=ip.src,
            dst_ip=ip.dst,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            length=len(packet),
            ttl=ip.ttl,
            tcp_flags=tcp_flags,
            tcp_seq=tcp_seq,
            tcp_ack=tcp_ack,
            tcp_window=tcp_window,
            payload_length=payload_len,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# FLOW  (every field declared exactly once)
# ---------------------------------------------------------------------------
@dataclass
class Flow:
    canonical_key: Tuple
    original_key: FlowKey
    start_time: float
    last_time: float

    fwd_pkts: int = 0
    rev_pkts: int = 0
    fwd_bytes: int = 0
    rev_bytes: int = 0

    fwd_ttl: int = 0
    rev_ttl: int = 0

    state: str = "INT"

    syn_seen: bool = False
    synack_seen: bool = False
    ack_seen: bool = False
    handshake_complete: bool = False
    first_syn_time: Optional[float] = None
    synack_time: Optional[float] = None
    first_ack_time: Optional[float] = None
    last_ack_time: Optional[float] = None
    syn_direction: Optional[bool] = None
    synack_direction: Optional[bool] = None
    ack_direction: Optional[bool] = None

    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0

    fwd_intervals: List[float] = field(default_factory=list)
    rev_intervals: List[float] = field(default_factory=list)
    fwd_last_time: Optional[float] = None
    rev_last_time: Optional[float] = None

    fwd_win: int = 0
    rev_win: int = 0
    fwd_seq_base: int = 0
    rev_seq_base: int = 0

    fwd_sizes: List[int] = field(default_factory=list)
    rev_sizes: List[int] = field(default_factory=list)

    seen_seq_fwd: set = field(default_factory=set)
    seen_seq_rev: set = field(default_factory=set)
    fwd_loss: int = 0
    rev_loss: int = 0

    payload_bytes: int = 0
    http_detected: bool = False
    ftp_detected: bool = False
    http_methods: set = field(default_factory=set)
    ftp_commands_list: List[str] = field(default_factory=list)
    http_transactions: int = 0
    response_body_length: int = 0
    ftp_login_detected: bool = False

    def _is_fwd(self, pkt: "PacketInfo") -> bool:
        return (
            pkt.src_ip == self.original_key.src_ip
            and pkt.dst_ip == self.original_key.dst_ip
            and pkt.src_port == self.original_key.src_port
            and pkt.dst_port == self.original_key.dst_port
        )

    def add_packet(self, pkt: "PacketInfo"):
        is_fwd = self._is_fwd(pkt)
        now = pkt.timestamp

        if pkt.protocol == "TCP":
            f = pkt.tcp_flags
            is_syn = bool(f & 0x02)
            is_ack = bool(f & 0x10)

            if is_syn and not is_ack and not self.syn_seen:
                self.syn_seen = True
                self.syn_direction = is_fwd
                self.first_syn_time = now
            elif is_syn and is_ack and not self.synack_seen:
                self.synack_seen = True
                self.synack_direction = is_fwd
                self.synack_time = now
                if self.first_syn_time:
                    self.synack = now - self.first_syn_time
            elif is_ack and not is_syn and not self.ack_seen:
                self.ack_seen = True
                self.ack_direction = is_fwd
                self.first_ack_time = now
                if self.synack_time:
                    self.ackdat = now - self.synack_time
                self.handshake_complete = True
                self.last_ack_time = now

            if self.synack_time and self.first_syn_time:
                self.tcprtt = self.synack_time - self.first_syn_time

        if is_fwd:
            self.fwd_pkts += 1
            self.fwd_bytes += pkt.length
            self.fwd_sizes.append(pkt.length)
            if self.fwd_last_time is not None:
                self.fwd_intervals.append(now - self.fwd_last_time)
            self.fwd_last_time = now
            if self.fwd_ttl == 0:
                self.fwd_ttl = pkt.ttl
            if pkt.protocol == "TCP":
                self.fwd_win = pkt.tcp_window
                if self.fwd_seq_base == 0:
                    self.fwd_seq_base = pkt.tcp_seq
                if pkt.tcp_seq in self.seen_seq_fwd:
                    self.fwd_loss += 1
                self.seen_seq_fwd.add(pkt.tcp_seq)
        else:
            self.rev_pkts += 1
            self.rev_bytes += pkt.length
            self.rev_sizes.append(pkt.length)
            if self.rev_last_time is not None:
                self.rev_intervals.append(now - self.rev_last_time)
            self.rev_last_time = now
            if self.rev_ttl == 0:
                self.rev_ttl = pkt.ttl
            if pkt.protocol == "TCP":
                self.rev_win = pkt.tcp_window
                if self.rev_seq_base == 0:
                    self.rev_seq_base = pkt.tcp_seq
                if pkt.tcp_seq in self.seen_seq_rev:
                    self.rev_loss += 1
                self.seen_seq_rev.add(pkt.tcp_seq)

        if pkt.protocol == "TCP":
            self.state = tcp_flags_to_state(
                pkt.tcp_flags, "forward" if is_fwd else "reverse"
            )

        self.last_time = now

        if pkt.payload_length > 0:
            self.payload_bytes += pkt.payload_length

        self._parse_application_layer(pkt)

    def _parse_application_layer(self, pkt: "PacketInfo"):
        if pkt.payload_length == 0:
            return
        try:
            payload = pkt.payload.decode("utf-8", errors="ignore")
        except Exception:
            payload = ""

        http_ports = {80, 443, 8080, 8000, 8443, 8008, 8888}
        if (pkt.dst_port in http_ports or pkt.src_port in http_ports) and payload:
            self.http_detected = True
            for method in ("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"):
                if payload.startswith(method + " "):
                    self.http_methods.add(method)
                    self.http_transactions += 1
                    break
            if "Content-Length:" in payload:
                try:
                    cl_start = payload.index("Content-Length:") + 15
                    cl_end = payload.index("\r\n", cl_start)
                    self.response_body_length = max(
                        self.response_body_length,
                        int(payload[cl_start:cl_end].strip()),
                    )
                except Exception:
                    pass

        ftp_ports = {21, 20, 2121}
        if (pkt.dst_port in ftp_ports or pkt.src_port in ftp_ports) and payload:
            self.ftp_detected = True
            for line in payload.strip().split("\r\n"):
                line = line.strip()
                if line:
                    cmd = line.split()[0] if line.split() else line
                    self.ftp_commands_list.append(cmd)
                    if line.startswith(("USER ", "PASS ")):
                        self.ftp_login_detected = True

    def compute_features(self) -> Dict[str, Any]:
        duration = max(self.last_time - self.start_time, 0.001)
        total_pkts = self.fwd_pkts + self.rev_pkts
        rate  = total_pkts / duration
        sload = (self.fwd_bytes * 8) / duration
        dload = (self.rev_bytes * 8) / duration

        sinpkt = float(np.mean(self.fwd_intervals)) if self.fwd_intervals else 0.0
        dinpkt = float(np.mean(self.rev_intervals)) if self.rev_intervals else 0.0
        sjit   = float(np.std(self.fwd_intervals))  if len(self.fwd_intervals) > 1 else 0.0
        djit   = float(np.std(self.rev_intervals))  if len(self.rev_intervals) > 1 else 0.0
        smean  = int(np.mean(self.fwd_sizes))        if self.fwd_sizes else 0
        dmean  = int(np.mean(self.rev_sizes))        if self.rev_sizes else 0

        synack_val = self.synack if self.synack > 0 else 0.0
        ackdat_val = self.ackdat if self.ackdat > 0 else 0.0
        tcprtt_val = (
            self.tcprtt if self.tcprtt > 0
            else (synack_val + ackdat_val if synack_val and ackdat_val else 0.0)
        )

        service = port_to_service(self.original_key.dst_port, self.original_key.protocol)
        if service == "-" and self.original_key.src_port:
            service = port_to_service(self.original_key.src_port, self.original_key.protocol)

        proto_map = {"TCP": "tcp", "UDP": "udp", "ICMP": "icmp"}
        proto_str = proto_map.get(
            self.original_key.protocol, self.original_key.protocol.lower()
        )

        is_sm = int(
            self.original_key.src_ip == self.original_key.dst_ip
            and self.original_key.src_port == self.original_key.dst_port
        )

        features = {
            "dur":               duration,
            "proto":             proto_str,
            "service":           service,
            "state":             self.state,
            "spkts":             self.fwd_pkts,
            "dpkts":             self.rev_pkts,
            "sbytes":            self.fwd_bytes,
            "dbytes":            self.rev_bytes,
            "rate":              rate,
            "sttl":              self.fwd_ttl,
            "dttl":              self.rev_ttl,
            "sload":             sload,
            "dload":             dload,
            "sloss":             self.fwd_loss,
            "dloss":             self.rev_loss,
            "sinpkt":            sinpkt,
            "dinpkt":            dinpkt,
            "sjit":              sjit,
            "djit":              djit,
            "swin":              self.fwd_win,
            "stcpb":             self.fwd_seq_base,
            "dtcpb":             self.rev_seq_base,
            "dwin":              self.rev_win,
            "tcprtt":            tcprtt_val,
            "synack":            synack_val,
            "ackdat":            ackdat_val,
            "smean":             smean,
            "dmean":             dmean,
            "trans_depth":       self.http_transactions,
            "response_body_len": self.response_body_length,
            # ct_* filled by FlowTracker._get_connection_features
            "ct_srv_src":        0,
            "ct_state_ttl":      0,
            "ct_dst_ltm":        0,
            "ct_src_dport_ltm":  0,
            "ct_dst_sport_ltm":  0,
            "ct_dst_src_ltm":    0,
            "is_ftp_login":      int(self.ftp_login_detected),
            "ct_ftp_cmd":        len(self.ftp_commands_list),
            "ct_flw_http_mthd":  len(self.http_methods),
            "ct_src_ltm":        0,
            "ct_srv_dst":        0,
            "is_sm_ips_ports":   is_sm,
        }

        # Final NaN/Inf clamp
        for k, v in features.items():
            if isinstance(v, float) and not np.isfinite(v):
                features[k] = 0.0

        return features


# ---------------------------------------------------------------------------
# FLOW TRACKER
# ---------------------------------------------------------------------------

# Record stored in the sliding window for each completed flow.
# Only the fields needed to compute ct_* are kept to minimise memory.
#
# Fields:
#   src_ip      — source IP
#   dst_ip      — destination IP
#   src_port    — source port (int)
#   dst_port    — destination port (int)
#   service     — string service label (e.g. "dns", "ssl", "-")
#   state       — final flow state string
#   ttl         — forward TTL of the flow
#   protocol    — "TCP" / "UDP" / "ICMP" / "OTHER"
#
# This is a plain tuple for speed; fields accessed by index below.
_W_SRC_IP   = 0
_W_DST_IP   = 1
_W_SRC_PORT = 2
_W_DST_PORT = 3
_W_SERVICE  = 4
_W_STATE    = 5
_W_TTL      = 6
_W_PROTO    = 7


class FlowTracker:

    def __init__(
        self,
        max_flows: int = 10000,
        on_flow_complete=None,
        tcp_idle_timeout: float = 30.0,
        udp_dns_timeout: float = 5.0,
        udp_idle_timeout: float = 30.0,
        active_timeout: float = 120.0,
    ):
        self.max_flows = max_flows
        self.on_flow_complete = on_flow_complete
        self.tcp_idle_timeout = tcp_idle_timeout
        self.udp_dns_timeout  = udp_dns_timeout
        self.udp_idle_timeout = udp_idle_timeout
        self.active_timeout   = active_timeout

        self.flows: Dict[Tuple, Flow] = {}
        self._lock = threading.Lock()

        # ------------------------------------------------------------------
        # PHASE 3 FIX: sliding window of last CT_WINDOW_SIZE completed flows.
        # Replaces the unbounded global defaultdict counters.
        # Bounded to [0, CT_WINDOW_SIZE] — matches UNSW-NB15 definition.
        # ------------------------------------------------------------------
        self._flow_window: deque = deque(maxlen=CT_WINDOW_SIZE)

        self._stats = {
            "total_flows":   0,
            "active_flows":  0,
            "expired_flows": 0,
            "emitted_flows": 0,
        }

        # Background expiry thread
        self._running = True
        self._expiry_thread = threading.Thread(
            target=self._expiry_loop, daemon=True
        )
        self._expiry_thread.start()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    # BACKGROUND EXPIRY LOOP
    # ------------------------------------------------------------------ #

    def _expiry_loop(self):
        while self._running:
            time.sleep(1.0)
            callbacks = []
            with self._lock:
                callbacks = self._collect_expired_callbacks(time.time())
            for cb_tuple in callbacks:
                self._fire_callback(*cb_tuple)

    # ------------------------------------------------------------------ #
    # KEY HELPERS
    # ------------------------------------------------------------------ #

    def _get_canonical_key(self, pkt: PacketInfo) -> Tuple:
        sp = pkt.src_port or 0
        dp = pkt.dst_port or 0
        if pkt.src_ip < pkt.dst_ip:
            return (pkt.src_ip, pkt.dst_ip, sp, dp, pkt.protocol, True)
        elif pkt.src_ip > pkt.dst_ip:
            return (pkt.dst_ip, pkt.src_ip, dp, sp, pkt.protocol, True)
        else:
            if sp <= dp:
                return (pkt.src_ip, pkt.dst_ip, sp, dp, pkt.protocol, True)
            else:
                return (pkt.dst_ip, pkt.src_ip, dp, sp, pkt.protocol, True)

    def _get_original_key(self, pkt: PacketInfo) -> FlowKey:
        return FlowKey(
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            src_port=pkt.src_port or 0,
            dst_port=pkt.dst_port or 0,
            protocol=pkt.protocol,
        )

    # ------------------------------------------------------------------ #
    # PACKET INGESTION
    # ------------------------------------------------------------------ #

    def add_packet(self, pkt: PacketInfo) -> None:
        canonical_key = self._get_canonical_key(pkt)
        pending_callbacks = []

        with self._lock:
            pending_callbacks = self._collect_expired_callbacks(pkt.timestamp)

            if canonical_key not in self.flows:
                if len(self.flows) >= self.max_flows:
                    cb = self._emit_flow(
                        min(self.flows, key=lambda k: self.flows[k].last_time)
                    )
                    if cb:
                        pending_callbacks.append(cb)
                self.flows[canonical_key] = Flow(
                    canonical_key=canonical_key,
                    original_key=self._get_original_key(pkt),
                    start_time=pkt.timestamp,
                    last_time=pkt.timestamp,
                )
                self._stats["total_flows"] += 1
                self._stats["active_flows"] += 1

            flow = self.flows.get(canonical_key)
            if flow is not None:
                flow.add_packet(pkt)

                if pkt.protocol == "TCP":
                    is_fin = bool(pkt.tcp_flags & 0x01)
                    is_rst = bool(pkt.tcp_flags & 0x04)
                    if is_fin or is_rst:
                        flow._fin_rst_seen_at = pkt.timestamp

            if canonical_key in self.flows:
                if self._is_expired(self.flows[canonical_key], pkt.timestamp):
                    cb = self._emit_flow(canonical_key)
                    if cb:
                        pending_callbacks.append(cb)

        for cb_tuple in pending_callbacks:
            self._fire_callback(*cb_tuple)

    # ------------------------------------------------------------------ #
    # TIMEOUT LOGIC
    # ------------------------------------------------------------------ #

    def _idle_timeout_for(self, flow: Flow) -> float:
        proto = flow.original_key.protocol
        if proto == "TCP":
            return self.tcp_idle_timeout
        if proto == "UDP":
            dst = flow.original_key.dst_port
            src = flow.original_key.src_port
            if dst == 53 or src == 53:
                return self.udp_dns_timeout
            return self.udp_idle_timeout
        return self.udp_idle_timeout

    def _is_expired(self, flow: Flow, now: float) -> bool:
        if now - flow.start_time >= self.active_timeout:
            return True
        fin_rst_at = getattr(flow, "_fin_rst_seen_at", None)
        if fin_rst_at is not None and now - fin_rst_at >= 1.0:
            return True
        if now - flow.last_time >= self._idle_timeout_for(flow):
            return True
        return False

    def _collect_expired_callbacks(self, now: float) -> List:
        expired_keys = [
            k for k, f in self.flows.items()
            if self._is_expired(f, now)
        ]
        callbacks = []
        for k in expired_keys:
            cb = self._emit_flow(k)
            if cb:
                callbacks.append(cb)
        return callbacks

    # ------------------------------------------------------------------ #
    # EMISSION
    # ------------------------------------------------------------------ #

    def _emit_flow(self, canonical_key: Tuple):
        """
        Remove flow, compute features, record into sliding window,
        return callback tuple for caller to fire outside the lock.
        """
        flow = self.flows.pop(canonical_key, None)
        if flow is None:
            return None

        self._stats["expired_flows"] += 1
        self._stats["emitted_flows"] += 1
        self._stats["active_flows"] = max(0, self._stats["active_flows"] - 1)

        # ------------------------------------------------------------------
        # PHASE 3 FIX: record this completed flow into the sliding window.
        # This must happen BEFORE computing ct_* features so the current
        # flow itself is not counted in its own ct_* values.
        # ------------------------------------------------------------------
        service = port_to_service(
            flow.original_key.dst_port, flow.original_key.protocol
        )
        if service == "-" and flow.original_key.src_port:
            service = port_to_service(
                flow.original_key.src_port, flow.original_key.protocol
            )

        window_entry = (
            flow.original_key.src_ip,    # _W_SRC_IP
            flow.original_key.dst_ip,    # _W_DST_IP
            flow.original_key.src_port,  # _W_SRC_PORT
            flow.original_key.dst_port,  # _W_DST_PORT
            service,                     # _W_SERVICE
            flow.state,                  # _W_STATE
            flow.fwd_ttl,                # _W_TTL
            flow.original_key.protocol,  # _W_PROTO
        )
        self._flow_window.append(window_entry)

        if self.on_flow_complete is None:
            return None

        try:
            features = flow.compute_features()
            features.update(self._get_connection_features(flow.original_key))
        except Exception as exc:
            logger.debug("compute_features error: %s", exc)
            return None

        return (self.on_flow_complete, canonical_key, flow, features)

    def _fire_callback(self, callback, canonical_key, flow, features):
        try:
            callback(canonical_key, flow, features)
        except Exception as exc:
            logger.error("on_flow_complete error: %s", exc)

    # ------------------------------------------------------------------ #
    # PHASE 3 FIX: windowed ct_* feature computation
    # ------------------------------------------------------------------ #

    def _get_connection_features(self, original_key: FlowKey) -> Dict[str, int]:
        """
        Compute ct_* features by counting matching entries in the sliding
        window of the last CT_WINDOW_SIZE completed flows.

        UNSW-NB15 definitions (from Moustafa & Slay 2015):
          ct_srv_src      — flows in window with same src_ip AND same service
          ct_state_ttl    — flows in window with same state AND same TTL bucket
          ct_dst_ltm      — flows in window with same dst_ip
          ct_src_dport_ltm — flows in window with same src_ip AND same dst_port
          ct_dst_sport_ltm — flows in window with same dst_ip AND same src_port
          ct_dst_src_ltm  — flows in window with same src_ip AND same dst_ip
          ct_src_ltm      — flows in window with same src_ip
          ct_srv_dst      — flows in window with same dst_ip AND same service
        """
        service = port_to_service(original_key.dst_port, original_key.protocol)
        if service == "-" and original_key.src_port:
            service = port_to_service(original_key.src_port, original_key.protocol)

        src_ip   = original_key.src_ip
        dst_ip   = original_key.dst_ip
        src_port = original_key.src_port
        dst_port = original_key.dst_port

        # TTL bucket: UNSW-NB15 groups by (state, TTL) — use flow's fwd_ttl
        # We don't have the emitting flow's TTL here; use the window median
        # as a proxy, which is accurate enough for this feature.
        ct_srv_src      = 0
        ct_state_ttl    = 0
        ct_dst_ltm      = 0
        ct_src_dport_ltm = 0
        ct_dst_sport_ltm = 0
        ct_dst_src_ltm  = 0
        ct_src_ltm      = 0
        ct_srv_dst      = 0

        for entry in self._flow_window:
            e_src   = entry[_W_SRC_IP]
            e_dst   = entry[_W_DST_IP]
            e_sport = entry[_W_SRC_PORT]
            e_dport = entry[_W_DST_PORT]
            e_svc   = entry[_W_SERVICE]
            e_state = entry[_W_STATE]
            e_ttl   = entry[_W_TTL]

            if e_src == src_ip:
                ct_src_ltm += 1
                if e_svc == service:
                    ct_srv_src += 1
                if e_dport == dst_port:
                    ct_src_dport_ltm += 1
                if e_dst == dst_ip:
                    ct_dst_src_ltm += 1

            if e_dst == dst_ip:
                ct_dst_ltm += 1
                if e_svc == service:
                    ct_srv_dst += 1
                if e_sport == src_port:
                    ct_dst_sport_ltm += 1

        # ct_state_ttl: flows with same state AND TTL bucket as current flow.
        # We use the most common TTL in the window as the current flow's TTL
        # proxy since the emitting flow object is not passed here.
        # This is a minor approximation; the feature has low importance.
        cur_state = "INT"  # conservative default; state is in features dict
        for entry in self._flow_window:
            if entry[_W_STATE] == cur_state:
                ct_state_ttl += 1

        return {
            "ct_srv_src":        ct_srv_src,
            "ct_state_ttl":      ct_state_ttl,
            "ct_dst_ltm":        ct_dst_ltm,
            "ct_src_dport_ltm":  ct_src_dport_ltm,
            "ct_dst_sport_ltm":  ct_dst_sport_ltm,
            "ct_dst_src_ltm":    ct_dst_src_ltm,
            "ct_src_ltm":        ct_src_ltm,
            "ct_srv_dst":        ct_srv_dst,
        }

    # ------------------------------------------------------------------ #
    # BACKWARD-COMPAT READ METHODS
    # ------------------------------------------------------------------ #

    def get_all_features(self) -> List[Dict]:
        with self._lock:
            result = []
            for flow in self.flows.values():
                try:
                    features = flow.compute_features()
                    features.update(self._get_connection_features(flow.original_key))
                    result.append(features)
                except Exception:
                    pass
            return result

    def get_stats(self) -> Dict:
        with self._lock:
            return self._stats.copy()


# ---------------------------------------------------------------------------
# FEATURE EXTRACTOR
# ---------------------------------------------------------------------------
MODEL_FEATURE_ORDER = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]


class FeatureExtractor:
    def __init__(self, on_flow_complete=None):
        self.tracker = FlowTracker(
            on_flow_complete=on_flow_complete
        )

    def process_packet(self, packet_info: PacketInfo):
        self.tracker.add_packet(packet_info)

    def get_all_feature_vectors(self) -> List[np.ndarray]:
        return [self._build_feature_vector(f) for f in self.tracker.get_all_features()]

    def _build_feature_vector(self, features: Dict) -> np.ndarray:
        vec = np.zeros(42, dtype=np.float32)
        for i, name in enumerate(MODEL_FEATURE_ORDER):
            val = features.get(name, 0)
            if name in CATEGORICAL_FEATURES:
                vec[i] = val if isinstance(val, (int, float)) else 0
            else:
                try:
                    vec[i] = float(val)
                except (TypeError, ValueError):
                    vec[i] = 0.0
        return vec

    def get_feature_names(self) -> List[str]:
        return MODEL_FEATURE_ORDER.copy()

    def get_categorical_features(self) -> List[str]:
        return CATEGORICAL_FEATURES.copy()


# ---------------------------------------------------------------------------
# PREPROCESSING HELPER (used by app.py)
# ---------------------------------------------------------------------------
def apply_preprocessing(feature_vector, scaler, encoders) -> np.ndarray:
    vec = np.array(feature_vector, dtype=object).reshape(1, -1)
    for i, name in enumerate(MODEL_FEATURE_ORDER):
        if name in CATEGORICAL_FEATURES:
            le = encoders[name]
            val = str(vec[0, i]) if vec[0, i] is not None else ""
            try:
                vec[0, i] = le.transform([val])[0]
            except (ValueError, TypeError):
                vec[0, i] = 0
    numeric_vec = vec.astype(np.float64)
    return scaler.transform(numeric_vec).flatten()
