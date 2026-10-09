"""
Packet Capture Module for ARGUS V2
Safe passive packet observation using Scapy.

FIX: ttl was referenced before assignment for ICMP/OTHER/ARP packets,
     causing UnboundLocalError caught silently, dropping those packets.
     ttl is now read from ip_layer.ttl before the protocol branch.
     payload_length is defaulted to 0 before the branch as well.
"""
import threading
import time
import platform
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    sniff = IP = TCP = UDP = ICMP = Raw = conf = None


@dataclass
class PacketMetadata:
    """Container for extracted packet metadata."""
    timestamp: str
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: Optional[int]
    dst_port: Optional[int]
    length: int
    tcp_flags: int = 0
    tcp_seq: int = 0
    tcp_ack: int = 0
    tcp_window: int = 0
    ttl: int = 64
    payload_length: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class PacketCapture:
    """
    Thread-safe packet capture using Scapy.
    Designed for passive IDS observation only.
    """

    def __init__(
        self,
        iface: Optional[str] = None,
        packet_callback: Optional[Callable] = None,
    ):
        if not SCAPY_AVAILABLE:
            raise RuntimeError(
                "Scapy not installed. Run: pip install scapy"
            )

        self.iface = iface
        self.packet_callback = packet_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._captured_packets: List[PacketMetadata] = []
        self._max_packets = 1000
        self._error_message: Optional[str] = None

        conf.sniff_promisc = False

        if platform.system() == "Windows":
            try:
                conf.use_pcap = True
            except Exception:
                pass

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error_message

    def _extract_metadata(self, packet) -> Optional[PacketMetadata]:
        """Extract metadata from a Scapy packet."""
        try:
            if not packet.haslayer(IP):
                return None

            ip_layer = packet[IP]

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            length = len(packet)

            # FIX: read ttl from IP layer BEFORE the protocol branch
            # so it is always defined regardless of protocol.
            ttl = ip_layer.ttl

            # FIX: default payload_length before branch so ICMP/OTHER
            # never cause an UnboundLocalError.
            payload_length = 0

            protocol = "UNKNOWN"
            src_port: Optional[int] = None
            dst_port: Optional[int] = None
            tcp_flags = 0
            tcp_seq = 0
            tcp_ack = 0
            tcp_window = 0

            if packet.haslayer(TCP):
                protocol = "TCP"
                tcp_layer = packet[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                tcp_flags = int(tcp_layer.flags)
                tcp_seq = tcp_layer.seq
                tcp_ack = tcp_layer.ack
                tcp_window = tcp_layer.window
                payload_length = (
                    len(tcp_layer.payload) if tcp_layer.payload else 0
                )

            elif packet.haslayer(UDP):
                protocol = "UDP"
                udp_layer = packet[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                payload_length = (
                    len(udp_layer.payload) if udp_layer.payload else 0
                )

            elif packet.haslayer(ICMP):
                protocol = "ICMP"
                # No ports for ICMP; payload_length stays 0

            else:
                protocol = "OTHER"
                # payload_length stays 0

            return PacketMetadata(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=protocol,
                src_port=src_port,
                dst_port=dst_port,
                length=length,
                tcp_flags=tcp_flags,
                tcp_seq=tcp_seq,
                tcp_ack=tcp_ack,
                tcp_window=tcp_window,
                ttl=ttl,
                payload_length=payload_length,
            )

        except Exception as e:
            logging.debug(f"Failed to extract metadata: {e}")
            return None

    def _packet_handler(self, packet):
        """Internal packet handler called by Scapy sniff."""
        metadata = self._extract_metadata(packet)
        if metadata is None:
            return

        with self._lock:
            self._captured_packets.append(metadata)
            if len(self._captured_packets) > self._max_packets:
                self._captured_packets.pop(0)

        if self.packet_callback:
            try:
                self.packet_callback(metadata)
            except Exception as e:
                logging.error(f"Packet callback error: {e}")

    def start(self) -> bool:
        """Start packet capture in background thread."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._captured_packets.clear()
            self._error_message = None

        self._thread = threading.Thread(
            target=self._sniff_loop, daemon=True
        )
        self._thread.start()
        return True

    def _sniff_loop(self):
        """Background sniffing loop."""
        try:
            sniff(
                iface=self.iface,
                prn=self._packet_handler,
                store=False,
                stop_filter=lambda _: not self._running,
                timeout=None,
            )
        except PermissionError:
            error_msg = (
                "Packet capture requires administrator/root privileges"
            )
            logging.error(error_msg)
            with self._lock:
                self._running = False
                self._error_message = error_msg
        except Exception as e:
            error_msg = f"Packet capture error: {e}"
            logging.error(error_msg)
            with self._lock:
                self._running = False
                self._error_message = str(e)

    def stop(self) -> bool:
        """Stop packet capture."""
        with self._lock:
            if not self._running:
                return False
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return True

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_captured_packets(
        self, limit: Optional[int] = None
    ) -> List[Dict]:
        with self._lock:
            packets = self._captured_packets.copy()
        if limit:
            packets = packets[-limit:]
        return [p.to_dict() for p in packets]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "running": self._running,
                "packets_captured": len(self._captured_packets),
                "interface": self.iface or "default",
                "error": self._error_message,
            }


def list_interfaces() -> List[str]:
    if not SCAPY_AVAILABLE:
        return []
    try:
        from scapy.arch.windows import get_windows_if_list
        interfaces = get_windows_if_list()
        return [i["name"] for i in interfaces if i.get("ips")]
    except Exception:
        try:
            from scapy.arch import get_if_list
            return get_if_list()
        except Exception:
            return []


if __name__ == "__main__":
    print(f"Scapy available: {SCAPY_AVAILABLE}")
    print(f"Platform: {platform.system()}")
    if SCAPY_AVAILABLE:
        print("Interfaces:", list_interfaces())
