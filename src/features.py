from __future__ import annotations
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Deque, Any
import numpy as np

try:
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
except Exception:
    IP = TCP = UDP = ICMP = IPv6 = object  # type: ignore

@dataclass
class SecondBucket:
    ts: int
    total_packets: int = 0
    total_bytes: int = 0
    tcp: int = 0
    udp: int = 0
    icmp: int = 0
    other: int = 0
    tcp_flags: Counter = field(default_factory=Counter)
    src_ips: Counter = field(default_factory=Counter)
    dst_ips: Counter = field(default_factory=Counter)
    src_ports: Counter = field(default_factory=Counter)
    dst_ports: Counter = field(default_factory=Counter)

class OnlineFeatureExtractor:
    """Maintains per-second buckets and exposes multi-window features at 1s cadence."""
    def __init__(self, windows: dict[str, int], resolution_sec: int = 1, max_retention_sec: int = 900):
        self.windows = windows
        self.resolution = max(1, int(resolution_sec))
        self.max_retention = max_retention_sec
        self._buckets: Deque[SecondBucket] = deque()
        self._current_second = int(time.time())
        self._current_bucket = SecondBucket(ts=self._current_second)

    def _finalize_second(self, sec: int):
        self._buckets.append(self._current_bucket)
        self._current_bucket = SecondBucket(ts=sec)
        cutoff = sec - self.max_retention
        while self._buckets and self._buckets[0].ts < cutoff:
            self._buckets.popleft()

    def on_packet(self, pkt: Any):
        now = int(time.time())
        if now > self._current_second:
            for s in range(self._current_second + 1, now + 1):
                self._finalize_second(s)
            self._current_second = now
        b = self._current_bucket
        b.total_packets += 1
        b.total_bytes += len(pkt)
        if hasattr(pkt, "haslayer") and pkt.haslayer(TCP):
            b.tcp += 1
            flags = int(pkt[TCP].flags)
            if flags & 0x02: b.tcp_flags['SYN'] += 1
            if flags & 0x10: b.tcp_flags['ACK'] += 1
            if flags & 0x04: b.tcp_flags['RST'] += 1
            if flags & 0x01: b.tcp_flags['FIN'] += 1
        elif hasattr(pkt, "haslayer") and pkt.haslayer(UDP):
            b.udp += 1
        elif hasattr(pkt, "haslayer") and pkt.haslayer(ICMP):
            b.icmp += 1
        else:
            b.other += 1
        if hasattr(pkt, "haslayer") and pkt.haslayer(IP):
            b.src_ips[pkt[IP].src] += 1
            b.dst_ips[pkt[IP].dst] += 1
        elif hasattr(pkt, "haslayer") and pkt.haslayer(IPv6):
            b.src_ips[pkt[IPv6].src] += 1
            b.dst_ips[pkt[IPv6].dst] += 1
        if hasattr(pkt, "haslayer") and pkt.haslayer(TCP):
            b.src_ports[pkt[TCP].sport] += 1
            b.dst_ports[pkt[TCP].dport] += 1
        elif hasattr(pkt, "haslayer") and pkt.haslayer(UDP):
            b.src_ports[pkt[UDP].sport] += 1
            b.dst_ports[pkt[UDP].dport] += 1

    def _agg_window(self, seconds: int) -> dict[str, float]:
        if seconds <= 0:
            return {}
        now = self._current_second
        start = now - seconds + 1
        buckets = [b for b in self._buckets if start <= b.ts <= now]
        if self._current_bucket.ts >= start:
            buckets.append(self._current_bucket)
        if not buckets:
            return {}
        tp = sum(b.total_packets for b in buckets)
        tb = sum(b.total_bytes for b in buckets)
        tcp = sum(b.tcp for b in buckets)
        udp = sum(b.udp for b in buckets)
        icmp = sum(b.icmp for b in buckets)
        other = sum(b.other for b in buckets)
        seconds_count = max(1, len(buckets))
        pps = tp / seconds_count
        bps = tb / seconds_count

        def entropy(counter: Counter) -> float:
            n = sum(counter.values())
            if n == 0: return 0.0
            import numpy as _np
            probs = _np.array(list(counter.values()), dtype=float) / n
            return float(-_np.sum(_np.where(probs>0, probs*_np.log2(probs), 0.0)))
        from collections import Counter as _C
        src_ips = _C(); dst_ips = _C(); src_ports = _C(); dst_ports = _C(); flags = _C()
        for b in buckets:
            src_ips.update(b.src_ips)
            dst_ips.update(b.dst_ips)
            src_ports.update(b.src_ports)
            dst_ports.update(b.dst_ports)
            flags.update(b.tcp_flags)
        features = {
            f"{seconds}s_pps": pps,
            f"{seconds}s_bps": bps,
            f"{seconds}s_tcp_rate": tcp/seconds_count,
            f"{seconds}s_udp_rate": udp/seconds_count,
            f"{seconds}s_icmp_rate": icmp/seconds_count,
            f"{seconds}s_other_rate": other/seconds_count,
            f"{seconds}s_srcip_unique": float(len(src_ips)),
            f"{seconds}s_dstip_unique": float(len(dst_ips)),
            f"{seconds}s_srcport_unique": float(len(src_ports)),
            f"{seconds}s_dstport_unique": float(len(dst_ports)),
            f"{seconds}s_srcip_entropy": entropy(src_ips),
            f"{seconds}s_dstip_entropy": entropy(dst_ips),
            f"{seconds}s_srcport_entropy": entropy(src_ports),
            f"{seconds}s_dstport_entropy": entropy(dst_ports),
            f"{seconds}s_syn_rate": flags.get('SYN',0)/seconds_count,
            f"{seconds}s_rst_rate": flags.get('RST',0)/seconds_count,
            f"{seconds}s_fin_rate": flags.get('FIN',0)/seconds_count,
            f"{seconds}s_ack_rate": flags.get('ACK',0)/seconds_count,
        }
        return features

    def build_feature_row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for _, secs in self.windows.items():
            row.update(self._agg_window(int(secs)))
        row["ts"] = float(self._current_second)
        return row
