import threading
import time
from queue import Queue
from scapy.all import sniff, conf

class PacketSniffer:
    """Thin wrapper around scapy.sniff that pushes packets into a thread-safe queue."""
    def __init__(self, iface: str, bpf_filter: str | None = None, queue_size: int = 10000):
        self.iface = iface
        self.bpf_filter = bpf_filter
        self.q: Queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _cb(self, pkt):
        try:
            self.q.put_nowait(pkt)
        except Exception:
            # Drop if queue is full to avoid backpressure that blocks sniffing
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        conf.sniff_promisc = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        sniff(
            iface=self.iface,
            filter=self.bpf_filter,
            prn=self._cb,
            store=False,
            stop_filter=lambda _: self._stop.is_set(),
        )

    def get_queue(self) -> Queue:
        return self.q

    def stop(self):
        self._stop.set()
        time.sleep(0.2)
