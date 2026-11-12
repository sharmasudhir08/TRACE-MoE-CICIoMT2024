"""Labels and the three-level hierarchy: label19 (attack type) -> label6 (category) -> label2 (Benign/Attack).

CICIoMT2024 CSVs have no label column; the label comes from the file name, e.g.
TCP_IP-DDoS-ICMP1_train.pcap.csv -> ("TCP_IP-DDoS-ICMP", "train"). The trailing number is a capture part.
"""
import re

LEVELS = ["label2", "label6", "label19"]

CICIOMT_CLASSES = [
    "ARP_Spoofing", "Benign",
    "MQTT-DDoS-Connect_Flood", "MQTT-DDoS-Publish_Flood", "MQTT-DoS-Connect_Flood", "MQTT-DoS-Publish_Flood",
    "MQTT-Malformed_Data",
    "Recon-OS_Scan", "Recon-Ping_Sweep", "Recon-Port_Scan", "Recon-VulScan",
    "TCP_IP-DDoS-ICMP", "TCP_IP-DDoS-SYN", "TCP_IP-DDoS-TCP", "TCP_IP-DDoS-UDP",
    "TCP_IP-DoS-ICMP", "TCP_IP-DoS-SYN", "TCP_IP-DoS-TCP", "TCP_IP-DoS-UDP",
]
# Order as in METHODOLOGY.md and the expert-by-category figure; Benign must be index 0 at every coarse level
# because attacks.py identifies benign classes by parent index 0.
CATEGORIES = ["Benign", "Spoofing", "Recon", "MQTT", "DoS", "DDoS"]
BINARY = ["Benign", "Attack"]

_FILE_RE = re.compile(r"^(?P<label>.+?)\d*_(?P<split>train|test)\.pcap\.(?:csv|xlsx)$", re.IGNORECASE)


def ciciomt_label_from_filename(name):
    """('TCP_IP-DDoS-ICMP', 'train') from 'TCP_IP-DDoS-ICMP1_train.pcap.csv'; None if the name does not match."""
    m = _FILE_RE.match(name)
    if not m:
        return None
    return m.group("label"), m.group("split").lower()


def ciciomt_category(label19):
    if label19 == "Benign":
        return "Benign"
    if "Spoofing" in label19:
        return "Spoofing"
    if label19.startswith("Recon"):
        return "Recon"
    if label19.startswith("MQTT"):
        return "MQTT"
    if label19.startswith("TCP_IP-DDoS"):
        return "DDoS"
    if label19.startswith("TCP_IP-DoS"):
        return "DoS"
    raise ValueError(f"Unknown CICIoMT2024 class '{label19}': add it to ciciomt_category()")


def binary_label(label6):
    return "Benign" if label6 == "Benign" else "Attack"


class Hierarchy:
    """Class names per level and the parent of every fine class (indices into the coarser level)."""

    def __init__(self, classes19=None):
        names = sorted(set(CICIOMT_CLASSES) | set(classes19 or []))
        self.classes = {"label19": names, "label6": list(CATEGORIES), "label2": list(BINARY)}
        self.parent19 = [CATEGORIES.index(ciciomt_category(c)) for c in names]
        self.parent6 = [BINARY.index(binary_label(c)) for c in CATEGORIES]

    def n(self, level):
        return len(self.classes[level])

    def index(self, level):
        return {c: i for i, c in enumerate(self.classes[level])}

    def to_dict(self):
        return {"classes": self.classes, "parent19": self.parent19, "parent6": self.parent6}
