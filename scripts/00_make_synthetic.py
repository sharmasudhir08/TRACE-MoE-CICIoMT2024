"""Create small synthetic CSVs with the exact CICIoMT2024 layout (45 features,
official file names, train/test files).

It exists only to test that every script runs end to end. The numbers mean nothing.
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
            "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
            "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
            "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
            "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
            "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
assert len(FEATURES) == 45

# class -> (protocol, intensity, number of train files)
CLASSES = {
    "Benign": ("mixed", 0.0, 1), "ARP_Spoofing": ("arp", 0.3, 1),
    "MQTT-DDoS-Connect_Flood": ("tcp", 1.6, 1), "MQTT-DDoS-Publish_Flood": ("tcp", 1.7, 1),
    "MQTT-DoS-Connect_Flood": ("tcp", 1.1, 1), "MQTT-DoS-Publish_Flood": ("tcp", 1.2, 1),
    "MQTT-Malformed_Data": ("tcp", 0.5, 1),
    "Recon-OS_Scan": ("tcp", 0.4, 1), "Recon-Ping_Sweep": ("icmp", 0.3, 1),
    "Recon-Port_Scan": ("tcp", 0.6, 1), "Recon-VulScan": ("tcp", 0.5, 1),
    "TCP_IP-DDoS-ICMP": ("icmp", 2.0, 2), "TCP_IP-DDoS-SYN": ("tcp", 2.0, 2),
    "TCP_IP-DDoS-TCP": ("tcp", 1.9, 2), "TCP_IP-DDoS-UDP": ("udp", 2.1, 2),
    "TCP_IP-DoS-ICMP": ("icmp", 1.4, 1), "TCP_IP-DoS-SYN": ("tcp", 1.4, 1),
    "TCP_IP-DoS-TCP": ("tcp", 1.3, 1), "TCP_IP-DoS-UDP": ("udp", 1.5, 1),
}
PROTO = {"icmp": 1, "tcp": 6, "udp": 17, "arp": 0}


def make_rows(name, proto, intensity, n, rng, shift=1.0):
    cls_rng = np.random.default_rng(zlib.crc32(name.encode()))
    sig = cls_rng.normal(0, 0.6, 45)                     # class signature
    z = np.zeros((n, 45))
    eps = rng.normal(0, 1, (n, 45))
    for t in range(1, n):                                  # AR(1): consecutive flows are related
        z[t] = 0.8 * z[t - 1] + 0.6 * eps[t]
    base = intensity + 0.7 * sig + 0.9 * z + rng.normal(0, 0.8, (n, 45))
    X = np.exp(np.clip(base, -6, 8)) * shift
    df = pd.DataFrame(X, columns=FEATURES)
    if proto == "mixed":
        p = rng.choice([6, 17, 1], size=n, p=[0.7, 0.25, 0.05]).astype(float)
    else:
        p = np.full(n, float(PROTO[proto]))
    df["Protocol Type"] = p
    for c in ["TCP", "UDP", "ICMP", "ARP"]:
        df[c] = 0.0
    df.loc[p == 6, "TCP"] = 1.0
    df.loc[p == 17, "UDP"] = 1.0
    df.loc[p == 1, "ICMP"] = 1.0
    df.loc[p == 0, "ARP"] = 1.0
    for c in ["fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
              "ece_flag_number", "cwr_flag_number", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
              "DHCP", "IGMP", "LLC"]:
        df[c] = (rng.random(n) < 1 / (1 + np.exp(-(sig[FEATURES.index(c)] * 2 + z[:, 0])))).astype(float)
    df["IPv"] = 1.0 - df["ARP"]
    df["LLC"] = 1.0 - df["ARP"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic"))
    ap.add_argument("--train-rows", type=int, default=1200)
    ap.add_argument("--test-rows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out) / "CICIoMT2024" / "WiFi_and_MQTT" / "attacks" / "CSV"
    for sub in ("train", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name, (proto, inten, n_files) in CLASSES.items():
        for k in range(1, n_files + 1):
            suffix = str(k) if n_files > 1 else ""
            tr = make_rows(name, proto, inten, a.train_rows // n_files, rng)
            if name in ("ARP_Spoofing", "Recon-Ping_Sweep"):             # some exact duplicates
                tr = pd.concat([tr, tr.sample(frac=0.05, random_state=1)], ignore_index=True)
            tr.to_csv(root / "train" / f"{name}{suffix}_train.pcap.csv", index=False)
        te = make_rows(name, proto, inten, a.test_rows, rng)
        if name == "Recon-Ping_Sweep":                                    # train rows leaking into test
            te = pd.concat([te, tr.head(20)], ignore_index=True)
        te.to_csv(root / "test" / f"{name}{'1' if n_files > 1 else ''}_test.pcap.csv", index=False)

    print(f"Synthetic CICIoMT2024 CSVs in {root}")


if __name__ == "__main__":
    main()
