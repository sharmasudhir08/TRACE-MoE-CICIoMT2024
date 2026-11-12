"""Step 12 (Digital Twin, task 20; added 2026-11-04). Who carries the traffic of each capture?

For every PCAP under a folder: the share of packets that involve each known device or infrastructure MAC
(configs/twin_devices.json) and the most frequent unknown MACs. MACs are read from the raw Ethernet header,
so malformed frames cannot stop the scan. Results are saved after every capture; a re-run continues where
it stopped (use --restart to scan everything again). This decides how much attack traffic a
per-device twin can see (TWIN_PROTOCOL.md, gate G0). Reads PCAPs only; no model, no training.

  python scripts/12_pcap_scan.py --pcaps data/raw/CICIoMT2024/WiFi_and_MQTT/attacks/PCAP --out results_rerun/twin
  python scripts/12_pcap_scan.py --inventory data/raw/CICIoMT2024/WiFi_and_MQTT/profiling/PCAP
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import dpkt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.pcapfeat import dominant_mac  # noqa: E402

BROADCAST = "ff:ff:ff:ff:ff:ff"


def mac_shares(path, names, top_unknown=5):
    counts, n = collections.Counter(), 0
    with open(path, "rb") as fh:
        for _, buf in dpkt.pcap.Reader(fh):
            n += 1
            # MACs straight from the Ethernet header: no payload decoding, so malformed frames can't stop the scan
            for m in {buf[0:6].hex(":"), buf[6:12].hex(":")}:
                counts[m] += 1
    row = {"file": path.name, "split": path.parent.name, "packets": n}
    for name, mac in names.items():
        row[name] = counts.pop(mac, 0) / max(n, 1)
    counts.pop(BROADCAST, None)
    row["top_unknown"] = "; ".join(f"{m} {c / n:.1%}" for m, c in counts.most_common(top_unknown))
    return row


def save(rows, cfg, target):
    df = pd.DataFrame(rows)
    df["profiled_devices"] = df[list(cfg["devices"])].max(axis=1)
    df.to_csv(target, index=False)


def inventory(profiling, exclude):
    found = collections.defaultdict(set)
    for f in sorted(list(profiling.glob("Power/*.pcap")) + list(profiling.glob("Interactions/**/*.pcap"))):
        mac, share = dominant_mac(f, exclude=exclude)
        found[mac].add((f.name, round(share, 3)))
        print(f"{f.name:45s} {mac}  {share:.1%}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", type=Path, help="folder with PCAPs (searched recursively)")
    ap.add_argument("--inventory", type=Path, help="profiling/PCAP folder: print the dominant MAC per capture")
    ap.add_argument("--devices", type=Path, default=ROOT / "configs" / "twin_devices.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results_rerun" / "twin")
    ap.add_argument("--restart", action="store_true", help="ignore an existing pcap_mac_shares.csv and scan everything")
    a = ap.parse_args()
    cfg = json.loads(a.devices.read_text())
    names = {**cfg["devices"], **cfg["infrastructure"]}
    if a.inventory:
        inventory(a.inventory, exclude={cfg["infrastructure"]["router"], BROADCAST})
    if a.pcaps:
        files = sorted(a.pcaps.rglob("*.pcap"))
        if not files:
            raise SystemExit(f"[PCAP_SCAN_NO_FILES] no .pcap under {a.pcaps}; check the --pcaps path")
        a.out.mkdir(parents=True, exist_ok=True)
        target = a.out / "pcap_mac_shares.csv"
        rows = pd.read_csv(target).to_dict("records") if (target.exists() and not a.restart) else []
        done = {(r["split"], r["file"]) for r in rows}
        for f in files:
            if (f.parent.name, f.name) in done:
                continue
            rows.append(mac_shares(f, names))
            print(f"{f.name}: {rows[-1]['packets']:,} packets", flush=True)
            save(rows, cfg, target)                  # after every capture, so a stop loses at most one file
        save(rows, cfg, target)
        print(f"-> {target} ({len(rows)} captures)")


if __name__ == "__main__":
    main()
