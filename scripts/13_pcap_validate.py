"""Step 13 (Digital Twin, gate G1; added 2026-11-04). Does our extractor reproduce CIC's CSVs?

Pairs every <name>.pcap with <name>.pcap.csv, recomputes the rows in CIC mode (trace_iomt.pcapfeat) and
reports, per file and per feature, the share of rows that match CIC's values. Reads data only.

  python scripts/13_pcap_validate.py --pcaps data/raw/CICIoMT2024/WiFi_and_MQTT/profiling/PCAP \
                                     --csvs  data/raw/CICIoMT2024/WiFi_and_MQTT/profiling/CSV
"""
import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt import pcapcheck, pcapfeat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", type=Path, required=True)
    ap.add_argument("--csvs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results_rerun" / "twin")
    ap.add_argument("--max-mb", type=float, default=None, help="skip PCAPs larger than this (quick checks)")
    a = ap.parse_args()
    csvs = {p.name[:-len(".pcap.csv")]: p for p in a.csvs.rglob("*.pcap.csv")}
    rows = []
    for pcap in sorted(a.pcaps.rglob("*.pcap")):
        csv_path = csvs.get(pcap.stem)
        if csv_path is None:
            print(f"[PCAP_VALIDATE_NO_CSV] {pcap.name}: no matching {pcap.stem}.pcap.csv under {a.csvs}; skipped")
            continue
        if a.max_mb and pcap.stat().st_size > a.max_mb * 1e6:
            continue
        official = pcapcheck.read_csv(csv_path)
        stats = collections.Counter()
        mine = pcapfeat.cic_rows(pcap, stats=stats)
        pairs, unplaced = pcapcheck.align(mine, official)
        ag = pcapcheck.agreement(pairs, pcapfeat.FEATURES)
        rows.append({"file": pcap.stem, "csv_rows": len(official), "our_rows": len(mine), "unplaced": unplaced,
                     "malformed_frames": stats["malformed"], **ag})
        worst = min(ag, key=ag.get) if ag else "-"
        print(f"{pcap.stem:40s} rows {len(official):8,} / {len(mine):8,}  unplaced {unplaced:6,}  "
              f"worst {worst} {ag.get(worst, 0):.3%}  malformed {stats['malformed']:,}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "pcap_validation.csv", index=False)
    feats = [c for c in pcapfeat.FEATURES if c in df]
    weights = df["csv_rows"] - df["unplaced"]
    print("\nRow-weighted agreement per feature:")
    for c in feats:
        print(f"  {c:18s} {(df[c] * weights).sum() / weights.sum():8.3%}")
    print(f"-> {a.out / 'pcap_validation.csv'}")


if __name__ == "__main__":
    main()
