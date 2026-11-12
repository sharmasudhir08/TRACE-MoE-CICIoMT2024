"""Step 14 (Digital Twin, task 20; added 2026-11-04). Extract packet-window rows from every PCAP.

Modes (trace_iomt.pcapfeat):
  device  windows of 10 packets per endpoint, for the twin experiments: the twinned endpoints of
          configs/twin_devices.json, or with --all-hosts every host except the router;
          every row records n_attacker_packets, so attack labels can be given by file (option a) or by
          attacker presence (option b) later without re-extracting;
  cic     CIC's own row definition (10 MB pieces, windows of 10 packets of the whole capture).
Captures are found under the CICIoMT2024 Wi-Fi/MQTT folder:
  attacks/PCAP/{train,test}/<Class><n>_<split>.pcap   -> label from the file name
  profiling/PCAP/<State>/.../<name>.pcap               -> Benign, state = Power/Idle/Active/Interactions/Broker
One Parquet file per capture; existing files are skipped unless --force, so the step can be restarted.
Files are processed in parallel (--workers). Reads PCAPs only; no model, no training.

  python scripts/14_pcap_extract.py --root data/raw/CICIoMT2024/WiFi_and_MQTT --mode device --workers 12
"""
import argparse
import collections
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt import pcapfeat  # noqa: E402
from trace_iomt.data.labels import ciciomt_label_from_filename  # noqa: E402
from trace_iomt.utils import code_hash, now_iso  # noqa: E402


def describe(pcap, root):
    """split, label19 and state of a capture from its place in the dataset folder."""
    rel = pcap.relative_to(root)
    if rel.parts[0] == "profiling":
        return {"split": "profiling", "label19": "Benign", "state": rel.parts[2]}
    parsed = ciciomt_label_from_filename(pcap.name + ".csv")
    if parsed is None:
        raise ValueError(f"[PCAP_EXTRACT_NAME] {pcap}: name does not look like <Class><n>_<train|test>.pcap; "
                         f"rename it or move it out of {root}")
    label, split = parsed
    return {"split": split, "label19": label, "state": ""}


def endpoints(cfg, with_extra):
    eps = dict(cfg["devices"])
    if with_extra:
        eps.update({k: v for k, v in cfg["extra_endpoints"].items() if not k.startswith("_")})
    return eps


def extract_one(pcap, root, out_dir, mode, eps, attackers, all_hosts=False, exclude=()):
    meta = describe(pcap, root)
    t0, stats = time.time(), collections.Counter()
    rows = (pcapfeat.device_rows(pcap, eps, attackers, stats, all_hosts=all_hosts, exclude=exclude)
            if mode == "device" else pcapfeat.cic_rows(pcap, stats=stats))
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    df["file"] = pcap.stem
    target = out_dir / meta["split"] / f"{pcap.stem}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(target)                      # a stopped run never leaves a half-written file behind
    return {"file": pcap.stem, **meta, "rows": len(df), "malformed_frames": stats["malformed"],
            "seconds": round(time.time() - t0, 1),
            "rows_per_endpoint": df["device"].value_counts().to_dict() if mode == "device" and len(df) else {}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the WiFi_and_MQTT folder of CICIoMT2024")
    ap.add_argument("--mode", choices=["device", "cic"], default="device")
    ap.add_argument("--devices", type=Path, default=ROOT / "configs" / "twin_devices.json")
    ap.add_argument("--with-extra", action="store_true", help="also twin the extra endpoints (broker, unprofiled)")
    ap.add_argument("--all-hosts", action="store_true",
                    help="rows for every host on the network (except the router); hosts outside the twin are named mac:<address>")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "twin")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.devices.read_text())
    eps = endpoints(cfg, a.with_extra)
    attackers = {v for k, v in cfg["attackers"].items() if not k.startswith("_")}
    exclude = {cfg["infrastructure"]["router"]}
    out_dir = a.out / a.mode
    pcaps = sorted(a.root.glob("attacks/PCAP/*/*.pcap")) + sorted(a.root.glob("profiling/PCAP/**/*.pcap"))
    if not pcaps:
        raise SystemExit(f"[PCAP_EXTRACT_NO_FILES] no PCAPs under {a.root}/attacks/PCAP or {a.root}/profiling/PCAP")
    todo = [p for p in pcaps if a.force or not (out_dir / describe(p, a.root)["split"] / f"{p.stem}.parquet").exists()]
    scope = "all hosts" if a.all_hosts else f"{len(eps)} endpoints"
    print(f"{len(pcaps)} captures, {len(todo)} to extract, mode {a.mode}, {scope} ({len(eps)} twinned), {a.workers} workers")
    done = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(extract_one, p, a.root, out_dir, a.mode, eps, attackers, a.all_hosts, exclude): p
                   for p in todo}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as exc:        # report and continue: one bad capture must not stop the others
                print(f"[PCAP_EXTRACT_FAILED] {futures[f].name}: {exc}. Re-run this file alone to see the traceback.")
                continue
            done.append(r)
            note = f"  [PCAP_MALFORMED] {r['malformed_frames']:,} undecodable IPv4/ARP frames skipped" if r["malformed_frames"] else ""
            print(f"{r['file']:45s} {r['split']:9s} {r['rows']:9,} rows  {r['seconds']:7.1f}s{note}", flush=True)
    manifest = {"created_at": now_iso(), "code_sha256": code_hash(), "mode": a.mode, "endpoints": eps,
                "all_hosts": a.all_hosts, "excluded": sorted(exclude),
                "attackers": sorted(attackers), "files": done}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(done)} files written under {out_dir}")


if __name__ == "__main__":
    main()
