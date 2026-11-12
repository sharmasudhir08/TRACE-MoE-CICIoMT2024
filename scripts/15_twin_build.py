"""Step 15 (Digital Twin; added 2026-11-04). Fit the twin and build the twin windows: a dry run before training.

Reads the device-mode Parquet files of step 14, fits the profiles (benign training rows only) and the state
estimator (profiling captures), builds the windows of one twin mode with one label rule, and writes:
  results_rerun/twin/build_<mode>_<rule>.json   windows per split and class, profile levels, state-estimator
                                                 accuracy on held-out time, Δ of benign vs attack test windows
  results_rerun/twin/twin_model.pkl              fitted profiles + state estimator (reused by training)
No detector is trained here.

  python scripts/15_twin_build.py --config configs/rerun.yaml --mode device_state --labels file
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from _common import setup
from trace_iomt import twin_data

ROOT = Path(__file__).resolve().parents[1]


def add(p):
    p.add_argument("--rows", type=Path, default=ROOT / "data" / "processed" / "twin" / "device")
    p.add_argument("--devices", type=Path, default=ROOT / "configs" / "twin_devices.json")
    p.add_argument("--mode", choices=twin_data.MODES, default="device_state")
    p.add_argument("--labels", choices=["file", "attacker"], default="file")
    p.add_argument("--cap", type=int, default=twin_data.DEFAULT_CAP, help="max consecutive rows per (capture, device)")
    p.add_argument("--min-support", type=int, default=200)
    p.add_argument("--shrinkage", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)


def delta_summary(wd, twinned_codes):
    """Mean of the window Δ (mean z over features) on test windows, benign vs attack, twinned devices only."""
    starts = wd.starts["test"]
    if len(starts) == 0:
        return {}
    d = twin_data.window_delta(wd, starts)
    z = d[:, :len(twin_data.CTX_NAMES) - 3].mean(axis=1)
    last = starts + wd.T - 1
    twinned = np.isin(wd.ctx[last, -1], twinned_codes)
    benign = wd.y["label2"][last] == wd.hierarchy.index("label2")["Benign"]
    out = {}
    for name, mask in (("benign", benign & twinned), ("attack", ~benign & twinned)):
        out[name] = {"windows": int(mask.sum()), "mean_delta": float(z[mask].mean()) if mask.any() else None}
    return out


def main():
    args, cfg, log = setup("Twin build (dry run)", "15_twin_build", add)
    t0 = time.time()
    twinned = twin_data.twinned_devices(args.devices)
    rows = twin_data.apply_labels(twin_data.load_attack_rows(args.rows, args.cap, args.seed, log), args.labels)
    profiling = twin_data.load_profiling_rows(args.rows)
    log.info(f"attack rows {len(rows):,} (cap {args.cap:,} per capture|device), profiling rows {len(profiling):,}")
    val_tail = float(cfg["data"].get("val_tail_fraction", 0.15))
    profiles, estimator, fit_report = twin_data.fit_twin(profiling, rows, twinned, val_tail, args.min_support,
                                                         args.shrinkage, args.seed, log)
    wd = twin_data.build_twin_windowed(cfg, rows, profiles, estimator, args.mode, args.seed, log)
    out = Path(cfg["paths"]["results_dir"]) / "twin"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "twin_model.pkl", "wb") as fh:
        pickle.dump({"profiles": profiles, "estimator": estimator, "twinned": sorted(twinned)}, fh)
    classes = wd.hierarchy.classes["label19"]
    per_class = {s: {classes[k]: int(v) for k, v in zip(*np.unique(wd.labels(s, "label19"), return_counts=True))}
                 for s in wd.starts}
    levels = {name: int((wd.ctx[:, -1] == code).sum()) for name, code in twin_data.LEVEL_CODE.items()}
    report = {"mode": args.mode, "labels": args.labels, "cap": args.cap, "seed": args.seed, **fit_report,
              "rows": int(len(rows)), "windows": wd.report["windows"], "windows_per_class": per_class,
              "row_profile_levels": levels, "test_delta": delta_summary(wd, [0, 1]),
              "seconds": round(time.time() - t0, 1)}
    path = out / f"build_{args.mode}_{args.labels}.json"
    path.write_text(json.dumps(report, indent=2))
    log.info(f"windows {wd.report['windows']}; profile levels per row {levels}; test Δ {report['test_delta']}")
    log.info(f"-> {path} and {out / 'twin_model.pkl'} ({report['seconds']}s)")


if __name__ == "__main__":
    main()
