"""Step 11 (post hoc, added 2026-11-03). Corrected abstention, hierarchy consistency, tuned binary thresholds.

No model is trained. For the newest copy of every run in results/main/:
  - neural models: model.pt is reloaded and validation and test windows are scored again, keeping the
    float32 19-class logits. Confidences are then computed in float64 log space, which does not saturate
    at 1.0 (the cause of the AURC tie artefact). Predicted labels are checked against predictions.npz.
  - classical models (their model is not saved): the saved probabilities (float16) are used.
For all models, AURC now treats tied confidences in expectation (abstain.risk_coverage).

Writes <run>/posthoc_v2.json (it refuses to overwrite one unless --force) and
results/tables/class_window_counts_<regime>.csv. Step 09 then uses posthoc_v2.json in place of the
abstain block of metrics.json, and step 10 the corrected values for H9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import setup
from trace_iomt.compare import latest_runs, load_runs
from trace_iomt.data.windows import build_windowed, class_counts
from trace_iomt.experiment import audit_drop_features, load_prepared
from trace_iomt.models import build_model
from trace_iomt.posthoc import analyse_run, classical_logits
from trace_iomt.train import predict
from trace_iomt.utils import resolve_device, save_json, set_seed

MIN_AGREEMENT = 0.999   # re-inference vs saved predictions; below this the reloaded model is not the saved one


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--regimes", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing posthoc_v2.json files")


def _selected(cfg, args):
    runs = [r for r in latest_runs(load_runs(cfg["paths"]["results_dir"])) if r.get("experiment") == "main"]
    keep = lambda r, key, allowed: allowed is None or r.get(key) in allowed
    runs = [r for r in runs if keep(r, "model", args.models) and keep(r, "regime", args.regimes)
            and keep(r, "seed", args.seeds)]
    return sorted(runs, key=lambda r: (r["regime"], r["seed"], r["family"], r["model"]))


def _neural_logits(r, wd, cfg, device, log):
    model = build_model(r["model"], len(wd.features), wd.hierarchy, cfg["model"])
    model.load_state_dict(torch.load(Path(r["_dir"]) / "model.pt", map_location="cpu"))
    model.to(device)
    out = {}
    for split in ("val", "test"):
        pred = predict(model, wd, wd.starts[split], device, amp=cfg["train"]["amp"], return_logits=True)
        out[split] = (pred["logits19"], wd.labels(split, "label19"))
    return out


def _check_against_saved(r, z, logits, log):
    """True if the re-scored test windows are the saved ones; logs agreement for the provenance record."""
    y_saved, y_new = z["y_true_label19"], logits["test"][1]
    if len(y_saved) != len(y_new) or not np.array_equal(y_saved, y_new):
        log.error(f"[POSTHOC_LABEL_MISMATCH] {r['run']}: test labels differ from predictions.npz "
                  f"({len(y_new)} vs {len(y_saved)} windows). The windows were built differently from the run; "
                  f"check that configs/rerun.yaml and data/processed are the ones the run used. Run skipped.")
        return None
    agreement = float(np.mean(logits["test"][0].argmax(1) == z["y_pred_label19"]))
    if agreement < MIN_AGREEMENT:
        log.warning(f"[POSTHOC_PRED_MISMATCH] {r['run']}: re-inference agrees with the saved predictions on only "
                    f"{agreement:.4%} of test windows (expected >= {MIN_AGREEMENT:.1%}). The reloaded model.pt may "
                    f"not be the one that produced predictions.npz; compare their timestamps before using this run.")
    return agreement


def _write_counts(wd, regime, cfg, log):
    names = wd.hierarchy.classes["label19"]
    counts = {split: class_counts(wd.labels(split, "label19"), len(names)) for split in ("train", "val", "test")}
    df = pd.DataFrame({"class": names, **counts})
    path = Path(cfg["paths"]["results_dir"]) / "tables" / f"class_window_counts_{regime}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Window counts per class (before oversampling) -> {path}")


def main():
    args, cfg, log = setup("Post-hoc abstention, consistency and tuned thresholds", "11_posthoc", add)
    device = resolve_device(cfg["train"]["device"])
    runs = _selected(cfg, args)
    log.info(f"{len(runs)} main runs selected")
    df, wd, wd_key, counted = None, None, None, set()
    for r in runs:
        d = Path(r["_dir"])
        if (d / "posthoc_v2.json").exists() and not args.force:
            log.info(f"{r['run']}: posthoc_v2.json exists, skipped (use --force to redo)")
            continue
        z = np.load(d / "predictions.npz")
        if r["family"] == "deep":
            key = (r["regime"], r["seed"], int(r["T"]))
            if key != wd_key:
                df = load_prepared(cfg) if df is None else df
                set_seed(r["seed"])
                cs = {**cfg, "train": {**cfg["train"], "seed": r["seed"]}}
                wd = build_windowed(df, cs, regime=r["regime"], T=key[2], seed=r["seed"], logger=log,
                                    drop_features=audit_drop_features(cfg))
                wd_key = key
                if r["regime"] not in counted:
                    _write_counts(wd, r["regime"], cfg, log)
                    counted.add(r["regime"])
            logits = _neural_logits(r, wd, cfg, device, log)
            agreement = _check_against_saved(r, z, logits, log)
            if agreement is None:
                continue
        else:
            logits, agreement = classical_logits(z), None
        out = analyse_run(r["hierarchy"], z, logits, agreement, cfg, has_energy=r["family"] == "deep")
        save_json(out, d / "posthoc_v2.json")
        a = out["abstain"]
        log.info(f"{r['run']}: AURC {100 * a['aurc']:.3f}  ties@max {a['ties_at_max']:.1%}  "
                 f"consistency {out['consistency']['joint']:.4%}  binary tuned {out['binary']['macro_f1_tuned']:.4f}"
                 + (f"  agreement {agreement:.4%}" if agreement is not None else ""))
    log.info("Done. Next: scripts/09_compare.py, then scripts/10_hypothesis_verdicts.py")


if __name__ == "__main__":
    main()
