"""Step 16 (twin study, H11; added 2026-11-05). Train and score the twin variants on the device rows.

Variants (TWIN_PROTOCOL.md §6; seeds per variant in configs/twin.yaml, twin.seeds):
  V0  No-Twin: TRACE-MoE on the device rows, no twin input
  V1  device-agnostic: Δ and flags against one fleet-wide benign profile
  V2  Twin-Δ, static: per-device profiles, no state
  V3  Twin-Δ+State, static: per-device, per-state profiles; the state also feeds the router
  V4  Twin-Δ+State, synchronised (DT-TRACE-MoE): the V3 model, with the twin updated on the test stream
  rf  random forest on single device rows (the classical reference)
Stages (each skips finished work, so a stopped stage can be started again):
  train  V0-V3 and rf             results_rerun/twin/<run>/  (model.pt, predictions.npz, posthoc_v2.json)
  tune   η, δ, budget on validation of V3 (twin.tune_seed)   results_rerun/twin/sync_params.json (frozen)
  sync   V4 on the test stream with the frozen values        results_rerun/twin/trace_moe_..._V4_seed<k>/

  python scripts/16_twin_train.py --config configs/twin.yaml --stage train --variants V3 V0 --set train.device=mps
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from _common import setup
from trace_iomt import twin_data, twin_sync
from trace_iomt.abstain import routing_report, selective_report
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import run_classical, run_deep, run_dir, run_done, run_name, save_run
from trace_iomt.metrics import evaluate_levels
from trace_iomt.posthoc import analyse_run, classical_logits
from trace_iomt.train import evaluate
from trace_iomt.twin_study import EXPERIMENT, VARIANTS, SeedData, deep_name, load_v3, model_cfg
from trace_iomt.utils import load_json, now_iso, resolve_device, save_json


def add(p):
    p.add_argument("--stage", choices=["train", "tune", "sync"], required=True)
    p.add_argument("--variants", nargs="+", default=None, help="train stage: any of V0 V1 V2 V3 rf")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="restrict to these seeds")
    p.add_argument("--force", action="store_true", help="tune stage: choose the values again")


def is_done(cfg, variant, seed):
    if variant == "rf":
        return run_done(cfg, EXPERIMENT, run_name("rf", "official", 1, "", seed))
    return run_done(cfg, EXPERIMENT, deep_name(cfg, variant, seed))


def window_arrays(wd, delta=None):
    """Per test and validation window: Δ (mean and max over features), device, state and time of the last row."""
    out = {}
    for split, prefix in (("test", ""), ("val", "val_")):
        st = wd.starts[split]
        last = st + wd.T - 1
        mean, mx = delta[split] if delta and split in delta else twin_sync.window_delta_summary(wd.ctx, st, wd.T)
        out.update({f"{prefix}twin_delta_mean": mean, f"{prefix}twin_delta_max": mx,
                    f"{prefix}device_id": wd.device_id[last].astype(np.int16),
                    f"{prefix}state_code": wd.state_code[last].astype(np.int8), f"{prefix}t_last": wd.t[last]})
    return out


def train_rf(data, seed, log):
    cfg = data.cfg
    wd1 = build_windowed(twin_data.feature_table(data.rows), cfg, regime="official", T=1, seed=seed, logger=log)
    d, _, _, _ = run_classical(None, cfg, "rf", "official", seed, EXPERIMENT, log, wd=wd1)
    z = np.load(d / "predictions.npz")
    save_json(analyse_run(load_json(d / "metrics.json")["hierarchy"], z, classical_logits(z), None, cfg, False),
              d / "posthoc_v2.json")


def stage_train(cfg, args, log):
    plan = {v: cfg["twin"]["seeds"][v] for v in (args.variants or [*VARIANTS, "rf"])}
    seeds = sorted({s for ss in plan.values() for s in ss if not args.seeds or s in args.seeds})
    for seed in seeds:
        todo = [v for v, ss in plan.items() if seed in ss and not is_done(cfg, v, seed)]
        if not todo:
            log.info(f"seed {seed}: {list(plan)} already done")
            continue
        data = SeedData(cfg, seed, log)
        for v in todo:
            if v == "rf":
                train_rf(data, seed, log)
                continue
            wd = data.windows(VARIANTS[v][1])
            run_deep(None, model_cfg(data.cfg, v), "trace_moe", "official", seed, EXPERIMENT, log, tag=v, wd=wd,
                     posthoc=True, extra_arrays=window_arrays(wd))
        del data


def static_val(model, wd, cfg, device):
    """Validation metrics and logits with the static twin, and the acceptance rule derived from them."""
    val_m, val_pred, val_true = evaluate(model, wd, "val", device, cfg, return_logits=True)
    rule = twin_sync.acceptance_rule(val_pred["logits19"], val_true["label19"], cfg["abstain"]["target_coverage"])
    return val_m, val_pred, val_true, rule


def stage_tune(cfg, args, log):
    path = Path(cfg["paths"]["results_dir"]) / EXPERIMENT / "sync_params.json"
    if path.exists() and not args.force:
        log.info(f"{path} exists (frozen); use --force to choose again")
        return
    seed, device = cfg["twin"]["tune_seed"], resolve_device(cfg["train"]["device"])
    data = SeedData(cfg, seed, log)
    wd = data.windows("device_state")
    model, _ = load_v3(cfg, seed, wd, device)
    val_m, _, val_true, rule = static_val(model, wd, data.cfg, device)
    benign = wd.hierarchy.index("label19")["Benign"]
    g = cfg["twin"]["sync_grid"]
    grid = []
    for eta, delta, budget in itertools.product(g["eta"], g["delta"], g["budget"]):
        res = twin_sync.stream(model, wd, "val", data.profiles, eta, delta, budget, rule, benign, device,
                               cfg["twin"]["stream_batch"], cfg["train"]["amp"])
        f1 = float(f1_score(val_true["label19"], res["proba"]["label19"].argmax(1), average="macro"))
        grid.append({"eta": eta, "delta": delta, "budget": budget, "val_macro_f1_label19": f1,
                     "updated": res["sync_counts"].get("updated", 0)})
        log.info(f"η {eta} δ {delta} budget {budget}: val 19-class macro-F1 {f1:.4f}, "
                 f"{grid[-1]['updated']} profile updates")
    best = max(grid, key=lambda r: r["val_macro_f1_label19"])       # first of equals keeps the smaller values
    save_json({"eta": best["eta"], "delta": best["delta"], "budget": best["budget"], "seed": seed,
               "static_val_macro_f1_label19": val_m["label19"]["macro_f1"], "acceptance_rule": list(rule),
               "grid": grid, "frozen_at": now_iso()}, path)
    log.info(f"chosen η {best['eta']} δ {best['delta']} budget {best['budget']} -> {path} (frozen)")


def sync_one(cfg, data, seed, params, log):
    c, device = model_cfg(data.cfg, "V4"), resolve_device(cfg["train"]["device"])
    wd = data.windows("device_state")
    model, v3_dir = load_v3(cfg, seed, wd, device)
    val_m, val_pred, val_true, rule = static_val(model, wd, c, device)
    benign = wd.hierarchy.index("label19")["Benign"]
    started = now_iso()
    res = twin_sync.stream(model, wd, "test", data.profiles, params["eta"], params["delta"], params["budget"], rule,
                           benign, device, cfg["twin"]["stream_batch"], cfg["train"]["amp"])
    last = wd.starts["test"] + wd.T - 1
    y_true = {lvl: wd.y[lvl][last] for lvl in res["proba"]}
    test_m = evaluate_levels(y_true, res["proba"], wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
    name = deep_name(cfg, "V4", seed)
    meta = dict(experiment=EXPERIMENT, run=name, model="trace_moe", family="deep", regime="official", seed=seed,
                T=wd.T, tag="V4", started_at=started, model_from=str(v3_dir), sync=params,
                sync_counts=res["sync_counts"], n_test_windows=int(len(last)),
                n_val_windows=int(len(wd.starts["val"])), hierarchy=wd.hierarchy.to_dict(), data_report=wd.report,
                abstain=selective_report(val_pred["proba"]["label19"], val_true["label19"],
                                         res["proba"]["label19"], y_true["label19"], c["abstain"]["target_coverage"]),
                routing=routing_report(res.get("route"), y_true["label6"], y_true["label19"],
                                       wd.hierarchy.classes["label6"], model.n_experts))
    arrays = {"route": res["route"].astype(np.int8),
              **window_arrays(wd, {"test": (res["delta_mean"], res["delta_max"])})}
    d = run_dir(c, EXPERIMENT, name)
    save_run(d, c, meta, test_m, val_m, None, y_true, res["proba"], arrays, val_y_true=val_true,
             val_proba=val_pred["proba"])
    logits = {"val": (val_pred["logits19"], val_true["label19"]), "test": (res["logits19"], y_true["label19"])}
    save_json(analyse_run(meta["hierarchy"], np.load(d / "predictions.npz"), logits, 1.0, c, True),
              d / "posthoc_v2.json")
    log.info(f"V4 seed {seed}: test 19-class macro-F1 {test_m['label19']['macro_f1']:.4f}, "
             f"sync {res['sync_counts']} -> {d}")


def stage_sync(cfg, args, log):
    path = Path(cfg["paths"]["results_dir"]) / EXPERIMENT / "sync_params.json"
    if not path.exists():
        raise SystemExit(f"[TWIN_NO_SYNC_PARAMS] {path} is missing. The synchronised runs use values chosen on "
                         f"validation: run --stage tune first.")
    params = {k: load_json(path)[k] for k in ("eta", "delta", "budget")}
    for seed in cfg["twin"]["seeds"]["V4"]:
        if (args.seeds and seed not in args.seeds) or is_done(cfg, "V4", seed):
            log.info(f"V4 seed {seed}: skipped (done or not selected)")
            continue
        sync_one(cfg, SeedData(cfg, seed, log), seed, params, log)


def main():
    args, cfg, log = setup("Twin study: train and score the variants", "16_twin_train", add)
    if "twin" not in cfg:
        raise SystemExit("[TWIN_NO_CONFIG] this step needs the twin block: use --config configs/twin.yaml")
    {"train": stage_train, "tune": stage_tune, "sync": stage_sync}[args.stage](cfg, args, log)


if __name__ == "__main__":
    main()
