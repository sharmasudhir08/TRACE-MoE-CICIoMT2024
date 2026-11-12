"""Shared experiment plumbing: load the prepared table, build windows, run one model, save one run folder.

Every run is saved as results/<experiment>/<run_name>/ with
  config.json       the exact configuration used
  metrics.json      test (and validation) metrics for label2 / label6 / label19 + run metadata,
                    including a reproducibility fingerprint (session, host, git commit, code SHA-256,
                    data SHA-256, started/finished timestamps); plus "abstain" and "routing"
  history.csv       per-epoch training log (neural models)
  predictions.npz   y_true / y_pred per level (+ probabilities if evaluation.save_proba is true)

Append-only: when run.append_only is true (the default), running the same configuration twice does NOT
overwrite the first result; the second one lands in <name>__v2, the third in <name>__v3, and so on.
Every run also appends one JSON line to results/registry.jsonl so you have one ledger of every run ever.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.labels import LEVELS
from .data.loaders import load_table
from .data.windows import build_windowed, class_balanced_sample
from .metrics import evaluate_levels
from .abstain import routing_report, selective_report
from .posthoc import analyse_run
from .models.classical import build_classical
from .models.factory import build_model, count_parameters
from .train import evaluate, train_model
from .utils import (code_hash, data_hash, git_info, load_json, now_iso, resolve_device, save_json,
                    session_id, set_seed)


def audit_drop_features(cfg):
    """data.drop_features may be a list, or 'auto' to use the audit's suggestion."""
    drop = cfg["data"].get("drop_features") or []
    if drop == "auto":
        p = Path(cfg["paths"]["results_dir"]) / "audit" / "suggested_drop_features.json"
        drop = load_json(p)["drop"] if p.exists() else []
    return list(drop)


def load_prepared(cfg, name="ciciomt2024"):
    return load_table(cfg["paths"]["processed_dir"], name)


def run_dir(cfg, experiment, name):
    """Return the folder for this run. If run.append_only is true (the default), a second run of the
    same configuration lands in <name>__v2, a third in <name>__v3, so previous results are never
    silently destroyed. Set run.append_only to false to restore the overwrite behaviour."""
    base = Path(cfg["paths"]["results_dir"]) / experiment
    base.mkdir(parents=True, exist_ok=True)
    append_only = (cfg.get("run") or {}).get("append_only", True)
    target = base / name
    if append_only and target.exists() and any(target.iterdir()):
        n = 2
        while (base / f"{name}__v{n}").exists():
            n += 1
        target = base / f"{name}__v{n}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_name(model_name, regime, T, tag, seed):
    return f"{model_name}_{regime}_T{T}{'_' + tag if tag else ''}_seed{seed}"


def run_done(cfg, experiment, name, required_key=None):
    """True when a finished copy of this run exists (metrics.json written, and required_key present in it).
    The steps use it to skip finished runs, so a stopped stage resumes and parallel lanes never repeat work."""
    d = latest_run_dir(cfg, experiment, name)
    if d is None:
        return False
    return required_key is None or required_key in load_json(d / "metrics.json")


def latest_run_dir(cfg, experiment, name):
    """Return the newest existing run folder matching <name> or <name>__vN, or None.
    Used by step 06 to pick up models trained by step 04 even under append-only."""
    base = Path(cfg["paths"]["results_dir"]) / experiment
    if not base.exists():
        return None
    cands = [base / name] + sorted(base.glob(f"{name}__v*"),
                                   key=lambda p: int(p.name.rsplit("__v", 1)[-1]))
    cands = [p for p in cands if p.is_dir() and (p / "metrics.json").exists()]
    return cands[-1] if cands else None


def _save_predictions(d, y_true, proba, cfg, extra=None, val_y_true=None, val_proba=None):
    """Save test (and, when supplied, validation) y_true/y_pred and class probabilities.
    Everything the models produced on the labelled splits is preserved so later analysis never
    needs to re-run inference."""
    arrays = {}
    save_proba = cfg["evaluation"].get("save_proba", False)
    for lvl in LEVELS:
        if lvl in proba:
            arrays[f"y_true_{lvl}"] = y_true[lvl].astype(np.int16)
            arrays[f"y_pred_{lvl}"] = proba[lvl].argmax(1).astype(np.int16)
            if save_proba:
                arrays[f"proba_{lvl}"] = proba[lvl].astype(np.float16)
    if val_y_true is not None and val_proba is not None:
        for lvl in LEVELS:
            if lvl in val_proba:
                arrays[f"val_y_true_{lvl}"] = val_y_true[lvl].astype(np.int16)
                arrays[f"val_y_pred_{lvl}"] = val_proba[lvl].argmax(1).astype(np.int16)
                if save_proba:
                    arrays[f"val_proba_{lvl}"] = val_proba[lvl].astype(np.float16)
    for k, v in (extra or {}).items():
        arrays[k] = v
    np.savez_compressed(d / "predictions.npz", **arrays)


def _fingerprint(cfg, run_name, started_at, experiment=""):
    """Reproducibility stamp attached to every metrics.json."""
    sha, dirty = git_info()
    parquet = Path(cfg["paths"]["processed_dir"]) / "ciciomt2024.parquet"
    env = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
           "host": platform.node(), "session": session_id()}
    try:
        import torch as _t
        env["torch"] = _t.__version__
        env["cuda_available"] = bool(_t.cuda.is_available())
        env["gpu"] = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "none"
    except ImportError:
        pass
    finished = now_iso()
    try:
        from datetime import datetime
        dur = (datetime.fromisoformat(finished) - datetime.fromisoformat(started_at)).total_seconds()
    except Exception:
        dur = None
    # run_id includes the experiment to avoid collisions when two experiments produce runs of the
    # same stem (e.g. step 04 writes main/trace_moe_official_T16_seed42 and step 06 writes
    # robustness/trace_moe_official_T16_seed42 — different runs, same stem).
    rid_prefix = f"{experiment}_" if experiment else ""
    return {"run_id": f"{session_id()}_{rid_prefix}{run_name}", "started_at": started_at, "finished_at": finished,
            "duration_seconds": dur, "host": env["host"], "session": env["session"],
            "git_commit": sha, "git_dirty": dirty, "code_sha256": code_hash(),
            "data_sha256": data_hash(parquet) if parquet.exists() else None, "environment": env}


def _append_registry(cfg, meta, test_metrics, run_path):
    """One JSON line per run, appended to <results>/registry.jsonl. Never overwritten.
    This is the single ledger of every run ever done; step 09 reads it alongside walking metrics.json."""
    reg = Path(cfg["paths"]["results_dir"]) / "registry.jsonl"
    reg.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": meta.get("finished_at", now_iso()), "run_id": meta.get("run_id"),
           "experiment": meta.get("experiment"), "run": meta.get("run"), "model": meta.get("model"),
           "family": meta.get("family"), "regime": meta.get("regime"), "seed": meta.get("seed"),
           "T": meta.get("T"), "tag": meta.get("tag"), "n_params": meta.get("n_params"),
           "train_seconds": meta.get("train_seconds"), "test_seconds": meta.get("test_seconds"),
           "git_commit": meta.get("git_commit"), "git_dirty": meta.get("git_dirty"),
           "code_sha256": meta.get("code_sha256"), "data_sha256": meta.get("data_sha256"),
           "macro_f1_label19": (test_metrics.get("label19") or {}).get("macro_f1"),
           "macro_f1_label6": (test_metrics.get("label6") or {}).get("macro_f1"),
           "macro_f1_label2": (test_metrics.get("label2") or {}).get("macro_f1"),
           "dir": str(run_path).replace("\\", "/"), "status": "ok"}
    with open(reg, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_run(d, cfg, meta, test_metrics, val_metrics=None, history=None, y_true=None, proba=None,
             extra_arrays=None, val_y_true=None, val_proba=None):
    started = meta.get("started_at") or now_iso()
    fp = _fingerprint(cfg, meta.get("run", ""), started, experiment=meta.get("experiment", ""))
    meta = {**fp, **meta, **{k: v for k, v in fp.items() if k not in meta}}   # fingerprint wins only where meta didn't set a value
    save_json(cfg, d / "config.json")
    save_json({**meta, "test": test_metrics, "val": val_metrics or {}}, d / "metrics.json")
    if history is not None and len(history):
        history.to_csv(d / "history.csv", index=False)
    if y_true is not None and proba is not None:
        _save_predictions(d, y_true, proba, cfg, extra_arrays, val_y_true=val_y_true, val_proba=val_proba)
    _append_registry(cfg, meta, test_metrics, d)


def run_deep(df, cfg, model_name, regime, seed, experiment, logger, T=None, tag="", wd=None, posthoc=False,
             extra_arrays=None):
    """Train and evaluate one neural model. Returns (run_dir, metrics, model, wd).
    posthoc: also write posthoc_v2.json from this run's float32 logits (what step 11 does for results/main);
    extra_arrays: more arrays for predictions.npz (the twin study saves Δ, device and time per window)."""
    set_seed(seed)
    cfg = {**cfg, "train": {**cfg["train"], "seed": seed}, "model": {**cfg["model"], "name": model_name}}
    T = int(T or cfg["windows"]["length"])
    if wd is None:
        wd = build_windowed(df, cfg, regime=regime, T=T, seed=seed, logger=logger,
                            drop_features=audit_drop_features(cfg))
    device = resolve_device(cfg["train"]["device"])
    model = build_model(model_name, len(wd.features), wd.hierarchy, cfg["model"])
    n_params = count_parameters(model)
    name = run_name(model_name, regime, T, tag, seed)
    started_at = now_iso()
    logger.info(f"=== {experiment}/{name}: {n_params:,} parameters on {device} | started {started_at}")
    t0 = time.time()
    model, hist = train_model(model, wd, cfg, device, logger)
    train_s = time.time() - t0
    t1 = time.time()
    test_m, pred, y_true = evaluate(model, wd, "test", device, cfg, return_logits=posthoc)
    test_s = time.time() - t1
    val_m, val_pred, val_true = (evaluate(model, wd, "val", device, cfg, return_logits=posthoc)
                                 if len(wd.starts["val"]) else ({}, None, None))
    abstain = selective_report(val_pred["proba"]["label19"] if val_pred else None,
                               val_true["label19"] if val_true else None,
                               pred["proba"]["label19"], y_true["label19"], cfg["abstain"]["target_coverage"])
    routing = routing_report(pred.get("route"), y_true["label6"], y_true["label19"], wd.hierarchy.classes["label6"],
                             getattr(model, "n_experts", 1)) if "route" in pred else {}
    meta = dict(experiment=experiment, run=name, model=model_name, family="deep", regime=regime, seed=seed,
                T=T, tag=tag, n_params=n_params, started_at=started_at,
                train_seconds=round(train_s, 1), test_seconds=round(test_s, 2),
                n_test_windows=int(len(wd.starts["test"])), n_val_windows=int(len(wd.starts.get("val", []))),
                epochs_run=int(len(hist)), hierarchy=wd.hierarchy.to_dict(), data_report=wd.report,
                abstain=abstain, routing=routing)
    d = run_dir(cfg, experiment, name)
    arrays = {**({"route": pred["route"].astype(np.int8)} if "route" in pred else {}), **(extra_arrays or {})}
    save_run(d, cfg, meta, test_m, val_m, hist, y_true, pred["proba"], arrays or None,
             val_y_true=val_true, val_proba=val_pred["proba"] if val_pred else None)
    torch.save(model.state_dict(), d / "model.pt")
    if posthoc and val_pred is not None:
        z = np.load(d / "predictions.npz")
        logits = {"val": (val_pred["logits19"], val_true["label19"]), "test": (pred["logits19"], y_true["label19"])}
        save_json(analyse_run(meta["hierarchy"], z, logits, 1.0, cfg, has_energy=True), d / "posthoc_v2.json")
    logger.info(f"test macro-F1  2:{test_m['label2']['macro_f1']:.4f}  6:{test_m['label6']['macro_f1']:.4f}  "
                f"19:{test_m['label19']['macro_f1']:.4f}  -> {d}")
    return d, {**meta, "test": test_m}, model, wd


def load_or_train_deep(df, cfg, model_name, regime, seed, experiment, logger, T=None, tag="", wd=None):
    """Reuse model.pt from the newest existing run with the same stem (handles append-only __vN suffixes),
    else train. Previous models are never silently overwritten."""
    T = int(T or cfg["windows"]["length"])
    name = run_name(model_name, regime, T, tag, seed)
    existing = latest_run_dir(cfg, experiment, name)
    p = (existing / "model.pt") if existing is not None else None
    if p is not None and p.exists():
        set_seed(seed)
        c2 = {**cfg, "train": {**cfg["train"], "seed": seed}}
        if wd is None:
            wd = build_windowed(df, c2, regime=regime, T=T, seed=seed, logger=logger,
                                drop_features=audit_drop_features(cfg))
        model = build_model(model_name, len(wd.features), wd.hierarchy, cfg["model"])
        model.load_state_dict(torch.load(p, map_location="cpu"))
        logger.info(f"Loaded trained model {p}")
        return model.to(resolve_device(cfg["train"]["device"])), wd
    _, _, model, wd = run_deep(df, cfg, model_name, regime, seed, experiment, logger, T=T, tag=tag, wd=wd)
    return model, wd


def rows_from_windows(wd, split, max_per_class=None, seed=0):
    """Feature rows (last row of each window) and labels for tree models."""
    starts = wd.starts[split]
    if max_per_class:
        idx = class_balanced_sample(wd.labels(split, "label19"), max_per_class, seed)
        starts = starts[np.sort(idx)]
    last = starts + wd.T - 1
    return wd.X[last], {lvl: wd.y[lvl][last] for lvl in LEVELS}


def fine_to_coarse(proba19, hierarchy):
    p6 = np.zeros((len(proba19), hierarchy.n("label6")))
    np.add.at(p6.T, np.array(hierarchy.parent19), proba19.T)
    p2 = np.zeros((len(proba19), 2))
    np.add.at(p2.T, np.array(hierarchy.parent6), p6.T)
    return {"label19": proba19, "label6": p6, "label2": p2}


def fit_classical(name, wd, cfg, seed):
    Xtr, ytr = rows_from_windows(wd, "train", cfg["classical"]["max_rows_per_class"], seed)
    clf = build_classical(name, cfg["classical"]["n_estimators"], seed, wd.hierarchy.n("label19"))
    t0 = time.time()
    clf.fit(Xtr, ytr["label19"])
    return clf, time.time() - t0, len(Xtr)


def classical_proba(clf, X, n_classes):
    p = clf.predict_proba(X)
    full = np.zeros((len(X), n_classes))
    full[:, np.asarray(clf.classes_, dtype=int)] = p
    return full


def run_classical(df, cfg, name, regime, seed, experiment, logger, wd=None):
    set_seed(seed)
    if wd is None:
        wd = build_windowed(df, cfg, regime=regime, T=1, seed=seed, logger=logger,
                            drop_features=audit_drop_features(cfg))
    run = run_name(name, regime, 1, "", seed)
    started_at = now_iso()
    logger.info(f"=== {experiment}/{run} | started {started_at}")
    clf, fit_s, n_train = fit_classical(name, wd, cfg, seed)
    Xte, yte = rows_from_windows(wd, "test")
    t0 = time.time()
    proba = fine_to_coarse(classical_proba(clf, Xte, wd.hierarchy.n("label19")), wd.hierarchy)
    pred_s = time.time() - t0
    test_m = evaluate_levels(yte, proba, wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
    pv, yv, val_proba_full, val_y_full = None, None, None, None
    if len(wd.starts["val"]):
        Xv, yvl = rows_from_windows(wd, "val")
        val_proba_full = fine_to_coarse(classical_proba(clf, Xv, wd.hierarchy.n("label19")), wd.hierarchy)
        val_y_full = yvl
        pv, yv = val_proba_full["label19"], yvl["label19"]
    abstain = selective_report(pv, yv, proba["label19"], yte["label19"], cfg["abstain"]["target_coverage"])
    meta = dict(experiment=experiment, run=run, model=name, family="classical", regime=regime, seed=seed, T=1,
                tag="", started_at=started_at, n_train_rows=int(n_train), abstain=abstain,
                train_seconds=round(fit_s, 1), test_seconds=round(pred_s, 2),
                predict_ms_per_1k=round(1000 * pred_s / max(len(Xte), 1) * 1000, 3),
                n_test_windows=int(len(Xte)), n_val_windows=int(len(wd.starts.get("val", []))),
                hierarchy=wd.hierarchy.to_dict(), data_report=wd.report)
    if name == "rf" and hasattr(clf, "feature_importances_"):
        meta["feature_importance"] = dict(zip(wd.features, np.round(clf.feature_importances_, 6).tolist()))
    d = run_dir(cfg, experiment, run)
    save_run(d, cfg, meta, test_m, None, None, yte, proba,
             val_y_true=val_y_full, val_proba=val_proba_full)
    logger.info(f"test macro-F1  2:{test_m['label2']['macro_f1']:.4f}  6:{test_m['label6']['macro_f1']:.4f}  "
                f"19:{test_m['label19']['macro_f1']:.4f}  -> {d}")
    return d, {**meta, "test": test_m}, clf, wd
