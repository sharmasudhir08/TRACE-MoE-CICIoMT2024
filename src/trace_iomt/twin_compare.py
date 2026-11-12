"""Twin study tables and the H11 verdict (TWIN_PROTOCOL.md §8; added 2026-11-05).

Works on the run folders of scripts/16_twin_train.py (results/twin/). Pieces:
  - ablation table: mean ± std over seeds per variant (V0-V4 and rf);
  - paired differences between variants over their common seeds, with a 95% t-interval;
  - file bootstrap of V4 - V0: test capture files resampled with replacement, macro-F1 recomputed from
    per-file confusion matrices, the difference averaged over seeds with the same resample for every seed;
  - Wilcoxon over the 19 seed-averaged class blocks, Holm-corrected;
  - AUROC of the twin discrepancy ‖Δ‖ for each attack class against benign test windows. The twin learns from
    benign traffic only, so every attack is unseen by the twin.
  - time to detect: first attacker packet of a segment -> end of the first window flagged Attack;
  - the H11 verdict from the decision rule.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .abstain import risk_coverage

VARIANT_NAMES = {"V0": "No-Twin", "V1": "Device-agnostic profile", "V2": "Twin-Δ (static)",
                 "V3": "Twin-Δ+State (static)", "V4": "Twin-Δ+State (synchronised)", "rf": "Random forest"}


def variant_of(run):
    """V0..V4 from the run tag, 'rf' for the random forest."""
    return "rf" if run.get("model") == "rf" else run.get("tag")


def run_row(run):
    """One row of summary numbers for a run (test split; post-hoc values from posthoc_v2.json)."""
    t, ph = run["test"], run.get("posthoc", {})
    return {"variant": variant_of(run), "seed": run["seed"], "run": run["run"], "dir": run["_dir"],
            "macro_f1_19": t["label19"]["macro_f1"], "min_recall_19": t["label19"]["min_class_recall"],
            "macro_f1_6": t["label6"]["macro_f1"], "macro_f1_2": t["label2"]["macro_f1"],
            "macro_f1_2_tuned": ph.get("binary", {}).get("macro_f1_tuned"),
            "aurc": ph.get("abstain", {}).get("aurc"), "consistency": ph.get("consistency", {}).get("joint")}


def paired_difference(rows, a, b, metric="macro_f1_19"):
    """a - b over the seeds both have: per-seed differences, mean and 95% t-interval."""
    piv = rows.pivot_table(index="seed", columns="variant", values=metric)
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna()
    n = len(d)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"a": a, "b": b, "metric": metric, "seeds": d.index.tolist(), "per_seed": d.round(6).tolist(),
            "mean": float(d.mean()), "t_low": float(d.mean() - half), "t_high": float(d.mean() + half)}


def confusion_by_file(y_true, y_pred, files, n_classes):
    """{file: n_classes x n_classes confusion matrix} for one run's test windows."""
    out = {}
    for f in np.unique(files):
        m = files == f
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (y_true[m], y_pred[m]), 1)
        out[f] = cm
    return out


def macro_f1_from_confusion(cm):
    """Macro-F1 over the classes present in the truth (a resample can miss a class entirely)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def file_bootstrap(cms_a, cms_b, n_resamples=1000, seed=0):
    """cms_a, cms_b: {seed: {file: confusion}} for two variants on the same windows. Returns the 95% interval of
    the seed-mean macro-F1 difference a - b, resampling test files with replacement (same files for all seeds)."""
    seeds = sorted(set(cms_a) & set(cms_b))
    files = sorted(set.intersection(*[set(cms_a[s]) for s in seeds]))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(0, len(files), len(files)), minlength=len(files))
        per_seed = []
        for s in seeds:
            ca = sum(c * cms_a[s][f] for c, f in zip(counts, files) if c)
            cb = sum(c * cms_b[s][f] for c, f in zip(counts, files) if c)
            per_seed.append(macro_f1_from_confusion(ca) - macro_f1_from_confusion(cb))
        diffs[i] = np.mean(per_seed)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {"seeds": seeds, "n_files": len(files), "n_resamples": n_resamples, "low": float(low), "high": float(high)}


def class_block_wilcoxon(per_class, reference, others, metric="f1"):
    """Wilcoxon of reference vs each other variant over the 19 seed-averaged class blocks, Holm-corrected."""
    blocks = per_class.groupby(["variant", "class"])[metric].mean().unstack("variant")
    rows = []
    for o in others:
        if reference not in blocks or o not in blocks:
            continue
        d = blocks[[reference, o]].dropna()
        p = 1.0 if np.allclose(d[reference], d[o]) else float(stats.wilcoxon(d[reference], d[o]).pvalue)
        rows.append({"reference": reference, "other": o, "n_blocks": len(d),
                     "mean_diff": float((d[reference] - d[o]).mean()), "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("p").reset_index(drop=True)
        k = len(out)
        out["p_holm"] = np.minimum(1, np.maximum.accumulate(out["p"] * (k - np.arange(k))))
    return out


def delta_auroc(delta, y19, classes, benign="Benign", mask=None):
    """AUROC of ‖Δ‖ separating each attack class from benign windows (higher Δ = attack)."""
    mask = np.ones(len(delta), dtype=bool) if mask is None else mask
    b = mask & (y19 == classes.index(benign))
    out = {}
    for k, name in enumerate(classes):
        a = mask & (y19 == k)
        if name == benign or not a.any() or not b.any():
            continue
        y = np.r_[np.zeros(b.sum()), np.ones(a.sum())]
        out[name] = float(roc_auc_score(y, np.r_[delta[b], delta[a]]))
    return out


def delta_aurc(delta, y_true, y_pred):
    """AURC with -‖Δ‖ as the confidence score (a window far from its profile is accepted last)."""
    return risk_coverage(-np.asarray(delta, dtype=np.float64), y_true == y_pred)[0]


def time_to_detect(segment, t_end, flagged, first_attack):
    """Per attacked segment: seconds from its first attacker packet to the end of the first window flagged Attack
    that ends after it (None if never flagged). segment, t_end, flagged: one value per window."""
    out = []
    for seg, t0 in first_attack.items():
        m = (segment == seg) & (t_end >= t0)
        if not m.any():
            continue                                    # no window ends after the first attack packet
        hit = t_end[m & flagged]
        out.append({"segment": str(seg), "seconds": float(hit.min() - t0) if len(hit) else None,
                    "windows": int(m.sum())})
    return out


def h11_verdict(primary, bootstrap):
    """TWIN_PROTOCOL.md §8: supported if mean > 0 and both the t-interval and the file bootstrap exclude 0."""
    if primary is None:
        return "not evaluated"
    if primary["mean"] <= 0:
        return "not supported"
    t_ok = primary["t_low"] > 0
    b_ok = bootstrap is not None and bootstrap["low"] > 0
    return "supported" if t_ok and b_ok else "partly supported"
