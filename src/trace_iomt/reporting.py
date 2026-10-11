"""Read everything in results/ and logs/ into one set of facts, used by the technical report (step 10)
and the paper writer (step 11). Every number in those documents comes from here."""
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def environment():
    env = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
           "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else "not available"
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        pass
    for lib in ("sklearn", "pandas", "numpy", "scipy"):
        try:
            env[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return env


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+) \| (.*)$")


def parse_logs(logs_dir):
    """Per log file: start, end, wall time, number of warnings/errors and the warning lines."""
    rows, issues = [], []
    for f in sorted(Path(logs_dir).glob("*.log")):
        times, n_warn, n_err = [], 0, 0
        for line in f.read_text(errors="ignore").splitlines():
            m = _TS.match(line)
            if not m:
                continue
            times.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            if m.group(2) == "WARNING":
                n_warn += 1; issues.append((f.stem, "WARNING", m.group(3)))
            elif m.group(2) in ("ERROR", "CRITICAL"):
                n_err += 1; issues.append((f.stem, m.group(2), m.group(3)))
        if times:
            rows.append({"log": f.name, "start": times[0].strftime("%Y-%m-%d %H:%M"),
                         "end": times[-1].strftime("%Y-%m-%d %H:%M"),
                         "minutes": round((times[-1] - times[0]).total_seconds() / 60, 1),
                         "lines": len(times), "warnings": n_warn, "errors": n_err})
    return pd.DataFrame(rows), pd.DataFrame(issues, columns=["log", "level", "message"])


def _ms(df, by, col):
    g = df.groupby(by)[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1).fillna(0), "n": g.size()})


def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def collect(results_dir, logs_dir, reference="trace_moe"):
    R = Path(results_dir)
    T = R / "tables"
    f = {"reference": reference, "results_dir": str(R), "logs_dir": str(logs_dir)}
    f["class_counts"] = _csv(R / "data" / "ciciomt2024_class_counts.csv")
    f["duplicates"] = _json(R / "audit" / "duplicates.json")
    f["shortcuts"] = _csv(R / "audit" / "shortcut_features_label19.csv")
    f["suggested_drop"] = (_json(R / "audit" / "suggested_drop_features.json") or {}).get("drop", [])
    f["audit_inflation"] = _csv(R / "audit" / "split_inflation.csv")
    ac = _csv(R / "audit" / "autocorrelation.csv")
    f["autocorr"] = None if ac is None or not len(ac) else (float(ac["lag1_autocorr"].mean()),
                                                             float(ac["lag1_autocorr_shuffled"].mean()))
    long = _csv(T / "all_runs_long.csv")
    f["long"] = long
    f["tables"] = {p.stem: pd.read_csv(p) for p in sorted(T.glob("*.csv"))} if T.exists() else {}
    f["friedman"] = _json(T / "friedman_nemenyi_official.json")
    f["efficiency"] = _csv(R / "efficiency" / "efficiency.csv")
    f["figures"] = sorted(str(p) for p in (R / "figures").glob("*.png")) if (R / "figures").exists() else []
    f["log_summary"], f["log_issues"] = parse_logs(logs_dir)
    f["env"] = environment()
    if long is None:
        return f

    # ---- main results
    main = long[(long["experiment"] == "main")]
    f["n_runs"] = int(long["run"].nunique())
    f["n_seeds"] = int(main["seed"].nunique()) if len(main) else 0
    regime = "official" if "official" in set(main["regime"]) else (main["regime"].iloc[0] if len(main) else None)
    f["regime"] = regime
    if len(main):
        m19 = _ms(main[(main["level"] == "label19") & (main["regime"] == regime)], "model", "macro_f1")
        m19 = m19.sort_values("mean", ascending=False)
        f["main19"] = m19
        f["best_model"] = m19.index[0]
        others = m19.drop(index=reference, errors="ignore")
        f["best_baseline"] = others.index[0] if len(others) else None
        f["ref_f1"] = m19.loc[reference] if reference in m19.index else None
        f["best_baseline_f1"] = others.iloc[0] if len(others) else None
        b = main[(main["level"] == "label2") & (main["regime"] == regime)]
        if "recall_at_fpr_0.01" in b:
            f["recall_fpr"] = _ms(b, "model", "recall_at_fpr_0.01")
        m6 = _ms(main[(main["level"] == "label6") & (main["regime"] == regime)], "model", "macro_f1")
        f["main6"] = m6.sort_values("mean", ascending=False)
        # inflation
        rr = _ms(main[main["level"] == "label19"], ["model", "regime"], "macro_f1")["mean"].unstack()
        f["regime_table"] = rr
        if {"random", regime} <= set(rr.columns) and regime != "random":
            f["inflation"] = (rr["random"] - rr[regime]).sort_values(ascending=False)
    w = f["tables"].get(f"wilcoxon_{reference}_vs_baselines_{regime}")
    f["wilcoxon"] = w
    # ---- ablation
    ab = long[(long["experiment"] == "ablation") & (long["level"] == "label19")]
    if len(ab):
        f["ablation"] = _ms(ab, ["tag", "T"], "macro_f1").reset_index()
    # ---- abstain layer and expert routing (read from every main run)
    arows, rrows = [], []
    for p in sorted((R / "main").glob("*/metrics.json")):
        r = json.loads(p.read_text())
        a = r.get("abstain") or {}
        if a:
            arows.append({"model": r["model"], "regime": r.get("regime"), "seed": r["seed"],
                          **{k: a.get(k) for k in ("coverage", "risk_full", "selective_risk", "aurc",
                                                   "accepted_macro_f1", "ece_raw", "ece_calibrated")}})
        ro = r.get("routing") or {}
        if ro.get("nmi_label6") is not None:
            rrows.append({"model": r["model"], "regime": r.get("regime"), "seed": r["seed"],
                          "nmi_label6": ro["nmi_label6"], "nmi_label19": ro["nmi_label19"]})
    f["abstain"] = pd.DataFrame(arows) if arows else None
    f["routing"] = pd.DataFrame(rrows) if rrows else None
    if len(main) and "min_class_recall" in main:
        f["min_recall"] = _ms(main[(main["level"] == "label19") & (main["regime"] == regime)], "model",
                              "min_class_recall").sort_values("mean", ascending=False)
    f["robustness"] = f["tables"].get("robustness_curves")
    rows = []
    for p in sorted((R / "zero_day").glob("*/metrics.json")):
        r = json.loads(p.read_text())
        if "ood" in r:
            o = r["ood"]
            rows.append({"model": r["model"], "held_out": o["held_out_class"], "seed": r["seed"],
                         "energy_auroc": o["energy"]["auroc"], "energy_fpr95": o["energy"]["fpr_at_95tpr"],
                         "msp_auroc": o["msp"]["auroc"], "caught_as_attack": o["held_out_detected_as_attack"],
                         "flagged_unknown": o.get("held_out_flagged_unknown"),
                         "false_alarm_known": o.get("known_false_alarm_rate"),
                         "caught_any": o.get("held_out_caught_any"),
                         "known_macro_f1": o["known_classes_macro_f1"]})
    f["zero_day"] = pd.DataFrame(rows) if rows else None
    return f


def mean_pm(row, d=2):
    return f"{100 * row['mean']:.{d}f} ± {100 * row['std']:.{d}f}"


# ---------------------------------------------------------------------------- hypothesis verdicts
# Same wording as METHODOLOGY.md Section 0 and the plan document; thresholds come from config "hypotheses".
HYPOTHESES = {
    "H1": ("RQ1", "TRACE-MoE has higher 19-class macro-F1 than each baseline"),
    "H2": ("RQ1", "TRACE-MoE raises the recall of the weakest class (minimum class recall) over the best baseline"),
    "H3": ("RQ1", "On binary detection TRACE-MoE is non-inferior to Random Forest"),
    "H4": ("RQ2", "Routing helps: one dense expert of the same size gives lower 19-class macro-F1"),
    "H5": ("RQ2", "The experts specialise by attack family (NMI between expert and category)"),
    "H6": ("RQ3", "Adversarially fine-tuned TRACE-MoE has a lower attack success rate than every baseline"),
    "H7": ("RQ3", "Adversarial fine-tuning costs little clean 19-class macro-F1"),
    "H8": ("RQ4", "The abstain score detects held-out attack types better than softmax confidence"),
    "H9": ("RQ4", "Abstention lowers the error on accepted traffic more than for any baseline (lowest AURC)"),
    "H10": ("RQ5", "TRACE-MoE is faster than the Transformer baseline and small enough for an edge gateway"),
}


def hypothesis_verdicts(F, criteria=None):
    """Return a DataFrame: hypothesis, RQ, statement, verdict, evidence. Verdicts are computed, never assumed."""
    c = {"alpha": 0.05, "min_recall_gain": 0.05, "noninferiority_margin": 0.005, "nmi_min": 0.5,
         "clean_drop_max": 0.01, "ood_auroc_min": 0.90, "max_params": 5_000_000, **(criteria or {})}
    ref, regime, T = F["reference"], F.get("regime"), F["tables"]
    rows = []

    def add(h, verdict, evidence):
        rows.append({"hypothesis": h, "RQ": HYPOTHESES[h][0], "statement": HYPOTHESES[h][1],
                     "verdict": verdict, "evidence": evidence})

    def tests_verdict(t, diff_col, label):
        better = int(((t[diff_col] > 0) & t["significant_0.05"]).sum())
        v = "supported" if better == len(t) else ("partly supported" if better else "not supported")
        return v, f"{label} significantly better (Holm p<{c['alpha']}) than {better}/{len(t)}: " + \
            ", ".join(f"{r['baseline']} {100 * r[diff_col]:+.2f} pts (p={r['p_holm']:.3g})" for _, r in t.iterrows())

    w = F.get("wilcoxon")
    if w is not None and len(w):
        add("H1", *tests_verdict(w, "mean_diff", "TRACE-MoE per-class F1"))
    else:
        add("H1", "not run", "needs main runs with >= 2 seeds and 09_compare.py")

    mr = F.get("min_recall")
    if mr is not None and ref in mr.index and len(mr) > 1:
        others = mr.drop(index=ref)
        gain = mr.loc[ref, "mean"] - others["mean"].max()
        add("H2", "supported" if gain >= c["min_recall_gain"] else ("partly supported" if gain > 0 else "not supported"),
            f"minimum class recall {100 * mr.loc[ref, 'mean']:.2f}% vs best baseline {others['mean'].idxmax()} "
            f"{100 * others['mean'].max():.2f}% (gain {100 * gain:+.2f} pts, needed {100 * c['min_recall_gain']:.0f})")
    else:
        add("H2", "not run", "needs main runs of TRACE-MoE and the baselines")

    ni = T.get("noninferiority_binary")
    if ni is not None and len(ni) and regime in set(ni["regime"]):
        r = ni[ni["regime"] == regime].iloc[0]
        ok = r["ci95_low"] > -c["noninferiority_margin"]
        add("H3", "supported" if ok else "not supported",
            f"binary macro-F1 TRACE-MoE minus RF = {100 * r['mean_diff']:+.3f} pts, 95% CI "
            f"[{100 * r['ci95_low']:+.3f}, {100 * r['ci95_high']:+.3f}] over {int(r['n_seeds'])} seeds "
            f"(margin -{100 * c['noninferiority_margin']:.1f} pts)")
    else:
        add("H3", "not run", "needs main runs of TRACE-MoE and RF")

    at = T.get("ablation_tests")
    if at is not None and len(at) and "dense_expert" in set(at["variant"]):
        r = at[at["variant"] == "dense_expert"].iloc[0]
        ok = r["mean_diff"] > 0 and bool(r["significant_0.05"])
        add("H4", "supported" if ok else ("partly supported" if r["mean_diff"] > 0 else "not supported"),
            f"full minus dense expert: {100 * r['mean_diff']:+.2f} pts per-class F1, Holm p={r['p_holm']:.3g}; "
            "other variants: " + ", ".join(f"{x['variant']} {100 * x['mean_diff']:+.2f}" for _, x in at.iterrows()
                                          if x["variant"] != "dense_expert"))
    else:
        add("H4", "not run", "needs 05_ablation.py")

    ro = F.get("routing")
    if ro is not None and len(ro) and ref in set(ro["model"]):
        g = ro[(ro["model"] == ref) & (ro["regime"] == regime)] if regime in set(ro["regime"]) else ro[ro["model"] == ref]
        nmi = g["nmi_label6"].mean()
        add("H5", "supported" if nmi >= c["nmi_min"] else ("partly supported" if nmi >= c["nmi_min"] / 2 else "not supported"),
            f"NMI(expert, category) = {nmi:.3f} (needed {c['nmi_min']}); NMI(expert, class) = {g['nmi_label19'].mean():.3f}")
    else:
        add("H5", "not run", "needs main runs of TRACE-MoE")

    rt = T.get("robustness_asr_tests")
    if rt is not None and len(rt):
        add("H6", *tests_verdict(rt, "mean_asr_reduction", "TRACE-MoE per-class ASR"))
    else:
        add("H6", "not run", "needs 06_robustness.py with at least one baseline")

    long = F.get("long")
    if long is not None and F.get("ref_f1") is not None:
        rl = long[(long["experiment"] == "robustness") & (long["model"] == ref) & (long["level"] == "label19")]
        if len(rl):
            drop = F["ref_f1"]["mean"] - rl["macro_f1"].mean()
            add("H7", "supported" if drop <= c["clean_drop_max"] else "not supported",
                f"clean 19-class macro-F1 {100 * F['ref_f1']['mean']:.2f}% before and {100 * rl['macro_f1'].mean():.2f}% "
                f"after adversarial fine-tuning (drop {100 * drop:+.2f} pts, allowed {100 * c['clean_drop_max']:.0f})")
        else:
            add("H7", "not run", "needs 06_robustness.py")
    else:
        add("H7", "not run", "needs 04_train_deep.py and 06_robustness.py")

    zd = F.get("zero_day")
    if zd is not None and len(zd) and ref in set(zd["model"]):
        z = zd[zd["model"] == ref]
        e, m = z["energy_auroc"].mean(), z["msp_auroc"].mean()
        ok1, ok2 = e >= c["ood_auroc_min"], e > m
        add("H8", "supported" if ok1 and ok2 else ("partly supported" if ok1 or ok2 else "not supported"),
            f"abstain-score AUROC {100 * e:.2f}% vs softmax confidence {100 * m:.2f}% (needed >= "
            f"{100 * c['ood_auroc_min']:.0f}%); held-out windows flagged unknown {100 * z['flagged_unknown'].mean():.1f}%, "
            f"caught as attack or unknown {100 * z['caught_any'].mean():.1f}%")
    else:
        add("H8", "not run", "needs 07_unknown_attacks.py")

    ab = F.get("abstain")
    if ab is not None and len(ab) and ref in set(ab["model"]):
        g = ab[ab["regime"] == regime] if regime in set(ab["regime"]) else ab
        a = g.groupby("model")["aurc"].mean().sort_values()
        ok = a.index[0] == ref
        rk = int(list(a.index).index(ref)) + 1
        add("H9", "supported" if ok else ("partly supported" if rk <= 2 else "not supported"),
            f"AURC (lower is better), rank {rk}/{len(a)}: " + ", ".join(f"{k} {100 * v:.3f}" for k, v in a.items()))
    else:
        add("H9", "not run", "needs main runs")

    eff = F.get("efficiency")
    if eff is not None and ref in set(eff["model"]):
        col = next((k for k in eff.columns if k.startswith("cpu_bs1024_ms")), None)
        e = eff.set_index("model")
        lat = e.loc[ref, col] if col else float("nan")
        small = e.loc[ref, "params"] < c["max_params"]
        faster = "transformer" in e.index and col and lat < e.loc["transformer", col]
        add("H10", "supported" if small and faster else ("partly supported" if small or faster else "not supported"),
            f"{int(e.loc[ref, 'params']):,} parameters ({e.loc[ref, 'size_mb_fp32']} MB); {lat:.4f} ms/window on CPU at "
            f"batch 1024" + (f" vs Transformer {e.loc['transformer', col]:.4f} ms" if "transformer" in e.index and col else ""))
    else:
        add("H10", "not run", "needs 08_efficiency.py")
    return pd.DataFrame(rows)
