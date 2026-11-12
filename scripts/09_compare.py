"""Step 9. Aggregate every run into the comparison tables, statistical tests and figures.

Reads results/<experiment>/<run>/metrics.json (steps 03-07) and writes results/tables/*.csv|.tex (+ provenance
sidecars for the main tables) and results/figures/*.png. The tables are the ones reporting.collect() reads to
compute the H1-H10 verdicts (step 10).

Besides the pre-specified tests over (class, seed) pairs, it writes a post-hoc version over 19 seed-averaged
class blocks (suffix _19blocks), because seeds of the same class are not independent.

Re-written on 2026-10-29: the original script is not in the public repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import setup
from trace_iomt.compare import (fig_bar, fig_cd, fig_confusion, fig_heatmap, fig_lines, friedman_nemenyi,
                                latest_runs, load_runs, long_table, mean_std, per_class_table, pretty_table,
                                wilcoxon_vs_reference, write_table)
from trace_iomt.utils import save_json

MAIN_METRICS = {"label19": ["macro_f1", "balanced_accuracy", "weighted_f1", "macro_pr_auc", "min_class_recall", "ece"],
                "label6": ["macro_f1", "balanced_accuracy", "weighted_f1", "min_class_recall"],
                "label2": ["macro_f1", "roc_auc", "pr_auc", "recall_at_fpr_0.01", "ece"]}
ABSTAIN_KEYS = ["coverage", "risk_full", "selective_risk", "aurc", "accepted_macro_f1", "ece_raw", "ece_calibrated"]


def seed_averaged(pc, metric="f1"):
    """One block per class: the metric averaged over seeds (post-hoc test without seed pseudo-replication)."""
    return pc.groupby(["model", "class"], as_index=False)[metric].mean().assign(seed=0)


def main_tables(long, pc_main, ref, out, figs, dpi):
    main = long[long["experiment"] == "main"]
    for regime in sorted(main["regime"].dropna().unique()):
        for lvl, metrics in MAIN_METRICS.items():
            g = main[(main["regime"] == regime) & (main["level"] == lvl)]
            metrics = [m for m in metrics if m in g and g[m].notna().any()]
            if not len(g) or not metrics:
                continue
            agg = mean_std(g, ["model", "family"], metrics).sort_values("macro_f1")
            write_table(pretty_table(agg, ["model", "family"], metrics), out, f"main_{regime}_{lvl}",
                        f"{lvl} test results, {regime} split, mean ± std over seeds (%)",
                        provenance=(g, ["model", "family"]))
    g19 = main[main["level"] == "label19"]
    piv = g19.groupby(["model", "regime"])["macro_f1"].mean().unstack()
    if {"official", "random"} <= set(piv.columns):
        infl = pd.DataFrame({"model": piv.index, "official": 100 * piv["official"], "random": 100 * piv["random"]})
        infl["inflation"] = infl["random"] - infl["official"]
        write_table(infl.round(2), out, "split_inflation", "19-class macro-F1 (%) per model and split protocol")
        bar = mean_std(g19, ["model", "regime"], ["macro_f1"])
        fig_bar(bar, "model", "regime", "macro_f1", figs / "split_inflation.png", "19-class macro-F1", dpi)
    for regime in sorted(pc_main["regime"].dropna().unique()):
        pc = pc_main[pc_main["regime"] == regime]
        for suffix, data in (("", pc), ("_19blocks", seed_averaged(pc))):
            w = wilcoxon_vs_reference(data, ref)
            if len(w):
                write_table(w, out, f"wilcoxon_{ref}_vs_baselines_{regime}{suffix}",
                            f"Wilcoxon signed-rank, {ref} vs each baseline, per-class F1, Holm ({regime}{suffix})")
            fn = friedman_nemenyi(data)
            if fn:
                save_json(fn, out / f"friedman_nemenyi_{regime}{suffix}.json")
                if regime == "official":
                    fig_cd(pd.Series(fn["average_ranks"]), fn["critical_difference"],
                           figs / f"critical_difference_{regime}{suffix}.png", dpi)
    if len(pc_main):
        pc = pc_main[pc_main["regime"] == "official"]
        bar = mean_std(pc, ["class", "model"], ["f1"])
        fig_bar(bar, "class", "model", "f1", figs / "per_class_f1_official.png", "F1 (mean over seeds)", dpi)


def noninferiority(long, ref, out, tree="rf"):
    rows = []
    b = long[(long["experiment"] == "main") & (long["level"] == "label2")]
    for regime in sorted(b["regime"].dropna().unique()):
        g = b[b["regime"] == regime].pivot_table(index="seed", columns="model", values="macro_f1")
        if not {ref, tree} <= set(g.columns):
            continue
        d = (g[ref] - g[tree]).dropna()
        if len(d) < 2:
            continue
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"regime": regime, "n_seeds": len(d), "mean_diff": d.mean(), "ci95_low": d.mean() - half,
                     "ci95_high": d.mean() + half})
    if rows:
        write_table(pd.DataFrame(rows), out, "noninferiority_binary",
                    f"Binary macro-F1 of {ref} minus {tree}, paired by seed, 95% t-interval")


def posthoc_tables(runs, ref, out, tree="rf"):
    """Tables from step 11 (posthoc_v2.json): AURC and selective risk per confidence score, hierarchy
    consistency, binary macro-F1 at the validation-tuned threshold and its non-inferiority test."""
    main = [r for r in runs if r.get("experiment") == "main" and r.get("posthoc")]
    if not main:
        return
    scores = pd.DataFrame([{"model": r["model"], "regime": r["regime"], "seed": r["seed"], "score": name, **v}
                           for r in main for name, v in r["posthoc"]["scores"].items()])
    rest = pd.DataFrame([{"model": r["model"], "regime": r["regime"], "seed": r["seed"],
                          **{f"consistency_{k}": v for k, v in r["posthoc"]["consistency"].items()},
                          **{f"binary_{k}": v for k, v in r["posthoc"]["binary"].items()}} for r in main])
    risk_cols = [c for c in scores.columns if c.startswith("selective_risk_")]
    for regime in sorted(rest["regime"].unique()):
        sc = scores[scores["regime"] == regime]
        agg = mean_std(sc, ["model", "score"], ["aurc"] + risk_cols)
        write_table(pretty_table(agg, ["model", "score"], ["aurc"] + risk_cols), out, f"selective_scores_{regime}",
                    f"AURC and selective risk by confidence score, ties in expectation, {regime} split (%)")
        g = rest[rest["regime"] == regime]
        cons = ["consistency_c19_6", "consistency_c6_2", "consistency_joint"]
        agg = mean_std(g, ["model"], cons)
        write_table(pretty_table(agg, ["model"], cons, digits=3), out, f"hierarchy_consistency_{regime}",
                    f"Cross-level consistency of predicted labels, {regime} split (%)")
        binc = ["binary_macro_f1_default", "binary_macro_f1_tuned"]
        agg = mean_std(g, ["model"], binc + ["binary_threshold_tuned"])
        tab = pretty_table(agg, ["model"], binc)
        tab["threshold_tuned"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(agg["binary_threshold_tuned"],
                                                                         agg["binary_threshold_tuned_std"])]
        write_table(tab, out, f"binary_tuned_{regime}", f"Binary macro-F1 at 0.5 and at the tuned threshold, {regime} (%)")
    rows = []
    for regime in sorted(rest["regime"].unique()):
        piv = rest[rest["regime"] == regime].pivot_table(index="seed", columns="model", values="binary_macro_f1_tuned")
        if not {ref, tree} <= set(piv.columns):
            continue
        d = (piv[ref] - piv[tree]).dropna()
        if len(d) < 2:
            continue
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"regime": regime, "n_seeds": len(d), "mean_diff": d.mean(), "ci95_low": d.mean() - half,
                     "ci95_high": d.mean() + half})
    if rows:
        write_table(pd.DataFrame(rows), out, "noninferiority_binary_tuned",
                    f"Binary macro-F1 at tuned thresholds, {ref} minus {tree}, paired by seed, 95% t-interval")


def abstain_and_routing(runs, out, figs, dpi):
    main = [r for r in runs if r.get("experiment") == "main"]
    ab = pd.DataFrame([{"model": r["model"], "regime": r.get("regime"), "seed": r["seed"],
                        **{k: (r.get("abstain") or {}).get(k) for k in ABSTAIN_KEYS}} for r in main if r.get("abstain")])
    for regime in sorted(ab["regime"].unique()) if len(ab) else []:
        agg = mean_std(ab[ab["regime"] == regime], ["model"], ABSTAIN_KEYS).sort_values("aurc")
        write_table(pretty_table(agg, ["model"], ABSTAIN_KEYS), out, f"abstain_{regime}",
                    f"Calibrated abstention, {regime} split, mean ± std over seeds (%)")
    curves = [{"model": r["model"], "seed": r["seed"], "coverage": c, "risk": k}
              for r in main if r.get("regime") == "official" and (r.get("abstain") or {}).get("curve")
              for c, k in zip(r["abstain"]["curve"]["coverage"], r["abstain"]["curve"]["risk"])]
    if curves:
        cv = pd.DataFrame(curves).groupby(["model", "coverage"], as_index=False)["risk"].mean()
        fig_lines(cv, "coverage", "risk", "model", figs / "risk_coverage.png", "Coverage", "Selective risk", dpi)
    ro = [r for r in main if (r.get("routing") or {}).get("nmi_label6") is not None]
    if ro:
        rows = [{"model": r["model"], "regime": r["regime"], "seed": r["seed"], "nmi_label6": r["routing"]["nmi_label6"],
                 "nmi_label19": r["routing"]["nmi_label19"],
                 **{f"e{k}": v for k, v in enumerate(r["routing"]["expert_share"])}} for r in ro]
        df = pd.DataFrame(rows)
        write_table(df.drop(columns="seed").groupby(["model", "regime"], as_index=False).mean().round(4), out,
                    "expert_routing", "Expert specialisation (NMI) and expert load share, mean over seeds")
        off = [r for r in ro if r["regime"] == "official"]
        if off:
            counts = pd.DataFrame([r["routing"]["counts_by_category"] for r in off]).apply(
                lambda col: np.sum(np.vstack(col.dropna().to_list()), axis=0))
            share = counts.apply(lambda v: v / v.sum()).T
            share.columns = [f"expert {k}" for k in share.columns]
            share.to_csv(out / "expert_by_category_official.csv")
            fig_heatmap(share, figs / "expert_by_category_official.png", dpi,
                        "Share of each category routed to each expert (official split)")


def ablation_tables(runs, long, out):
    ab = long[long["experiment"] == "ablation"]
    if not len(ab):
        return
    ab = ab.assign(variant=np.where(ab["tag"] == "window", "T=" + ab["T"].astype(str), ab["tag"]))
    piv = mean_std(ab, ["variant", "level"], ["macro_f1"]).pivot(index="variant", columns="level")
    write_table(piv.reset_index().round(4), out, "ablation", "TRACE-MoE ablation, macro-F1 mean and std over seeds")
    pc = per_class_table([r for r in runs if r.get("experiment") == "ablation"])
    if not len(pc):
        return
    # window-length runs are compared in the ablation table only; the tests pair each variant with "full"
    pc = pc[pc["tag"] != "window"].assign(model=lambda d: d["tag"])
    tests = wilcoxon_vs_reference(pc, "full")
    if len(tests):
        write_table(tests.rename(columns={"baseline": "variant"}), out, "ablation_tests",
                    "Full TRACE-MoE minus each ablation variant, per-class F1, Wilcoxon + Holm")


def robustness_tables(runs, long, ref, out, figs, dpi, main_eps):
    rob = [r for r in runs if r.get("experiment") == "robustness" and r.get("curves")]
    if not rob:
        return
    rows, asr = [], []
    for r in rob:
        for curve, by_eps in r["curves"].items():
            for eps, m in by_eps.items():
                rows.append({"model": r["model"], "seed": r["seed"], "curve": curve, "eps": float(eps),
                             "macro_f1_label19": m["label19"]["macro_f1"], "macro_f1_label2": m["label2"]["macro_f1"],
                             "attack_detection_rate": m.get("attack_detection_rate")})
                for tname, tm in (m.get("transfer") or {}).items():
                    rows.append({"model": f"{tname} (transfer from {r['model']})", "seed": r["seed"], "curve": curve,
                                 "eps": float(eps), "macro_f1_label19": tm["label19"]["macro_f1"],
                                 "macro_f1_label2": tm["label2"]["macro_f1"],
                                 "attack_detection_rate": tm.get("attack_detection_rate")})
                if curve == "adv_finetuned" and abs(float(eps) - main_eps) < 1e-9:
                    asr += [{"model": r["model"], "seed": r["seed"], "class": c, "asr": v}
                            for c, v in m["asr_per_class"].items()]
    rc = pd.DataFrame(rows)
    write_table(rc, out, "robustness_curves", "Robustness curves per model, seed and epsilon")
    clean = rc[(rc["curve"] == "clean") & ~rc["model"].str.contains("transfer")]
    fig_lines(clean.groupby(["model", "eps"], as_index=False)["attack_detection_rate"].mean(), "eps",
              "attack_detection_rate", "model", figs / "robustness_curve_clean_model.png", "epsilon",
              "Attack detection rate", dpi)
    tuned = rc[(rc["curve"] == "adv_finetuned") & ~rc["model"].str.contains("transfer")]
    fig_lines(tuned.groupby(["model", "eps"], as_index=False)["attack_detection_rate"].mean(), "eps",
              "attack_detection_rate", "model", figs / "robustness_curve_adv_finetuned.png", "epsilon",
              "Attack detection rate", dpi)
    main19 = long[(long["experiment"] == "main") & (long["regime"] == "official") & (long["level"] == "label19")]
    after = long[(long["experiment"] == "robustness") & (long["level"] == "label19")]
    summ = []
    for m in sorted(set(r["model"] for r in rob)):
        at_eps = rc[(rc["model"] == m) & (np.isclose(rc["eps"], main_eps))]
        summ.append({"model": m, "clean_macro_f1": main19[main19["model"] == m]["macro_f1"].mean(),
                     "clean_macro_f1_after_finetune": after[after["model"] == m]["macro_f1"].mean(),
                     "attacked_macro_f1_clean_model": at_eps[at_eps["curve"] == "clean"]["macro_f1_label19"].mean(),
                     "attacked_macro_f1_finetuned": at_eps[at_eps["curve"] == "adv_finetuned"]["macro_f1_label19"].mean(),
                     "asr_clean_model": 1 - at_eps[at_eps["curve"] == "clean"]["attack_detection_rate"].mean(),
                     "asr_finetuned": 1 - at_eps[at_eps["curve"] == "adv_finetuned"]["attack_detection_rate"].mean()})
    s = pd.DataFrame(summ)
    s["clean_drop"] = s["clean_macro_f1"] - s["clean_macro_f1_after_finetune"]
    write_table(s.round(4), out, "robustness_asr", f"Clean vs attacked macro-F1 and ASR at eps={main_eps}")
    if asr:
        t = wilcoxon_vs_reference(pd.DataFrame(asr), ref, metric="asr")
        if len(t):
            t["mean_asr_reduction"] = -t["mean_diff"]          # baseline ASR minus reference ASR
            write_table(t, out, "robustness_asr_tests",
                        f"Per-class ASR at eps={main_eps} after fine-tuning: {ref} vs each baseline, Wilcoxon + Holm")
    tm = []
    for p in sorted((Path(out).parent / "robustness").glob("transfer_matrix_seed*/transfer_matrix.json")):
        tm += pd.read_json(p).to_dict("records")
    if tm:
        mat = pd.DataFrame(tm).pivot_table(index="surrogate", columns="target", values="asr", aggfunc="mean")
        write_table(mat.reset_index().round(4), out, "transfer_attack_matrix",
                    "Black-box transfer ASR (rows: surrogate, columns: target), mean over seeds")


def zero_day_tables(runs, out):
    rows = [{"model": r["model"], "held_out": r["ood"]["held_out_class"], "seed": r["seed"],
             "energy_auroc": r["ood"]["energy"]["auroc"], "energy_fpr95": r["ood"]["energy"]["fpr_at_95tpr"],
             "msp_auroc": r["ood"]["msp"]["auroc"], "flagged_unknown": r["ood"]["held_out_flagged_unknown"],
             "caught_as_attack": r["ood"]["held_out_detected_as_attack"], "caught_any": r["ood"]["held_out_caught_any"],
             "false_alarm_known": r["ood"]["known_false_alarm_rate"],
             "known_macro_f1": r["ood"]["known_classes_macro_f1"]}
            for r in runs if r.get("experiment") == "zero_day" and r.get("ood")]
    if not rows:
        return
    z = pd.DataFrame(rows)
    cols = [c for c in z.columns if c not in ("model", "held_out", "seed")]
    write_table(z.groupby(["model", "held_out"], as_index=False)[cols].mean().round(4), out, "zero_day",
                "Unknown-attack detection per held-out class, mean over seeds")
    write_table(z.groupby("model", as_index=False)[cols].mean().round(4), out, "zero_day_overall",
                "Unknown-attack detection, mean over held-out classes and seeds")


def main():
    args, cfg, log = setup("Compare runs", "09_compare", None)
    R = Path(cfg["paths"]["results_dir"])
    out, figs = R / "tables", R / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    dpi = cfg["figures"]["dpi"]
    ref = cfg["main"]["reference_model"]
    all_runs = load_runs(R)
    if not all_runs:
        raise SystemExit(f"No runs under {R}: run steps 03-07 first")
    runs = latest_runs(all_runs)
    if len(runs) < len(all_runs):
        log.info(f"{len(all_runs) - len(runs)} older copies of re-run runs (<name>__vN) ignored; newest kept")
    long = long_table(runs)
    long.to_csv(out / "all_runs_long.csv", index=False)
    log.info(f"{len(runs)} runs, {len(long)} run x level rows")
    pc_main = per_class_table([r for r in runs if r.get("experiment") == "main"])
    main_tables(long, pc_main, ref, out, figs, dpi)
    noninferiority(long, ref, out)
    abstain_and_routing(runs, out, figs, dpi)
    posthoc_tables(runs, ref, out)
    ablation_tables(runs, long, out)
    robustness_tables(runs, long, ref, out, figs, dpi, float(cfg["robustness"]["main_epsilon"]))
    zero_day_tables(runs, out)
    off = [r for r in runs if r.get("experiment") == "main" and r.get("model") == ref and r.get("regime") == "official"]
    if off:
        r = min(off, key=lambda x: x["seed"])
        names = r["hierarchy"]["classes"]["label19"]
        fig_confusion(np.array(r["test"]["label19"]["confusion_matrix"]), names,
                      figs / f"confusion_{ref}_official.png", dpi, f"{ref}, official split, seed {r['seed']}")
    log.info(f"Tables in {out}, figures in {figs}")


if __name__ == "__main__":
    main()
