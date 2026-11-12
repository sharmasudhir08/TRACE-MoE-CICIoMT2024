"""Step 17 (twin study, H11; added 2026-11-05). Tables and the H11 verdict from the runs of step 16.

Reads results/twin/<run>/ (metrics.json, posthoc_v2.json, predictions.npz) and writes to results/tables/:
  twin_ablation        mean ± std over seeds per variant (V0-V4, rf): 19-class, min recall, 6-category, binary
                       (tuned), AURC (MSP), AURC with ‖Δ‖ as the confidence score, mean AUROC of ‖Δ‖ over attack
                       classes on twinned hosts, hierarchy consistency
  twin_differences     paired differences between variants over their common seeds, with 95% t-intervals
  twin_wilcoxon        V4 vs V0 and V4 vs rf over the 19 seed-averaged class blocks, Holm-corrected
  twin_per_class       per-class F1 and recall per variant (mean over seeds)
  twin_delta_auroc     AUROC of ‖Δ‖ for each attack class against benign test windows, all hosts and twinned only
  twin_sync            the synchroniser's counts per V4 seed
and results/twin_h11.json (verdict and evidence). The file bootstrap of V4 - V0 needs the test capture file of
every window; it rebuilds the windows of each seed once (no model, about 2-3 minutes per seed) and caches them
in results/twin/window_meta_seed<k>.npz. The same cache gives
  twin_time_to_detect  first attacker packet -> end of the first window flagged Attack, per variant. Run it after the training lanes, not alongside them (memory).

  python scripts/17_twin_compare.py --config configs/twin.yaml
"""
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup
from trace_iomt import twin_compare as tc
from trace_iomt import twin_data
from trace_iomt.compare import latest_runs, load_runs, long_table, mean_std, per_class_table, pretty_table, write_table
from trace_iomt.data.windows import build_windowed
from trace_iomt.twin_study import project_path
from trace_iomt.utils import save_json, set_seed

METRICS = ["macro_f1_19", "min_recall_19", "macro_f1_6", "macro_f1_2_tuned", "aurc", "aurc_delta", "delta_auroc",
           "consistency"]
ORDER = ["V0", "V1", "V2", "V3", "V4", "rf"]
PAIRS = [("V4", "V0"), ("V1", "V0"), ("V2", "V0"), ("V3", "V0"), ("V2", "V1"), ("V3", "V2"), ("V4", "V3")]


def add(p):
    p.add_argument("--skip-bootstrap", action="store_true", help="tables only; the H11 verdict stays provisional")
    p.add_argument("--resamples", type=int, default=1000)


def twin_runs(cfg):
    runs = [r for r in latest_runs(load_runs(cfg["paths"]["results_dir"])) if r.get("experiment") == "twin"]
    return sorted(runs, key=lambda r: (ORDER.index(tc.variant_of(r)), r["seed"]))


def run_rows(cfg, runs):
    """Summary row per run, plus the twin's own scores for V1-V4: AURC with ‖Δ‖ as the confidence and the mean
    over attack classes of the AUROC of ‖Δ‖ against benign windows on twinned hosts (the twin never sees an
    attack, so every class is unseen by it; this replaces the leave-one-attack-out flag rate in the table)."""
    twinned = list(twin_data.twinned_devices(project_path(cfg["twin"]["devices"])))
    rows = []
    for r in runs:
        row = tc.run_row(r)
        row["aurc_delta"] = row["delta_auroc"] = np.nan
        if row["variant"] in ("V1", "V2", "V3", "V4"):
            z = np.load(Path(r["_dir"]) / "predictions.npz")
            row["aurc_delta"] = tc.delta_aurc(z["twin_delta_mean"], z["y_true_label19"], z["y_pred_label19"])
            on_twinned = np.isin(np.array(r["data_report"]["twin"]["devices"])[z["device_id"]], twinned)
            aucs = tc.delta_auroc(z["twin_delta_mean"], z["y_true_label19"], r["hierarchy"]["classes"]["label19"],
                                  mask=on_twinned)
            row["delta_auroc"] = float(np.mean(list(aucs.values()))) if aucs else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(rows, runs, out, log):
    agg = mean_std(rows, ["variant"], METRICS)
    agg["order"] = agg["variant"].map(ORDER.index)
    agg = agg.sort_values("order").drop(columns="order")
    table = pretty_table(agg, ["variant", "n_seeds"], METRICS, pct=True)
    table.insert(0, "name", table["variant"].map(tc.VARIANT_NAMES))
    long = long_table(runs)
    long["variant"] = np.where(long["model"] == "rf", "rf", long["tag"])
    write_table(table, out, "twin_ablation", "Digital Twin ablation on the device rows, official split (%)",
                provenance=(long, ["variant"]))
    log.info("\n" + table.to_string(index=False))


def window_meta(cfg, seed, log):
    """Per test window of this seed (cached): capture file, segment (capture|device), end time of the last row and
    the 19-class label (to check alignment); per segment: time of its first packet to or from an attacker."""
    cache = Path(cfg["paths"]["results_dir"]) / "twin" / f"window_meta_seed{seed}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    t = cfg["twin"]
    set_seed(seed)
    rows = twin_data.apply_labels(twin_data.load_attack_rows(project_path(t["rows_dir"]), t["cap"], seed), t["labels"])
    c = {**cfg, "train": {**cfg["train"], "seed": seed}}
    wd = build_windowed(twin_data.feature_table(rows), c, regime="official", seed=seed, logger=log)
    last = wd.row_index[wd.starts["test"] + wd.T - 1]
    seg = rows["source_file"].to_numpy()[last].astype(str)
    hit = rows[(rows["split"] == "test") & (rows["n_attacker_packets"] > 0)]
    first = hit.groupby("source_file")["t_start"].min()
    meta = {"files": np.array([f.split("|")[0] for f in seg]), "segment": seg,
            "t_end_last": rows["t_end"].to_numpy(dtype=np.float64)[last],
            "y_true_label19": wd.labels("test", "label19").astype(np.int16),
            "attack_segments": first.index.to_numpy().astype(str), "first_attack_t": first.to_numpy(dtype=np.float64)}
    np.savez_compressed(cache, **meta)
    return meta


def bootstrap(cfg, runs, resamples, log):
    """File bootstrap of V4 - V0 (19-class macro-F1) over the seeds both have."""
    by = {(tc.variant_of(r), r["seed"]): r for r in runs}
    seeds = sorted(s for (v, s) in by if v == "V4" and ("V0", s) in by)
    if not seeds:
        return None
    cms = {"V4": {}, "V0": {}}
    for s in seeds:
        meta = window_meta(cfg, s, log)
        files = meta["files"]
        for v in ("V4", "V0"):
            z = np.load(Path(by[(v, s)]["_dir"]) / "predictions.npz")
            if not aligned(z, meta):
                raise SystemExit(f"[TWIN_WINDOW_MISMATCH] {v} seed {s}: the rebuilt test windows differ from the run's "
                                 f"predictions.npz. Check that configs/twin.yaml and the device rows are the ones the "
                                 f"run used, then delete window_meta_seed{s}.npz and run again.")
            n = len(by[(v, s)]["hierarchy"]["classes"]["label19"])
            cms[v][s] = tc.confusion_by_file(z["y_true_label19"], z["y_pred_label19"], files, n)
    return tc.file_bootstrap(cms["V4"], cms["V0"], resamples)


def aligned(z, meta):
    return np.array_equal(z["y_true_label19"], meta["y_true_label19"])


def time_to_detect_table(cfg, runs, log):
    """Time from the first attacker packet of each attacked segment to the end of the first window flagged Attack
    (binary head), per variant: median and 90th percentile over segments and seeds, and the share never flagged.
    The detector starts observing at the start of each segment's kept block (at most twin.cap rows)."""
    rows = []
    for r in runs:
        v = tc.variant_of(r)
        if v == "rf":
            continue                                    # T = 1 windows: no aligned window metadata
        meta = window_meta(cfg, r["seed"], log)
        z = np.load(Path(r["_dir"]) / "predictions.npz")
        if not aligned(z, meta):
            raise SystemExit(f"[TWIN_WINDOW_MISMATCH] {r['run']}: rebuilt test windows differ from predictions.npz")
        attack = r["hierarchy"]["classes"]["label2"].index("Attack")
        ttd = tc.time_to_detect(meta["segment"], meta["t_end_last"], z["y_pred_label2"] == attack,
                                dict(zip(meta["attack_segments"], meta["first_attack_t"])))
        rows += [{"variant": v, "seed": r["seed"], **x} for x in ttd]
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    g = df.groupby("variant")
    out = pd.DataFrame({"segments": g.size(), "missed_share": g["seconds"].apply(lambda x: x.isna().mean()),
                        "median_s": g["seconds"].median(), "p90_s": g["seconds"].quantile(0.9)}).round(4)
    return out.reset_index()


def delta_auroc_table(cfg, runs):
    twinned = twin_data.twinned_devices(project_path(cfg["twin"]["devices"]))
    rows = []
    for r in runs:
        v = tc.variant_of(r)
        if v not in ("V1", "V2", "V3", "V4"):
            continue
        z = np.load(Path(r["_dir"]) / "predictions.npz")
        classes = r["hierarchy"]["classes"]["label19"]
        devices = np.array(r["data_report"]["twin"]["devices"])
        is_twinned = np.isin(devices[z["device_id"]], list(twinned))
        for scope, mask in (("all hosts", None), ("twinned", is_twinned)):
            for cls, auc in tc.delta_auroc(z["twin_delta_mean"], z["y_true_label19"], classes, mask=mask).items():
                rows.append({"variant": v, "seed": r["seed"], "scope": scope, "class": cls, "auroc": auc})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    return df.groupby(["scope", "class", "variant"])["auroc"].mean().unstack("variant").round(4).reset_index()


def main():
    args, cfg, log = setup("Twin study tables and H11", "17_twin_compare", add)
    runs = twin_runs(cfg)
    if not runs:
        raise SystemExit(f"[TWIN_NO_RUNS] no runs in {cfg['paths']['results_dir']}/twin: run step 16 first")
    out = Path(cfg["paths"]["results_dir"]) / "tables"
    rows = run_rows(cfg, runs)
    log.info(f"{len(runs)} twin runs: " + ", ".join(f"{v} x{n}" for v, n in rows["variant"].value_counts().items()))
    ablation_table(rows, runs, out, log)

    diffs = [d for m in ("macro_f1_19", "min_recall_19") for a, b in PAIRS
             if (d := tc.paired_difference(rows, a, b, m)) is not None]
    write_table(pd.DataFrame(diffs), out, "twin_differences", "Paired differences between twin variants")
    pc = per_class_table(runs)
    pc["variant"] = ["rf" if m == "rf" else t for m, t in zip(pc["model"], pc["tag"])]
    write_table(tc.class_block_wilcoxon(pc, "V4", ["V0", "rf"]), out, "twin_wilcoxon",
                "V4 vs V0 and random forest over 19 seed-averaged class blocks (Holm)")
    write_table(pc.groupby(["class", "variant"])[["f1", "recall"]].mean().round(4).reset_index(), out,
                "twin_per_class", "Per-class F1 and recall per twin variant (mean over seeds)")
    write_table(delta_auroc_table(cfg, runs), out, "twin_delta_auroc", "AUROC of the twin discrepancy per attack class")
    if not args.skip_bootstrap:
        write_table(time_to_detect_table(cfg, runs, log), out, "twin_time_to_detect",
                    "Time to detect per twin variant, T = 16 (s)")
    sync = pd.DataFrame([{"seed": r["seed"], **r.get("sync", {}), **r.get("sync_counts", {})}
                         for r in runs if tc.variant_of(r) == "V4"])
    write_table(sync, out, "twin_sync", "Synchroniser counts per V4 seed")

    primary = next((d for d in diffs if (d["a"], d["b"], d["metric"]) == ("V4", "V0", "macro_f1_19")), None)
    boot = None if args.skip_bootstrap else bootstrap(cfg, runs, args.resamples, log)
    verdict = tc.h11_verdict(primary, boot) + ("" if boot is not None or primary is None else " (provisional: no bootstrap)")
    save_json({"hypothesis": "H11", "verdict": verdict, "primary": primary, "file_bootstrap": boot,
               "secondary": [d for d in diffs if d is not primary]}, Path(cfg["paths"]["results_dir"]) / "twin_h11.json")
    log.info(f"H11: {verdict} | V4 - V0 {primary and round(100 * primary['mean'], 2)} points, "
             f"t-interval {primary and [round(100 * primary['t_low'], 2), round(100 * primary['t_high'], 2)]}, "
             f"file bootstrap {boot and [round(100 * boot['low'], 2), round(100 * boot['high'], 2)]}")


if __name__ == "__main__":
    main()
