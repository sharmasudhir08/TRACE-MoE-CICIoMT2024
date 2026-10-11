"""Step 2. Data sanity checks (a supporting step, not a contribution of TRACE-MoE).

  duplicates.json / duplicates_per_class.csv  exact duplicates within and across train/test
  shortcut_features_label19.csv / _label6.csv macro-F1 of a tree that sees ONE feature only
  suggested_drop_features.json                features above data.shortcut_threshold
  autocorrelation.csv                         lag-1 autocorrelation in file order vs shuffled
  split_inflation.csv                         RF macro-F1 under each split regime, with/without dedup
  audit_report.md                             a readable summary of all of the above
"""
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup
from trace_iomt import audit
from trace_iomt.compare import _plt, write_table
from trace_iomt.data.loaders import feature_cols
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import classical_proba, fine_to_coarse, fit_classical, load_prepared, rows_from_windows
from trace_iomt.metrics import evaluate_levels
from trace_iomt.utils import save_json


def add(p):
    p.add_argument("--rf-trees", type=int, default=100)
    p.add_argument("--rf-rows-per-class", type=int, default=20000)


def main():
    args, cfg, log = setup("Data sanity checks", "02_audit", add)
    out = Path(cfg["paths"]["results_dir"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    dpi = cfg["figures"]["dpi"]
    df = load_prepared(cfg)
    feats = feature_cols(df)

    log.info("1/4 duplicates")
    dup, per_class = audit.duplicate_report(df, feats)
    save_json(dup, out / "duplicates.json")
    per_class.to_csv(out / "duplicates_per_class.csv", index=False)
    log.info(dup)

    log.info("2/4 single-feature shortcut test")
    sc = {}
    for lvl in ("label19", "label6"):
        sc[lvl] = audit.shortcut_test(df, feats, lvl, seed=cfg["train"]["seed"])
        sc[lvl].to_csv(out / f"shortcut_features_{lvl}.csv", index=False)
    drop = audit.suggest_drop(sc["label19"], cfg["data"]["shortcut_threshold"])
    save_json({"threshold": cfg["data"]["shortcut_threshold"], "drop": drop}, out / "suggested_drop_features.json")
    plt = _plt()
    top = sc["label19"].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.barh(top["feature"], top["macro_f1"]); ax.axvline(cfg["data"]["shortcut_threshold"], color="r", ls="--")
    ax.set_xlabel("Macro-F1 (19 classes) from one feature"); plt.tight_layout()
    plt.savefig(out / "shortcut_features.png", dpi=dpi); plt.close()

    log.info("3/4 row-order autocorrelation")
    ac = audit.row_order_autocorrelation(df, feats)
    ac.to_csv(out / "autocorrelation.csv", index=False)

    log.info("4/4 split inflation (RF under each regime, with and without duplicate removal)")
    rows = []
    c = {**cfg, "classical": {**cfg["classical"], "n_estimators": args.rf_trees,
                              "max_rows_per_class": args.rf_rows_per_class}}
    for regime in ("random", "official", "temporal"):
        for dedup in (False, True):
            c2 = {**c, "data": {**c["data"], "drop_exact_duplicates": dedup}}
            wd = build_windowed(df, c2, regime=regime, T=1, logger=log, drop_features=[])
            clf, _, _ = fit_classical("rf", wd, c2, cfg["train"]["seed"])
            X, y = rows_from_windows(wd, "test")
            pr = fine_to_coarse(classical_proba(clf, X, wd.hierarchy.n("label19")), wd.hierarchy)
            m = evaluate_levels(y, pr, wd.hierarchy)
            rows.append({"regime": regime, "dedup": dedup, "n_test": len(X),
                         **{f"macro_f1_{l}": m[l]["macro_f1"] for l in ("label2", "label6", "label19")}})
            log.info(rows[-1])
    infl = pd.DataFrame(rows)
    write_table(infl.round(4), out, "split_inflation", "Random-forest macro-F1 under different split protocols")

    rep = ["# Leakage audit", "",
           f"Rows: {dup['n_rows']:,}; features: {dup['n_features']}", "",
           "## Duplicates", ""] + [f"- {k}: {v}" for k, v in dup.items()] + [
           "", "## Single-feature shortcuts (top 10, label19)", "",
           "```", sc["label19"].head(10).round(4).to_string(index=False), "```",
           "", f"Suggested features to drop (macro-F1 >= {cfg['data']['shortcut_threshold']}): {drop or 'none'}", "",
           "## Row order", "",
           f"Mean lag-1 autocorrelation in file order: {np.nanmean(ac['lag1_autocorr']) if len(ac) else float('nan'):.3f} "
           f"vs shuffled {np.nanmean(ac['lag1_autocorr_shuffled']) if len(ac) else float('nan'):.3f}. "
           "Clearly higher in file order means consecutive flows carry information, which windows use.", "",
           "## Split inflation (RF)", "", "```", infl.round(4).to_string(index=False), "```"]
    (out / "audit_report.md").write_text("\n".join(rep))
    log.info(f"Audit written to {out}")


if __name__ == "__main__":
    main()
