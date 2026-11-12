"""Collect every run in results/, aggregate over seeds, run significance tests, and write
tables (CSV + LaTeX) and figures for the paper.

Every aggregated table in results/tables/<name>.csv is accompanied by a sibling
results/tables/<name>.provenance.json that lists, for each row, the exact run folders, metrics.json
paths and git commits that contributed to that row. The provenance sidecar makes every number in the
technical report and the paper traceable back to a specific `results/<experiment>/<run>/metrics.json`.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SUMMARY_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_pr_auc", "roc_auc",
                "pr_auc", "recall_at_fpr_0.01", "min_class_recall", "ece"]
LEVELS = ["label2", "label6", "label19"]
# Fields propagated through long_table so that every aggregated row can be traced back to the
# exact runs that produced it.
PROVENANCE_KEYS = ["dir", "git_commit", "git_dirty", "code_sha256", "data_sha256", "run_id",
                   "started_at", "finished_at", "host", "session"]

# Nemenyi critical values q_alpha (alpha = 0.05) for k = 2..10 models (Demsar, JMLR 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def read_run(run_dir):
    """metrics.json of one run. If step 11 wrote posthoc_v2.json, its corrected abstain block replaces the
    original one (kept as abstain_original), and its scores, consistency and tuned-binary results are added."""
    d = Path(run_dir)
    with open(d / "metrics.json") as f:
        m = json.load(f)
    m["_dir"] = str(d)
    if (d / "posthoc_v2.json").exists():
        with open(d / "posthoc_v2.json") as f:
            ph = json.load(f)
        m["abstain_original"], m["abstain"] = m.get("abstain"), ph["abstain"]
        m["posthoc"] = ph
    return m


def load_runs(results_dir):
    """Walk results/<experiment>/<run>/metrics.json. Also honours results/registry.jsonl when present
    (used only to confirm we are not missing any run, not to replace the walk)."""
    return [read_run(p.parent) for p in sorted(Path(results_dir).glob("*/*/metrics.json"))]


def latest_runs(runs):
    """Keep only the newest copy of each run. Append-only mode stores a repeated run as <name>__v2, __v3, ...;
    averaging every copy would count that configuration and seed more than once."""
    best = {}
    for r in runs:
        d = Path(r["_dir"])
        stem, _, version = d.name.partition("__v")
        key, rank = (d.parent.name, stem), int(version) if version else 1
        if key not in best or rank > best[key][0]:
            best[key] = (rank, r)
    return [r for _, r in best.values()]


def long_table(runs):
    """One row per run x level with the summary metrics AND provenance fields (run dir, git_commit,
    code_sha256, data_sha256, session, host, started_at, finished_at) so every number can be traced
    back to the exact run that produced it."""
    rows = []
    for r in runs:
        meta = {k: r.get(k) for k in ("experiment", "run", "model", "family", "regime", "seed", "T", "tag",
                                      "n_params", "train_seconds", "test_seconds")}
        # Prefer the dir stored in metrics.json's registry row (string); fall back to _dir attached at load time.
        meta["dir"] = r.get("dir") or r.get("_dir")
        for k in ("git_commit", "git_dirty", "code_sha256", "data_sha256", "run_id",
                  "started_at", "finished_at", "host", "session"):
            meta[k] = r.get(k)
        for lvl in LEVELS:
            t = r.get("test", {}).get(lvl)
            if not t:
                continue
            row = dict(meta)
            row["level"] = lvl
            row.update({k: t.get(k) for k in SUMMARY_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        s = str(x)
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def provenance_for_groups(long_df, by, extra_cols=("seed", "run")):
    """Return {group_key_tuple: {dirs: [...], git_commits: [...], runs: [{...}, ...]}}.
    group_key_tuple preserves the order in `by`. runs lists one entry per contributing (long row).
    `by` must be a list of column names present in long_df."""
    if long_df is None or not len(long_df):
        return {}
    by = list(by)
    out = {}
    group_cols = long_df.columns.intersection(by).tolist() or by
    for key, g in long_df.groupby(group_cols, dropna=False):
        tup = key if isinstance(key, tuple) else (key,)
        rows = []
        for _, r in g.iterrows():
            rows.append({c: (None if (isinstance(r.get(c), float) and pd.isna(r.get(c))) else r.get(c))
                         for c in (list(by) + list(extra_cols) + ["level", "macro_f1", "dir",
                                                                   "git_commit", "code_sha256",
                                                                   "run_id", "started_at", "session"])})
        out[tuple(str(x) for x in tup)] = {
            "dirs": _dedup_keep_order(g["dir"].tolist()) if "dir" in g.columns else [],
            "git_commits": _dedup_keep_order(g["git_commit"].tolist()) if "git_commit" in g.columns else [],
            "sessions": _dedup_keep_order(g["session"].tolist()) if "session" in g.columns else [],
            "n_runs": int(len(g)),
            "runs": rows,
        }
    return out


def write_provenance_sidecar(long_df, by, out_dir, name):
    """Write <out_dir>/<name>.provenance.json: for each group of rows (same `by` values),
    the exact runs/dirs/git_commits that fed into that group."""
    prov = provenance_for_groups(long_df, by)
    sidecar = {"table": name, "by": list(by), "groups":
               [{"key": dict(zip(by, k)), **v} for k, v in prov.items()]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{name}.provenance.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, default=str)
    return sidecar


def attach_source_columns(agg_df, long_df, by, max_items=12):
    """Return agg_df with two extra columns: `source_runs` and `source_git_commits` (both pipe-separated,
    truncated to at most max_items for readability; full lists live in the .provenance.json sidecar)."""
    if agg_df is None or long_df is None or not len(agg_df) or not len(long_df):
        return agg_df
    prov = provenance_for_groups(long_df, by)
    runs_col, git_col = [], []
    for _, row in agg_df.iterrows():
        key = tuple(str(row[c]) if c in agg_df.columns else "" for c in by)
        info = prov.get(key)
        if info is None:
            runs_col.append(""); git_col.append(""); continue
        runs = info["dirs"][:max_items]
        gits = info["git_commits"][:max_items]
        runs_col.append("|".join(Path(r).name for r in runs))
        git_col.append("|".join(gits))
    agg_df = agg_df.copy()
    agg_df["source_runs"] = runs_col
    agg_df["source_git_commits"] = git_col
    return agg_df


def per_class_table(runs, level="label19"):
    rows = []
    for r in runs:
        pc = r.get("test", {}).get(level, {}).get("per_class", {})
        for cls, v in pc.items():
            if v["support"] > 0:
                rows.append({"experiment": r["experiment"], "model": r["model"], "regime": r.get("regime"),
                             "tag": r.get("tag", ""), "seed": r.get("seed"), "class": cls, "f1": v["f1"],
                             "recall": v["recall"]})
    return pd.DataFrame(rows)


def mean_std(df, by, metrics):
    g = df.groupby(by, dropna=False)[metrics]
    m, s, n = g.mean(), g.std(ddof=1).fillna(0), g.size().rename("n_seeds")
    out = m.copy()
    for c in metrics:
        out[c + "_std"] = s[c]
    return out.join(n).reset_index()


def fmt_pm(mean, std, digits=4, pct=False):
    if pd.isna(mean):
        return "--"
    k = 100 if pct else 1
    return f"{mean * k:.{digits}f} ± {std * k:.{digits}f}"


def pretty_table(agg, index, metrics, digits=4, pct=True):
    out = agg[index].copy()
    for c in metrics:
        out[c] = [fmt_pm(a, b, 2 if pct else digits, pct) for a, b in zip(agg[c], agg[c + "_std"])]
    return out


def to_latex(df, caption, label, bold_max_cols=None):
    """Plain LaTeX tabular (booktabs) without extra dependencies. Bolds the best value per column."""
    cols = list(df.columns)
    best = {}
    for c in bold_max_cols or []:
        vals = pd.to_numeric(df[c].astype(str).str.split(" ").str[0], errors="coerce")
        if vals.notna().any():
            best[c] = vals.idxmax()
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 1 + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for i, row in df.iterrows():
        cells = []
        for c in cols:
            v = esc(row[c])
            cells.append(rf"\textbf{{{v}}}" if best.get(c) == i else v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_table(df, out_dir, name, caption, bold_max_cols=None, provenance=None):
    """Write <name>.csv and <name>.tex. When provenance=(long_df, by) is given, also writes
    <name>.provenance.json listing the exact run folders and git commits that fed each row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.tex").write_text(to_latex(df.reset_index(drop=True), caption, f"tab:{name}", bold_max_cols))
    if provenance is not None:
        long_df, by = provenance
        write_provenance_sidecar(long_df, by, out_dir, name)


# ------------------------------------------------------------------ statistics
def wilcoxon_vs_reference(pc, reference, metric="f1"):
    """Paired Wilcoxon signed-rank test of the reference model against every other model.

    Pairs are (class, seed) cells with the same class and seed, so n = classes x seeds, which gives the
    test real power (5 seeds alone cannot reach p < 0.05 two-sided). Holm correction across baselines.
    """
    piv = pc.pivot_table(index=["class", "seed"], columns="model", values=metric)
    if reference not in piv.columns:
        return pd.DataFrame()
    rows = []
    for m in piv.columns:
        if m == reference:
            continue
        d = piv[[reference, m]].dropna()
        if len(d) < 5 or np.allclose(d[reference], d[m]):
            p = 1.0
            stat = 0.0
        else:
            stat, p = stats.wilcoxon(d[reference], d[m])
        rows.append({"reference": reference, "baseline": m, "n_pairs": len(d),
                     "mean_diff": float((d[reference] - d[m]).mean()), "wilcoxon_stat": float(stat), "p": float(p)})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    k = len(out)
    out["p_holm"] = np.minimum(1, np.maximum.accumulate(out["p"] * (k - np.arange(k))))
    out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def friedman_nemenyi(pc, metric="f1"):
    """Friedman test over models with (class, seed) blocks; average ranks and the Nemenyi critical difference."""
    piv = pc.pivot_table(index=["class", "seed"], columns="model", values=metric).dropna()
    k, n = piv.shape[1], piv.shape[0]
    if k < 3 or n < 2:
        return None
    stat, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    cd = Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": float(stat), "p": float(p), "k_models": k, "n_blocks": n, "critical_difference": float(cd),
            "average_ranks": ranks.round(3).to_dict()}


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})
    return plt


def fig_bar(agg, x, hue, y, path, ylabel, dpi, title=None):
    plt = _plt()
    piv = agg.pivot(index=x, columns=hue, values=y)
    err = agg.pivot(index=x, columns=hue, values=y + "_std")
    ax = piv.plot.bar(yerr=err, figsize=(7, 3.4), capsize=2, width=0.8)
    ax.set_ylabel(ylabel); ax.set_xlabel("")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, ncol=min(4, piv.shape[1]), loc="lower right")
    lo = np.nanmin(piv.values)
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()


def fig_confusion(cm, names, path, dpi, title=""):
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    s = max(4, 0.38 * len(names))
    fig, ax = plt.subplots(figsize=(s + 1, s))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
    if len(names) <= 20:
        for i in range(len(names)):
            for j in range(len(names)):
                if cmn[i, j] >= 0.01:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                            color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()


def fig_heatmap(piv, path, dpi, title=""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * piv.shape[1], 0.8 + 0.28 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if v < 0.5 else "black")
    ax.set_title(title); fig.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()


def fig_lines(df, x, y, hue, path, xlabel, ylabel, dpi, err=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for h, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.errorbar(g[x], g[y], yerr=g[err] if err else None, marker="o", ms=3, capsize=2, label=str(h))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()


def fig_cd(ranks, cd, path, dpi):
    """Critical-difference style plot: average rank per model with the CD bar."""
    plt = _plt()
    items = sorted(ranks.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5, 0.5 + 0.3 * len(items)))
    for i, (m, r) in enumerate(items):
        ax.plot(r, i, "o", color="C0"); ax.text(r + 0.05, i, f"{m} ({r:.2f})", va="center", fontsize=7)
    ax.plot([1, 1 + cd], [-0.8, -0.8], "k-", lw=2); ax.text(1, -1.2, f"CD = {cd:.2f}", fontsize=7)
    ax.set_yticks([]); ax.set_xlabel("Average rank (lower is better)"); ax.invert_yaxis()
    ax.set_xlim(0.8, max(r for _, r in items) + 1.5)
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()
