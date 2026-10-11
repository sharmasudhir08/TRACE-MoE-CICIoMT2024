"""Leakage audit (C1): duplicates, single-feature shortcuts, row-order structure and split inflation."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from .data.loaders import feature_cols


def _row_hash(df, cols):
    return pd.util.hash_pandas_object(df[cols].round(6), index=False).to_numpy()


def duplicate_report(df, feats=None):
    """Exact duplicate rows within each split, across train/test, and identical rows with different labels."""
    feats = feats or feature_cols(df)
    h = _row_hash(df, feats)
    split = df["split"].astype(str).to_numpy()
    lab = df["label19"].astype(str).to_numpy()
    out = {"n_rows": int(len(df)), "n_features": len(feats)}
    for s in np.unique(split):
        m = split == s
        n_dup = int(m.sum() - len(np.unique(h[m])))
        out[f"{s}_rows"] = int(m.sum())
        out[f"{s}_duplicate_rows"] = n_dup
        out[f"{s}_duplicate_pct"] = round(100 * n_dup / max(m.sum(), 1), 3)
    if {"train", "test"} <= set(split):
        tr = pd.Index(np.unique(h[split == "train"]))
        te_mask = split == "test"
        in_train = pd.Index(h[te_mask]).isin(tr)
        out["test_rows_also_in_train"] = int(in_train.sum())
        out["test_rows_also_in_train_pct"] = round(100 * in_train.mean(), 3)
    g = pd.DataFrame({"h": h, "y": lab}).groupby("h")["y"].nunique()
    conflict = g[g > 1].index
    out["feature_vectors_with_conflicting_labels"] = int(len(conflict))
    out["rows_with_conflicting_labels"] = int(pd.Index(h).isin(conflict).sum())
    per_class = pd.DataFrame({"h": h, "y": lab, "s": split})
    pc = per_class.groupby(["y", "s"]).agg(rows=("h", "size"), unique=("h", "nunique")).reset_index()
    pc["duplicate_pct"] = (100 * (1 - pc["unique"] / pc["rows"])).round(2)
    return out, pc


def _balanced(df, label, n, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in df.groupby(label, observed=True).indices.items():
        keep.append(idx if len(idx) <= n else rng.choice(idx, n, replace=False))
    return df.iloc[np.sort(np.concatenate(keep))] if keep else df.iloc[:0]


def shortcut_test(df, feats=None, label="label19", max_per_class=5000, max_depth=6, seed=42):
    """Macro-F1 of a small decision tree that sees ONE feature at a time (train split -> test split).

    A feature that alone nearly separates all classes is a shortcut (e.g. a capture artefact),
    which inflates scores and does not generalise.
    """
    feats = feats or feature_cols(df)
    tr = _balanced(df[df["split"] == "train"], label, max_per_class, seed)
    te = _balanced(df[df["split"] == "test"], label, max_per_class, seed)
    if len(te) == 0:   # no test split: hold out 30% of the sample
        te = tr.sample(frac=0.3, random_state=seed)
        tr = tr.drop(te.index)
    ytr, yte = tr[label].astype(str).to_numpy(), te[label].astype(str).to_numpy()
    rows = []
    for f in feats:
        xtr = np.nan_to_num(tr[[f]].to_numpy(np.float64))
        xte = np.nan_to_num(te[[f]].to_numpy(np.float64))
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(xtr, ytr)
        rows.append({"feature": f, "macro_f1": f1_score(yte, clf.predict(xte), average="macro"),
                     "n_unique_train": int(tr[f].nunique())})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def row_order_autocorrelation(df, feats=None, max_rows_per_file=20000, seed=42):
    """Lag-1 autocorrelation of each feature inside each file, versus the same rows shuffled.

    Clearly positive values mean consecutive flows are related, which is what windows exploit.
    Values near the shuffled baseline mean the file order carries no temporal signal.
    """
    feats = feats or feature_cols(df)
    rng = np.random.default_rng(seed)
    rows = []
    for (f, lab), g in df[df["split"] == "train"].groupby(["source_file", "label19"], observed=True):
        g = g.sort_values("row_in_file").head(max_rows_per_file)
        X = g[feats].to_numpy(np.float64)
        if len(X) < 50:
            continue
        def ac(M):
            a, b = M[:-1], M[1:]
            a, b = a - a.mean(0), b - b.mean(0)
            den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (a * b).sum(0) / den
            return np.nanmean(r)
        rows.append({"file": f, "label19": lab, "rows": len(X), "lag1_autocorr": ac(X),
                     "lag1_autocorr_shuffled": ac(X[rng.permutation(len(X))])})
    return pd.DataFrame(rows)


def suggest_drop(shortcuts, threshold):
    return shortcuts.loc[shortcuts["macro_f1"] >= threshold, "feature"].tolist()
