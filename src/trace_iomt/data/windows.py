"""Split regimes and windows of consecutive flows.

A window is T consecutive flows inside one file segment (it never crosses a file or split boundary); its label
is the label of its last flow. Regimes:
  official  official train/test files; validation = last data.val_tail_fraction of each training file
  random    windows from all files pooled and split at random (stratified by class): 20 % test, then
            data.val_tail_fraction of the rest for validation (the protocol most published papers use)
  temporal  first part of every file for training, last split.temporal_test_fraction for testing
  loao      official, with split.loao_class removed from training and validation (unknown-attack test)

Strides: windows.stride_train / stride_val / stride_eval, never larger than T, so T = 1 keeps every row.
The random regime uses stride_train for every window before splitting.

Duplicate windows (data.drop_exact_duplicates): identical training windows are kept once, and validation or
test windows identical to a training window are removed. Windows are compared on their scaled values rounded
to _DEDUP_DECIMALS. The scaler is fitted on a seed-dependent sample of training rows, so the counts vary
slightly by seed.
"""
import numpy as np

from .labels import LEVELS, Hierarchy
from .loaders import feature_cols
from .preprocess import build_scaler, protocol_tokens

SPLITS = ("train", "val", "test")
_TRAIN, _VAL, _TEST = 0, 1, 2
_DEFAULT_SCALER_FIT_ROWS = 1_000_000
# Closest match to the original code's logged counts (official, T = 1, seed 42: train 2,788,114 removed,
# val 346,261): 6 decimals gives train -0.7 % and val -9.5 %; exact values, float16, other roundings and
# counting repeats inside validation were all further off (2026-10-30, CICIoMT2024 Wi-Fi/MQTT).
_DEDUP_DECIMALS = 6
_HASH_MULT = np.uint64(0x100000001B3)
_WINDOW_MULT = np.uint64(0x9E3779B97F4A7C15)


def class_counts(labels, n_classes):
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)


def class_balanced_sample(labels, max_per_class, seed=0, oversample=False):
    """Indices with at most max_per_class per class, shuffled. With oversample, smaller classes are drawn
    with replacement up to max_per_class, so every class contributes the same number of windows."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        elif oversample and max_per_class and len(idx) < max_per_class:
            idx = rng.choice(idx, max_per_class, replace=True)
        picked.append(idx)
    if not picked:
        return np.zeros(0, dtype=np.int64)
    return rng.permutation(np.concatenate(picked)).astype(np.int64)


class WindowedData:
    """Scaled flows in file order plus the window start positions of every split."""

    def __init__(self, X, proto, y, starts, T, features, hierarchy, scaler, report, row_index, regime):
        self.X, self.proto, self.y, self.starts = X, proto, y, starts
        self.T, self.features, self.hierarchy, self.scaler = T, features, hierarchy, scaler
        self.report, self.row_index, self.regime = report, row_index, regime

    def gather(self, starts):
        idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(self.T)
        last = idx[:, -1]
        return self.X[idx], self.proto[idx], {lvl: self.y[lvl][last] for lvl in LEVELS}

    def labels(self, split, level):
        return self.y[level][self.starts[split] + self.T - 1]


def _segment_parts(file_code, pos, n_in_file, split, regime, data_cfg, split_cfg):
    """Split part (train/val/test) of every row, in file order."""
    val_frac = float(data_cfg.get("val_tail_fraction", 0.15))
    part = np.full(len(pos), _TRAIN, dtype=np.int8)
    if regime in ("official", "loao"):
        is_test = split == "test"
        n_val = np.rint(val_frac * n_in_file).astype(np.int64)
        part[(~is_test) & (pos >= n_in_file - n_val)] = _VAL
        part[is_test] = _TEST
    elif regime == "temporal":
        n_test = np.rint(float(split_cfg.get("temporal_test_fraction", 0.2)) * n_in_file).astype(np.int64)
        n_train = n_in_file - n_test
        n_val = np.rint(val_frac * n_train).astype(np.int64)
        part[pos >= n_train - n_val] = _VAL
        part[pos >= n_train] = _TEST
    elif regime != "random":
        raise ValueError(f"split.regime must be official, random, temporal or loao, got '{regime}'")
    return part


def _starts_in_segments(seg_id, T, stride):
    """Window starts inside each run of equal seg_id, every `stride` rows."""
    if len(seg_id) == 0:
        return np.zeros(0, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(seg_id)) + 1
    begins = np.r_[0, bounds]
    ends = np.r_[bounds, len(seg_id)]
    out = [np.arange(a, b - T + 1, stride, dtype=np.int64) for a, b in zip(begins, ends) if b - a >= T]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _stratified_window_split(labels, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SPLITS}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_test = int(round(test_frac * len(idx)))
        rest = idx[n_test:]
        n_val = int(round(val_frac * len(rest)))
        out["test"].append(idx[:n_test])
        out["val"].append(rest[:n_val])
        out["train"].append(rest[n_val:])
    return {s: np.sort(np.concatenate(v)) if v else np.zeros(0, dtype=np.int64) for s, v in out.items()}


def _row_hashes(X):
    """64-bit hash of every scaled row (bit pattern of the float32 values)."""
    bits = np.ascontiguousarray(X + np.float32(0.0)).view(np.uint32)   # + 0.0 turns -0.0 into +0.0
    h = np.full(len(X), 0xCBF29CE484222325, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(bits.shape[1]):
            h ^= bits[:, j].astype(np.uint64)
            h *= _HASH_MULT
    return h


def _window_hashes(row_hash, starts, T):
    h = np.zeros(len(starts), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(T):
            h = h * _WINDOW_MULT + row_hash[starts + k]
    return h


def _dedup(starts, row_hash, T):
    h = {s: _window_hashes(row_hash, starts[s], T) for s in SPLITS}
    _, first = np.unique(h["train"], return_index=True)
    keep_train = np.sort(first)
    train_set = h["train"][keep_train]
    keep_val = ~np.isin(h["val"], train_set)
    keep_test = ~np.isin(h["test"], train_set)
    removed = {"train_duplicate_windows_removed": int(len(starts["train"]) - len(keep_train)),
               "val_windows_also_in_train_removed": int((~keep_val).sum()),
               "test_windows_also_in_train_removed": int((~keep_test).sum())}
    return {"train": starts["train"][keep_train], "val": starts["val"][keep_val],
            "test": starts["test"][keep_test]}, removed


def _fit_scaler(X_raw, fit_rows, data_cfg, seed):
    scaler = build_scaler(data_cfg.get("scaler", "robust"), data_cfg.get("clip_value", 10.0))
    rows = np.flatnonzero(fit_rows)
    limit = int(data_cfg.get("scaler_fit_rows", _DEFAULT_SCALER_FIT_ROWS))
    if len(rows) > limit:
        rows = np.sort(np.random.default_rng(seed).choice(rows, limit, replace=False))
    scaler.fit(X_raw[rows])
    return scaler, len(rows)


def build_windowed(df, cfg, regime=None, T=None, seed=None, logger=None, drop_features=None):
    data_cfg, split_cfg, win_cfg = cfg["data"], cfg["split"], cfg["windows"]
    regime = regime or split_cfg.get("regime", "official")
    T = int(T or win_cfg["length"])
    seed = int(cfg["train"]["seed"] if seed is None else seed)
    drop = set(drop_features or [])
    features = [c for c in feature_cols(df) if c not in drop]

    file_codes = df["source_file"].astype("category").cat.codes.to_numpy()
    order = np.lexsort((df["row_in_file"].to_numpy(), file_codes))
    file_code = file_codes[order]
    split = df["split"].astype(str).to_numpy()[order]
    _, first_of_file, n_per_file = np.unique(file_code, return_index=True, return_counts=True)
    pos = np.arange(len(order)) - np.repeat(first_of_file, n_per_file)
    n_in_file = np.repeat(n_per_file, n_per_file)

    part = _segment_parts(file_code, pos, n_in_file, split, regime, data_cfg, split_cfg)
    seg_id = file_code.astype(np.int64) * 3 + part

    hierarchy = Hierarchy(df["label19"].astype(str).unique().tolist())
    y = {}
    for lvl in LEVELS:
        col = df[lvl].astype("category")
        lut = np.array([hierarchy.index(lvl)[str(c)] for c in col.cat.categories], dtype=np.int64)
        y[lvl] = lut[col.cat.codes.to_numpy()][order]

    stride = {"train": win_cfg.get("stride_train", 1), "val": win_cfg.get("stride_val", 1),
              "test": win_cfg.get("stride_eval", 1)}
    stride = {s: max(1, min(int(v), T)) for s, v in stride.items()}
    loao_removed = 0
    if regime == "random":
        cand = _starts_in_segments(seg_id, T, stride["train"])
        parts = _stratified_window_split(y["label19"][cand + T - 1], float(data_cfg.get("val_tail_fraction", 0.15)),
                                         float(split_cfg.get("random_test_fraction", 0.2)), seed)
        starts = {s: cand[parts[s]] for s in SPLITS}
        covered = np.zeros(len(order) + 1, dtype=np.int64)
        np.add.at(covered, starts["train"], 1)
        np.add.at(covered, starts["train"] + T, -1)
        fit_rows = np.cumsum(covered[:-1]) > 0
    else:
        starts = {}
        for s, code in zip(SPLITS, (_TRAIN, _VAL, _TEST)):
            seg = np.where(part == code, seg_id, -1)
            st = _starts_in_segments(seg, T, stride[s])
            starts[s] = st[seg[st] >= 0]
        fit_rows = part == _TRAIN
        if regime == "loao":
            held = split_cfg.get("loao_class")
            if not held:
                raise ValueError("split.regime=loao needs split.loao_class (e.g. ARP_Spoofing)")
            k = hierarchy.index("label19")[held]
            for s in ("train", "val"):
                keep = y["label19"][starts[s] + T - 1] != k
                loao_removed += int((~keep).sum())
                starts[s] = starts[s][keep]
            fit_rows &= y["label19"] != k

    X_raw = df[features].to_numpy(dtype=np.float32)[order]
    scaler, n_fit = _fit_scaler(X_raw, fit_rows, data_cfg, seed)
    X = scaler.transform(X_raw)
    del X_raw
    proto = (protocol_tokens(df["Protocol Type"].to_numpy()[order]) if "Protocol Type" in df.columns
             else np.zeros(len(order), dtype=np.int64))

    removed = {}
    if data_cfg.get("drop_exact_duplicates", True):
        starts, removed = _dedup(starts, _row_hashes(np.round(X, _DEDUP_DECIMALS)), T)
        if logger:
            logger.info(f"Duplicate removal: {removed}")
    if logger:
        logger.info(f"Windows ({regime}, T={T}): " + ", ".join(f"{s}={len(starts[s]):,}" for s in SPLITS))

    report = {"regime": regime, "T": T, "seed": seed, "strides": stride, "n_rows": int(len(order)),
              "n_features": len(features), "dropped_features": sorted(drop), "scaler_fit_rows": int(n_fit),
              "windows": {s: int(len(starts[s])) for s in SPLITS}, "duplicates": removed}
    if regime == "loao":
        report["loao_class"] = split_cfg.get("loao_class")
        report["loao_windows_removed"] = loao_removed
    return WindowedData(X, proto, y, starts, T, features, hierarchy, scaler, report,
                        df.index.to_numpy()[order], regime)
