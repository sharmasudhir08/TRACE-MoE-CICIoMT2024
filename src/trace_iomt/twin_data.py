"""Twin dataset builder: per-device rows -> windows with twin context (TWIN_PROTOCOL.md §5-6; added 2026-11-04).

Input: the Parquet files of scripts/14_pcap_extract.py (device mode, --all-hosts). Steps:
  1. load attack rows (train/test) with a contiguous block of at most `cap` rows per segment, a segment
     being one device in one capture (as the CSV loader caps each file), so windows stay consecutive;
  2. label rows by file (rule "file", option a) or as attack only if they contain attacker packets
     (rule "attacker", option b);
  3. fit the twin on benign training rows only: the profiling captures plus Benign training rows outside
     the validation tail (never validation or test rows); fit the state estimator on the profiling captures;
  4. per row: z = |x - μ| / (σ + ε) against the row's profile, the new-peer and new-topic flags and the
     profile level; the state is estimated causally from the device's previous W rows;
  5. windows from data.windows.build_windowed with "source_file" = capture|device, so a window is T
     consecutive rows of one device and never crosses a device, a capture or a split. The mean of z over a
     window is Δ of Eq. 1.
Modes (TWIN_PROTOCOL.md §6): "fleet" = one benign profile for everyone (V1, device-agnostic), "device" =
per-device profiles without states (V2), "device_state" = per-device, per-state profiles (V3, V4).

Model input (window_inputs): log1p of Δ (z can reach thousands of σ on floods), the window means of the two
novelty flags and the share of the window's rows at each profile level, plus the last row's state code.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import twin
from .data.labels import binary_label, ciciomt_category
from .data.loaders import META_COLS, _contiguous_block
from .data.windows import build_windowed
from .pcapfeat import FEATURES

DEFAULT_CAP = 50_000
MODES = ("fleet", "device", "device_state")
LEVEL_CODE = {"state": 0, "device": 1, "fleet": 2}
CTX_NAMES = [f"z_{f}" for f in FEATURES] + ["new_peer", "new_topic", "profile_level"]
TWIN_INPUT_DIM = len(FEATURES) + 2 + len(LEVEL_CODE)
STATE_CODE = {s: i + 1 for i, s in enumerate(twin.STATES)}       # 0 = no state (modes fleet and device)


def twinned_devices(devices_json):
    """Twinned hosts: the profiled devices plus the extra endpoints of configs/twin_devices.json."""
    import json
    cfg = json.loads(Path(devices_json).read_text())
    return set(cfg["devices"]) | {k for k in cfg.get("extra_endpoints", {}) if not k.startswith("_")}


def _read(path, columns=None):
    return pd.read_parquet(path, columns=columns)


def load_attack_rows(device_dir, cap=DEFAULT_CAP, seed=42, logger=None):
    """Attack-capture rows (train and test), at most `cap` consecutive rows per (capture, device)."""
    rng = np.random.default_rng(seed)
    frames = []
    for path in sorted(Path(device_dir).glob("t*/*.parquet")):          # train/ and test/
        df = _read(path)
        for device, seg in df.sort_values("t_start", kind="stable").groupby("device", sort=True):
            a, b = _contiguous_block(len(seg), cap, rng)
            seg = seg.iloc[a:b].copy()
            seg["row_in_file"] = np.arange(a, b, dtype=np.int64)
            seg["source_file"] = f"{path.stem}|{device}"
            frames.append(seg)
        if logger:
            logger.info(f"{path.name}: {len(df):,} rows, {df['device'].nunique()} hosts")
    rows = pd.concat(frames, ignore_index=True)
    rows["state"] = ""
    return rows


def load_profiling_rows(device_dir):
    return pd.concat([_read(p) for p in sorted(Path(device_dir).glob("profiling/*.parquet"))], ignore_index=True)


def apply_labels(rows, rule="file"):
    """label19/label6/label2. rule "file" keeps the capture's label (as the official CSVs do); rule "attacker"
    labels a row of an attack capture as Benign unless it contains packets to or from an attacker. The
    capture's own label is kept in capture_label (the twin learns only from Benign *captures*)."""
    if "capture_label" not in rows.columns:
        rows = rows.assign(capture_label=rows["label19"].astype(str))
    lab = rows["capture_label"].astype(str)
    if rule == "attacker":
        lab = lab.where((rows["n_attacker_packets"] > 0) | (lab == "Benign"), "Benign")
    elif rule != "file":
        raise ValueError(f"label rule must be 'file' or 'attacker', got '{rule}'")
    rows = rows.copy()
    rows["label19"] = lab
    rows["label6"] = lab.map(ciciomt_category)
    rows["label2"] = rows["label6"].map(binary_label)
    rows["dataset"] = "CICIoMT2024-device"
    return rows


def profile_fit_rows(profiling, attack_rows, val_tail_fraction):
    """Benign rows the twin may learn from: the profiling captures and Benign *training* rows outside the
    validation tail of their segment (the same tail build_windowed uses for validation)."""
    label = attack_rows["capture_label"] if "capture_label" in attack_rows else attack_rows["label19"]
    benign = attack_rows[(attack_rows["split"] == "train") & (label.astype(str) == "Benign")]
    n = benign.groupby("source_file")["row_in_file"].transform("count")
    pos = benign.groupby("source_file").cumcount()
    benign = benign[pos < n - np.rint(val_tail_fraction * n)]
    return pd.concat([profiling, benign], ignore_index=True)


def causal_states(rows, estimator, w=16):
    """State per row from the device's current and previous w-1 rows of the same segment (causal)."""
    x = pd.DataFrame(twin.feature_matrix(rows), index=rows.index)
    keys = rows["source_file"]
    mean = x.groupby(keys).rolling(w, min_periods=1).mean().droplevel(0).loc[rows.index]
    std = x.groupby(keys).rolling(w, min_periods=1).std(ddof=0).droplevel(0).loc[rows.index].fillna(0.0)
    return estimator.model.predict(np.hstack([mean.to_numpy(), std.to_numpy()]))


def profile_keys(devices, states, profiles, mode):
    """Profile key and level per row, resolved once per unique (device, state) pair."""
    pairs = pd.DataFrame({"device": np.asarray(devices, dtype=str), "state": np.asarray(states, dtype=str)})
    uniq = pairs.drop_duplicates()
    resolved = {}
    for d, st in zip(uniq["device"], uniq["state"]):
        if mode == "fleet":
            resolved[(d, st)] = ((twin.FLEET, None), "fleet")
        else:
            resolved[(d, st)] = profiles.lookup_key(d, st if mode == "device_state" else None)
    res = [resolved[(d, st)] for d, st in zip(pairs["device"], pairs["state"])]
    return [r[0] for r in res], [r[1] for r in res]


def _any_new(cells, known, unparsable_is_new=False):
    """Per row: does the ';'-joined cell hold a value outside `known` (or '?' when unparsable_is_new)?"""
    s = pd.Series(cells).fillna("").astype(str).str.split(";").explode()
    s = s[s != ""]
    new = ~s.isin(known)
    if unparsable_is_new:
        new |= s == "?"
    return new.groupby(level=0).any().reindex(range(len(cells)), fill_value=False).to_numpy()


def row_twin_features(rows, profiles, estimator=None, mode="device_state", w=16, return_keys=False):
    """(ctx [n x len(CTX_NAMES)] float32, state [n] str) in the order of `rows` (needs a RangeIndex and rows
    sorted by time inside each segment, as load_attack_rows returns them). With return_keys, also the profile
    key of every row."""
    if mode not in MODES:
        raise ValueError(f"twin mode must be one of {MODES}, got '{mode}'")
    states = causal_states(rows, estimator, w) if mode == "device_state" else np.full(len(rows), "", dtype=object)
    keys, levels = profile_keys(rows["device"], states, profiles, mode)
    x = twin.feature_matrix(rows)
    ctx = np.zeros((len(rows), len(CTX_NAMES)), dtype=np.float32)
    key_arr = np.array([f"{k[0]}|{k[1]}" for k in keys])
    peers, topics = rows["peers"].to_numpy(), rows["topics"].to_numpy()
    nf = len(FEATURES)
    for key in np.unique(key_arr):
        idx = np.flatnonzero(key_arr == key)
        p = profiles.profiles[keys[idx[0]]]
        ctx[idx, :nf] = np.abs(x[idx] - p.mu) / (p.sigma + twin.EPSILON)
        ctx[idx, nf] = _any_new(peers[idx], p.peers)
        ctx[idx, nf + 1] = _any_new(topics[idx], p.topics, unparsable_is_new=True)
    ctx[:, -1] = [LEVEL_CODE[lv] for lv in levels]
    return (ctx, states, keys) if return_keys else (ctx, states)


def fit_twin(profiling, rows, twinned, val_tail, min_support=200, shrinkage=50, seed=42, logger=None):
    """Profiles (benign training rows only) and the state estimator (profiling captures). Returns (profiles,
    estimator, report); the report has the estimator's accuracy on held-out time (last 30% of each capture)."""
    from sklearn.metrics import accuracy_score
    fit_rows = profile_fit_rows(profiling, rows, val_tail)
    profiles = twin.fit_profiles(fit_rows, twinned, min_support, shrinkage)
    X, y, _, starts, lengths = twin.state_samples(profiling[profiling["device"].isin(twinned)])
    train, test = twin.holdout_by_time(starts, lengths)
    held = twin.StateEstimator(seed=seed).fit(X[train], y[train])
    acc = float(accuracy_score(y[test], held.model.predict(X[test]))) if test.any() else float("nan")
    estimator = twin.StateEstimator(seed=seed).fit(X, y)          # final estimator uses all profiling windows
    if logger:
        logger.info(f"profiles from {len(fit_rows):,} benign rows; state estimator held-out accuracy {acc:.3f}")
    return profiles, estimator, {"profile_rows": int(len(fit_rows)), "state_holdout_accuracy": acc,
                                 "profiles": {f"{k[0]}|{k[1]}": v.n for k, v in profiles.profiles.items()}}


def feature_table(rows):
    """The rows as build_windowed reads them: features and meta columns only, RangeIndex. A single-packet row
    has no standard deviation or variance (NaN in pcapfeat); it is counted as 0, as in twin.feature_matrix.
    Left as NaN it passes through the scaler and makes every model output NaN (Mac Studio, 2026-11-05)."""
    meta = [c for c in META_COLS if c in rows.columns]
    table = rows[list(FEATURES) + meta].reset_index(drop=True)
    table[list(FEATURES)] = table[list(FEATURES)].fillna(0.0)
    return table


def build_twin_windowed(cfg, rows, profiles, estimator, mode, seed, logger=None):
    """WindowedData for the twin study, with the twin context of `mode` attached (see attach_twin)."""
    wd = build_windowed(feature_table(rows), cfg, regime="official", seed=seed, logger=logger)
    return attach_twin(wd, rows, profiles, estimator, mode, cfg["windows"]["length"])


def attach_twin(wd, rows, profiles, estimator, mode, w=16):
    """Attach the twin context of `mode` to a WindowedData built from these rows; all arrays follow wd.X:
      ctx         twin features per row (CTX_NAMES)
      state       estimated state per row (str), state_code its code (STATE_CODE, 0 = none)
      key_id      index into twin_keys, the profile each row is scored against
      device_id   index into devices; t (row start time, s); xlog (features on the twin's log scale)
    The windows do not depend on the mode, so one WindowedData can be re-attached for every variant."""
    rows = rows.reset_index(drop=True)
    ctx, states, keys = row_twin_features(rows, profiles, estimator, mode, w, return_keys=True)
    order = wd.row_index
    key_str = np.array([f"{k[0]}|{k[1]}" for k in keys])
    uniq, key_id = np.unique(key_str, return_inverse=True)
    first = {k: keys[i] for k, i in zip(*np.unique(key_str, return_index=True))}
    wd.ctx = ctx[order]
    wd.state = np.asarray(states, dtype=object)[order]
    wd.state_code = np.array([STATE_CODE.get(twin.STATE_ALIASES.get(s, s), 0) for s in wd.state], dtype=np.int64)
    wd.twin_keys = [first[k] for k in uniq]
    wd.key_id = key_id.astype(np.int32)[order]
    wd.devices, dev = np.unique(rows["device"].astype(str).to_numpy(), return_inverse=True)
    wd.device_id = dev.astype(np.int32)[order]
    wd.t = rows["t_start"].to_numpy(dtype=np.float64)[order]
    wd.xlog = twin.feature_matrix(rows).astype(np.float32)[order]
    wd.report["twin"] = {"mode": mode, "rows": int(len(rows)), "fallbacks": dict(profiles.fallbacks),
                         "devices": wd.devices.tolist()}
    return wd


def window_inputs(ctx, state_code, starts, T):
    """Model input per window: (vector [B x TWIN_INPUT_DIM] float32, state code of the last row [B])."""
    return window_inputs_at(ctx, state_code, np.asarray(starts, dtype=np.int64)[:, None] + np.arange(T))


def window_inputs_at(ctx, state_code, idx):
    """window_inputs for windows given as row indices into ctx, idx [B x T]."""
    c = ctx[idx]                                                   # (B, T, len(CTX_NAMES))
    nf = len(FEATURES)
    level = c[..., -1].astype(np.int64)
    share = np.stack([(level == code).mean(axis=1) for code in LEVEL_CODE.values()], axis=1)
    vec = np.concatenate([np.log1p(c[..., :nf].mean(axis=1)), c[..., nf:nf + 2].mean(axis=1), share], axis=1)
    return vec.astype(np.float32), state_code[idx[:, -1]]


def window_delta(wd, starts):
    """Δ (Eq. 1) and mean novelty flags per window: the mean of the rows' twin features over the window."""
    idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(wd.T)
    return wd.ctx[idx].mean(axis=1)
