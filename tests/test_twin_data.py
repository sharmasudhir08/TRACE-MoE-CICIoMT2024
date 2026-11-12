"""Twin dataset builder (trace_iomt.twin_data) on a small synthetic extraction tree."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trace_iomt import twin, twin_data
from trace_iomt.config import load_config
from trace_iomt.pcapfeat import FEATURES


def device_rows(n, device, base, split, label, attacker_every=0, seed=0, peers="10.0.0.1", topics=""):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(np.abs(base + rng.standard_normal((n, len(FEATURES)))), columns=FEATURES)
    df["Protocol Type"] = 6.0
    df["device"], df["peers"], df["topics"] = device, peers, topics
    df["n_attacker_packets"] = [(1 if attacker_every and i % attacker_every == 0 else 0) for i in range(n)]
    df["t_start"] = np.arange(n, dtype=float)
    df["t_end"] = df["t_start"] + 0.5
    df["split"], df["label19"], df["state"] = split, label, ""
    return df


def write(root, split, stem, frames):
    path = root / split / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df["file"] = stem
    df.to_parquet(path, index=False)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "device"
    write(root, "train", "Benign_train", [device_rows(120, "cam", 5, "train", "Benign"),
                                           device_rows(80, "mac:20:7b", 20, "train", "Benign", seed=1)])
    write(root, "train", "TCP_IP-DDoS-ICMP1_train", [device_rows(100, "cam", 40, "train", "TCP_IP-DDoS-ICMP", attacker_every=2, seed=2)])
    write(root, "test", "TCP_IP-DDoS-ICMP1_test", [device_rows(60, "cam", 40, "test", "TCP_IP-DDoS-ICMP", attacker_every=2, seed=3,
                                                               peers="10.0.0.1;10.9.9.9", topics="?")])
    prof = device_rows(300, "cam", 5, "profiling", "Benign", seed=4)
    prof["state"] = "Idle"
    write(root, "profiling", "Idle", [prof])
    return root


def cfg():
    c = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", [])
    c["windows"].update(length=4, stride_train=2, stride_val=4, stride_eval=1)
    c["data"]["drop_exact_duplicates"] = False
    return c


def test_load_caps_each_segment_with_a_contiguous_block(tree):
    rows = twin_data.load_attack_rows(tree, cap=50, seed=0)
    seg = rows[rows["source_file"] == "Benign_train|cam"]
    assert len(seg) == 50 and np.all(np.diff(seg["row_in_file"]) == 1) and np.all(np.diff(seg["t_start"]) > 0)
    assert set(rows["source_file"]) == {"Benign_train|cam", "Benign_train|mac:20:7b", "TCP_IP-DDoS-ICMP1_train|cam",
                                        "TCP_IP-DDoS-ICMP1_test|cam"}
    assert set(rows["split"]) == {"train", "test"}                    # profiling rows are loaded separately


def test_label_rules(tree):
    rows = twin_data.load_attack_rows(tree, cap=None)
    a = twin_data.apply_labels(rows, "file")
    b = twin_data.apply_labels(rows, "attacker")
    attack = a["capture_label"] == "TCP_IP-DDoS-ICMP"
    assert set(a.loc[attack, "label19"]) == {"TCP_IP-DDoS-ICMP"}
    assert set(b.loc[attack & (b["n_attacker_packets"] == 0), "label19"]) == {"Benign"}
    assert set(b.loc[attack & (b["n_attacker_packets"] > 0), "label6"]) == {"DDoS"}
    with pytest.raises(ValueError):
        twin_data.apply_labels(rows, "other")


def test_profile_rows_are_benign_training_rows_outside_the_validation_tail(tree):
    rows = twin_data.apply_labels(twin_data.load_attack_rows(tree, cap=None), "attacker")
    fit = twin_data.profile_fit_rows(twin_data.load_profiling_rows(tree), rows, val_tail_fraction=0.15)
    attack_files = fit["file"].isin(["TCP_IP-DDoS-ICMP1_train", "TCP_IP-DDoS-ICMP1_test"])
    assert not attack_files.any()                       # rule b relabels attack rows Benign; they still stay out
    cam = fit[(fit["file"] == "Benign_train") & (fit["device"] == "cam")]
    assert len(cam) == 120 - round(0.15 * 120) and cam["t_start"].max() < 102
    assert (fit["state"] == "Idle").sum() == 300


def _fitted(tree, mode_rule="file"):
    rows = twin_data.apply_labels(twin_data.load_attack_rows(tree, cap=None), mode_rule)
    profiling = twin_data.load_profiling_rows(tree)
    prof = twin.fit_profiles(twin_data.profile_fit_rows(profiling, rows, 0.15), twinned={"cam"}, min_support=50)
    X, y, *_ = twin.state_samples(profiling, w=4, stride=2)
    est = twin.StateEstimator(w=4).fit(X, y)
    return rows, prof, est


def test_row_features_by_mode(tree):
    rows, prof, est = _fitted(tree)
    rows = rows.reset_index(drop=True)
    fleet, _ = twin_data.row_twin_features(rows, prof, est, mode="fleet", w=4)
    dev, _ = twin_data.row_twin_features(rows, prof, est, mode="device", w=4)
    ds, states = twin_data.row_twin_features(rows, prof, est, mode="device_state", w=4)
    cam = (rows["device"] == "cam").to_numpy()
    assert set(fleet[:, -1]) == {2.0}
    assert set(dev[cam, -1]) == {1.0} and set(dev[~cam, -1]) == {2.0}      # untwinned host -> fleet
    assert set(states) == {"Idle"}
    p = prof.profiles[("cam", None)]
    x = twin.feature_matrix(rows.iloc[[0]])
    np.testing.assert_allclose(dev[0, :len(FEATURES)], np.abs(x[0] - p.mu) / (p.sigma + twin.EPSILON), rtol=1e-5)
    test = (rows["split"] == "test").to_numpy()
    assert dev[test, len(FEATURES)].min() == 1.0                           # 10.9.9.9 never seen
    assert dev[test, len(FEATURES) + 1].min() == 1.0                       # '?' topic


def test_build_aligns_twin_features_with_windows(tree):
    rows, prof, est = _fitted(tree)
    wd = twin_data.build_twin_windowed(cfg(), rows, prof, est, mode="device", seed=0)
    assert wd.ctx.shape == (len(wd.X), len(twin_data.CTX_NAMES))
    ordered = rows.reset_index(drop=True).iloc[wd.row_index]
    expected, _ = twin_data.row_twin_features(rows.reset_index(drop=True), prof, est, "device", 4)
    np.testing.assert_allclose(wd.ctx, expected[wd.row_index])
    for split, starts in wd.starts.items():
        for s in starts[:20]:
            assert ordered["source_file"].iloc[s:s + wd.T].nunique() == 1      # one device, one capture
    delta = twin_data.window_delta(wd, wd.starts["test"][:3])
    assert delta.shape == (3, len(twin_data.CTX_NAMES))
