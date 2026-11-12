"""Checks of the re-implemented windowing rules (data/windows.py):  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.data.windows import build_windowed, class_balanced_sample  # noqa: E402

N_PER_FILE = 200


def _frame(duplicate_test_rows=0):
    """Two classes, one train and one test file each, plus optional test rows copied from training."""
    rng = np.random.default_rng(1)
    parts = []
    for lab, cat in (("Benign", "Benign"), ("TCP_IP-DoS-SYN", "DoS")):
        for split in ("train", "test"):
            d = pd.DataFrame(rng.normal(size=(N_PER_FILE, 3)), columns=["a", "b", "Protocol Type"])
            d["label19"], d["label6"] = lab, cat
            d["label2"] = "Benign" if cat == "Benign" else "Attack"
            d["split"], d["source_file"], d["row_in_file"], d["dataset"] = split, f"{lab}_{split}", np.arange(N_PER_FILE), "toy"
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if duplicate_test_rows:
        test = df.index[(df["split"] == "test") & (df["label19"] == "Benign")][:duplicate_test_rows]
        train = df.index[(df["split"] == "train") & (df["label19"] == "Benign")][:duplicate_test_rows]
        df.loc[test, ["a", "b", "Protocol Type"]] = df.loc[train, ["a", "b", "Protocol Type"]].to_numpy()
    return df


def _cfg(**windows):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["windows"].update(windows)
    return cfg


def test_t1_keeps_every_row_and_validation_is_the_file_tail():
    # logs_v2 show T = 1 windows covering all 8,775,013 rows, so strides never exceed T
    df = _frame()
    wd = build_windowed(df, _cfg(), regime="official", T=1)
    n = {s: len(wd.starts[s]) for s in ("train", "val", "test")}
    assert sum(n.values()) == len(df)
    assert n["val"] == 2 * round(0.15 * N_PER_FILE)
    val_pos = df.loc[wd.row_index[wd.starts["val"]], "row_in_file"].to_numpy()
    assert val_pos.min() >= N_PER_FILE - round(0.15 * N_PER_FILE)


def test_strides_per_split():
    df = _frame()
    wd = build_windowed(df, _cfg(stride_train=8, stride_val=16, stride_eval=1), regime="official", T=16)
    n_train_rows = N_PER_FILE - round(0.15 * N_PER_FILE)
    assert len(wd.starts["train"]) == 2 * len(range(0, n_train_rows - 16 + 1, 8))
    assert len(wd.starts["test"]) == 2 * (N_PER_FILE - 15)
    assert np.all(np.diff(wd.starts["test"][: N_PER_FILE - 15]) == 1)


def test_test_windows_identical_to_training_are_removed():
    df = _frame(duplicate_test_rows=20)
    wd = build_windowed(df, _cfg(), regime="official", T=1)
    assert wd.report["duplicates"]["test_windows_also_in_train_removed"] == 20


def test_rows_equal_to_six_decimals_count_as_duplicates():
    # rounding the scaled values to 6 decimals reproduces the original duplicate counts most closely
    df = _frame(duplicate_test_rows=20)
    df.insert(0, "z", 0.0)                     # constant feature: scaled value = raw value (median 0, scale 1)
    test = df.index[(df["split"] == "test") & (df["label19"] == "Benign")][:20]
    df.loc[test, "z"] = 3e-7                   # survives float32, disappears at 6 decimals
    wd = build_windowed(df, _cfg(), regime="official", T=1)
    assert wd.report["duplicates"]["test_windows_also_in_train_removed"] == 20


def test_random_regime_splits_windows_by_fraction():
    df = _frame()
    wd = build_windowed(df, _cfg(), regime="random", T=1)
    n_test = len(wd.starts["test"])
    assert abs(n_test - 0.2 * len(df)) <= 2
    assert len(np.intersect1d(wd.starts["train"], wd.starts["test"])) == 0


def test_loao_removes_class_from_training_only():
    cfg = _cfg()
    cfg["split"]["loao_class"] = "TCP_IP-DoS-SYN"
    wd = build_windowed(_frame(), cfg, regime="loao", T=4)
    k = wd.hierarchy.index("label19")["TCP_IP-DoS-SYN"]
    assert k not in set(wd.labels("train", "label19")) and k not in set(wd.labels("val", "label19"))
    assert k in set(wd.labels("test", "label19"))


def test_class_balanced_sample_oversamples_small_classes():
    labels = np.r_[np.zeros(100, dtype=int), np.ones(5, dtype=int)]
    idx = class_balanced_sample(labels, 50, seed=0, oversample=True)
    assert np.bincount(labels[idx]).tolist() == [50, 50]
    assert np.bincount(labels[class_balanced_sample(labels, 50, seed=0)]).tolist() == [50, 5]
