"""Statistics of the twin study (trace_iomt.twin_compare)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from trace_iomt import twin_compare as tc


def test_macro_f1_from_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, p), 1)                                           # class 4 absent: left out of the mean
    assert tc.macro_f1_from_confusion(cm) == np.float64(f1_score(y, p, labels=range(4), average="macro"))


def test_file_bootstrap_of_identical_variants_is_zero_and_detects_a_gain():
    rng = np.random.default_rng(1)
    files = np.repeat(["f1", "f2", "f3", "f4"], 100)
    y = rng.integers(0, 3, 400)
    noisy = np.where(rng.random(400) < 0.5, rng.integers(0, 3, 400), y)
    same = {s: tc.confusion_by_file(y, noisy, files, 3) for s in (42, 43)}
    b = tc.file_bootstrap(same, same, n_resamples=50)
    assert b["low"] == b["high"] == 0.0 and b["n_files"] == 4
    better = {s: tc.confusion_by_file(y, y, files, 3) for s in (42, 43)}
    assert tc.file_bootstrap(better, same, n_resamples=50)["low"] > 0


def test_paired_difference_and_h11_rule():
    rows = pd.DataFrame({"variant": ["V4"] * 3 + ["V0"] * 3, "seed": [42, 43, 44] * 2,
                         "macro_f1_19": [0.820, 0.821, 0.819, 0.80, 0.80, 0.80]})
    d = tc.paired_difference(rows, "V4", "V0")
    assert np.isclose(d["mean"], 0.02) and d["t_low"] > 0
    assert tc.h11_verdict(d, {"low": 0.001}) == "supported"
    assert tc.h11_verdict(d, {"low": -0.001}) == "partly supported"
    assert tc.h11_verdict({**d, "mean": -0.01}, {"low": 0.1}) == "not supported"


def test_delta_auroc_per_attack_class():
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    delta = np.r_[np.zeros(50), np.ones(50), np.r_[np.zeros(25), np.ones(25)]]
    out = tc.delta_auroc(delta, y, ["Benign", "A", "B"])
    assert out == {"A": 1.0, "B": 0.75}


def test_time_to_detect_from_first_attack_packet():
    seg = np.array(["a"] * 4 + ["b"] * 3)
    t_end = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    flagged = np.array([True, False, False, True, False, False, False])
    out = tc.time_to_detect(seg, t_end, flagged, {"a": 1.5, "b": 0.5, "c": 9.0})
    assert out == [{"segment": "a", "seconds": 2.5, "windows": 3}, {"segment": "b", "seconds": None, "windows": 3}]
