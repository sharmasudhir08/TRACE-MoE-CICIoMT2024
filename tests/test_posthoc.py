"""AURC with tied confidences, non-saturating confidence scores, and the post-hoc helpers (2026-11-03)."""
import itertools
import json

import numpy as np
import pytest

from trace_iomt.abstain import confidence_scores, risk_coverage, selective_risk_at, selective_scores
from trace_iomt.posthoc import binary_at_thresholds, hierarchy_consistency, tune_binary_threshold


def _plain_aurc(conf, correct):
    """AURC for confidences without ties: mean selective risk over k = 1..n."""
    c = correct[np.argsort(-conf)].astype(float)
    return float((1 - np.cumsum(c) / np.arange(1, len(c) + 1)).mean())


def test_aurc_does_not_depend_on_the_order_of_tied_windows():
    # 8 windows all tied at confidence 1.0; the errors first or last in file order must not matter.
    conf = np.ones(8)
    errors_first = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    errors_last = errors_first[::-1].copy()
    assert risk_coverage(conf, errors_first)[0] == pytest.approx(risk_coverage(conf, errors_last)[0])


def test_aurc_with_ties_equals_the_average_over_all_orders():
    conf = np.array([0.9, 0.9, 0.9, 0.5, 0.5])
    correct = np.array([1, 0, 1, 0, 1], dtype=bool)
    brute = []
    for p in itertools.permutations(range(3)):
        for q in itertools.permutations(range(3, 5)):
            order = list(p) + list(q)
            c = correct[order].astype(float)
            brute.append((1 - np.cumsum(c) / np.arange(1, 6)).mean())
    assert risk_coverage(conf, correct)[0] == pytest.approx(np.mean(brute))


def test_aurc_without_ties_is_unchanged():
    rng = np.random.default_rng(0)
    conf, correct = rng.random(200), rng.random(200) > 0.3
    assert risk_coverage(conf, correct)[0] == pytest.approx(_plain_aurc(conf, correct))


def test_selective_risk_at_coverage_uses_the_most_confident_share():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0], dtype=bool)
    assert selective_risk_at(conf, correct, 0.5) == pytest.approx(0.0)
    assert selective_risk_at(conf, correct, 1.0) == pytest.approx(0.5)


def test_log_space_confidence_separates_windows_that_softmax_ties_at_one():
    logits = np.array([[60.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float32)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).astype(np.float32)
    assert p[0].max() == p[1].max() == 1.0                      # float32 softmax saturates
    s = confidence_scores(logits)
    for name in ("msp_raw", "margin", "neg_entropy", "energy"):
        assert s[name][0] > s[name][1], name                    # the larger gap is more confident


def test_classical_scores_skip_energy():
    assert "energy" not in confidence_scores(np.log(np.array([[0.7, 0.3]])), has_energy=False)


def test_selective_scores_report_and_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    logits = rng.normal(size=(600, 3)) + 3 * np.eye(3)[y]
    report, scores = selective_scores(logits[:300], y[:300], logits[300:], y[300:], target_coverage=0.9)
    assert 0.8 < report["coverage"] <= 1.0
    assert report["selective_risk"] <= report["risk_full"]
    assert set(scores) == {"msp_raw", "msp_calibrated", "margin", "neg_entropy", "energy"}
    assert all(0 <= v["aurc"] <= 1 and "selective_risk_90" in v for v in scores.values())


def test_hierarchy_consistency():
    parent19, parent6 = [0, 1, 1], [0, 1]           # 3 fine classes -> 2 categories -> 2 binary labels
    out = hierarchy_consistency(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]),
                                parent19, parent6)
    assert out == {"c19_6": 0.75, "c6_2": 0.75, "joint": 0.5}


def test_tuned_threshold_beats_default_when_scores_are_shifted():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.15 + 0.3 * y + rng.normal(0, 0.05, 2000), 0, 1)     # attacks score ~0.45, below 0.5
    t, _ = tune_binary_threshold(p, y)
    assert 0.25 < t < 0.45
    out = binary_at_thresholds(p, y, p, y)
    assert out["macro_f1_tuned"] > out["macro_f1_default"]


def test_read_run_uses_posthoc_results_when_present(tmp_path):
    from trace_iomt.compare import read_run
    (tmp_path / "metrics.json").write_text(json.dumps({"model": "m", "abstain": {"aurc": 0.5}}))
    assert read_run(tmp_path)["abstain"]["aurc"] == 0.5
    (tmp_path / "posthoc_v2.json").write_text(json.dumps({"abstain": {"aurc": 0.1}, "scores": {}}))
    r = read_run(tmp_path)
    assert r["abstain"]["aurc"] == 0.1 and r["abstain_original"]["aurc"] == 0.5 and "posthoc" in r
