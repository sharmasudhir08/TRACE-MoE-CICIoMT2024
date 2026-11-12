"""Abstain layer (contribution C3): calibrated confidence, a reject threshold, and the risk-coverage analysis.

After training, a temperature T is fitted on the validation windows (Guo et al., 2017), so the 19-class
confidence is calibrated. The reject threshold is the validation confidence that keeps `target_coverage`
of the windows; below it the model abstains and raises an "unknown / uncertain" alert instead of a class.
The same procedure is applied to every model (neural and RF), so the comparison is fair.

Selective-classification terms (Geifman and El-Yaniv, 2017):
  coverage        share of windows the model accepts
  selective risk  error rate on the accepted windows
  AURC            area under the risk-coverage curve (lower is better), threshold-free
"""
import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score

from .metrics import expected_calibration_error


def _scaled(proba, T):
    z = np.log(np.clip(proba, 1e-12, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(proba_val, y_val, grid=None):
    """Temperature that minimises the validation negative log-likelihood (grid search, robust and fast)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60)) if grid is None else grid
    best, best_nll = 1.0, np.inf
    for T in grid:
        p = _scaled(proba_val, T)
        nll = -np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)).mean()
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


def _expected_cum_errors(conf, correct):
    """Expected number of errors among the k most confident windows, k = 1..n, when windows with equal
    confidence are taken in random order. Inside a block of m tied windows with e errors, the expected
    errors grow linearly by e/m per window, so the result does not depend on the order of the input."""
    order = np.argsort(-conf, kind="stable")
    c, wrong = conf[order], (~correct[order].astype(bool)).astype(float)
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    sizes = np.diff(np.r_[starts, len(c)])
    errors = np.add.reduceat(wrong, starts) if len(c) else np.zeros(0)
    before = np.r_[0.0, np.cumsum(errors)[:-1]]
    pos = np.arange(1, len(c) + 1) - np.repeat(starts, sizes)       # 1..m inside each block
    return np.repeat(before, sizes) + np.repeat(errors / sizes, sizes) * pos


def risk_coverage(conf, correct, n_points=20):
    """Risk-coverage curve (most confident first) and its area (AURC), with ties handled in expectation.

    Before 2026-11-03 a stable sort kept tied windows in file order, which groups them by
    class; with 80-99% of neural-model windows tied at confidence 1.0 the AURC then measured file order."""
    k = np.arange(1, len(conf) + 1)
    risk = _expected_cum_errors(np.asarray(conf), np.asarray(correct)) / k
    cov = k / len(conf)
    pts = np.linspace(0.05, 1.0, n_points)
    idx = np.clip(np.ceil(pts * len(conf)).astype(int) - 1, 0, len(conf) - 1)
    return float(risk.mean()), {"coverage": cov[idx].round(4).tolist(), "risk": risk[idx].round(6).tolist()}


def selective_risk_at(conf, correct, coverage):
    """Expected error rate on the `coverage` share of most confident windows (ties in expectation)."""
    k = max(1, int(np.ceil(coverage * len(conf))))
    return float(_expected_cum_errors(np.asarray(conf), np.asarray(correct))[k - 1] / k)


def _log_softmax(z):
    """log-softmax in float64 that stays distinct near confidence 1: log(1 + s) with s the summed share of the
    other classes is computed with log1p, because 1 + 1e-26 == 1 even in float64."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    e[np.arange(len(z)), z.argmax(1)] = 0.0          # the top class contributes exp(0) = 1, added by log1p
    return z - np.log1p(e.sum(1, keepdims=True))


def fit_temperature_logits(logits_val, y_val, grid=None):
    """Temperature minimising validation NLL, on logits (no saturation, unlike probabilities)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60)) if grid is None else grid
    nll = [-_log_softmax(logits_val / T)[np.arange(len(y_val)), y_val].mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


def confidence_scores(logits, T=1.0, has_energy=True):
    """Confidence scores, higher = more confident, computed in float64 log space so they do not saturate.
    `logits` may be log-probabilities (classical models); energy is then meaningless and skipped."""
    lp = _log_softmax(np.asarray(logits, dtype=np.float64))
    top2 = np.sort(lp, 1)[:, -2:]
    p = np.exp(lp)
    out = {"msp_raw": top2[:, 1],                                   # log max-softmax: same order as MSP
           "msp_calibrated": _log_softmax(np.asarray(logits, dtype=np.float64) / T).max(1),
           "margin": top2[:, 1] - top2[:, 0],                            # logit gap of the top two classes
           "neg_entropy": (p * lp).sum(1)}
    if has_energy:
        z = np.asarray(logits, dtype=np.float64)
        out["energy"] = z.max(1) + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1))
    return out


def selective_scores(logits_val, y_val, logits_test, y_test, target_coverage=0.95, coverages=(0.90, 0.95),
                     has_energy=True):
    """Corrected abstain report (same keys as selective_report) plus AURC and selective risk per score."""
    T = fit_temperature_logits(logits_val, y_val)
    lp_test, lp_val = _log_softmax(logits_test / T), _log_softmax(logits_val / T)
    pt = np.exp(lp_test)
    pred = lp_test.argmax(1)
    correct = pred == y_test
    conf, conf_val = lp_test.max(1), lp_val.max(1)
    thr = float(np.quantile(conf_val, 1 - target_coverage))
    acc = conf >= thr
    labels = np.arange(logits_test.shape[1])
    aurc, curve = risk_coverage(conf, correct)
    report = {"temperature": T, "threshold_log": thr, "target_coverage": target_coverage,
              "coverage": float(acc.mean()), "risk_full": float(1 - correct.mean()),
              "selective_risk": float(1 - correct[acc].mean()) if acc.any() else float("nan"),
              "accepted_macro_f1": float(f1_score(y_test[acc], pred[acc], labels=labels, average="macro",
                                                  zero_division=0)) if acc.any() else float("nan"),
              "aurc": aurc, "ece_raw": expected_calibration_error(y_test, np.exp(_log_softmax(logits_test))),
              "ece_calibrated": expected_calibration_error(y_test, pt), "curve": curve,
              "ties_at_max": float(np.mean(conf == conf.max()))}
    raw_correct = _log_softmax(logits_test).argmax(1) == y_test
    scores = {}
    for name, sc in confidence_scores(logits_test, T, has_energy).items():
        ok = correct if name == "msp_calibrated" else raw_correct
        scores[name] = {"aurc": risk_coverage(sc, ok)[0],
                        **{f"selective_risk_{int(round(100 * c))}": selective_risk_at(sc, ok, c) for c in coverages}}
    return report, scores


def selective_report(proba_val, y_val, proba_test, y_test, target_coverage=0.95):
    """Calibrate on validation, pick the reject threshold on validation, report on test."""
    if len(proba_test) == 0:
        return {}
    have_val = proba_val is not None and len(proba_val) > 0
    T = fit_temperature(proba_val, y_val) if have_val else 1.0
    pt = _scaled(proba_test, T)
    conf = pt.max(1)
    pred = pt.argmax(1)
    correct = pred == y_test
    conf_val = _scaled(proba_val, T).max(1) if have_val else conf
    thr = float(np.quantile(conf_val, 1 - target_coverage))
    acc = conf >= thr
    aurc, curve = risk_coverage(conf, correct)
    labels = np.arange(proba_test.shape[1])
    out = {"temperature": T, "threshold": thr, "target_coverage": target_coverage,
           "coverage": float(acc.mean()), "risk_full": float(1 - correct.mean()),
           "selective_risk": float(1 - correct[acc].mean()) if acc.any() else float("nan"),
           "accepted_macro_f1": float(f1_score(y_test[acc], pred[acc], labels=labels, average="macro",
                                               zero_division=0)) if acc.any() else float("nan"),
           "aurc": aurc, "ece_raw": expected_calibration_error(y_test, proba_test),
           "ece_calibrated": expected_calibration_error(y_test, pt), "curve": curve}
    return out


def routing_report(route, y6, y19, classes6, n_experts):
    """How the experts split the traffic: NMI with the 6 categories and the 19 classes, and counts per category."""
    if route is None or len(route) == 0:
        return {}
    counts = {}
    for k, name in enumerate(classes6):
        m = y6 == k
        if m.any():
            counts[name] = np.bincount(route[m], minlength=n_experts).tolist()
    return {"nmi_label6": float(normalized_mutual_info_score(y6, route)),
            "nmi_label19": float(normalized_mutual_info_score(y19, route)),
            "expert_share": (np.bincount(route, minlength=n_experts) / len(route)).round(4).tolist(),
            "counts_by_category": counts}
