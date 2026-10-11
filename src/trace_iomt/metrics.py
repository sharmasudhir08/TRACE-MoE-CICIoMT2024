"""Evaluation metrics used in every experiment, so that all models are compared the same way."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)


def recall_at_fpr(y_true_bin, score_attack, max_fpr=0.01):
    """Highest attack recall (TPR) reachable while the false-positive rate stays <= max_fpr."""
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_bin, score_attack)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def expected_calibration_error(y_true, proba, n_bins=15):
    """ECE: gap between confidence and accuracy, averaged over confidence bins (lower is better)."""
    conf = proba.max(1)
    correct = (proba.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y_true, y_pred, proba=None, class_names=None, attack_index=None, max_fpr=0.01):
    """Return a dict of summary metrics plus per-class precision/recall/F1 and the confusion matrix."""
    labels = np.arange(len(class_names)) if class_names is not None else np.unique(np.r_[y_true, y_pred])
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = class_names if class_names is not None else [str(l) for l in labels]
    out["per_class"] = {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                   "support": int(s[i])} for i in range(len(labels))}
    present = s > 0
    out["macro_recall_present"] = float(r[present].mean()) if present.any() else float("nan")
    out["min_class_recall"] = float(r[present].min()) if present.any() else float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    if proba is not None:
        proba = np.asarray(proba, dtype=np.float64)
        out["ece"] = expected_calibration_error(y_true, proba)
        if proba.shape[1] == 2 and attack_index is not None:
            yb = (y_true == attack_index).astype(int)
            score = proba[:, attack_index]
            if len(np.unique(yb)) == 2:
                out["roc_auc"] = float(roc_auc_score(yb, score))
                out["pr_auc"] = float(average_precision_score(yb, score))
                out[f"recall_at_fpr_{max_fpr:g}"] = recall_at_fpr(yb, score, max_fpr)
        else:
            aps = []
            for k in labels:
                yk = (y_true == k).astype(int)
                if 0 < yk.sum() < len(yk):
                    aps.append(average_precision_score(yk, proba[:, k]))
            out["macro_pr_auc"] = float(np.mean(aps)) if aps else float("nan")
    return out


def ood_metrics(scores_known, scores_unknown):
    """Zero-day detection from an in-distribution score (higher = known). Unknown is the positive class."""
    y = np.r_[np.zeros(len(scores_known)), np.ones(len(scores_unknown))]
    s = -np.r_[scores_known, scores_unknown]              # higher = more likely unknown
    if len(scores_known) == 0 or len(scores_unknown) == 0:
        return {"auroc": float("nan"), "fpr_at_95tpr": float("nan")}
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, 0.95, side="left")
    return {"auroc": float(roc_auc_score(y, s)),
            "fpr_at_95tpr": float(fpr[min(idx, len(fpr) - 1)]),
            "n_known": int(len(scores_known)), "n_unknown": int(len(scores_unknown))}


def evaluate_levels(y_true_by_level, proba_by_level, hierarchy, max_fpr=0.01):
    """Metrics for every level (label2, label6, label19) from predicted probabilities."""
    res = {}
    for lvl, proba in proba_by_level.items():
        names = hierarchy.classes[lvl]
        y = y_true_by_level[lvl]
        pred = proba.argmax(1)
        attack_idx = names.index("Attack") if lvl == "label2" else None
        res[lvl] = classification_metrics(y, pred, proba, names, attack_idx, max_fpr)
    return res
