"""Post-hoc analyses on saved predictions (added 2026-11-03): hierarchy consistency and tuned binary thresholds.

Both work for every model from predictions.npz alone, so the random forest (whose model is not saved) is
treated exactly like the neural models.
"""
import numpy as np
from sklearn.metrics import f1_score

from .abstain import selective_scores
from .utils import code_hash, now_iso

ATTACK = 1   # index of "Attack" in labels.BINARY


def hierarchy_consistency(pred19, pred6, pred2, parent19, parent6):
    """Share of windows whose predicted labels agree across levels: 19 -> 6, 6 -> 2 and all three."""
    p19to6 = np.asarray(parent19)[pred19] == pred6
    p6to2 = np.asarray(parent6)[pred6] == pred2
    return {"c19_6": float(p19to6.mean()), "c6_2": float(p6to2.mean()), "joint": float((p19to6 & p6to2).mean())}


def tune_binary_threshold(p_attack_val, y_val, n_candidates=1001):
    """Threshold on the attack probability that maximises validation macro-F1 (candidates: score quantiles)."""
    p = np.asarray(p_attack_val, dtype=np.float64)
    cands = np.unique(np.quantile(p, np.linspace(0, 1, n_candidates)))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_score(y_val, (p >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, float(best_f1)


def binary_at_thresholds(p_attack_val, y_val, p_attack_test, y_test):
    """Test binary macro-F1 at the default 0.5 and at the validation-tuned threshold."""
    t, val_f1 = tune_binary_threshold(p_attack_val, y_val)
    test = np.asarray(p_attack_test, dtype=np.float64)
    f1 = lambda thr: float(f1_score(y_test, (test >= thr).astype(int), average="macro", zero_division=0))
    return {"threshold_tuned": t, "val_macro_f1_tuned": val_f1, "macro_f1_default": f1(0.5), "macro_f1_tuned": f1(t)}


def analyse_run(hierarchy, z, logits, agreement, cfg, has_energy):
    """posthoc_v2.json content for one run. z: the arrays of predictions.npz; logits: {"val"|"test": (19-class
    logits or log-probabilities, y_true)}; hierarchy: the run's hierarchy dict (parent19, parent6)."""
    report, scores = selective_scores(*logits["val"], *logits["test"], cfg["abstain"]["target_coverage"],
                                      has_energy=has_energy)
    return {"created_at": now_iso(), "code_sha256": code_hash(),
            "source": "re-inference, float32 logits" if has_energy else "saved float16 probabilities",
            "prediction_agreement": agreement, "abstain": report, "scores": scores,
            "consistency": hierarchy_consistency(z["y_pred_label19"], z["y_pred_label6"], z["y_pred_label2"],
                                                 hierarchy["parent19"], hierarchy["parent6"]),
            "binary": binary_at_thresholds(z["val_proba_label2"][:, ATTACK].astype(np.float64), z["val_y_true_label2"],
                                           z["proba_label2"][:, ATTACK].astype(np.float64), z["y_true_label2"])}


def classical_logits(z):
    """Log-probabilities from the saved float16 probabilities (classical models keep no model file)."""
    log_p = lambda p: np.log(np.clip(p.astype(np.float32), 1e-12, 1.0))
    return {"val": (log_p(z["val_proba_label19"]), z["val_y_true_label19"]),
            "test": (log_p(z["proba_label19"]), z["y_true_label19"])}
