"""Tree-ensemble baseline on single rows (window length 1).

Random Forest is the classifier that wins most often in the published CICIoMT2024 papers we reviewed
(e.g. Naeem 2024, Doménech 2026, Lipsa 2025, Misbah 2025, Alami-Kamouri 2026, and the special-issue
balancing study of Alkhodaidi et al. 2026), so it is the one tree model kept.
"""
from sklearn.ensemble import RandomForestClassifier


def build_classical(name, n_estimators=200, seed=42, n_classes=2):
    if name == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                      class_weight="balanced_subsample")
    raise ValueError(f"Unknown classical model '{name}' (only 'rf' is used in TRACE-MoE)")
