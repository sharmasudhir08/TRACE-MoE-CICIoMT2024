"""Diagnostic (read-only): are AURC values distorted by tied confidences?

risk_coverage() sorts windows by confidence with a stable sort, so windows with the
same confidence keep their file order, which groups them by class. If many windows
tie (e.g. confidence exactly 1.0), AURC then depends on file order, not on the model.

For every run under results_rerun/main/, this prints:
  - the share of test windows tied at the maximum confidence,
  - AURC with the current stable tie-break (as in the run, but from float16 proba),
  - AURC with ties broken at random (mean of 5 shuffles), which removes the file-order effect.
A large gap between the two AURC columns means the reported AURC is a tie artefact.

Usage (from the repository root):  python scripts/check_aurc_ties.py [results_rerun]
Writes nothing; nothing in results_rerun is changed.
"""
import sys
from pathlib import Path

import numpy as np


def aurc(conf, correct, rng=None):
    if rng is None:
        order = np.argsort(-conf, kind="stable")
    else:
        order = np.lexsort((rng.random(len(conf)), -conf))   # sort by confidence, ties at random
    c = correct[order].astype(float)
    return float((1 - np.cumsum(c) / np.arange(1, len(c) + 1)).mean())


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_rerun") / "main"
    newest = {}   # stem -> (version, path): use the newest copy of a repeated run (<name>__vN), as step 09 does
    for p in root.glob("*/predictions.npz"):
        stem, _, version = p.parent.name.partition("__v")
        rank = int(version) if version else 1
        if stem not in newest or rank > newest[stem][0]:
            newest[stem] = (rank, p)
    runs = [newest[stem][1] for stem in sorted(newest)]
    if not runs:
        raise SystemExit(f"[AURC_TIES] no predictions.npz under {root}; run from the repository root")
    print(f"{'run':55s} {'tied@max':>9s} {'AURC stable':>12s} {'AURC random':>12s}")
    for path in runs:
        z = np.load(path)
        if "proba_label19" not in z:
            print(f"{path.parent.name:55s} no proba_label19 saved")
            continue
        proba = z["proba_label19"].astype(np.float32)
        conf, correct = proba.max(1), proba.argmax(1) == z["y_true_label19"]
        tied = float(np.mean(conf == conf.max()))
        rng = np.random.default_rng(0)
        shuffled = np.mean([aurc(conf, correct, rng) for _ in range(5)])
        print(f"{path.parent.name:55s} {100 * tied:8.1f}% {100 * aurc(conf, correct):12.3f} {100 * shuffled:12.3f}")


if __name__ == "__main__":
    main()
