"""Step 10. Hypothesis verdicts H1-H10 from the tables of step 09, using the decision rules fixed a priori in
reporting.hypothesis_verdicts and the thresholds in configs/default.yaml -> hypotheses.

Results: results/tables/hypotheses.csv (+ .tex) and results/hypotheses.json
Replaces the original 10_technical_report.py (not in the public repository), which also wrote a Word report;
the paper is now built separately, so only the verdicts are produced here.
"""
from pathlib import Path

from _common import setup
from trace_iomt.compare import write_table
from trace_iomt.reporting import collect, hypothesis_verdicts
from trace_iomt.utils import save_json


def main():
    args, cfg, log = setup("Hypothesis verdicts", "10_hypothesis_verdicts", None)
    R = Path(cfg["paths"]["results_dir"])
    F = collect(R, cfg["paths"]["logs_dir"], cfg["main"]["reference_model"])
    v = hypothesis_verdicts(F, cfg.get("hypotheses"))
    write_table(v, R / "tables", "hypotheses", "Hypothesis verdicts H1-H10 (decision rules fixed a priori)")
    save_json(v.to_dict("records"), R / "hypotheses.json")
    for _, r in v.iterrows():
        log.info(f"{r['hypothesis']:>4} {r['verdict']:<17} {r['evidence']}")


if __name__ == "__main__":
    main()
