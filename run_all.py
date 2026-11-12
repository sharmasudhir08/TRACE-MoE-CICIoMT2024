"""Run the pipeline steps in order, stopping at the first failure.

  python run_all.py                              steps 01-08, 11, 09, 10 with configs/default.yaml
  python run_all.py --config configs/default_v2.yaml --set train.device=cuda
  python run_all.py --from 5 --to 7              only steps 05-07
  python run_all.py --smoke                      synthetic data (step 00) + every step with configs/smoke.yaml

Every step of one invocation shares one TRACE_SESSION id, so the session logs group together.
--seeds is passed to the steps that train models (03-07), e.g. --seeds 42 43.
Re-written on 2026-10-29: the original script is not in the public repository.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {0: "00_make_synthetic.py", 1: "01_prepare.py", 2: "02_audit.py", 3: "03_train_rf.py",
         4: "04_train_deep.py", 5: "05_ablation.py", 6: "06_robustness.py", 7: "07_unknown_attacks.py",
         8: "08_efficiency.py", 9: "09_compare.py", 10: "10_hypothesis_verdicts.py", 11: "11_posthoc.py"}
# Run order: step 11 (post hoc, added 2026-11-03) reads the trained runs and must run before 09 and 10.
ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 9, 10]
# Steps that train or reload models accept --seeds; the others would reject it.
SEEDED = {3, 4, 5, 6, 7}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--from", dest="start", type=int, default=None)
    ap.add_argument("--to", dest="end", type=int, default=ORDER[-1])
    ap.add_argument("--seeds", nargs="+", default=None, help="passed to the steps that train models")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.config = str(ROOT / "configs" / "smoke.yaml")
    if a.start is None:
        a.start = 0 if a.smoke else 1
    os.environ.setdefault("TRACE_SESSION", time.strftime("run_%Y-%m-%d_%H%M%S"))
    for k in ORDER[ORDER.index(a.start):ORDER.index(a.end) + 1]:
        cmd = [sys.executable, str(ROOT / "scripts" / STEPS[k])]
        if k > 0:
            cmd += ["--config", a.config] + [x for s in a.set for x in ("--set", s)]
            if a.seeds and k in SEEDED:
                cmd += ["--seeds", *a.seeds]
        print(f"\n=== step {k:02d}: {STEPS[k]} | session {os.environ['TRACE_SESSION']} ===", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"=== step {k:02d} exit code {rc} ===", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
