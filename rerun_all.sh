#!/bin/zsh
# Full re-run on the training machine (plan of 2026-10-31).
#   main comparison (steps 03-04): 5 seeds, both splits
#   H4 ablation (step 05):   full + dense expert on seeds 42-44; the other variants and window lengths on seed 42
#   H6/H7 robustness (06):   seeds 42-44
#   H8 unknown attacks (07): TRACE-MoE on seeds 42-44; Transformer and BiLSTM-DNN baselines on seed 42
# Every step skips runs that are already finished, so any lane can be stopped and started again at no cost.
#
#   ./rerun_all.sh <device> <lane>
#     lane A      RF, neural models (main), robustness, ablation seeds 43-44     terminal 1
#     lane B      unknown attacks, ablation seed 42                              terminal 2, in parallel with A
#     lane final  efficiency, post-hoc (step 11), tables, verdicts               after A and B have finished
#     lane posthoc  step 11 then tables and verdicts only (no training; 2026-11-03 AURC fix)
#     lane all    everything in one terminal
#   e.g. ./rerun_all.sh mps A      and, in a second terminal, ./rerun_all.sh mps B
set -euo pipefail
cd "${0:A:h}"
DEV=${1:-mps}
LANE=${2:-all}
CFG=${CFG:-configs/rerun.yaml}           # override only for testing, e.g. CFG=configs/smoke.yaml
ALL_SEEDS=(42 43 44 45 46)
D=(--set train.device=$DEV)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TRACE_SESSION="rerun_${LANE}_$(date +%Y-%m-%d_%H%M%S)"

step() { echo "\n=== $(date '+%F %T') | lane $LANE | $* ==="; caffeinate -i python "$@" --config $CFG; }

main_runs()   { step scripts/03_train_rf.py --seeds $ALL_SEEDS
                step scripts/04_train_deep.py $D --seeds $ALL_SEEDS; }
robustness()  { step scripts/06_robustness.py $D --seeds 42 43 44; }
ablation_42() { step scripts/05_ablation.py $D --seeds 42; }
ablation_rest() { step scripts/05_ablation.py $D --seeds 43 44 --variants full dense_expert --skip-windows; }
unknown()     { step scripts/07_unknown_attacks.py $D --seeds 42
                step scripts/07_unknown_attacks.py $D --seeds 43 44 --models trace_moe; }
final()       { step scripts/08_efficiency.py
                step scripts/11_posthoc.py $D
                step scripts/09_compare.py
                step scripts/10_hypothesis_verdicts.py; }

case $LANE in
  A)     main_runs; robustness; ablation_rest ;;
  B)     unknown; ablation_42 ;;
  final) final ;;
  posthoc) step scripts/11_posthoc.py $D; step scripts/09_compare.py; step scripts/10_hypothesis_verdicts.py ;;
  all)   main_runs; robustness; ablation_rest; ablation_42; unknown; final ;;
  *)     echo "lane must be A, B, final, posthoc or all (got '$LANE')"; exit 2 ;;
esac
echo "\n=== $(date '+%F %T') | lane $LANE done ==="
