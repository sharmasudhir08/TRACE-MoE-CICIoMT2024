#!/bin/zsh
# Twin study (H11) on the training machine (plan of 2026-11-05, configs/twin.yaml). Needs the device rows of
# step 14 in data/processed/twin/device. Every stage skips finished runs, so any lane can be stopped and
# started again at no cost.
#
#   ./twin_all.sh <device> <lane>
#     lane check  1 epoch of V3 seed 42, then tune and sync, into results/_twin_check (about 30-45 min)
#     lane A      V3 then V0, 5 seeds each (10 runs)                         terminal 1
#     lane B      V1 and V2 on 3 seeds, random forest on 5 (6 + 5 runs)      terminal 2, in parallel with A
#     lane final  choose η, δ, budget on validation, then V4 on 5 seeds      after lane A has finished
#   e.g. ./twin_all.sh mps A      and, in a second terminal, ./twin_all.sh mps B
set -euo pipefail
cd "${0:A:h}"
DEV=${1:-mps}
LANE=${2:-check}
CFG=${CFG:-configs/twin.yaml}
D=(--set train.device=$DEV)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TRACE_SESSION="twin_${LANE}_$(date +%Y-%m-%d_%H%M%S)"

step() { echo "\n=== $(date '+%F %T') | lane $LANE | $* ==="; caffeinate -i python scripts/16_twin_train.py --config $CFG "$@"; }

case $LANE in
  check) C=($D --set train.epochs=1 --set paths.results_dir=results/_twin_check --set paths.logs_dir=logs/_twin_check)
         step --stage train --variants V3 --seeds 42 $C
         step --stage tune $C
         step --stage sync --seeds 42 $C ;;
  A)     step --stage train --variants V3 V0 $D ;;
  B)     step --stage train --variants V1 V2 rf $D ;;
  final) step --stage tune $D
         step --stage sync $D ;;
  *)     echo "lane must be check, A, B or final (got '$LANE')"; exit 2 ;;
esac
echo "\n=== $(date '+%F %T') | lane $LANE done ==="
