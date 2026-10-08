#!/usr/bin/env bash
# Unity (Linux build) search-new runs, docs §24: GPU for the search server, then sequential runs
set -u
G=${1:-1}; W=${2:-8}
PY=~/blackout-env/.venv/bin/python
N=checkpoints/rollout_net/arch/f3_x_d2.pt
V=checkpoints/value/s2v_f3xd2/iter2.pt
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2
run() {  # out tags extra...
  local out=$1 tags=$2; shift 2
  echo "=== $(date +%T) $out $tags $*"
  PYTHONPATH=. $PY examples/unity_search_eval.py play --opponent p11,v23 --seeds 150 --seed0 4000000 --tags "$tags" \
      --net $N --value $V --server spawn --server-gpus $G --workers $W --out "$out" "$@" || echo "FAILED $out"
}
run reports/s3_newnet/unity/full search
run reports/s3_newnet/unity/race10 race10 --search-until-s 10
run reports/s3_newnet/unity/plain plain
echo "=== $(date +%T) done"
