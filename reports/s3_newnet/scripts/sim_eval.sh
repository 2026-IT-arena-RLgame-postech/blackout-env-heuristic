#!/usr/bin/env bash
# Sim Gate (docs §24): one GPU, sequential runs.  usage: sim_eval.sh GPU OUT "tag|net|value|opponent|seeds|extra" ...
# run from the repo clone; every row appends to OUT/matches.jsonl (reruns skip done rows)
set -u
GPU=$1; OUT=$2; shift 2
PY=~/blackout-env/.venv/bin/python
for spec in "$@"; do
  IFS='|' read -r tag net value opp seeds extra <<< "$spec"
  echo "=== $(date +%T) $tag net=$net value=$value vs $opp seeds=$seeds $extra"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY examples/s3_search_eval.py play --opponent "$opp" --net "$net" --value "$value" \
      --margin 3 --seeds "$seeds" --seed0 2000000 --tag "$tag" --out "$OUT" $extra || echo "FAILED $spec"
done
echo "=== $(date +%T) done"
