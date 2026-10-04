#!/bin/bash
# One lane of the S4 simulator matrix: runs the jobs of a job file one after another on GPU $G.
# job line: tag|opponent|seeds|extra args   (the ref tag s8e25m3 also plays the plain r5 rows)
# usage: G=1 setsid nohup bash reports/s4_deep/lane.sh jobs_a.txt > logs/lane_a.log 2>&1 &
cd ~/blackout-env-s4; set -f
PY=~/blackout-env/.venv/bin/python
export CUDA_VISIBLE_DEVICES=$G NUMBA_NUM_THREADS=${NT:-4} OMP_NUM_THREADS=${NT:-4}
while IFS='|' read -r tag opp seeds extra; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  out=reports/s4_deep/parts/${tag}_${opp}
  plain=--no-plain; [[ "$tag" == s8e25m3 ]] && plain=
  echo "=== $tag vs $opp ($seeds seeds) $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"
  PYTHONPATH=. $PY examples/s3_search_eval.py play --opponent $opp --seeds $seeds --seed0 2000000 --chunk 128 \
      --margin 3 --packed --device cuda --tag $tag --out $out $plain $extra 2>&1 | grep -E '^\[|Error|error' | tail -3
done < "$1"
echo "=== lane done $(date +%T)"
