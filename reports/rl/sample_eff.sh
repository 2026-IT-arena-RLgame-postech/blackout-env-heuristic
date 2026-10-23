#!/bin/bash
# Sample-efficiency eval on the WSL laptop (RTX 3060): every saved policy ckpt of grpo_r3b, ppo_r1, ppo_r2_reg (+ p24c_v321 = step 0)
# vs p24c_v321 / p22_v31 / V32.1, seeds 980000-980063, both sides. 3 jobs in parallel, health log every 30 s, resumable (skips existing jsons).
# Usage (WSL):  cd ~/blackout-env-eval && setsid nohup reports/rl/sample_eff.sh > reports/rl/sample_eff/driver.log 2>&1 &
cd ~/blackout-env-eval
D=reports/rl/sample_eff; F=checkpoints/distill/frozen; mkdir -p $D/logs
( while :; do echo "$(date +%H:%M:%S) $(free -m | awk '/Mem/{print "used="$3" avail="$7}') gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')" >> $D/wsl_health.log; sync; sleep 30; done ) &
HP=$!
{
for o in "p24c_v321 net:$F/p24c_v321.pt" "p22_v31 net:$F/p22_v31.pt" "v32_1 V32.1"; do
  echo "step0__p24c_v321 $F/p24c_v321.pt $o"
done
for run in grpo_r3b ppo_r1 ppo_r2_reg; do
  for ck in $(ls checkpoints/rl/$run/iter_*.pt | sed 's/.*iter_\([0-9]*\)\.pt/\1/' | sort -n) final; do
    f=checkpoints/rl/$run/iter_$ck.pt; [ $ck = final ] && f=checkpoints/rl/$run/final.pt
    for o in "p24c_v321 net:$F/p24c_v321.pt" "p22_v31 net:$F/p22_v31.pt" "v32_1 V32.1"; do echo "${run}__$ck $f $o"; done
  done
done
} | xargs -P 3 -L 1 bash -c '
  name=$0; cand=$1; ot=$2; os=$3; out='$D'/${name}__${ot}.json
  [ -s $out ] && exit 0
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 env PYTHONPATH=. .venv/bin/python examples/eval_grpo_ckpt.py \
    --cand $cand --opp $os --lo 980000 --hi 980064 --out $out --device cuda:0 > '$D'/logs/${name}__${ot}.log 2>&1
  echo "$(date) done $name $ot rc=$?"; sync'
kill $HP
echo "$(date) ALL DONE $(ls $D/*.json | wc -l)/69"
