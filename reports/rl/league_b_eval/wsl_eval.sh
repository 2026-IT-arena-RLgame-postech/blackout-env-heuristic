#!/bin/bash
# League B 256-seed eval on the WSL laptop (RTX 3060), run from ~/blackout-env-league.
# Phase 1 (GPU, 3 parallel): league_b final vs p24c/p22/V32.1/p5a/p16 and head-to-head vs ppo_r2_reg final,
#   seeds 970000-970255 x 2 sides in 64-seed chunks (same as ppo_r2_eval) -> league_b__<opp>__<lo>.json
# Phase 2 (CPU): OOD-26 suite minus run16_final (checkpoint unavailable; docs §56) for league_b final / iter_600, ppo_r2_reg final, p24c; 32 seeds x 2 from 2600000.
cd ~/blackout-env-league
D=reports/rl/league_b_eval; F=checkpoints/distill/frozen; C=checkpoints/rl/league_b
mkdir -p $D/logs
( while :; do echo "$(date +%H:%M:%S) $(free -m | awk '/Mem/{print "used="$3" avail="$7}')" >> $D/health_mem.log
  [ $(( $(date +%s) / 30 % 4 )) = 0 ] && echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')" >> $D/health_gpu.log
  sync; sleep 30; done ) &
HP=$!
for o in "p24c_v321 net:$F/p24c_v321.pt" "p22_v31 net:$F/p22_v31.pt" "v32_1 V32.1" "p5a_v22 net:$F/p5a_v22.pt" "p16_v27 net:$F/p16_v27.pt" "ppo_r2_reg net:checkpoints/rl/ppo_r2_reg/final.pt"; do
  for lo in 970000 970064 970128 970192; do echo "$o $lo $((lo + 64))"; done
done | xargs -P 3 -L 1 bash -c '
  ot=$0; os=$1; lo=$2; hi=$3; out='$D'/league_b__${ot}__${lo}.json
  [ -s $out ] && exit 0
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 env PYTHONPATH=. .venv/bin/python examples/eval_grpo_ckpt.py \
    --cand '$C'/final.pt --opp $os --lo $lo --hi $hi --out $out --device cuda:0 > '$D'/logs/league_b__${ot}__${lo}.log 2>&1
  rc=$?; [ -s $out ] && echo "$(date) done $ot $lo" || echo "$(date) FAIL $ot $lo rc=$rc"; sync'
echo "$(date) PHASE1 DONE $(ls $D/league_b__*.json | wc -l)/24"; sync
# Phase 2: OOD-26 on CPU (nets on CPU)
export PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt
nice -n 15 .venv/bin/python examples/v27_eval.py play --seeds 32 --seed0 2600000 --cap-s 425 --stop-exhausted \
  --device cpu --chunk 16 --workers ${EW:-6} --out $D/ood26 \
  --ours "net:$C/final.pt;net:$C/iter_600.pt;net:checkpoints/rl/ppo_r2_reg/final.pt;net:$F/p24c_v321.pt" --opponent "ood26=v30_h4hunt,v30_h2,v30_camp,v30_stage3,v30_nostage,v30_c0,v30_hunt,v30_reckless,early_rush,fortress,raider,duelist,cargo_hunter,rush_all,v4,v19,v21,v26,random,idle,v31_eps10,v30_lag4,v30_freeze10,p20_t1,v31" > $D/logs/ood26.log 2>&1
.venv/bin/python examples/ood_report.py --out $D/ood26 --base "net:$F/p24c_v321.pt" > $D/ood26_report.txt 2>&1
kill $HP
echo "$(date) ALL DONE"; sync
