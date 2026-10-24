#!/bin/bash
# PPO "league arm B" on the WSL laptop (RTX 3060 6 GB): the ppo_r2.sh ARM=reg recipe (init P24c, 9-opponent pool + weights,
# pfsp=hard, fp16 update/critic, compile, critic_warmup=10, eval every 150 its) PLUS per-opponent advantage centering and gated
# self-play snapshots with a share ramp (blackout_env/rl/league.py, PPOTrainer flags; setup note reports/rl/ppo_league_b_setup.md).
#
# Launch (from ~/blackout-env-league on wsl_gpu; needs a free GPU):
#   MAXDEC=19100000 reports/rl/ppo_league_b.sh
# Resume after a crash / reboot (same run dir, SAME env/flags; continues from <OUT>/latest.pt, in-flight games restart):
#   RESUME=1 MAXDEC=19100000 reports/rl/ppo_league_b.sh
set -e
cd ~/blackout-env-league
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" = 0 ] || { echo "GPU has $n compute processes"; exit 1; }
F=checkpoints/distill/frozen
OUT=${OUT:-checkpoints/rl/league_b}
mkdir -p $OUT
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture"
W="0.5,1,1,1,1,1,1,1,1"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
RES=""; [ -z "$RESUME" ] || RES="--resume"
# league flags (defaults of the design: gate 0.85 / gap 100 / start 150 / max 4 / shares 5-10-20 % / anchors >= 40 % / 2 critic-only its)
LEAGUE="adv_center_opp=true league=true league_gate=${GATE:-0.85} league_min_gap=100 league_min_start=150 league_max=${NSNAP:-4} \
league_steps=0.05,0.10,0.20 league_ramp_every=20 league_ramp_min_games=${RAMP_GAMES:-100} league_adv_ratio=1.5 league_ev_min=0.4 \
league_anchor_min=0.4 league_warmup=2"
setsid nohup nice -n 10 env PYTHONPATH=. .venv/bin/python examples/rl_train.py \
    --algo ppo --out $OUT --hours ${HOURS:-30} --max-decisions ${MAXDEC:?set MAXDEC} --eval-every ${EVAL_EVERY:-150} --eval-seeds ${EVAL_SEEDS:-16} \
    --save-every 10 $RES \
    --set "opponents=$OPPS" "opp_weights=$W" update_fp16=true critic_fp16=true update_compile=true critic_warmup=10 critic_head=reg pfsp=hard \
    games=${GAMES:-256} segment=${SEGMENT:-128} heur_workers=${HWORKERS:-6} \
    graph_shared_pool=true league_snap_graphs=true update_micro=512 cuda_mem_fraction=${MEMFRAC:-0.70} critic_read_batch=1024 $LEAGUE ${EXTRA} \
    >> $OUT/run.log 2>&1 &
echo "launched pid $! -> $OUT"
setsid nohup bash reports/rl/health_logger.sh $OUT > /dev/null 2>&1 &
