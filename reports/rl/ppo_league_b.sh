#!/bin/bash
# PPO "league arm B" on the WSL laptop (RTX 3060 6 GB): the ppo_r2.sh ARM=reg recipe (init P24c, 9-opponent pool + weights,
# pfsp=hard, fp16 update/critic, compile, critic_warmup=10, eval every 150 its) PLUS per-opponent advantage centering and gated
# self-play snapshots with a share ramp (blackout_env/rl/league.py, PPOTrainer flags; setup note reports/rl/ppo_league_b_setup.md).
#
# NOTE league_ramp_min_games=30 (design says 100): a snapshot at 5 % of ~3 finished games/it finishes only ~0.16 games/it, so 100 would take ~600 its.
# Launch (from ~/blackout-env-league on wsl_gpu; needs a free GPU):
#   MAXDEC=19100000 reports/rl/ppo_league_b.sh
# On poscat_gpu2 (multi-GPU, never GPU 3): REPO=~/blackout-env-<dir> GPU=2,0 (learner,actor: async 2-GPU, default) or GPU=2 (sync) OUT=checkpoints/rl/league_b2 MAXDEC=... reports/rl/ppo_league_b.sh
# Resume after a crash / reboot (same run dir, SAME env/flags; continues from <OUT>/latest.pt, in-flight games restart):
#   RESUME=1 MAXDEC=19100000 reports/rl/ppo_league_b.sh
set -e
cd ${REPO:-~/blackout-env-league}
# GPU may be a list ("2,0"): the first is the learner, the second the async actor (PPO async_collect=auto turns it on).
if [ -n "$GPU" ]; then export CUDA_VISIBLE_DEVICES=$GPU
  for g in ${GPU//,/ }; do [ "$g" != 3 ] || [ -n "$ALLOW_GPU3" ] || { echo "GPU 3 is off limits (ALLOW_GPU3=1 when the user okays it)"; exit 1; }
    if [ "$g" = 3 ]; then   # (ALLOW_GPU3) Xorg lives there as a graphics process: refuse only if a compute process runs
      u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" || { echo "GPU 3 has compute processes"; exit 1; }
    elif [ -n "$ALLOW_OWN" ]; then   # tolerate only my own processes on that GPU (e.g. a running eval of the previous trial)
      u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2}'); do
        [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || { echo "GPU $g has a foreign process $p"; exit 1; }; done
    else nvidia-smi -i $g | grep -q "No running processes found" || { nvidia-smi -i $g; echo "GPU $g busy"; exit 1; }; fi
  done; n=0; else n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); fi
[ "$n" = 0 ] || { echo "GPU has $n compute processes"; exit 1; }
F=checkpoints/distill/frozen
OUT=${OUT:-checkpoints/rl/league_b}
mkdir -p $OUT
OPPS=${OPPS:-"net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture"}
W=${W:-"0.5,1,1,1,1,1,1,1,1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
RES=""; [ -z "$RESUME" ] || RES="--resume"
# league flags (defaults of the design: gate 0.85 / gap 100 / start 150 / max 4 / shares 5-10-20 % / anchors >= 40 % / 2 critic-only its)
LEAGUE="adv_center_opp=true league=true league_gate=${GATE:-0.85} league_min_gap=100 league_min_start=150 league_max=${NSNAP:-4} \
league_steps=0.05,0.10,0.20 league_ramp_every=20 league_ramp_min_games=${RAMP_GAMES:-30} league_adv_ratio=1.5 league_ev_min=0.4 \
league_anchor_min=0.4 league_warmup=2"
setsid nohup nice -n 10 env PYTHONPATH=. .venv/bin/python examples/rl_train.py \
    --algo ppo --out $OUT --hours ${HOURS:-30} --max-decisions ${MAXDEC:?set MAXDEC} --eval-every ${EVAL_EVERY:-150} --eval-seeds ${EVAL_SEEDS:-16} \
    ${EVAL_OPPS:+--eval-opps $EVAL_OPPS} \
    --save-every 10 $RES \
    --set "opponents=$OPPS" "opp_weights=$W" update_fp16=true critic_fp16=true update_compile=true critic_warmup=10 critic_head=reg pfsp=hard \
    games=${GAMES:-256} segment=${SEGMENT:-128} heur_workers=${HWORKERS:-6} \
    graph_shared_pool=true league_snap_graphs=true update_micro=512 cuda_mem_fraction=${MEMFRAC:-0.70} critic_read_batch=1024 $LEAGUE ${EXTRA} \
    >> $OUT/run.log 2>&1 &
echo "launched pid $! -> $OUT"
setsid nohup bash reports/rl/health_logger.sh $OUT > /dev/null 2>&1 &
