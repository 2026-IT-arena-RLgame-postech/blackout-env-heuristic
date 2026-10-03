#!/bin/bash
# P14 simulator eval chain on wsl_gpu (docs/policy_v2_design_ko.md §29): waits for the P14b run to finish, freezes it,
# then plays the paired eval (rollout_net_strength sim: 128 held-out seeds 1100000+ x both sides, full matches;
# candidates p11 (reference) / p14a / p14b vs V23, V19, P5a, V25 and head-to-head vs P11).
# Usage: scripts/p14_wsl_eval.sh   (detached; pid in ~/p14tmp/eval.pid)
R=$HOME/blackout-env-p14
cd $R
export PYTHONPATH=$R BLACKOUT_MAIN=$HOME/p14assets BLACKOUT_CKPTS=p14a=$HOME/p14frozen/p14a.pt,p14b=$HOME/p14frozen/p14b.pt
B=$HOME/p14runs/p14b
setsid nohup bash -c "
  while kill -0 \$(cat $B/run.pid) 2>/dev/null; do sleep 30; done
  grep -q 'done:' $B/stdout.log && cp $B/latest.pt $HOME/p14frozen/p14b.pt
  .venv/bin/python examples/rollout_net_strength.py sim --candidates p11 p14a p14b --opponents v23 v19 p5a v25 \
      --h2h p14a,p11 p14b,p11 p14b,p14a --seeds 128 --seed0 1100000 --chunk 32 --workers 11 --device cuda \
      --out reports/p14_sim >> reports/p14_sim/stdout.log 2>&1
  echo 'P14 EVAL DONE' >> reports/p14_sim/stdout.log" > /dev/null 2>&1 < /dev/null &
echo $! > $HOME/p14tmp/eval.pid
