#!/bin/zsh
# Held-out eval of D1, started when the D1 training process (pid 75449) exits.
cd /Users/mac/project/26rl/blackout-env
while kill -0 75449 2>/dev/null; do sleep 30; done
D=checkpoints/distill/d1
PY=.venv/bin/python
$PY examples/distill_heldout_eval.py $D/v_0020.pt $D/v_0030.pt $D/v_0040.pt $D/latest.pt \
    --inference iqn --workers 8 --out reports/distill_heldout_d1
$PY examples/distill_heldout_eval.py $D/v_0030.pt $D/latest.pt \
    --inference fixed_tau tie_mean --no-teacher-baseline --workers 8 --out reports/distill_heldout_d1
echo "HELDOUT DONE"
