#!/bin/zsh
# Held-out eval of D2 on the D1 seeds (teacher and D1 rows already in matches.jsonl), after D2 exits.
cd /Users/mac/project/26rl/blackout-env
while pgrep -f "out-dir checkpoints/distill/d2" >/dev/null; do sleep 30; done
D=checkpoints/distill/d2
.venv/bin/python examples/distill_heldout_eval.py $D/v_0020.pt $D/v_0030.pt $D/v_0040.pt $D/latest.pt \
    --inference iqn --workers 8 --out reports/distill_heldout
echo "HELDOUT D2 DONE"
