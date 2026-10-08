#!/bin/zsh
# Held-out eval of D3b (v20-teach) on the shared seeds, after D3b exits.  The v20 teacher gets
# its own baseline rows (teacher:v20-teach) next to v17-teach's.
cd /Users/mac/project/26rl/blackout-env
while pgrep -f "out-dir checkpoints/distill/d3b" >/dev/null; do sleep 30; done
while pgrep -f "distill_heldout_eval.py" >/dev/null; do sleep 30; done   # D2's eval, if still running
D=checkpoints/distill/d3b
.venv/bin/python examples/distill_heldout_eval.py $D/v_0020.pt $D/v_0030.pt $D/v_0040.pt $D/latest.pt \
    --teacher v20-teach --inference iqn --workers 8 --out reports/distill_heldout
echo "HELDOUT D3B DONE"
