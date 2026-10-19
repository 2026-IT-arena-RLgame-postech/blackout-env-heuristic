#!/bin/bash
# Waits until the shared dataset has TARGET rows, stops the collection (pid file), then runs the six A/B runs through queue.sh.
R=$HOME/blackout-env-vitab; cd $R; export PYTHONPATH=$R
D=checkpoints/distill/vitab/data; TARGET=${TARGET:-900000}
while :; do
  n=$(.venv/bin/python - <<PY
import glob, numpy as np
print(sum(int(np.load(p, mmap_mode="r")["labels"].shape[0]) for p in glob.glob("$D/shards/*.npz") if not p.endswith(".tmp.npz")))
PY
)
  echo "$(date) rows $n" >> checkpoints/distill/vitab/queue.log
  [ "$n" -ge "$TARGET" ] && break
  sleep 300
done
pid=$(cat $D/run.pid); kill -TERM -- -$pid 2>/dev/null || kill -TERM $pid; sleep 20
echo "$(date) collection stopped at $n rows" >> checkpoints/distill/vitab/queue.log
exec reports/vitab/queue.sh "A:cnn:0 B1:patch4:0 B2:patch6:0 A:cnn:1 B1:patch4:1 B2:patch6:1" >> checkpoints/distill/vitab/queue.log 2>&1
