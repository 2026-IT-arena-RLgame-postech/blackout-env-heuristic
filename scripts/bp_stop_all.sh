#!/bin/bash
# Stop bp_loop and every econ_stats it started under reports/bp/g* (local + remote hosts); experiments (reports/bp/x*) are kept.
for p in $(pgrep -f "[s]cripts/bp_loop.py"); do kill $p; done
sleep 2
for p in $(ps -Ao pid,command | grep -E "[s]sh -o ConnectTimeout=15 .*reports/bp/g" | awk '{print $1}'); do kill $p; done
for p in $(ps -Ao pid,command | grep -E "[e]con_stats.py play .*reports/bp/g" | awk '{print $1}'); do
  for c in $(pgrep -P $p); do kill $c 2>/dev/null; done; kill $p 2>/dev/null; done
for p in $(ps -Ao pid,command | grep -E "^ *[0-9]+ bash -c cd .*reports/bp/g" | awk '{print $1}'); do kill $p 2>/dev/null; done
for h in poscat_gpu2 wsl_gpu poscat_gpu1; do
  ssh -o ConnectTimeout=10 $h 'for p in $(ps -u $USER -o pid=,args= | grep -E "[e]con_stats.py play .*reports/bp/g" | awk "{print \$1}"); do for c in $(pgrep -P $p); do kill $c 2>/dev/null; done; kill $p; done; for p in $(ps -u $USER -o pid=,args= | grep -E "bash -c cd ~/blackout-env-bp.*reports/bp/g" | awk "{print \$1}"); do kill $p 2>/dev/null; done; sleep 1; echo "$(hostname) left: $(ps -u $USER -o args= | grep -c "[r]eports/bp/g")"' &
done
wait
# orphaned pool workers (parent killed): local + remote
for p in $(ps -Ao pid,ppid,command | awk '$2==1' | grep "[b]lackout-env-balance/.venv/bin/python -B -c" | awk '{print $1}'); do kill $p; done
for h in poscat_gpu2 wsl_gpu poscat_gpu1; do
  ssh -o ConnectTimeout=10 $h 'for p in $(ps -u $USER -o pid=,ppid=,args= | awk "\$2==1" | grep "[b]lackout-env-bp/.venv/bin/python -c" | awk "{print \$1}"); do kill $p; done' &
done
wait
echo "note: orphan cleanup also hits orphaned workers of experiments started with nohup only if their parent died"
