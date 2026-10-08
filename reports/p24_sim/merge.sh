#!/bin/bash
# P24 (docs §60): merge the gate rows (wsl: V32/V31/V30/P22/h4/h4hunt; gpu1 CPU: V28/V29/V27/P14b/V25/V23/P17/P20/P21e)
# into reports/p24_sim/matches.jsonl and write report.txt (base P22).  Usage (Mac, repo root): reports/p24_sim/merge.sh [dir]
set -e
D=${1:-reports/p24_sim}; mkdir -p $D/parts
scp -q wsl_gpu:blackout-env-p24/$D/matches.jsonl $D/parts/wsl.jsonl 2>/dev/null || true
scp -q poscat_gpu1:blackout-env-p24/$D/matches.jsonl $D/parts/gpu1.jsonl 2>/dev/null || true
cat $D/parts/*.jsonl | python3 -c 'import sys,json
seen=set()
for l in sys.stdin:
    r=json.loads(l); k=(r["ours"],r["opponent"],r["seed"],r["side"])
    if k not in seen: seen.add(k); sys.stdout.write(l)' > $D/matches.jsonl
PYTHONPATH=. ${PY:-.venv/bin/python} examples/v27_eval.py report --base p22 --out $D > $D/report.txt
