#!/bin/bash
# P21 (docs §50): merge the gate rows (wsl: 64 seeds x 9 + 192 more vs V28/P17/V30/(V29/P18bp2); gpu1: 192 more vs the rest)
# into <dir>/matches.jsonl and write report.txt (base P20).  Usage (Mac, repo root): merge.sh [dir=reports/p21_sim] [candidate=p21c]
set -e
D=${1:-reports/p21_sim}; C=${2:-p21c}; mkdir -p $D/parts
scp -q wsl_gpu:blackout-env-p21/$D/matches.jsonl $D/parts/wsl.jsonl 2>/dev/null || true
scp -q poscat_gpu1:blackout-env-p21/$D/matches.jsonl $D/parts/gpu1.jsonl 2>/dev/null || true  # gpu1 clone removed after the run: parts/gpu1.jsonl kept
cat $D/parts/*.jsonl | python3 -c 'import sys,json
seen=set()
for l in sys.stdin:
    r=json.loads(l); k=(r["ours"],r["opponent"],r["seed"],r["side"])
    if k not in seen: seen.add(k); sys.stdout.write(l)' > $D/matches.jsonl  # one row per (ours, opponent, seed, side)
PYTHONPATH=. ${PY:-.venv/bin/python} examples/v27_eval.py report --base p20 --out $D > $D/report.txt
grep -A1 "^$C " $D/report.txt | grep -E "p20:" | sed 's/  */ /g' | cut -c1-80
