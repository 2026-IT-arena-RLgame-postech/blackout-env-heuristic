#!/bin/bash
# P24c 256-seed gate (docs §60): the 64-seed gate rows (reports/p24_sim, seeds 2200000+) + 192 more seeds from 2200064
# (wsl: P22 h2h/V31/V30/P20; gpu1 CPU: V23/P14b/V25) for P24c and P22; p24a rows and unpaired cells dropped.
# Writes reports/p24c_sim/matches.jsonl + report.txt (base P22).  Usage (Mac, repo root): reports/p24c_sim/merge.sh
set -e
D=reports/p24c_sim; mkdir -p $D/parts
scp -q wsl_gpu:blackout-env-p24/reports/p24c_gate256/matches.jsonl $D/parts/wsl.jsonl 2>/dev/null || true
scp -q poscat_gpu1:blackout-env-p24/reports/p24c_gate256/matches.jsonl $D/parts/gpu1.jsonl 2>/dev/null || true
cat reports/p24_sim/matches.jsonl $D/parts/*.jsonl | python3 -c 'import sys,json
seen=set(); rows=[]
for l in sys.stdin:
    r=json.loads(l); k=(r["ours"],r["opponent"],r["seed"],r["side"])
    if r["ours"]!="p24a" and k not in seen: seen.add(k); rows.append((k,l))
# keep only (opponent, seed, side) cells that both P24c and P22 played (drops the unpaired P24c-only V32 rows of a killed lane)
for k,l in rows:
    if all((n,)+k[1:] in seen for n in ("p24c","p22")): sys.stdout.write(l)' > $D/matches.jsonl
PYTHONPATH=. ${PY:-.venv/bin/python} examples/v27_eval.py report --base p22 --out $D > $D/report.txt
