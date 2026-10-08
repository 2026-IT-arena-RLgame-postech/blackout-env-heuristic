#!/bin/bash
# P22 (docs §55): merge P22b's 256-seed gate = 64 seeds (reports/p22_sim, wsl) + 192 more (seeds 2000064+; wsl: V31/V30/P21e/P20/P17(/V28/V29),
# gpu1: V23/V25/P14b/V27 in p22b_gate256, V29/V28 in p22b_gate256b) into reports/p22b_sim/matches.jsonl (P22a rows dropped), report.txt (base P21e).
# Usage (Mac, repo root): reports/p22b_sim/merge.sh   (parts/ not committed; remote dirs removed after the run, so matches.jsonl is the record)
set -e
D=reports/p22b_sim; mkdir -p $D/parts
scp -q wsl_gpu:blackout-env-p22/reports/p22b_gate256/matches.jsonl $D/parts/wsl.jsonl 2>/dev/null || true
scp -q poscat_gpu1:blackout-env-p22/reports/p22b_gate256/matches.jsonl $D/parts/gpu1.jsonl 2>/dev/null || true
scp -q poscat_gpu1:blackout-env-p22/reports/p22b_gate256b/matches.jsonl $D/parts/gpu1b.jsonl 2>/dev/null || true
cat reports/p22_sim/matches.jsonl $D/parts/*.jsonl | python3 -c 'import sys,json
seen=set()
for l in sys.stdin:
    r=json.loads(l); k=(r["ours"],r["opponent"],r["seed"],r["side"])
    if r["ours"]!="p22a" and k not in seen: seen.add(k); sys.stdout.write(l)' > $D/matches.jsonl
PYTHONPATH=. ${PY:-.venv/bin/python} examples/v27_eval.py report --base p21e --out $D > $D/report.txt
