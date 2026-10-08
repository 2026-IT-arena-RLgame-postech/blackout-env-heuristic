#!/bin/bash
# P20 (docs §47): merge the 256-seed gate rows (gpu1 / wsl / gpu2 parts + the 64-seed rows of reports/p20_sim with p19v5 -> p20)
# into reports/p20_gate256/matches.jsonl and write report.txt (base P17).  Run on the Mac from the repo root.
set -e
D=reports/p20_gate256; mkdir -p $D/parts
scp -q poscat_gpu1:blackout-env-p20/reports/p20_gate256/matches.jsonl $D/parts/gpu1.jsonl
scp -q wsl_gpu:blackout-env-p18b/reports/p20_gate256/matches.jsonl $D/parts/wsl.jsonl
scp -q poscat_gpu2:blackout-env-p20/reports/p20_gate256/matches.jsonl $D/parts/gpu2.jsonl || true
sed 's/"p19v5"/"p20"/g' reports/p20_sim/matches.jsonl > $D/parts/sim64.jsonl
cat $D/parts/*.jsonl | awk '!seen[$0]++' > $D/matches.jsonl
.venv/bin/python examples/v27_eval.py report --base p17 --out $D > $D/report.txt
grep -A1 "^p20 " $D/report.txt | grep -E "^p20|p17:" | sed 's/  */ /g' | cut -c1-160
