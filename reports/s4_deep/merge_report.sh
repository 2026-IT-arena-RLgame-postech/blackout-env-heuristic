#!/bin/bash
# Merge the per-run matches.jsonl of the S4 matrix and write the report (vs plain r5 and paired vs s8e25m3).
cd "$(dirname "$0")/../.."
cat reports/s4_deep/parts/*/matches.jsonl > reports/s4_deep/matches.jsonl
PY=${PY:-~/blackout-env/.venv/bin/python}
PYTHONPATH=. $PY examples/s3_search_eval.py report --out reports/s4_deep --ref s8e25m3
