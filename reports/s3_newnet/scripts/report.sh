#!/usr/bin/env bash
# merge the per-process sim dirs and print paired reports (search_new - search_r5, each - plain)
set -u
D=${1:-reports/s3_newnet}
mkdir -p $D/merged
cat $D/sim/matches.jsonl $D/sim_new/matches.jsonl $D/sim_plainnew/matches.jsonl $D/sim_abl/matches.jsonl 2>/dev/null > $D/merged/matches.jsonl
PYTHONPATH=. ${PY:-python} examples/s3_search_eval.py report --out $D/merged \
    --pairs search_new:search_r5 search_new:plain search_r5:plain plain_new:plain search_new:plain_new search_new:search_new_v5 search_new_v5:search_r5
