#!/bin/bash
# Q12 stacked candidate (sentry_radius=8.0 + threat_radius=0.13 + camp_engage_radius=6.5), 256 seeds 1700000+, pool 10.
# Base rows come from gate64/gate256a (same seeds).  gate_stack.sh [pid-to-wait-for]
set -eu
cd "$(dirname "$0")/../../.."
if [ -n "${1:-}" ]; then while kill -0 "$1" 2>/dev/null; do sleep 60; done; fi
K="reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
B="strategic_v30:$K"
POOL="$B;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;strategic_v28:reserve_stage=0.5,trade_hold=advance;p17;p16;strategic_v27:reserve_stage=1.5;strategic_v25;p14b;strategic_v23;strategic_v30:reserve_stage=0.75,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
exec reports/q12_v31/scripts/gate.sh reports/q12_v31/${OUT:-gate_stack} ${SEED0:-1700000} ${SEEDS:-256} ${WORKERS:-8} \
  "$B,sentry_radius=8.0,threat_radius=0.13,camp_engage_radius=6.5" "$POOL"
