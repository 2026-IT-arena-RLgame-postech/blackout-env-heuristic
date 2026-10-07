#!/bin/bash
# V32: run gate.sh $2 (env as gate.sh) after reports/v32/$1 has finished.  Detached by the caller.
R=${R:-$HOME/blackout-env-v32}; cd $R
until grep -q "GATE DONE" reports/v32/$1/stdout.log 2>/dev/null; do sleep 30; done
bash reports/v32/scripts/gate.sh $2
