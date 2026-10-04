#!/bin/zsh
# usage: launch.sh NAME CMD... -- detached run, log + pid under reports/late_search_why/logs
cd "$(dirname $0)/../.."
name=$1; shift
export PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
nohup caffeinate -i "$@" > reports/late_search_why/logs/$name.log 2>&1 &
echo $! > reports/late_search_why/logs/$name.pid
echo "$name pid $(cat reports/late_search_why/logs/$name.pid)"
