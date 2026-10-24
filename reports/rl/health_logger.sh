#!/bin/bash
# Health logger for long WSL runs: free -m every 30 s -> $1/health_mem.log, nvidia-smi every 2 min -> $1/health_gpu.log (sync after writes).
OUT=${1:?run dir}
n=0
while true; do
    { date +%FT%T; free -m | sed -n 2p; } >> "$OUT/health_mem.log"
    if [ $((n % 4)) = 0 ]; then
        { date +%FT%T; nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm --format=csv,noheader; } >> "$OUT/health_gpu.log"
    fi
    sync
    n=$((n + 1))
    sleep 30
done
