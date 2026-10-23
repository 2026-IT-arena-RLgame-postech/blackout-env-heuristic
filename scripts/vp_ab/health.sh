#!/bin/bash
# 30 s health logger (RAM / swap / GPU temp, power, memory, util) -> runs/health.log
cd ~/blackout-env-vp
while true; do
  echo "$(date +%H:%M:%S) $(free -m | awk '/Mem/{print "used="$3" avail="$7}') $(free -m | awk '/Swap/{print "swap="$3}') gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')" >> runs/health.log
  sync runs/health.log; sleep 30
done
