#!/bin/bash
# 30 s RAM/swap log; nvidia-smi (temp, power, memory, util) only every 2 min -> runs/health.log
cd ~/blackout-env-vp
n=0
while true; do
  g=""
  if [ $((n % 4)) = 0 ]; then
    g=" gpu=$(timeout 20 nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"
  fi
  echo "$(date +%H:%M:%S) $(free -m | awk '/Mem/{print "used="$3" avail="$7}') $(free -m | awk '/Swap/{print "swap="$3}')$g" >> runs/health.log
  sync; n=$((n+1)); sleep 30
done
