#!/bin/bash
# GPU-resident data: single GPU (default / CUDA graphs) and DDP strong + weak scaling.
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
P=~/blackout-env/.venv/bin/python
TR=~/blackout-env/.venv/bin/torchrun
O=~/cuda_work/ab4
mkdir -p $O
COMMON="--data checkpoints/rollout_net/s3data_x1 --init checkpoints/rollout_net/r5.pt --steps 3000 --lr 3e-4 --log-every 250 --eval-every 1000"
FAST="--amp fp16 --compile"
uptime > $O/uptime_start.txt
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train $COMMON $FAST --out $O/g1.pt > $O/g1.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train $COMMON $FAST --compile-mode reduce-overhead --out $O/g1_cg.pt > $O/g1_cg.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train $COMMON $FAST --gpu-data 0 --out $O/g1_host.pt > $O/g1_host.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/rollout_net_train.py train $COMMON $FAST --out $O/g2.pt > $O/g2.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/rollout_net_train.py train $COMMON $FAST --out $O/g4.pt > $O/g4.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train $COMMON $FAST --batch 4096 --out $O/g1_b4096.pt > $O/g1_b4096.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/rollout_net_train.py train $COMMON $FAST --batch 2048 --out $O/g2_b2048.pt > $O/g2_b2048.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/rollout_net_train.py train $COMMON $FAST --batch 4096 --out $O/g4_b4096.pt > $O/g4_b4096.log 2>&1
uptime > $O/uptime_end.txt
rm -f $O/*.pt
echo AB4 DONE
