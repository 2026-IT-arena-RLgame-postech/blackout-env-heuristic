#!/bin/bash
# DDP scaling of the RolloutNet trainer after the OMP_NUM_THREADS fix; then the value-step bench.
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
P=~/blackout-env/.venv/bin/python
TR=~/blackout-env/.venv/bin/torchrun
O=~/cuda_work/ab
COMMON="--data checkpoints/rollout_net/s3data_x1 --init checkpoints/rollout_net/r5.pt --steps 3000 --lr 3e-4 --log-every 250 --eval-every 1000"
FAST="--amp fp16 --compile"
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/rollout_net_train.py train $COMMON $FAST --out $O/ddp2.pt > $O/ddp2.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/rollout_net_train.py train $COMMON $FAST --out $O/ddp4.pt > $O/ddp4.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/rollout_net_train.py train $COMMON $FAST --batch 2048 --out $O/ddp2_b2048.pt > $O/ddp2_b2048.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/rollout_net_train.py train $COMMON $FAST --batch 4096 --out $O/ddp4_b4096.pt > $O/ddp4_b4096.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P ~/cuda_work/value_bench.py 256,1024 none:0,fp16:0,fp16:trunk,fp16:all,fp16:cg > $O/value_bench.log 2>&1
echo AB3 DONE
