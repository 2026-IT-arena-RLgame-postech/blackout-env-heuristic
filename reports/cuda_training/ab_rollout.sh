#!/bin/bash
# A/B of the RolloutNet trainer on CUDA: S3-style retrain (init r5, s3data_x1, lr 3e-4), 3000 steps.
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
P=~/blackout-env/.venv/bin/python
D=checkpoints/rollout_net/s3data_x1
O=~/cuda_work/ab
mkdir -p $O
COMMON="--data $D --init checkpoints/rollout_net/r5.pt --steps 3000 --lr 3e-4 --log-every 250 --eval-every 1000"
CUDA_VISIBLE_DEVICES=0 nohup $P ~/cuda_work/ref_rollout_net_train.py train $COMMON --device cuda --out $O/ref.pt > $O/ref.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $P examples/rollout_net_train.py train $COMMON --channels-last 0 --out $O/eager32.pt > $O/eager32.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $P examples/rollout_net_train.py train $COMMON --amp fp16 --compile --out $O/fast.pt > $O/fast.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $P examples/rollout_net_train.py train $COMMON --amp fp16 --compile --compile-mode reduce-overhead --out $O/fast_cg.pt > $O/fast_cg.log 2>&1 &
wait
echo AB1 DONE
