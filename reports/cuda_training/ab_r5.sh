#!/bin/bash
# r5-style A/B (P11 shards + simulator DAgger extra 50% + P11 KL 0.5, from r4, lr 1e-3), 2000 steps:
# the code before the CUDA work (eager fp32, everything in RAM) vs the CUDA path (fp16 + compile, mmap).
P=~/blackout-env/.venv/bin/python
O=~/cuda_work/abr5
mkdir -p $O
R=~/blackout-env/checkpoints/rollout_net
COMMON="--data $R/data --extra $R/simdata/train_ex23.npz --extra-frac 0.5 --init $R/r4.pt --p11-kl 0.5 --lr 1e-3 --steps 2000 --batch 1024 --log-every 250 --eval-every 2000"
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 $P ~/cuda_work/ref_rollout_net_train.py train $COMMON --device cuda --out $O/ref.pt > $O/ref.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train $COMMON --amp fp16 --compile --out $O/fast.pt > $O/fast.log 2>&1
rm -f $O/*.pt
echo ABR5 DONE
