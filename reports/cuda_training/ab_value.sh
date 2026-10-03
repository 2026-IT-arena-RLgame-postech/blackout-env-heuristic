#!/bin/bash
# A/B of the S1 value trainer on CUDA (S1-short-like: p5a init, lr 3e-4, warmup 500), 3000 steps, on the
# 1/4 match subset of datasets/value_s1 (datasets/value_s1_sub4).
P=~/blackout-env/.venv/bin/python
TR=~/blackout-env/.venv/bin/torchrun
O=~/cuda_work/abv
mkdir -p $O
C=~/blackout-env/datasets/value_s1_sub4
I=~/blackout-env/checkpoints/distill/frozen/p5a_v22.pt
COMMON="--cache $C --init $I --steps 3000 --lr 3e-4 --warmup 500 --eval-every 1000 --eval-rows 20000"
# reference: the code before the CUDA work (feat/policy-v2 b87bc52), eager fp32
(cd ~/cuda_work/ref_repo && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD $P examples/value_train.py train $COMMON --device cuda --out-dir $O/ref > $O/ref.log 2>&1)
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 $P examples/value_train.py train $COMMON --out-dir $O/eager32 > $O/eager32.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/value_train.py train $COMMON --amp fp16 --compile --out-dir $O/fast > $O/fast.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/value_train.py train $COMMON --amp fp16 --compile --out-dir $O/ddp4 > $O/ddp4.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/value_train.py train $COMMON --amp fp16 --compile --batch-size 512 --out-dir $O/ddp2_b512 > $O/ddp2_b512.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/value_train.py train $COMMON --amp fp16 --compile --batch-size 1024 --out-dir $O/ddp4_b1024 > $O/ddp4_b1024.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/value_train.py train $COMMON --amp fp16 --compile --batch-size 1024 --out-dir $O/fast_b1024 > $O/fast_b1024.log 2>&1
find $O -name "*.pt" -delete
echo ABV DONE
