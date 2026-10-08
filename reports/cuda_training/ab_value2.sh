#!/bin/bash
# Value A/B after the dynamic=False fix: fp16 + compile on 1 GPU, and DDP correctness (2 / 4 ranks).
P=~/blackout-env/.venv/bin/python
TR=~/blackout-env/.venv/bin/torchrun
O=~/cuda_work/abv2
mkdir -p $O
C=~/blackout-env/datasets/value_s1_sub4
I=~/blackout-env/checkpoints/distill/frozen/p5a_v22.pt
COMMON="--cache $C --init $I --steps 3000 --lr 3e-4 --warmup 500 --eval-every 1000 --eval-rows 20000"
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
uptime > $O/uptime_start.txt
$P -m pytest -q -p no:cacheprovider tests/test_accel.py > $O/pytest_accel.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/value_train.py train $COMMON --amp fp16 --compile --out-dir $O/fast > $O/fast.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/value_train.py train $COMMON --amp fp16 --compile --out-dir $O/ddp2 > $O/ddp2.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node 4 examples/value_train.py train $COMMON --amp fp16 --compile --batch-size 1024 --out-dir $O/ddp4_b1024 > $O/ddp4_b1024.log 2>&1
uptime > $O/uptime_end.txt
find $O -name "*.pt" -delete
echo ABV2 DONE
