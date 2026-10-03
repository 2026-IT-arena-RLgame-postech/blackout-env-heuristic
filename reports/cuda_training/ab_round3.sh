#!/bin/bash
# Round 3 after the submodule-compile fix + in-memory default: value fast / DDP, distill fast, r5 fast.
P=~/blackout-env/.venv/bin/python
TR=~/blackout-env/.venv/bin/torchrun
cd ~/blackout-env-cuda
export PYTHONPATH=$PWD
O=~/cuda_work/round3
mkdir -p $O
uptime > $O/uptime_start.txt
$P -m pytest -q -p no:cacheprovider tests/test_accel.py > $O/pytest_accel.log 2>&1
C=~/blackout-env/datasets/value_s1_sub4
I=~/blackout-env/checkpoints/distill/frozen/p5a_v22.pt
VC="--cache $C --init $I --steps 3000 --lr 3e-4 --warmup 500 --eval-every 1000 --eval-rows 20000"
CUDA_VISIBLE_DEVICES=0 $P examples/value_train.py train $VC --amp fp16 --compile --out-dir $O/value_fast > $O/value_fast.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node 2 examples/value_train.py train $VC --amp fp16 --compile --out-dir $O/value_ddp2 > $O/value_ddp2.log 2>&1
S=~/blackout-env/checkpoints/distill/p11/shards
REC="--policy-head logits --action-bins 32 --act-arc-bins 1 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --opening-seconds 10 --lr 1e-4 --device cuda"
CUDA_VISIBLE_DEVICES=0 $P examples/distill_offline_ab.py $S --rows 600000 --steps 3000 --init ~/blackout-env/checkpoints/distill/frozen/p10_v22.pt --out $O/distill_fast.json -- $REC --amp fp16 --compile > $O/distill_fast.log 2>&1
R=~/blackout-env/checkpoints/rollout_net
CUDA_VISIBLE_DEVICES=0 $P examples/rollout_net_train.py train --data $R/data --extra $R/simdata/train_ex23.npz --extra-frac 0.5 --init $R/r4.pt --p11-kl 0.5 --lr 1e-3 --steps 2000 --batch 1024 --log-every 250 --eval-every 2000 --amp fp16 --compile --out $O/r5_fast.pt > $O/r5_fast.log 2>&1
uptime > $O/uptime_end.txt
find $O -name "*.pt" -delete
echo ROUND3 DONE
