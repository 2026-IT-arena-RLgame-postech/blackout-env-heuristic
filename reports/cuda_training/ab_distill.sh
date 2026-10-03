#!/bin/bash
# Distill learner on CUDA: offline A/B (P11 recipe from P10, 3000 steps on the P11 shards) + step bench.
P=~/blackout-env/.venv/bin/python
O=~/cuda_work/abd
mkdir -p $O
S=~/blackout-env/checkpoints/distill/p11/shards
I=~/blackout-env/checkpoints/distill/frozen/p10_v22.pt
AB="$S --rows 600000 --steps 3000 --init $I"
REC="--policy-head logits --action-bins 32 --act-arc-bins 1 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --opening-seconds 10 --lr 1e-4 --device cuda"
cd ~/blackout-env-cuda
# reference: the learner before the CUDA work (b87bc52), eager fp32; the A/B script itself is new
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/cuda_work/ref_repo $P examples/distill_offline_ab.py $AB --out $O/ref.json -- $REC > $O/ref.log 2>&1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 $P examples/distill_offline_ab.py $AB --out $O/eager32.json -- $REC > $O/eager32.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/distill_offline_ab.py $AB --out $O/fast.json -- $REC --amp fp16 --compile > $O/fast.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/distill_step_bench.py $S --rows 200000 --policy-heads logits --batch-sizes 256 512 1024 --amp none fp16 --compile --device cuda > $O/bench_compile.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P examples/distill_step_bench.py $S --rows 200000 --policy-heads logits --batch-sizes 256 1024 --amp none fp16 --device cuda > $O/bench_eager.log 2>&1
echo ABD DONE
