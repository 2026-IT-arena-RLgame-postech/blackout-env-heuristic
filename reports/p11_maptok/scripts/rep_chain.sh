#!/bin/bash
# §25 replication: the two fine-tune arms again with --seed 1 (other batch order, other map-path init), sequentially.
cd ~/blackout-env-p11m
I=checkpoints/distill/frozen/p11_v22.pt
SEED=1 ./ab_arm.sh ft_f3_s1 2 20000 900000 $I -- --lr 1e-4 --map-xattn f3
SEED=1 ./ab_arm.sh ft_base_s1 2 20000 900000 $I -- --lr 1e-4
echo REP DONE > runs/rep_done
