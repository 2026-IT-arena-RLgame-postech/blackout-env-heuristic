#!/bin/bash
# V32.1 (docs §58.7) = C = B + adapt_k=3, on gpu2 (EW workers): the full pool at 64 seeds, the Hunter-poor pool
# opponents to 256 seeds, OOD-26 at 32 seeds, and the OOD margin-regression set at 128 seeds.
cd ${R:-$HOME/blackout-env-v32}
G=reports/v32/scripts/gate.sh
OURS=C OPP="V31;V30;V29;V28;V27;V25;V23;p14b;p17;p20;p21e;p22;H4;H4HUNT" SEEDS=64 bash $G c64
OURS=C OPP="V23;p14b;V25" SEEDS=192 SEED0=5200064 bash $G c256
OURS=C OPP=ood26 SEEDS=32 SEED0=2600000 bash $G c_ood
OURS=C OPP="ood26=v30_h2,v4,v21,v31_eps10,duelist,v30_nostage,v26" SEEDS=128 SEED0=2600000 bash $G c_ood128
OURS=V31 OPP="ood26=v30_h2,v4,v21,v31_eps10,duelist,v30_nostage,v26" SEEDS=128 SEED0=2600000 bash $G c_ood128
echo V321 DONE >> reports/v32/c_ood128/stdout.log
