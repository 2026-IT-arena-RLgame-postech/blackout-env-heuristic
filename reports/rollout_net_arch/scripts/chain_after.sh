#!/bin/bash
# start an eval chain only once no other eval_variant.sh of mine runs (CPU headroom)
cd ~/blackout-env-arch
while ps -eo args | grep -q "^bash eval_variant[.]sh"; do sleep 60; done
exec bash chain_eval.sh "$@"
