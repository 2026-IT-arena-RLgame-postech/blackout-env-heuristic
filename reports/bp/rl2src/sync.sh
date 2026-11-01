#!/bin/zsh
# push HEAD to gpu1 and check it out in ~/blackout-env-bp (bp_loop.sync_code does not reach that repo)
cd /Users/mac/project/26rl/blackout-env-teach
git push -qf poscat_gpu1 feat/bp-teacher:bp-incoming && ssh poscat_gpu1 'cd ~/blackout-env-bp && git fetch -q ~/blackout-env bp-incoming && git checkout -q -f -B feat/balance-patch FETCH_HEAD && git log --oneline | head -1 && find blackout_env -name "*.nb[ic]" -delete'
