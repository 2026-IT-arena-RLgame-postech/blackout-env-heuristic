"""Replay official competition replays (.npz: seed + every step's actions) in our simulator at 50 Hz and compare the
final scores with the official build's (meta["final"]).

    PYTHONPATH=. .venv/bin/python scripts/comp/replay_in_sim.py <replay.npz> ...
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from blackout_env.sim.batch import BatchSim  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402

ok = 0
for path in sys.argv[1:]:
    d = np.load(path)
    actions, present, seed = d["actions"], d["present"], int(d["seed"])
    meta = json.loads(str(d["meta"]))
    bs = BatchSim([SimState.from_seed(seed)])
    bs.ticks_per_decision = 1
    t = 0
    for t in range(len(actions)):
        if bs.over[0]:
            break
        mv = np.where(present[t][:, None], actions[t], 0.0).astype(np.float32)[None]
        bs.step(np.clip(mv, -1, 1))
    sim = bs.scores[0].tolist()
    off = meta.get("final", {})
    off_sc = [round(off.get("score_0", 0) * 100), round(off.get("score_1", 0) * 100)]
    same = sim == off_sc
    ok += same
    print(f"{path.split('/')[-1]}: steps {len(actions)} sim-steps {t + 1} sim {sim} official {off_sc} {'OK' if same else 'DIFF'}")
print(f"{ok}/{len(sys.argv) - 1} identical final scores")
