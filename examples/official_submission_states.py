"""Fork broadcasts (raw_state / raw_map) from short sim matches, P24c vs V30, for build_official_submission --states.

    PYTHONPATH=. .venv/bin/python examples/official_submission_states.py 4 1500 states.npz
"""
import sys, numpy as np, torch
from blackout_env.sim.state import SimState
from blackout_env.train.distill.inference import load_checkpoint_net, make_student
from blackout_env.train.distill.opponents import make_opponent
net = load_checkpoint_net("checkpoints/distill/frozen/p24c_v321.pt")
stu = make_student(net, "iqn")
raws, maps = [], []
for seed in range(int(sys.argv[1])):
    sim = SimState.from_seed(1000 + seed)
    opp = make_opponent("strategic_v30", seed)
    for t in range(int(sys.argv[2])):
        if sim.over: break
        obs = sim.obs()
        raws.append(sim.raw_state()); maps.append(sim.raw_map())
        mv = np.zeros((10, 2), np.float32)
        a = stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5)}) if seed % 2 == 0 else opp.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5)})
        b = opp.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5, 10)}) if seed % 2 == 0 else stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5, 10)})
        for n, v in {**a, **b}.items(): mv[int(n[5:])] = v
        sim.step(mv); sim.step(mv)
np.savez_compressed(sys.argv[3], raw=np.stack(raws), map=np.stack(maps))
print(len(raws))
