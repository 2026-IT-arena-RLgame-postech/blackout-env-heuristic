import hashlib, sys, numpy as np
from blackout_env.sim.state import SimState, obs_from_raw
from blackout_env.train.distill.opponents import make_opponent
V30 = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28 = "strategic_v28:reserve_stage=0.5,trade_hold=advance"
def match_hash(a, b, seed, ticks):
    sim = SimState.from_seed(seed); pols = [make_opponent(a, 0), make_opponent(b, 1)]
    h = hashlib.sha256()
    for t in range(ticks):
        if sim.over: break
        obs = obs_from_raw(sim.raw_state(), sim.raw_map()); mv = np.zeros((10, 2), np.float32)
        for team, pol in enumerate(pols):
            for name, act in pol.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * team, 5 * team + 5)}).items():
                mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
        h.update(mv.tobytes()); sim.step(mv)
    return h.hexdigest()[:16], sim.scores
for a, b, seed in ((V30, V28, 4_000_003), (V28, V30, 4_000_005), ("strategic_v25", "strategic_v27:reserve_stage=1.5", 4_000_007), ("strategic_v21", "strategic_v19", 4_000_011)):
    print(a.split(":")[0], b.split(":")[0], seed, match_hash(a, b, seed, 700))
