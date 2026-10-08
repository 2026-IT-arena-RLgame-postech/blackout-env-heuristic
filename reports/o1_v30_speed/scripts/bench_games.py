"""Closed-loop sim games, single process: per-game move hash, scores, per-policy act time, wall time.
usage: bench_games.py OUT.jsonl"""
import hashlib, json, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
from blackout_env.sim import SimState
from blackout_env.train.distill.opponents import make_opponent
D = {
    "v30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
    "v29": "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve",
    "v28": "strategic_v28:reserve_stage=0.5,trade_hold=advance",
    "v27": "strategic_v27:reserve_stage=1.5",
    "v25": "strategic_v25",
    "p20": "ckpt:/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p20_v30.pt",
    "p17": "ckpt:/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p17_v28.pt",
}
PLAN = [("v30", "p20", 8), ("v30", "v28", 8), ("v30", "v29", 4), ("v30", "v27", 4), ("v30", "v25", 4), ("v30", "p17", 4),
        ("v30", "v30", 4), ("v25", "v28", 4), ("v27", "v28", 4)]
def names(t): return [f"unit_{i}" for i in range(5 * t, 5 * t + 5)]
out = open(sys.argv[1], "w")
games = [(a, b, 9_200_000 + 101 * i + 10_000 * j, side) for j, (a, b, k) in enumerate(PLAN) for i in range(k) for side in (0, 1)]
# warm-up: JIT/numba cache load, checkpoint load
make_opponent(D["p20"], 0); make_opponent(D["p17"], 0)
for idx, g in enumerate([games[0]] + games):
    a, b, seed, side = g
    sim = SimState.from_seed(seed)
    descs = {side: D[a], 1 - side: D[b]}
    pols = {t: make_opponent(descs[t], seed + 7919 * t) for t in (0, 1)}
    for p in pols.values():
        if hasattr(p, "reset"): p.reset()
    act = [0.0, 0.0]; h = hashlib.sha256(); n = 0
    t0 = time.perf_counter()
    while not sim.over and n < 420 * 25:
        obs = sim.obs(); mv = np.zeros((10, 2), np.float32)
        for t in (0, 1):
            ta = time.perf_counter()
            acts = pols[t].act({k: obs[k] for k in names(t)})
            act[t] += time.perf_counter() - ta
            for k, v in acts.items(): mv[int(k.split("_")[1])] = np.clip(np.asarray(v, np.float32), -1, 1)
        h.update(mv.tobytes()); sim.step(mv); n += 1
    wall = time.perf_counter() - t0
    s = sim.scores
    row = dict(a=a, b=b, seed=seed, a_side=side, ticks=n, scores=list(s), margin=s[side] - s[1 - side], hash=h.hexdigest()[:16],
               a_act_s=round(act[side], 4), b_act_s=round(act[1 - side], 4), wall_s=round(wall, 4))
    if idx == 0:
        continue  # warm-up game
    out.write(json.dumps(row) + "\n"); out.flush()
