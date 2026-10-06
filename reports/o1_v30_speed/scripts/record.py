"""Record a corpus of sim games with V30 on one or both sides: raw_state/raw_map per decision."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
OUT = sys.argv[1]
V30 = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
OPP = {
    "p20": "ckpt:/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p20_v30.pt",
    "p17": "ckpt:/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p17_v28.pt",
    "v28": "strategic_v28:reserve_stage=0.5,trade_hold=advance",
    "v29": "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve",
    "v27": "strategic_v27:reserve_stage=1.5",
    "v25": "strategic_v25",
    "v30": V30,
}
PLAN = {"p20": 8, "v28": 8, "v29": 4, "v27": 4, "v25": 4, "p17": 4, "v30": 4}  # seeds per opponent, x2 sides

def names(t): return [f"unit_{i}" for i in range(5 * t, 5 * t + 5)]

def play(task):
    import torch; torch.set_num_threads(1)
    from blackout_env.sim import SimState
    from blackout_env.train.distill.opponents import make_opponent
    opp, seed, side = task
    sim = SimState.from_seed(seed)
    descs = {side: V30, 1 - side: OPP[opp]}
    pols = {t: make_opponent(descs[t], seed + 7919 * t) for t in (0, 1)}
    for p in pols.values():
        if hasattr(p, "reset"): p.reset()
    rs, rm, mv = [], [], []
    t0 = time.time()
    n = 0
    while not sim.over and n < 420 * 25:
        rs.append(sim.raw_state()); rm.append(sim.raw_map())
        obs = sim.obs()
        moves = np.zeros((10, 2), np.float32)
        for t, p in pols.items():
            for k, a in p.act({k: obs[k] for k in names(t)}).items():
                moves[int(k.split("_")[1])] = np.clip(np.asarray(a, np.float32), -1, 1)
        mv.append(moves); sim.step(moves); n += 1
    s = sim.scores
    meta = dict(opp=opp, seed=seed, v30_side=side, descs=[descs[0], descs[1]], scores=list(s), ticks=n,
                seconds=round(time.time() - t0, 2))
    np.savez_compressed(os.path.join(OUT, f"{opp}_s{seed}_side{side}.npz"), raw_state=np.array(rs),
                        raw_map=np.array(rm), moves=np.array(mv), meta=json.dumps(meta))
    return meta

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tasks = [(o, 9_100_000 + 37 * i + 1000 * j, side) for j, (o, k) in enumerate(PLAN.items()) for i in range(k) for side in (0, 1)]
    tasks = [t for t in tasks if not os.path.exists(os.path.join(OUT, f"{t[0]}_s{t[1]}_side{t[2]}.npz"))]
    with ProcessPoolExecutor(6) as ex:
        for m in ex.map(play, tasks):
            print(json.dumps(m), flush=True)
