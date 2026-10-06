"""Replay recorded corpus observations through V30 (code tree given by PYTHONPATH); per-tick hashes of
actions + V30's inspection state; per-tick act() time.  Usage: replay.py CORPUS OUT.json [workers] [only-substr]"""
import glob, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
V30 = os.environ.get("POLICY", "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true")
STATE_ATTRS = ("staged", "joining", "trade_held", "sentries", "committed", "last_step", "waiting", "campers",
               "_race_now", "sizes", "anyangle_stats")

def names(t): return [f"unit_{i}" for i in range(5 * t, 5 * t + 5)]

def state_repr(p):
    out = []
    for a in STATE_ATTRS:
        v = getattr(p, a, None)
        if isinstance(v, dict): v = sorted(v.items())
        out.append(repr(v))
    out.append(repr(sorted(getattr(p, "_staged_xy", set()))))
    out.append(repr(sorted(getattr(p, "_held_xy", {}).items())))
    return "|".join(out).encode()

def run(path):
    from blackout_env.sim.state import obs_from_raw
    from blackout_env.train.distill.opponents import make_opponent
    d = np.load(path); meta = json.loads(str(d["meta"]))
    rs, rm = d["raw_state"], d["raw_map"]
    sides = [t for t in (0, 1) if meta["descs"][t] == V30] if not os.environ.get("ALLSIDES") else [0, 1]
    res = {}
    for side in sides:
        p = make_opponent(V30, meta["seed"] + 7919 * side); p.reset()
        hashes, times = [], []
        for t in range(len(rs)):
            obs = obs_from_raw(rs[t], rm[t], names(side))
            t0 = time.perf_counter(); acts = p.act(obs); times.append(time.perf_counter() - t0)
            h = hashlib.sha256()
            for k in sorted(acts):
                h.update(k.encode()); h.update(np.asarray(acts[k], np.float32).tobytes())
            h.update(state_repr(p))
            hashes.append(h.hexdigest()[:16])
        res[f"{os.path.basename(path)}#{side}"] = dict(hashes=hashes, act_s=sum(times), ticks=len(times),
                                                       first_tick_s=times[0])
    return res

if __name__ == "__main__":
    corpus, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    only = sys.argv[4] if len(sys.argv) > 4 else ""
    import re; files = sorted(f for f in glob.glob(os.path.join(corpus, "*.npz")) if re.search(only, os.path.basename(f)))
    allres = {}
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(run, files): allres.update(r)
    else:
        run(files[0])  # warm-up (numba cache load / JIT)
        for f in files: allres.update(run(f))
    json.dump(allres, open(out, "w"))
    tot = sum(v["act_s"] for v in allres.values()); n = sum(v["ticks"] for v in allres.values())
    print(f"{len(allres)} sequences, {n} ticks, {tot:.2f}s act, {tot / n * 1e6:.0f} us/tick")
