"""Bit-exactness hash for heuristic speed work (balance-patch heuristics V32.1 .. V34b).

Plays full games (both sides, several descriptor pairings, fixed seeds) on the batched Python
simulator and hashes every move array plus the final game state.  Run it on the unmodified code
(``--write``), then after each optimisation (``--check``): any mismatch means an action changed.

    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. python examples/heur_perf_hash.py --write base.json
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. python examples/heur_perf_hash.py --check base.json

``examples/heur_perf_hash_base.json`` holds the hashes of the code before the speed-up commits
(8 games per pairing, 3000 decisions cap).  Uses 4 processes.
"""
import sys, hashlib, json, time, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
from multiprocessing import Pool
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, HeuristicPolicy
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState

PAIRS = [('V34b','V34b'),('V34a','V34a'),('V33bp','V33bp'),('V32.1','V32.1'),
         ('V34b,camp_path=true','V34b,camp_path=true'),('V34b,storm_units=2','V34b,storm_units=2'),
         ('V34b,carrier_retreat=4','V34b,carrier_retreat=4'),('V34b','V33bp'),('V33bp','V34b,camp_path=true')]

def play(job):
    da, db, n, T, seed0 = job
    a=HeuristicPolicy(da); b=HeuristicPolicy(db); a.reset(n); b.reset(n)
    bs=BatchSim([SimState.from_seed(seed0+i) for i in range(n)])
    F=np.float32; bufs=(np.zeros((n,2,N_CH,H,W),F),np.zeros((n,2,4),F),np.zeros((n,2,10,N_AGENT_COLS),F)); mv=np.zeros((n,10,2),F)
    hs=[hashlib.sha1() for _ in range(n)]
    t=0
    for t in range(T):
        live=np.nonzero(~bs.over)[0]
        if not len(live): break
        bs.render(live,*bufs); sub=tuple(x[:len(live)] for x in bufs)
        mv[live,0:5]=a.moves(bs,live,(0,),sub)[0]; mv[live,5:10]=b.moves(bs,live,(1,),sub)[1]
        for j in live: hs[j].update(np.ascontiguousarray(mv[j]).tobytes())
        bs.step(mv,live)
    from blackout_env.sim import core
    for j in range(n): hs[j].update(np.ascontiguousarray(bs.g[j]).tobytes())
    return (da, db, [h.hexdigest()[:12] for h in hs], t, int(bs.over.sum()))

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write'); g.add_argument('--check')
    ap.add_argument('--games', type=int, default=8); ap.add_argument('--cap', type=int, default=3000)
    a = ap.parse_args()
    out = a.write or '/dev/null'; n, T = a.games, a.cap
    jobs=[(a,b,n,T,12000000+100*i) for i,(a,b) in enumerate(PAIRS)]
    t0=time.time()
    with Pool(4) as p: res=p.map(play, jobs, chunksize=1)
    d={f'{a}|{b}':(h,t,o) for a,b,h,t,o in res}
    print('done %.0fs' % (time.time()-t0))
    for k,v in d.items(): print(k,v[1],v[2],v[0][0])
    if a.write: json.dump(d,open(a.write,'w'),indent=1)
    else:
        ref = json.load(open(a.check)); d = json.loads(json.dumps(d)); bad = [k for k in d if ref.get(k) != d[k]]
        print('MISMATCH: %s' % bad if bad else 'ALL %d PAIRINGS IDENTICAL (%d games each)' % (len(d), n))
        sys.exit(1 if bad else 0)
