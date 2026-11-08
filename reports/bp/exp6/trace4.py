import sys, numpy as np, collections
sys.path.insert(0, "examples")
import v27_eval as V
from blackout_env.sim import core, batch as _batch
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
desc, netp = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
n = len(seeds)
IT = ["bat","BuffSpd","DebSpd","BuffSz","DebSz"]
for sd in (0, 1):
    ours, opp = V._policy(desc), V._policy("net:" + netp)
    ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])
    F32 = np.float32
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    log = [[] for _ in range(n)]
    for t in range(40 * 50 // _batch.DEFAULT_TPD):
        live = np.nonzero(~bs.over)[0]
        if not len(live): break
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        for team in (0, 1):
            mine = np.full(len(live), sd == team)
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if not sel.any(): continue
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                moves[live[sel], 5*team:5*team+5] = pol.moves(bs, live[sel], (team,), s_in)[team]
        bs.step(moves, live)
        for b in live:
            ne = int(bs.g[b, core.G_NEV]); sec = (t+1)*_batch.DEFAULT_TPD/50
            for e in bs.ev[b, :ne]:
                if e[1] in (core.EV_PICKUP, core.EV_DEPOSIT, core.EV_STEAL) and e[3] != 0:
                    u = int(e[2]); who = "D" if (u < 5) == (sd == 0) else "R"
                    log[b].append(f"{sec:.1f}{['pick','dep','x','x','x','x','x','x','STEAL'][e[1]]}:{who}{u%5}:{IT[e[3]]}")
    for j, s in enumerate(seeds):
        print("side", sd, s%1000, " ".join(log[j]))
