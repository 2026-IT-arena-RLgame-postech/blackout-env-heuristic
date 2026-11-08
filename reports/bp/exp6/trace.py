"""trace.py <descriptor> <net> seeds... : net plays descriptor, summarise events (per side)."""
import sys, numpy as np, collections
sys.path.insert(0, "examples")
import v27_eval as V
from blackout_env.sim import core, batch as _batch
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
desc, netp = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
CAP = 60
for sd in (0, 1):  # side of the descriptor
    n = len(seeds)
    ours, opp = V._policy(desc), V._policy("net:" + netp)
    ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])
    F32 = np.float32
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    ev_log = [[] for _ in range(n)]
    snaps = [dict() for _ in range(n)]
    for t in range(CAP * 50 // _batch.DEFAULT_TPD):
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
        tick = (t + 1) * _batch.DEFAULT_TPD
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                ev_log[b].append((int(e[0]), int(e[1]), int(e[2]), int(e[3]), int(e[4]), int(e[5])))
            if tick % 250 == 0:
                snaps[b][tick // 50] = (int(bs.scores[b, sd]), int(bs.scores[b, 1 - sd]))
    for j, s in enumerate(seeds):
        ev = ev_log[j]
        # unit idx<5 => team A(0). team of idx
        def team(u): return 0 if u < 5 else 1
        def me(u): return team(u) == sd
        kills = collections.Counter(); steals = collections.Counter(); tf = []
        for e in ev:
            k = e[1]
            if k == core.EV_KILL:
                killer, victim, vcls, kcls = e[2], e[3], e[4], e[5]
                if killer >= 0:
                    kills[("D" if me(killer) else "R", "HCa"[kcls], "HCa"[vcls] if False else "CHa"[vcls])] += 1
                else: kills[("env",)] += 1
            elif k == core.EV_STEAL:
                steals["D" if me(e[2]) else "R"] += e[4]
            elif k == core.EV_TRANSFORM:
                tf.append((e[0]/50, "D" if me(e[2]) else "R", "CHa"[e[3]]))
        print(f"side{sd} seed{s} score D/R {int(bs.scores[j, sd])}/{int(bs.scores[j, 1-sd])} ticks {int(bs.ticks[j])} snaps {snaps[j]}")
        print("   kills", dict(kills), "steals", dict(steals))
        print("   transforms", [(round(a,1),b,c) for a,b,c in tf][:14])
