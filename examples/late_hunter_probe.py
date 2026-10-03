"""P15 probes (docs/policy_v2_design_ko.md §30), simulator, seeds 4000000.. x both sides.

opening OURS OPP N: in the race-lost games, Hunters per team at decisions 101 / 105 / 125 / 250
and Collectors killed by decision 125.
after_trade OURS OPP N: after every Hunter-Hunter trade past decision 110 -- the Hunter counts
before it, which team has the next Hunter first, Collectors each side loses in the next 10 s.

    PYTHONPATH=. .venv/bin/python examples/late_hunter_probe.py opening 'strategic_v25:race_walk=false' strategic_v25 32
"""
import collections
import sys

import numpy as np

sys.path.insert(0, ".")
from examples.late_hunter_eval import _policy  # noqa: E402
from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.batch import BatchSim  # noqa: E402
from blackout_env.sim.mapdata import COLLECTOR, HUNTER  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402


def opening(ours_d, opp_d, nseed):
    seeds = [s for s in range(4_000_000, 4_000_000 + nseed) for _ in (0, 1)]
    sides = np.array([sd for _ in range(nseed) for sd in (0, 1)])
    n = len(seeds)
    ours, opp = _policy(ours_d), _policy(opp_d); ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])
    moves = np.zeros((n, 10, 2), np.float32)
    snap = {}; killed = np.zeros((n, 2)); first = np.full((n, 2), -1)
    for t in range(250):
        live = np.nonzero(~bs.over)[0]
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if sel.any():
                    moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), None)[team]
        bs.step(moves, live)
        cls = bs.ui[:, :, core.U_CLS]
        for b in live:
            sd = sides[b]
            h = [(cls[b, 5*sd:5*sd+5] == HUNTER).sum(), (cls[b, 5*(1-sd):5*(1-sd)+5] == HUNTER).sum()]
            for k in (0, 1):
                if first[b, k] < 0 and h[k]: first[b, k] = t + 1
            if t + 1 in (101, 105, 125, 250): snap.setdefault(t + 1, {})[b] = h
            for e in bs.ev[b, :int(bs.g[b, core.G_NEV])]:
                if e[1] == core.EV_KILL and e[2] >= 0 and e[4] == COLLECTOR and t + 1 <= 125:
                    killed[b, 0 if (e[3] < 5) == (sd == 0) else 1] += 1
    lost = [b for b in range(n) if first[b, 1] >= 0 and (first[b, 0] < 0 or first[b, 0] > first[b, 1])]
    print(ours_d, 'vs', opp_d, 'race lost', len(lost), '/', n)
    for t, d in sorted(snap.items()):
        c = collections.Counter(tuple(int(x) for x in d[b]) for b in lost if b in d)
        print(f'  decision {t}: (ours, theirs) Hunters', sorted(c.items(), key=lambda x: -x[1])[:6])
    print('  Collectors killed by decision 125 (ours, theirs):', killed[lost].mean(0))

def after_trade(ours_d, opp_d, nseed):
    seeds = [s for s in range(4_000_000, 4_000_000 + nseed) for _ in (0, 1)]
    sides = np.array([sd for _ in range(nseed) for sd in (0, 1)])
    n = len(seeds)
    ours, opp = _policy(ours_d), _policy(opp_d); ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])

    moves = np.zeros((n, 10, 2), np.float32)
    pending = []  # (game, tick, ours_h_before, theirs_h_before)
    res = collections.Counter(); lossC = []
    hist = {}
    for t in range(150 * 25):
        live = np.nonzero(~bs.over)[0]
        if not len(live): break
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if sel.any():
                    moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), None)[team]
        before = bs.ui[:, :, core.U_CLS].copy()
        bs.step(moves, live)
        for b in live:
            sd = sides[b]
            evs = bs.ev[b, :int(bs.g[b, core.G_NEV])]
            hh = [e for e in evs if e[1] == core.EV_KILL and e[2] < 0 and e[4] == HUNTER]
            if hh and t + 1 > 110:
                cb = before[b]
                pending.append([b, t + 1, int((cb[5*sd:5*sd+5] == HUNTER).sum()), int((cb[5*(1-sd):5*(1-sd)+5] == HUNTER).sum()), None, [0, 0]])
            cls = bs.ui[b, :, core.U_CLS]
            for p in pending:
                if p[0] != b or t + 1 <= p[1] or t + 1 > p[1] + 250: continue
                for e in evs:
                    if e[1] == core.EV_KILL and e[2] >= 0 and e[4] == COLLECTOR:
                        p[5][0 if (e[3] < 5) == (sd == 0) else 1] += 1
                if p[4] is None:
                    for e in evs:
                        if e[1] == core.EV_TRANSFORM and e[3] == HUNTER:
                            p[4] = 'us' if (e[2] < 5) == (sd == 0) else 'them'; break
    for p in pending:
        res[(p[2], p[3], p[4])] += 1
    print(ours_d, 'vs', opp_d, 'trades after 110:', len(pending), 'per game', len(pending) / n)
    print(' (our H, their H before) -> next Hunter first:', sorted(res.items(), key=lambda x: -x[1])[:10])
    nx = collections.Counter(p[4] for p in pending); print(' next first', dict(nx))
    print(' Collectors lost in 10 s after (ours, theirs):', np.mean([p[5] for p in pending], axis=0) if pending else None)

if __name__ == "__main__":
    {"opening": opening, "after_trade": after_trade}[sys.argv[1]](sys.argv[2], sys.argv[3], int(sys.argv[4]))
