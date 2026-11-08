"""trace2.py <descriptor> <net> seeds... : where/when does the net's hunters kill descriptor's units; hunter counts."""
import sys, numpy as np, collections
sys.path.insert(0, "examples")
import v27_eval as V
from blackout_env.sim import core, batch as _batch
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
desc, netp = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
CAP = 45
Z = collections.Counter(); T = collections.Counter(); HC = collections.defaultdict(list)
def zone(p, sd):
    x, y = p  # world
    # own base corner: team A (1,22) ; B (22,1). normalise to descriptor's perspective
    if sd == 1: x, y = y, x   # mirror so descriptor base is at (1,22)
    if x < 5 and y > 18: return "ownbase"
    if 10 <= x <= 14 and 10 <= y <= 14: return "sanct"
    if 18 <= x and y <= 5: return "enemybase"
    if np.hypot(x - 12, y - 12) < 5: return "centre"
    # approach: own half vs enemy half by x+? (own base at top-left? y big) 
    return "ownhalf" if (y - x) > 0 else "enemyhalf"
for sd in (0, 1):
    n = len(seeds)
    ours, opp = V._policy(desc), V._policy("net:" + netp)
    ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])
    F32 = np.float32
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
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
        pre = bs.uf.copy(); precls = bs.ui[:, :, core.U_CLS].copy()
        bs.step(moves, live)
        tick = (t + 1) * _batch.DEFAULT_TPD
        sec = tick / 50
        for b in live:
            if tick % 50 == 0:
                for team, key in ((sd, "D"), (1 - sd, "R")):
                    c = bs.ui[b, 5*team:5*team+5, core.U_CLS]
                    HC[(key, int(sec))].append(int((c == 1).sum()))
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] == core.EV_KILL and e[2] >= 0:
                    killer, victim = int(e[2]), int(e[3])
                    if (killer < 5) == (sd == 0): continue  # only R kills D
                    if (victim < 5) != (sd == 0): continue
                    p = tuple(pre[b, victim][:2])
                    Z[(zone(p, sd), "CHa"[precls[b, victim]])] += 1
                    T[(int(sec // 5) * 5, "CHa"[precls[b, victim]])] += 1
print("zones of D deaths by R:", sorted(Z.items(), key=lambda x: -x[1]))
print("time:", sorted(T.items()))
for k in ("D", "R"):
    print(k, "mean hunters by sec", [round(float(np.mean(HC[(k, s)])), 2) for s in range(1, 45, 3)])
