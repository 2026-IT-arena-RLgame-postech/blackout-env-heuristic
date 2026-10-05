"""Q6: why does the search beat the heuristics / students after the opening race?
(docs/policy_v2_design_ko.md §33 "탐색이 후반에 강한 이유".)  Simulator only.

``play``: for each (seed, side) one tag plays the full match (cap 150 s) against one opponent, and
every decision is recorded: both teams' positions, classes, held item (type, amount), scores, our
team's executed moves (so a match can be replayed bit-exactly), the executing branch, sim events;
every search root: tick, all branch leaf values, the choice.  Tags: ``plain`` (the acting net,
no search), ``search`` (search-new: B8 ``alt*,samp:1.0x2``, every 25, window 25, margin 3, V =
s2v_f3xd2 iter2), ``o4fmean`` (+ opponent = 3 r5-style samples + base for the whole rollout,
mean).  One npz per (tag, opponent, chunk) under ``<out>/raw``.

``cf``: counterfactual value of the search's deviations.  Roots where the search chose a
non-base branch are sampled; each match is replayed (recorded own moves, the opponent re-acts --
a heuristic is deterministic, checked against the recording) to the root, the state and the
opponent's instance are copied, and the root is continued twice -- base (the net for the rest)
and chosen (the chosen branch for the window, then the net) -- under K reseeds (k = 0: the
match's own RNG), common to both arms, against ``--cont-opp`` (``true`` = the recorded
opponent, ``model`` = the acting net itself, i.e. the search's own opponent model) for
``--horizon`` decisions.  Every arm records the lead at +1/+6/+15 s / horizon, V_sym at the
horizon, and per-unit event counts in the first 6 s (deaths, kills, pickups, steals, deposits,
transforms, item losses).  -> ``<out>/cf_<tag>_<opponent>_<cont>.npz``.

``report``: match-level paired search - plain by race outcome; deviation categories with the
predicted (search) gain and the realised gain vs the model and the true opponent.

    PYTHONPATH=. .venv/bin/python examples/late_search_why.py play --tag plain --opponent strategic_v25 --seeds 128
    PYTHONPATH=. .venv/bin/python examples/late_search_why.py cf --tag search --opponent strategic_v25 --n-roots 400
    PYTHONPATH=. .venv/bin/python examples/late_search_why.py report
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NET = "checkpoints/rollout_net/arch/f3_x_d2.pt"
VALUE = "checkpoints/value/s2v_f3xd2/iter2.pt"
NETS = {"p14b": "checkpoints/distill/frozen/p14_v25.pt", "p11": "checkpoints/distill/frozen/p11_v22.pt"}
SANCT = (11.0, 13.0)
F32 = np.float32
TAGS = {
    "plain": None,
    "search": {},
    "o4fmean": {"opp_branches": "samp:1.0x3", "opp_agg": "mean", "opp_left": 150},
}


def make_net(device):
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    return NetPolicy(RolloutNet.from_checkpoint(str(REPO / NET)), device=device)


def make_searcher(tag, net, device, branches="alt*,samp:1.0x2", margin=3.0):
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches, searcher_kwargs

    deep = TAGS[tag]
    if deep is None:
        return None
    return Searcher(ModNetPolicy(net), ValueNetLeaf(str(REPO / VALUE), device), parse_branches(branches),
                    window=25, margin=margin, **searcher_kwargs(deep))


def make_opp(name, our, device):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name == "model":
        return our
    if name in NETS:
        return NetPolicy.from_checkpoint(str(REPO / NETS[name]), device=device)
    return HeuristicPolicy(name)


def held_info(bs):
    """[n, 10] held item type (-1 none) and amount."""
    from blackout_env.sim import core

    h = bs.ui[:, :, core.U_HELD]
    n = h.shape[0]
    typ = np.full(h.shape, -1, np.int8)
    amt = np.zeros(h.shape, np.int8)
    b, u = np.nonzero(h >= 0)
    typ[b, u] = bs.it[b, h[b, u], core.I_TYPE]
    amt[b, u] = bs.it[b, h[b, u], core.I_AMT]
    assert typ.shape == (n, 10)
    return typ, amt


def step_both(bs, live, sides, our, opp, bufs, moves, own_moves=None):
    """One decision for games ``live``: our team = ``our`` (or the given ``own_moves`` [len(live), 5, 2]),
    the other = ``opp``; returns the per-game own moves."""
    need = own_moves is None or opp is our or getattr(opp, "needs_inputs", True)
    if need:
        bs.render(live, *bufs)
    sub = tuple(x[: len(live)] for x in bufs)
    ours = sides[live]
    if own_moves is None or opp is our:
        mo = our.moves(bs, live, (0, 1), sub)
    else:
        mo = None
    if opp is our:
        oo = mo
    else:
        oo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
        for team in (0, 1):
            sel = ours != team
            if sel.any():
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                oo[team][sel] = opp.moves(bs, live[sel], (team,), s_in)[team]
    own = np.zeros((len(live), 5, 2), F32)
    for team in (0, 1):
        mine = ours == team
        if own_moves is not None:
            own[mine] = own_moves[mine]
        else:
            own[mine] = mo[team][mine]
        moves[live, 5 * team:5 * team + 5] = np.where(mine[:, None, None], own, oo[team])
    bs.step(moves, live)
    return own


# ---------------------------------------------------------------------------------- play

def run_chunk(seeds, sides, net, opp_name, device, searcher, cap, every=25, window=25, log=None):
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.search import ModNetPolicy
    from blackout_env.sim.state import SimState

    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    our = ModNetPolicy(net)
    opp = make_opp(opp_name, our, device)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    our.configure(n)
    if opp is not our:
        opp.reset(n)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    pos = np.zeros((cap + 1, n, 10, 2), F32)
    cls = np.zeros((cap + 1, n, 10), np.int8)
    htyp = np.full((cap + 1, n, 10), -1, np.int8)
    hamt = np.zeros((cap + 1, n, 10), np.int8)
    score = np.zeros((cap + 1, n, 2), np.int16)
    ownmv = np.zeros((cap, n, 5, 2), F32)
    allmv = np.zeros((cap, n, 10, 2), F32)
    branch = np.zeros((cap, n), np.int16)
    nb = len(searcher.branches) if searcher else 1
    roots = {"game": [], "tick": [], "values": [], "choice": []}
    events = []
    cur = np.zeros(n, np.int64)
    end = np.full(n, cap, np.int64)

    def snap(t):
        pos[t], cls[t] = bs.uf, bs.ui[:, :, core.U_CLS]
        htyp[t], hamt[t] = held_info(bs)
        score[t] = bs.scores

    snap(0)
    t_s = 0.0
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if searcher is not None and t % every == 0:
            c0 = time.time()
            rts = [bs.state(int(b)) for b in live]
            reseeds = [(int(seeds[b]) * 7919 + t * 104729 + 12345) % (2 ** 31) for b in live]
            out = searcher.search(rts, sides[live], reseeds)
            for j, b in enumerate(live):
                k = int(out.choice[j])
                our.set_game(b, sides[b], searcher.branches[k], window)
                cur[b] = k
            roots["game"].append(live.copy())
            roots["tick"].append(np.full(len(live), t, np.int32))
            roots["values"].append(np.asarray(out.values, np.float32))
            roots["choice"].append(np.asarray(out.choice, np.int16))
            t_s += time.time() - c0
        branch[t, live] = np.where(our.left[live] > 0, cur[live], 0)
        own = step_both(bs, live, sides, our, opp, bufs, moves)
        ownmv[t, live] = own
        allmv[t, live] = moves[live]
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] in (core.EV_KILL, core.EV_TRANSFORM, core.EV_PICKUP, core.EV_STEAL, core.EV_DEPOSIT,
                            core.EV_ABSORB, core.EV_LOST):
                    events.append((int(b), t + 1, *map(int, e[1:])))
        snap(t + 1)
        end[live[bs.over[live]]] = t + 1
        if log and t % 500 == 0:
            log(f"   t={t // 25}s live {len(live)} search {t_s:.0f}s")
    out = {"seed": np.asarray(seeds), "side": sides, "pos": pos, "cls": cls, "htyp": htyp, "hamt": hamt,
           "score": score, "ownmv": ownmv, "allmv": allmv, "branch": branch, "end": end,
           "events": np.asarray(events, np.int64).reshape(-1, 7), "final": bs.scores, "search_s": np.float64(t_s)}
    for k, v in roots.items():
        out["root_" + k] = np.concatenate(v) if v else np.zeros((0,) if k != "values" else (0, nb))
    if searcher is not None:
        out["branches"] = np.asarray(json.dumps([b.name for b in searcher.branches]))
    return out


def cmd_play(args):
    import torch

    from blackout_env.sim.search import default_device

    torch.set_num_threads(args.threads)
    device = args.device or default_device()
    out = Path(args.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    net = make_net(device)
    searcher = make_searcher(args.tag, net, device)
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    for c in range(0, len(tasks), args.chunk):
        path = out / f"{args.tag}_{args.opponent}_{c:04d}.npz"
        if path.exists():
            print("exists", path, flush=True)
            continue
        chunk = tasks[c:c + args.chunk]
        t0 = time.time()
        res = run_chunk([s for s, _ in chunk], [sd for _, sd in chunk], net, args.opponent, device, searcher,
                        args.cap, log=print if args.verbose else None)
        np.savez_compressed(path, **res)
        lead = np.where(res["side"] == 0, res["final"][:, 0] - res["final"][:, 1], res["final"][:, 1] - res["final"][:, 0])
        print(f"[{args.tag} vs {args.opponent}] {c + len(chunk)}/{len(tasks)} margin {lead.mean():+.1f} "
              f"{time.time() - t0:.0f}s (search {float(res['search_s']):.0f}s)", flush=True)


def load_tag(out: Path, tag: str, opponent: str) -> dict:
    parts = [dict(np.load(p)) for p in sorted((out / "raw").glob(f"{tag}_{opponent}_*.npz"))]
    if not parts:
        raise FileNotFoundError(f"{tag}_{opponent}")
    m, off = {}, 0
    for p in parts:
        p["events"] = p["events"].copy()
        p["events"][:, 0] += off
        p["root_game"] = p["root_game"] + off
        off += len(p["seed"])
    for k in parts[0]:
        if k in ("branches", "search_s"):
            m[k] = parts[0][k]
        elif k in ("seed", "side", "final", "end", "events") or k.startswith("root_"):
            m[k] = np.concatenate([p[k] for p in parts])
        else:
            m[k] = np.concatenate([p[k] for p in parts], axis=1)
    return m


# ---------------------------------------------------------------------------------- counterfactual

EV_KEYS = ("death", "kill", "pickup", "steal", "deposit", "deposit_amt", "transform_h", "transform_c", "lost_amt")


def ev_counts(ev_rows, side, acc):
    """Accumulate per-unit event counts [10, len(EV_KEYS)] for one game's step events."""
    from blackout_env.sim import core
    from blackout_env.sim.mapdata import HUNTER

    for e in ev_rows:
        kind, a, b, c, d = int(e[1]), int(e[2]), int(e[3]), int(e[4]), int(e[5])
        if kind == core.EV_KILL:
            acc[b, 0] += 1
            if a >= 0:
                acc[a, 1] += 1
        elif kind == core.EV_PICKUP:
            acc[a, 2] += 1
        elif kind == core.EV_STEAL:
            acc[a, 3] += 1
        elif kind == core.EV_DEPOSIT:
            acc[a, 4] += 1
            acc[a, 5] += c
        elif kind == core.EV_TRANSFORM:
            acc[a, 6 if b == HUNTER else 7] += 1
        elif kind == core.EV_LOST:
            acc[a, 8] += c


def select_roots(S, n_roots, seed, min_tick=0, control=0):
    """Indices into the root arrays: ``n_roots`` deviations (choice != 0) sampled uniformly, plus
    ``control`` base roots whose best non-base branch is counterfactually tried as a control."""
    rng = np.random.default_rng(seed)
    ch, tk = S["root_choice"], S["root_tick"]
    dev = np.nonzero((ch > 0) & (tk >= min_tick))[0]
    pick = rng.choice(dev, size=min(n_roots, len(dev)), replace=False)
    out = [(int(i), int(ch[i])) for i in pick]
    if control:
        base = np.nonzero((ch == 0) & (tk >= min_tick))[0]
        cp = rng.choice(base, size=min(control, len(base)), replace=False)
        for i in cp:
            v = S["root_values"][i].copy()
            v[0] = -np.inf
            out.append((int(i), -int(v.argmax())))  # negative: control (not chosen, best non-base)
    return out


def cmd_cf(args):
    import torch

    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, HeuristicPolicy
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.search import ModNetPolicy, ValueNetLeaf, parse_branches
    from blackout_env.sim.state import SimState

    torch.set_num_threads(args.threads)
    from blackout_env.sim.search import default_device

    device = args.device or default_device()
    out = Path(args.out)
    S = load_tag(out, args.tag, args.opponent)
    branches = parse_branches("alt*,samp:1.0x2")
    assert [b.name for b in branches] == json.loads(str(S["branches"]))
    sel = select_roots(S, args.n_roots, args.select_seed, args.min_tick, args.control)
    if args.part is not None:
        k, m = map(int, args.part.split("/"))
        sel = sel[k::m]
    path = out / f"cf_{args.tag}_{args.opponent}_{args.cont_opp}{args.suffix}.npz"
    print(f"{len(sel)} roots -> {path}", flush=True)
    net = make_net(device)
    leaf = ValueNetLeaf(str(REPO / VALUE), device)
    K = args.reseeds
    games = sorted({int(S["root_game"][i]) for i, _ in sel})
    by_game = {g: [(i, c) for i, c in sel if int(S["root_game"][i]) == g] for g in games}
    res_rows = []
    t_all = time.time()
    for c0 in range(0, len(games), args.replay_chunk):
        gch = games[c0:c0 + args.replay_chunk]
        # ---- replay to every root of these games, snapshot state + opponent instance
        seeds, sides = S["seed"][gch], S["side"][gch].astype(np.int64)
        bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
        n = len(gch)
        dummy = ModNetPolicy(net)
        dummy.configure(n)
        opp = make_opp(args.opponent, dummy, device)
        opp.reset(n)
        if isinstance(opp, HeuristicPolicy):  # the instances' seeds of the recorded chunk layout
            from blackout_env.train.distill.opponents import make_opponent

            for j, g in enumerate(gch):
                for tm in (0, 1):
                    opp.instances[(j, tm)] = make_opponent(args.opponent, 2 * (g % args.play_chunk) + tm)
        bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
        moves = np.zeros((n, 10, 2), F32)
        want = {}
        for j, g in enumerate(gch):
            for i, c in by_game[g]:
                want.setdefault(int(S["root_tick"][i]), []).append((j, i, c))
        snaps = []
        tmax = max(want)
        mism = 0
        for t in range(tmax + 1):
            for j, i, c in want.get(t, []):
                inst = copy.deepcopy(opp.instances.get((j, 1 - int(sides[j])))) if isinstance(opp, HeuristicPolicy) else None
                snaps.append((i, c, bs.state(j), inst, int(sides[j]), int(S["seed"][gch[j]]), gch[j]))
                mism += int(np.abs(bs.uf[j] - S["pos"][t, gch[j]]).max() > 0)
            if t == tmax:
                break
            live = np.nonzero(~bs.over)[0]
            step_both(bs, live, sides, dummy, opp, bufs, moves, own_moves=S["ownmv"][t, np.asarray(gch)[live]])
        # ---- continue every root x arm x reseed
        R = len(snaps)
        sims, teams, arm_br, reseeds, insts = [], [], [], [], {}
        for r, (i, c, st, inst, side, seed, g) in enumerate(snaps):
            for arm in (0, 1):
                for k in range(K):
                    sims.append(st)
                    teams.append(side)
                    arm_br.append(branches[abs(c)] if arm == 1 else branches[0])
                    reseeds.append(None if k == 0 else (seed * 31 + int(S["root_tick"][i]) * 977 + k * 7) % (2 ** 31))
                    if inst is not None:
                        insts[(len(sims) - 1, 1 - side)] = copy.deepcopy(inst)
        N = len(sims)
        cb = BatchSim(sims)
        rs = [x for x in reseeds]
        idx_rs = [q for q, x in enumerate(rs) if x is not None]
        if idx_rs:
            from blackout_env.sim.unity_random import init_state

            for q in idx_rs:
                cb.rng[q] = init_state(int(rs[q]))
                cb.g[q, core.G_SPAWN_MODE] = core.SPAWN_RNG
        teams = np.asarray(teams, np.int64)
        our = ModNetPolicy(net)
        our.configure(N)
        for q in range(N):
            our.set_game(q, teams[q], arm_br[q], args.window)
        if args.cont_opp == "true":
            copp = make_opp(args.opponent, our, device)
            copp.reset(N)
            if isinstance(copp, HeuristicPolicy):
                copp.instances = insts
        else:
            copp = our
        cbufs = (np.zeros((N, 2, N_CH, H, W), F32), np.zeros((N, 2, 4), F32), np.zeros((N, 2, 10, N_AGENT_COLS), F32))
        cmoves = np.zeros((N, 10, 2), F32)
        acc = np.zeros((N, 10, len(EV_KEYS)), np.int32)
        lead0 = np.where(teams == 0, cb.scores[:, 0] - cb.scores[:, 1], cb.scores[:, 1] - cb.scores[:, 0])
        marks = {25: None, 150: None, 375: None}
        pos_w = np.zeros((N, 10, 2), F32)
        t1 = time.time()
        for t in range(args.horizon):
            live = np.nonzero(~cb.over)[0]
            if not len(live):
                break
            step_both(cb, live, teams, our, copp, cbufs, cmoves)
            if t < 150:
                for q in live:
                    ne = int(cb.g[q, core.G_NEV])
                    if ne:
                        ev_counts(cb.ev[q, :ne], teams[q], acc[q])
            if t + 1 == args.window:
                pos_w = cb.uf.copy()
            if t + 1 in marks:
                marks[t + 1] = np.where(teams == 0, cb.scores[:, 0] - cb.scores[:, 1], cb.scores[:, 1] - cb.scores[:, 0])
            if args.verbose and t % 125 == 0:
                print(f"   cont t={t} live {len(live)} {time.time() - t1:.0f}s", flush=True)
        lead_h = np.where(teams == 0, cb.scores[:, 0] - cb.scores[:, 1], cb.scores[:, 1] - cb.scores[:, 0])
        for k_, v in marks.items():
            if v is None:
                marks[k_] = lead_h
        vleaf = leaf(cb, teams)
        for r, (i, c, st, inst, side, seed, g) in enumerate(snaps):
            sl = slice(r * 2 * K, (r + 1) * 2 * K)
            res_rows.append({"root": i, "branch": c, "game": g, "lead0": lead0[sl], "lead1": marks[25][sl],
                             "lead6": marks[150][sl], "lead15": marks[375][sl], "leadH": lead_h[sl], "vH": vleaf[sl],
                             "over": cb.over[sl].copy(), "ev": acc[sl], "pos_w": pos_w[sl]})
        print(f"games {c0 + len(gch)}/{len(games)} roots {len(res_rows)} replay-mismatch {mism} "
              f"cont {N} games {time.time() - t1:.0f}s total {time.time() - t_all:.0f}s", flush=True)
    keys = res_rows[0].keys()
    np.savez_compressed(path, K=np.int64(K), horizon=np.int64(args.horizon),
                        **{k: np.stack([np.asarray(r[k]) for r in res_rows]) for k in keys})
    print("wrote", path)


# ---------------------------------------------------------------------------------- report

def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def paired(seeds, x):
    """Mean and seed-level SE of per-match values ``x`` (both sides of a seed averaged first)."""
    d = {}
    for s, v in zip(seeds, x):
        d.setdefault(int(s), []).append(v)
    ps = [np.mean(v) for v in d.values()]
    return float(np.mean(ps)) if ps else float("nan"), se(ps), len(x)


def first_hunter(D, team):
    """[n] first decision with a Hunter on ``team`` (10**6 if never)."""
    from blackout_env.sim.mapdata import HUNTER

    side = D["side"]
    cols = np.where(side[None, :, None] == team, 0, 5)  # placeholder (unused)
    del cols
    out = np.full(len(side), 10 ** 6, np.int64)
    for g in range(len(side)):
        u = slice(0, 5) if team[g] == 0 else slice(5, 10)
        h = (D["cls"][: D["end"][g] + 1, g, u] == HUNTER).any(-1)
        if h.any():
            out[g] = int(h.argmax())
    return out


def match_facts(D):
    """Per-match lead, race and event stats (our team = D['side'])."""
    from blackout_env.sim import core
    from blackout_env.sim.mapdata import CARRIER, HUNTER

    side = D["side"].astype(np.int64)
    n = len(side)
    sc = D["final"]
    lead = np.where(side == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])
    fo, fp = first_hunter(D, side), first_hunter(D, 1 - side)
    race = np.sign(fp - fo)
    f = {"lead": lead, "race": race, "first_ours": fo, "first_opp": fp, "end": D["end"]}
    ts = (250, 500, 750, 1125, 1500, 2250, 3000, 3750)
    for t in ts:
        tt = np.minimum(t, D["end"])
        s_ = D["score"][tt, np.arange(n)]
        f[f"lead@{t // 25}"] = np.where(side == 0, s_[:, 0] - s_[:, 1], s_[:, 1] - s_[:, 0]).astype(np.int64)
    ev = D["events"]
    keys = ("deaths_c", "deaths_h", "deaths_car", "kills_h", "opp_deaths_c", "lost_bat", "opp_lost_bat",
            "dep_amt", "opp_dep_amt", "steal", "opp_steal", "tr_h", "opp_tr_h", "tr_car", "opp_tr_car", "hh_trade")
    for k in keys:
        f[k] = np.zeros(n, np.int64)
    for e in ev:
        g, t, kind, a, b, c, d = map(int, e)
        mine = lambda u: (u < 5) == (side[g] == 0)  # noqa: E731
        if kind == core.EV_KILL:
            if c == HUNTER and a < 0:  # Hunter vs Hunter: both die, no killer
                f["hh_trade"][g] += 1 if mine(b) else 0
            if mine(b):
                f["deaths_c" if c == 0 else ("deaths_h" if c == HUNTER else "deaths_car")][g] += 1
            else:
                if c == 0:
                    f["opp_deaths_c"][g] += 1
                if a >= 0 and mine(a) and d == HUNTER:
                    f["kills_h"][g] += 1
        elif kind == core.EV_LOST and d == 1 and b == 0:
            f["lost_bat" if mine(a) else "opp_lost_bat"][g] += c
        elif kind == core.EV_DEPOSIT and b == 0:
            f["dep_amt" if mine(a) else "opp_dep_amt"][g] += c
        elif kind == core.EV_STEAL:
            f["steal" if mine(a) else "opp_steal"][g] += 1
        elif kind == core.EV_TRANSFORM:
            if b == HUNTER:
                f["tr_h" if mine(a) else "opp_tr_h"][g] += 1
            elif b == CARRIER:
                f["tr_car" if mine(a) else "opp_tr_car"][g] += 1
    hcount = np.zeros((2, n))
    for g in range(n):
        e = D["end"][g]
        u_o = slice(0, 5) if side[g] == 0 else slice(5, 10)
        u_p = slice(5, 10) if side[g] == 0 else slice(0, 5)
        hcount[0, g] = (D["cls"][:e, g, u_o] == HUNTER).sum() / 25.0
        hcount[1, g] = (D["cls"][:e, g, u_p] == HUNTER).sum() / 25.0
    f["hunter_s"], f["opp_hunter_s"] = hcount
    return f


def match_report(out, opponents, tags, plain="plain"):
    lines = []
    for opp in opponents:
        try:
            P = load_tag(out, plain, opp)
        except FileNotFoundError:
            continue
        fp_ = match_facts(P)
        for tag in tags:
            try:
                S = load_tag(out, tag, opp)
            except FileNotFoundError:
                continue
            n = min(len(S["seed"]), len(P["seed"]))
            assert np.array_equal(S["seed"][:n], P["seed"][:n]) and np.array_equal(S["side"][:n], P["side"][:n])
            fs = match_facts(S)
            seeds = S["seed"][:n]
            d = fs["lead"][:n] - fp_["lead"][:n]
            m, e, k = paired(seeds, d)
            rs, rp = fs["race"][:n], fp_["race"][:n]
            lines.append(f"== {tag} - plain vs {opp}: n={k}  {fs['lead'][:n].mean():+.1f} vs {fp_['lead'][:n].mean():+.1f}  "
                         f"diff {m:+.1f} ± {e:.1f}   race w/t/l search {[int((rs == x).sum()) for x in (1, 0, -1)]} "
                         f"plain {[int((rp == x).sum()) for x in (1, 0, -1)]}")
            for (a, b) in sorted({(int(x), int(y)) for x, y in zip(rp, rs)}):
                msk = (rp == a) & (rs == b)
                mm, ee, kk = paired(seeds[msk], d[msk])
                lines.append(f"   race plain {a:+d} -> search {b:+d}: {kk:4d} matches  diff {mm:+6.1f} ± {ee:4.1f}  "
                             f"share of total {d[msk].sum() / max(abs(d.sum()), 1e-9):+.2f}")
            tied = (rp == 0) & (rs == 0)
            lines.append("   lead trajectory (all / both race-tied), search - plain:")
            for t in (10, 20, 30, 45, 60, 90, 120, 150):
                dd = fs[f"lead@{t}"][:n] - fp_[f"lead@{t}"][:n]
                lines.append(f"      t={t:3d}s  {paired(seeds, dd)[0]:+6.1f} ± {paired(seeds, dd)[1]:.1f}   "
                             f"tied {paired(seeds[tied], dd[tied])[0]:+6.1f} ± {paired(seeds[tied], dd[tied])[1]:.1f}")
            lines.append("   per-match stats, search vs plain (all | both race-tied):")
            for kx in ("deaths_c", "deaths_h", "deaths_car", "lost_bat", "kills_h", "opp_deaths_c", "opp_lost_bat", "dep_amt",
                       "opp_dep_amt", "steal", "opp_steal", "tr_h", "opp_tr_h", "tr_car", "opp_tr_car", "hh_trade",
                       "hunter_s", "opp_hunter_s"):
                a_, b_ = fs[kx][:n].astype(float), fp_[kx][:n].astype(float)
                lines.append(f"      {kx:13s} {a_.mean():7.2f} vs {b_.mean():7.2f} (d {paired(seeds, a_ - b_)[0]:+6.2f} ± "
                             f"{paired(seeds, a_ - b_)[1]:.2f}) | {a_[tied].mean():7.2f} vs {b_[tied].mean():7.2f}")
            lines.append(f"   match length s: {fs['end'][:n].mean() / 25:.1f} vs {fp_['end'][:n].mean() / 25:.1f}")
    return lines


def root_features(S):
    """Per search root: phase, chosen branch, deviating unit's class / cargo / enemy-Hunter distance, ..."""
    from blackout_env.sim.mapdata import COLLECTOR, HUNTER

    br = json.loads(str(S["branches"]))
    g, t, ch, v = S["root_game"], S["root_tick"].astype(np.int64), S["root_choice"].astype(np.int64), S["root_values"]
    side = S["side"][g].astype(np.int64)
    fo = first_hunter(S, S["side"].astype(np.int64))
    fp = first_hunter(S, 1 - S["side"].astype(np.int64))
    f = {"game": g, "tick": t, "choice": ch, "gain_pred": v[np.arange(len(g)), ch] - v[:, 0],
         "best_nonbase": v[:, 1:].max(1) - v[:, 0], "v_base": v[:, 0]}
    f["prerace"] = t < np.minimum(fo[g], fp[g])
    unit = np.full(len(g), -1)
    kind = np.array([br[c][:3] if c > 0 else "base" for c in ch])
    for i, c in enumerate(ch):
        if c > 0 and br[c].startswith("alt"):
            unit[i] = int(br[c][3])
    f["kind"], f["unit"] = kind, unit
    own = np.stack([np.arange(5) + 5 * s for s in side])          # [R, 5] global unit index
    oppu = np.stack([np.arange(5) + 5 * (1 - s) for s in side])
    P = S["pos"][t, g]                                              # [R, 10, 2]
    C = S["cls"][t, g]
    HT = S["htyp"][t, g]
    HA = S["hamt"][t, g]
    r = np.arange(len(g))
    sc = S["score"][t, g]
    f["lead"] = np.where(side == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])
    f["own_h"] = (C[r[:, None], own] == HUNTER).sum(1)
    f["opp_h"] = (C[r[:, None], oppu] == HUNTER).sum(1)
    gu = np.where(unit >= 0, own[r, np.maximum(unit, 0)], -1)
    f["u_cls"] = np.where(gu >= 0, C[r, np.maximum(gu, 0)], -1)
    f["u_bat"] = np.where(gu >= 0, np.where(HT[r, np.maximum(gu, 0)] == 0, HA[r, np.maximum(gu, 0)], 0), 0)
    f["u_item"] = np.where(gu >= 0, HT[r, np.maximum(gu, 0)], -1)
    # distance from the deviating unit to the nearest enemy Hunter (inf if none)
    dpos = P[r[:, None], oppu] - P[r, np.maximum(gu, 0)][:, None]
    dist = np.hypot(dpos[..., 0], dpos[..., 1])
    dist = np.where(C[r[:, None], oppu] == HUNTER, dist, np.inf)
    f["u_dhunt"] = np.where(gu >= 0, dist.min(1), np.nan)
    # nearest enemy Collector/Carrier (for our Hunters)
    dprey = np.where(C[r[:, None], oppu] != HUNTER, np.hypot(dpos[..., 0], dpos[..., 1]), np.inf)
    f["u_dprey"] = np.where(gu >= 0, dprey.min(1), np.nan)
    # team-level: any own collector carrying battery within 4 tiles of an enemy Hunter
    f["team_bat"] = np.where(HT[r[:, None], own] == 0, HA[r[:, None], own], 0).sum(1)
    # opening walker or reserve: the deviating unit's distance to the sanctuary just before the race
    tr = np.minimum(np.minimum(fo[g], fp[g]), S["end"][g]) - 1
    pr = S["pos"][np.maximum(tr, 0), g, np.maximum(gu, 0)]
    dx = np.maximum(np.maximum(SANCT[0] - pr[:, 0], pr[:, 0] - SANCT[1]), 0)
    dy = np.maximum(np.maximum(SANCT[0] - pr[:, 1], pr[:, 1] - SANCT[1]), 0)
    f["u_walker"] = (gu >= 0) & (np.hypot(dx, dy) <= 1.0)
    f["t_race"] = np.minimum(fo[g], fp[g])
    del COLLECTOR
    return f


def situation(f):
    """Situational category of a deviation (feature-based)."""
    from blackout_env.sim.mapdata import CARRIER, COLLECTOR, HUNTER

    cat = np.full(len(f["tick"]), "base", dtype=object)
    for i in range(len(cat)):
        if f["choice"][i] == 0:
            continue
        if f["prerace"][i]:
            cat[i] = ("race: joint sample" if f["kind"][i] == "sam" else
                      "race: walker" if f["u_walker"][i] else "race: reserve (non-walker)")
        elif f["tick"][i] < f["t_race"][i] + 150:
            cat[i] = "second race (<= 6 s after the race)" + (": sample" if f["kind"][i] == "sam" else
                                                              ": Hunter" if f["u_cls"][i] == HUNTER else "")
        elif f["kind"][i] == "sam":
            cat[i] = "joint sample (all units)"
        elif f["u_cls"][i] == HUNTER:
            cat[i] = "Hunter: prey <= 4" if f["u_dprey"][i] <= 4 else "Hunter: no prey near"
        elif f["u_cls"][i] == CARRIER:
            cat[i] = "Carrier"
        elif f["u_cls"][i] == COLLECTOR:
            near = f["u_dhunt"][i] <= 4
            if f["u_bat"][i] > 0:
                cat[i] = "Collector carrying, Hunter <= 4" if near else "Collector carrying, safe"
            elif f["u_item"][i] > 0:
                cat[i] = "Collector special item"
            else:
                cat[i] = "Collector empty, Hunter <= 4" if near else "Collector empty, safe"
    return cat


CH = ("own death", "own kill", "own Hunter transform", "own deposit", "own pickup/steal", "own battery lost",
      "opp deposit", "opp Hunter transform")


def cf_effects(C, sides):
    """[R, len(CH)] arm difference (chosen - base, mean over reseeds) of 6 s event totals, plus the
    deviating-unit-only part of the own channels."""
    K = int(C["K"])
    ev = C["ev"].astype(float)                                   # [R, 2K, 10, 9]
    R = ev.shape[0]
    own = np.stack([np.arange(5) + 5 * s for s in sides])
    opp = np.stack([np.arange(5) + 5 * (1 - s) for s in sides])
    r = np.arange(R)[:, None]
    eo, ep = ev[r, :, own].transpose(0, 2, 1, 3), ev[r, :, opp].transpose(0, 2, 1, 3)  # [R, 2K, 5, 9]
    tot_o, tot_p = eo.sum(2), ep.sum(2)                         # [R, 2K, 9]
    # EV_KEYS = death, kill, pickup, steal, deposit, deposit_amt, transform_h, transform_c, lost_amt
    ch = np.stack([tot_o[..., 0], tot_o[..., 1], tot_o[..., 6], tot_o[..., 5], tot_o[..., 2] + tot_o[..., 3],
                   tot_o[..., 8], tot_p[..., 5], tot_p[..., 6]], -1)  # [R, 2K, C]
    d = ch[:, K:].mean(1) - ch[:, :K].mean(1)
    return d, eo


def classify_effect(d):
    thr = np.array([0.25, 0.25, 0.25, 1.0, 0.25, 1.0, 1.0, 0.25])
    out = np.full(len(d), "no event change in 6 s", dtype=object)
    for i in range(len(d)):
        for c in range(len(CH)):
            if abs(d[i, c]) >= thr[c]:
                good = (d[i, c] > 0) != (c in (0, 5, 6, 7))
                out[i] = f"{CH[c]} {'+' if d[i, c] > 0 else '-'}" + (" (good)" if good else " (bad)")
                break
    return out


def cf_gains(C):
    K = int(C["K"])
    g = {}
    for key in ("vH", "leadH", "lead6", "lead15", "lead1"):
        x = C[key].astype(float)
        g[key] = x[:, K:].mean(1) - x[:, :K].mean(1)
        g[key + "_k0"] = x[:, K] - x[:, 0]
    return g


def cf_report(out, tag, opponents):
    lines = []
    for opp in opponents:
        try:
            S = load_tag(out, tag, opp)
        except FileNotFoundError:
            continue
        rf = root_features(S)
        sit = situation(rf)
        cfs = {}
        for cont in ("true", "model"):
            for pth in sorted(out.glob(f"cf_{tag}_{opp}_{cont}*.npz")):
                d = dict(np.load(pth))
                if cont in cfs:
                    cfs[cont] = {k: (np.concatenate([cfs[cont][k], v]) if np.ndim(v) else v) for k, v in d.items()}
                else:
                    cfs[cont] = d
        if "true" not in cfs:
            continue
        C = cfs["true"]
        roots = C["root"]
        sides = S["side"][S["root_game"][roots]].astype(np.int64)
        ctrl = C["branch"] < 0
        gain = cf_gains(C)
        eff, _ = cf_effects(C, sides)
        ecat = classify_effect(eff)
        gm = None
        if "model" in cfs:
            M = cfs["model"]
            pos = {int(r): j for j, r in enumerate(M["root"])}
            jm = np.array([pos.get(int(r), -1) for r in roots])
            gmv = cf_gains(M)["vH"]
            gm = np.where(jm >= 0, gmv[np.maximum(jm, 0)], np.nan)
        pred = rf["gain_pred"][roots]
        best = rf["best_nonbase"][roots]
        seeds = S["seed"][S["root_game"][roots]]
        n_dev = int((rf["choice"] > 0).sum())
        n_m = len(S["seed"])
        dv = ~ctrl
        lines.append(f"== cf {tag} vs {opp}: {dv.sum()} deviations sampled of {n_dev} ({n_dev / n_m:.1f}/match), "
                     f"{ctrl.sum()} controls, K={int(C['K'])}, horizon {int(C['horizon']) // 25}s")
        for key in ("vH", "leadH", "lead15", "lead6", "lead1"):
            m, e, _ = paired(seeds[dv], gain[key][dv])
            lines.append(f"   realised gain per deviation [{key:6s}] {m:+6.2f} ± {e:.2f}   (k=0 own RNG only "
                         f"{paired(seeds[dv], gain[key + '_k0'][dv])[0]:+.2f})")
        lines.append(f"   predicted (search) gain per deviation {pred[dv].mean():+.2f}; implied per match "
                     f"{gain['vH'][dv].mean() * n_dev / n_m:+.1f} (realised) vs {pred[dv].mean() * n_dev / n_m:+.1f} (predicted)")
        if gm is not None:
            ok = dv & ~np.isnan(gm)
            m, e, _ = paired(seeds[ok], gm[ok])
            lines.append(f"   realised vs the model opponent (same roots, n={ok.sum()}): {m:+.2f} ± {e:.2f};  "
                         f"true on the same roots {paired(seeds[ok], gain['vH'][ok])[0]:+.2f}")
        if ctrl.any():
            m, e, _ = paired(seeds[ctrl], gain["vH"][ctrl])
            lines.append(f"   controls (base chosen; best non-base branch, predicted {best[ctrl].mean():+.2f}): realised {m:+.2f} ± {e:.2f}")
        corr = np.corrcoef(pred[dv], gain["vH"][dv])[0, 1]
        lines.append(f"   corr(predicted, realised) over deviations {corr:+.3f}")
        tot = gain["vH"][dv].sum()
        for name, cat in (("situation", sit[roots]), ("6 s effect (first channel, chosen - base)", ecat)):
            lines.append(f"   by {name}:  n  share | predicted | realised true ± se | share of realised | realised model")
            for c in sorted(set(cat[dv]), key=lambda c: -np.sum(dv & (cat == c))):
                msk = dv & (cat == c)
                m, e, k = paired(seeds[msk], gain["vH"][msk])
                mm = np.nanmean(gm[msk]) if gm is not None else np.nan
                lines.append(f"      {c:42s} {k:4d} {k / dv.sum():.2f} | {pred[msk].mean():+6.2f} | {m:+6.2f} ± {e:4.2f} | "
                             f"{gain['vH'][msk].sum() / tot if tot else np.nan:+.2f} | {mm:+6.2f}")
        # phase
        lines.append("   by root time:  n | predicted | realised true | realised model")
        tk = rf["tick"][roots] / 25
        for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 90), (90, 151)):
            msk = dv & (tk >= lo) & (tk < hi)
            if msk.any():
                m, e, k = paired(seeds[msk], gain["vH"][msk])
                mm = np.nanmean(gm[msk]) if gm is not None else np.nan
                lines.append(f"      {lo:3d}-{hi:3d}s {k:4d} | {pred[msk].mean():+6.2f} | {m:+6.2f} ± {e:4.2f} | {mm:+6.2f}")
        # by predicted gain bin
        lines.append("   by predicted gain:  n | realised true | realised model")
        for lo, hi in ((3, 5), (5, 8), (8, 15), (15, 1e9)):
            msk = dv & (pred >= lo) & (pred < hi)
            if msk.any():
                m, e, k = paired(seeds[msk], gain["vH"][msk])
                mm = np.nanmean(gm[msk]) if gm is not None else np.nan
                lines.append(f"      [{lo:g},{hi:g}) {k:4d} | {m:+6.2f} ± {e:4.2f} | {mm:+6.2f}")
    return lines


def cmd_report(args):
    out = Path(args.out)
    opps = args.opponents.split(",")
    lines = match_report(out, opps, args.tags.split(","))
    for tag in args.tags.split(","):
        lines += cf_report(out, tag, opps)
    txt = "\n".join(lines)
    print(txt)
    (out / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--tag", required=True, choices=sorted(TAGS))
    q.add_argument("--opponent", default="strategic_v25")
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=64)
    q.add_argument("--cap", type=int, default=3750)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/late_search_why"))
    q = sub.add_parser("cf")
    q.add_argument("--tag", default="search")
    q.add_argument("--opponent", default="strategic_v25")
    q.add_argument("--cont-opp", default="true", choices=("true", "model"))
    q.add_argument("--n-roots", type=int, default=400)
    q.add_argument("--control", type=int, default=0)
    q.add_argument("--min-tick", type=int, default=0)
    q.add_argument("--select-seed", type=int, default=0)
    q.add_argument("--part", default=None, help="k/m: every m-th selected root from k")
    q.add_argument("--suffix", default="")
    q.add_argument("--reseeds", type=int, default=4)
    q.add_argument("--horizon", type=int, default=750)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--replay-chunk", type=int, default=64)
    q.add_argument("--play-chunk", type=int, default=64, help="the play run's --chunk (heuristic instance seeds)")
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/late_search_why"))
    q = sub.add_parser("report")
    q.add_argument("--opponents", default="strategic_v25,strategic_v23")
    q.add_argument("--tags", default="search,o4fmean")
    q.add_argument("--out", default=str(REPO / "reports/late_search_why"))
    args = p.parse_args()
    {"play": cmd_play, "cf": cmd_cf, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
