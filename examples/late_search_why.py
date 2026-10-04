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
    args = p.parse_args()
    {"play": cmd_play, "cf": cmd_cf}[args.cmd](args)


if __name__ == "__main__":
    main()
