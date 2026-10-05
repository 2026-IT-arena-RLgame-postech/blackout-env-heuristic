"""Q11: what does the blind search add on top of P17 after the opening (docs/policy_v2_design_ko.md §44)?

The §40 search champion = P17 acting + o4fmean search from 5 s (B8 ``alt*,samp:1.0x2``, every 25, window 25,
margin 3, overlay; rollout / opponent model v28d256, leaf V s2v_f3xd2 iter2) beats plain P17 by +17.6 pooled.
This records where it deviates from P17 and what each deviation is worth against the true opponent, as §33 (Q6)
did for search-new, so the robust part can become a V29 rule.

``play``: one opponent, tag ``plain`` (P17 alone) or ``search`` (``o4f_v28r`` of ``blind_search_eval.py``, or any
of its ``VARIANTS`` with ``--variant``), seeds x both sides in lockstep chunks; every decision records both teams'
positions, classes, held item, scores, our executed moves (``ownmv``), **P17's own move for our units**
(``basemv``: what plain P17 would have done there), the executing branch and sim events; every root its 8 branch
values and choice.  Same npz layout as ``late_search_why.py`` (its loaders / features are reused) under
``<out>/raw/<tag>_<opponent>_<chunk>.npz``.

``cf``: sampled deviation roots (choice != base) are replayed (recorded own moves; a heuristic opponent re-acts
bit-exactly, checked against the recorded positions), the state and the opponent's instance copied and continued
twice -- base (P17 from the root) and chosen (the recorded executed moves for the window, then P17) -- under K
reseeds (k = 0 the match's own RNG) for ``--horizon`` decisions against the true opponent.  Each arm records the
lead at +1/+6/+15 s / horizon, V_sym at the horizon, and 6 s per-unit event counts -> ``<out>/cf_<opp>.npz``.
Our continuation policy = P17 (CPU is fine: the arms share it).

``report``: match-level search - plain, and the deviation table (category x realised gain, ``late_search_why``'s
classification plus the deviation direction relative to P17's move).

    PYTHONPATH=. python examples/q11_why.py play --tag search --opponent v28 --seeds 64 --device cuda
    PYTHONPATH=. python examples/q11_why.py cf --opponent v28 --n-roots 600 --device cpu
    PYTHONPATH=. python examples/q11_why.py report --opponents v29,v28,v25
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

import blind_search_eval as bse  # noqa: E402
import late_search_why as lsw  # noqa: E402

F32 = np.float32
P17 = bse.P17
V29 = ("strategic_v29", {"reserve_stage": 0.5, "trade_hold": "advance", "reserve_join": "reserve"})
V30 = ("strategic_v30", {"reserve_stage": 0.5, "trade_hold": "advance", "reserve_join": "reserve", "solo_sentry": True})
HBASES = {"v29": V29, "v30": V30}  # heuristic acting bases (--base); tags get the suffix _b29 / _b30


def base_suffix(base: str) -> str:
    return "" if base == "p17" else "_b" + base[1:]
SANCT_C = np.array([12.0, 12.0], F32)


def make_opp(name: str, device: str):
    if name in HBASES:
        return bse.FixedHeuristicPolicy(*HBASES[name])
    return bse.make_opponent(name, device)


def is_heuristic(opp) -> bool:
    from blackout_env.sim.batch import HeuristicPolicy

    return isinstance(opp, HeuristicPolicy)


def search_args(args) -> SimpleNamespace:
    return SimpleNamespace(v25r=bse.V25R, value=bse.VALUE, branches="alt*,samp:1.0x2", window=args.window, margin=3.0)


# ---------------------------------------------------------------------------------- play

def run_chunk(seeds, sides, opp_name, base, searcher, net, device, cap, every=25, window=25, search_from=125, log=None):
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.search import K_BASE, ModNetPolicy, overlay_base
    from blackout_env.sim.state import SimState

    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    our = ModNetPolicy(net)
    our.configure(n)
    opp = make_opp(opp_name, device)
    opp.reset(n)
    base.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    pos = np.zeros((cap + 1, n, 10, 2), F32)
    cls = np.zeros((cap + 1, n, 10), np.int8)
    htyp = np.full((cap + 1, n, 10), -1, np.int8)
    hamt = np.zeros((cap + 1, n, 10), np.int8)
    score = np.zeros((cap + 1, n, 2), np.int16)
    ownmv = np.zeros((cap, n, 5, 2), F32)
    basemv = np.zeros((cap, n, 5, 2), F32)
    branch = np.zeros((cap, n), np.int16)
    nb = len(searcher.branches) if searcher else 1
    roots = {"game": [], "tick": [], "values": [], "choice": []}
    events = []
    cur = np.zeros(n, np.int64)
    end = np.full(n, cap, np.int64)

    def snap(t):
        pos[t], cls[t] = bs.uf, bs.ui[:, :, core.U_CLS]
        htyp[t], hamt[t] = lsw.held_info(bs)
        score[t] = bs.scores

    snap(0)
    t_s = 0.0
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if searcher is not None and t % every == 0 and t >= search_from:
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
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        ours = sides[live]
        pre_left, pre_kind, pre_unit = our.left[live].copy(), our.kind[live].copy(), our.unit[live].copy()
        branch[t, live] = np.where(pre_left > 0, cur[live], 0)
        if (pre_left > 0).any():
            mo = our.moves(bs, live, (0, 1), sub)
        else:
            mo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
            our.t[live] += 1
        own = np.zeros((len(live), 5, 2), F32)
        bm = np.zeros((len(live), 5, 2), F32)
        for team in (0, 1):
            rows = np.nonzero(ours == team)[0]
            if not len(rows):
                continue
            s_in = sub if len(rows) == len(live) else tuple(x[rows] for x in sub)
            ext = base.moves(bs, live[rows], (team,), s_in)[team]
            kd = np.where(pre_left[rows] > 0, pre_kind[rows], K_BASE)
            own[rows] = overlay_base(mo[team][rows], ext, kd, pre_unit[rows], True)
            bm[rows] = ext
        oo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
        for team in (0, 1):
            sel = ours != team
            if sel.any():
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                oo[team][sel] = opp.moves(bs, live[sel], (team,), s_in)[team]
        for team in (0, 1):
            mine = ours == team
            moves[live, 5 * team:5 * team + 5] = np.where(mine[:, None, None], own, oo[team])
        ownmv[t, live] = own
        basemv[t, live] = bm
        bs.step(moves, live)
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
           "score": score, "ownmv": ownmv, "basemv": basemv, "branch": branch, "end": end,
           "events": np.asarray(events, np.int64).reshape(-1, 7), "final": bs.scores, "search_s": np.float64(t_s)}
    for k, v in roots.items():
        out["root_" + k] = np.concatenate(v) if v else np.zeros((0,) if k != "values" else (0, nb))
    out["branches"] = np.asarray(json.dumps([b.name for b in searcher.branches] if searcher else ["base"]))
    return out


def cmd_play(args):
    import torch

    from blackout_env.sim.batch import NetPolicy

    torch.set_num_threads(args.threads)
    out = Path(args.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    if args.base in HBASES:  # a heuristic base: the acting instances and the search's own (reset per search) apart
        base, sbase = bse.FixedHeuristicPolicy(*HBASES[args.base]), bse.FixedHeuristicPolicy(*HBASES[args.base])
    else:
        base = sbase = NetPolicy.from_checkpoint(str(REPO / P17), device=args.device)
    if args.tag == "plain":
        searcher, net = None, bse.rollout_policy(bse.V28R, args.device)
    else:
        searcher, net, _ = bse.build_searcher(args.variant, sbase, args.device, search_args(args))
    tag = args.tag + base_suffix(args.base)
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    for c in range(0, len(tasks), args.chunk):
        path = out / f"{tag}_{args.opponent}_{c:04d}.npz"
        if path.exists():
            print("exists", path, flush=True)
            continue
        chunk = tasks[c:c + args.chunk]
        t0 = time.time()
        res = run_chunk([s for s, _ in chunk], [sd for _, sd in chunk], args.opponent, base, searcher, net, args.device,
                        args.cap, window=args.window, log=print if args.verbose else None)
        np.savez_compressed(path, **res)
        lead = np.where(res["side"] == 0, res["final"][:, 0] - res["final"][:, 1], res["final"][:, 1] - res["final"][:, 0])
        print(f"[{tag} vs {args.opponent}] {c + len(chunk)}/{len(tasks)} margin {lead.mean():+.1f} "
              f"{time.time() - t0:.0f}s (search {float(res['search_s']):.0f}s)", flush=True)


def load_tag(out: Path, tag: str, opponent: str) -> dict:
    """``late_search_why.load_tag`` + ``basemv`` and each game's index inside its play chunk (``jidx``)."""
    parts = sorted((out / "raw").glob(f"{tag}_{opponent}_*.npz"))
    D = lsw.load_tag(out, tag, opponent)
    D["jidx"] = np.concatenate([np.arange(len(np.load(p)["seed"])) for p in parts])
    D["chunk0"] = np.concatenate([np.full(len(np.load(p)["seed"]), int(p.stem.rsplit("_", 1)[1])) for p in parts])
    return D


# ---------------------------------------------------------------------------------- counterfactual

def cmd_cf(args):
    import torch

    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, NetPolicy
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.search import ValueNetLeaf
    from blackout_env.sim.state import SimState
    from blackout_env.sim.unity_random import init_state

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    S = load_tag(out, "search" + base_suffix(args.base), args.opponent)
    rng = np.random.default_rng(args.select_seed)
    ch, tk = S["root_choice"].astype(np.int64), S["root_tick"].astype(np.int64)
    dev = np.nonzero((ch > 0) & (tk >= args.min_tick)
                     & ((S["chunk0"][S["root_game"]] == args.chunk0) if args.chunk0 >= 0 else True))[0]
    sel = np.sort(rng.choice(dev, size=min(args.n_roots, len(dev)), replace=False))
    if args.part is not None:
        k, m = map(int, args.part.split("/"))
        sel = sel[k::m]
    path = out / f"cf_{args.opponent}{base_suffix(args.base)}{args.suffix}.npz"
    print(f"{len(sel)} of {len(dev)} deviation roots -> {path}", flush=True)
    # our continuation: P17, or fresh V29 instances at the root (as inside the search)
    base = (bse.FixedHeuristicPolicy(*HBASES[args.base]) if args.base in HBASES
            else NetPolicy.from_checkpoint(str(REPO / P17), device=args.device))
    leaf = ValueNetLeaf(str(REPO / bse.VALUE), args.device)
    K, Wd = args.reseeds, args.window
    rows = []
    t_all = time.time()
    chunks = sorted({int(x) for x in S["chunk0"]})
    for c0 in chunks:
        gall = np.nonzero(S["chunk0"] == c0)[0]               # the whole play chunk (heuristic instance seeds)
        mine = [int(i) for i in sel if S["chunk0"][S["root_game"][i]] == c0]
        if not mine:
            continue
        n = len(gall)
        seeds, sides = S["seed"][gall], S["side"][gall].astype(np.int64)
        bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
        opp = make_opp(args.opponent, args.device)
        assert is_heuristic(opp), "cf replays need a deterministic heuristic opponent"
        opp.reset(n)
        bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
        moves = np.zeros((n, 10, 2), F32)
        want = {}
        for i in mine:
            want.setdefault(int(tk[i]), []).append(i)
        snaps, mism = [], 0
        tmax = max(want)
        for t in range(tmax + 1):
            for i in want.get(t, []):
                g = int(S["root_game"][i])
                j = int(S["jidx"][g])
                side = int(sides[j])
                snaps.append((i, g, bs.state(j), copy.deepcopy(opp.instances.get((j, 1 - side))), side))
                mism += int(np.abs(bs.uf[j] - S["pos"][t, g]).max() > 0)
            if t == tmax:
                break
            live = np.nonzero(~bs.over)[0]
            lsw.step_both(bs, live, sides, None, opp, bufs, moves, own_moves=S["ownmv"][t, gall[live]])
        # ---- arms: 0 = P17 from the root, 1 = the recorded executed moves for the window, then P17
        sims, teams, arm, rec, insts = [], [], [], [], {}
        for r, (i, g, st, inst, side) in enumerate(snaps):
            t = int(tk[i])
            for a in (0, 1):
                for k in range(K):
                    sims.append(st)
                    teams.append(side)
                    arm.append(a)
                    rec.append((g, t))
                    insts[(len(sims) - 1, 1 - side)] = copy.deepcopy(inst)
        N = len(sims)
        cb = BatchSim(sims)
        for q in range(N):
            k = q % K
            if k:
                i = snaps[q // (2 * K)][0]
                cb.rng[q] = init_state(int((int(S["seed"][S["root_game"][i]]) * 31 + int(tk[i]) * 977 + k * 7) % (2 ** 31)))
                cb.g[q, core.G_SPAWN_MODE] = core.SPAWN_RNG
        teams = np.asarray(teams, np.int64)
        arm = np.asarray(arm)
        copp = make_opp(args.opponent, args.device)
        copp.reset(N)
        copp.instances = insts
        base.reset(N)
        cbufs = (np.zeros((N, 2, N_CH, H, W), F32), np.zeros((N, 2, 4), F32), np.zeros((N, 2, 10, N_AGENT_COLS), F32))
        cm = np.zeros((N, 10, 2), F32)
        acc = np.zeros((N, 10, len(lsw.EV_KEYS)), np.int32)
        lead = lambda: np.where(teams == 0, cb.scores[:, 0] - cb.scores[:, 1], cb.scores[:, 1] - cb.scores[:, 0])  # noqa: E731
        lead0 = lead()
        marks = {25: None, 150: None, 375: None}
        pos_w = cb.uf.copy()
        t1 = time.time()
        for t in range(args.horizon):
            live = np.nonzero(~cb.over)[0]
            if not len(live):
                break
            cb.render(live, *cbufs)
            sub = tuple(x[: len(live)] for x in cbufs)
            own = np.zeros((len(live), 5, 2), F32)
            for team in (0, 1):
                rws = np.nonzero(teams[live] == team)[0]
                if len(rws):
                    s_in = sub if len(rws) == len(live) else tuple(x[rws] for x in sub)
                    own[rws] = base.moves(cb, live[rws], (team,), s_in)[team]
            if t < Wd:
                for jj, q in enumerate(live):
                    if arm[q] == 1:
                        g, t0 = rec[q]
                        own[jj] = S["ownmv"][t0 + t, g]
            oo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
            for team in (0, 1):
                sl = teams[live] != team
                if sl.any():
                    s_in = sub if sl.all() else tuple(x[sl] for x in sub)
                    oo[team][sl] = copp.moves(cb, live[sl], (team,), s_in)[team]
            for team in (0, 1):
                mn = teams[live] == team
                cm[live, 5 * team:5 * team + 5] = np.where(mn[:, None, None], own, oo[team])
            cb.step(cm, live)
            if t < 150:
                for q in live:
                    ne = int(cb.g[q, core.G_NEV])
                    if ne:
                        lsw.ev_counts(cb.ev[q, :ne], teams[q], acc[q])
            if t + 1 == Wd:
                pos_w = cb.uf.copy()
            if t + 1 in marks:
                marks[t + 1] = lead()
            if args.verbose and t % 125 == 0:
                print(f"   cont t={t} live {len(live)} {time.time() - t1:.0f}s", flush=True)
        lead_h = lead()
        for k_, v in marks.items():
            if v is None:
                marks[k_] = lead_h
        vleaf = leaf(cb, teams)
        for r, (i, g, st, inst, side) in enumerate(snaps):
            sl = slice(r * 2 * K, (r + 1) * 2 * K)
            rows.append({"root": i, "branch": int(ch[i]), "game": g, "lead0": lead0[sl], "lead1": marks[25][sl],
                         "lead6": marks[150][sl], "lead15": marks[375][sl], "leadH": lead_h[sl], "vH": vleaf[sl],
                         "over": cb.over[sl].copy(), "ev": acc[sl], "pos_w": pos_w[sl]})
        print(f"chunk {c0}: roots {len(rows)} replay-mismatch {mism} cont {N} games {time.time() - t1:.0f}s "
              f"total {time.time() - t_all:.0f}s", flush=True)
    keys = rows[0].keys()
    np.savez_compressed(path, K=np.int64(K), horizon=np.int64(args.horizon),
                        **{k: np.stack([np.asarray(r[k]) for r in rows]) for k in keys})
    print("wrote", path)


# ---------------------------------------------------------------------------------- report

def deviation_dir(S, f):
    """Per root: the deviating unit's executed vs P17 move over the window -- mean cosine, and the radial
    component toward the Hunter sanctuary / away from the nearest enemy Hunter (executed - P17, tiles/decision)."""
    from blackout_env.sim.mapdata import HUNTER

    R = len(f["tick"])
    out = {k: np.full(R, np.nan) for k in ("cos", "d_sanct", "d_flee", "d_prey", "stop_exec", "stop_base")}
    for r in range(R):
        u = int(f["unit"][r])
        if u < 0:
            continue
        g, t = int(f["game"][r]), int(f["tick"][r])
        side = int(S["side"][g])
        gu = u + 5 * side
        ex, bm = S["ownmv"][t:t + 25, g, u].astype(float), S["basemv"][t:t + 25, g, u].astype(float)
        ne, nb = np.linalg.norm(ex, axis=1), np.linalg.norm(bm, axis=1)
        ok = (ne > 1e-3) & (nb > 1e-3)
        out["cos"][r] = float(np.mean((ex[ok] * bm[ok]).sum(1) / (ne[ok] * nb[ok]))) if ok.any() else np.nan
        out["stop_exec"][r], out["stop_base"][r] = float((ne < 0.05).mean()), float((nb < 0.05).mean())
        p = S["pos"][t, g, gu].astype(float)
        dv = (ex - bm).mean(0)
        to_s = SANCT_C - p
        out["d_sanct"][r] = float(dv @ to_s / max(np.linalg.norm(to_s), 1e-6))
        opp = np.arange(5) + 5 * (1 - side)
        oc = S["cls"][t, g, opp]
        op = S["pos"][t, g, opp].astype(float)
        dist = np.linalg.norm(op - p, axis=1)
        h = np.nonzero(oc == HUNTER)[0]
        if len(h):
            e = op[h[dist[h].argmin()]] - p
            out["d_flee"][r] = float(-(dv @ e) / max(np.linalg.norm(e), 1e-6))
        pr = np.nonzero(oc != HUNTER)[0]
        if len(pr):
            e = op[pr[dist[pr].argmin()]] - p
            out["d_prey"][r] = float(dv @ e / max(np.linalg.norm(e), 1e-6))
    return out


def cmd_report(args):
    out = Path(args.out)
    opps = args.opponents.split(",")
    sfx = base_suffix(args.base)
    lines = lsw.match_report(out, opps, ["search" + sfx], plain="plain" + sfx)
    for opp in opps:
        try:
            S = load_tag(out, "search" + sfx, opp)
        except FileNotFoundError:
            continue
        f = lsw.root_features(S)
        cat = lsw.situation(f)
        dvi = deviation_dir(S, f)
        nm = len(S["seed"])
        devm = f["choice"] > 0
        lines += ["", f"== deviations vs {opp}: roots {len(cat)}, deviations {int(devm.sum())} ({devm.mean():.2f}), "
                      f"{devm.sum() / nm:.1f} per match"]
        tsec = f["tick"] // 25
        for lo, hi in ((5, 10), (10, 30), (30, 60), (60, 151)):
            m = (tsec >= lo) & (tsec < hi)
            lines.append(f"   {lo:3d}-{hi:3d}s: roots {int(m.sum()):6d} dev rate {devm[m].mean() if m.any() else 0:.2f}")
        C = None
        cpath = out / f"cf_{opp}{sfx}.npz"
        if cpath.exists():
            C = dict(np.load(cpath))
            gains = lsw.cf_gains(C)
            idx = C["root"].astype(np.int64)
            lines.append(f"   cf: {len(idx)} roots, K={int(C['K'])}, horizon {int(C['horizon'])}: realised vH gain "
                         f"{gains['vH'].mean():+.2f}±{lsw.se(gains['vH']):.2f}, lead6 {gains['lead6'].mean():+.2f}, "
                         f"predicted {f['gain_pred'][idx].mean():+.2f}; implied per match "
                         f"{gains['vH'].mean() * devm.sum() / nm:+.1f}")
            eff, _ = lsw.cf_effects(C, S["side"][C["game"].astype(np.int64)])
            ecat = lsw.classify_effect(eff)
        lines.append(f"   {'category':42s} {'n':>5s} {'share':>6s} {'pred':>6s}" +
                     (f" {'cf n':>5s} {'real vH':>12s} {'lead6':>6s} {'sum share':>9s}" if C is not None else "") +
                     f" {'cos':>5s} {'d_sanct':>7s} {'d_flee':>6s} {'d_prey':>6s} {'stopX':>5s} {'stopB':>5s}")
        tot = gains["vH"].sum() if C is not None else 0.0
        for c in sorted(set(cat[devm]), key=lambda c: -int((cat == c).sum())):
            m = (cat == c) & devm
            row = f"   {c:42s} {int(m.sum()):5d} {m.sum() / devm.sum():6.2f} {np.mean(f['gain_pred'][m]):+6.1f}"
            if C is not None:
                cm = cat[idx] == c
                gv = gains["vH"][cm]
                row += (f" {int(cm.sum()):5d} {f'{gv.mean():+.2f}±{lsw.se(gv):.2f}' if cm.sum() else '':>12s} "
                        f"{gains['lead6'][cm].mean() if cm.sum() else np.nan:+6.2f} {gv.sum() / tot if tot else np.nan:+9.2f}")
            row += "".join(f" {np.nanmean(dvi[k][m]) if np.isfinite(dvi[k][m]).any() else np.nan:{w}.2f}"
                           for k, w in (("cos", 5), ("d_sanct", 7), ("d_flee", 6), ("d_prey", 6), ("stop_exec", 5),
                                        ("stop_base", 5)))
            lines.append(row)
        if C is not None:
            lines.append("   cf by 6 s event effect (chosen - base):")
            for c in sorted(set(ecat), key=lambda c: -int((ecat == c).sum())):
                m = ecat == c
                lines.append(f"      {c:40s} {int(m.sum()):5d} real {gains['vH'][m].mean():+6.2f}±{lsw.se(gains['vH'][m]):.2f} "
                             f"share {gains['vH'][m].sum() / tot if tot else np.nan:+.2f}")
            lines.append("   cf by phase:")
            for lo, hi in ((5, 10), (10, 30), (30, 60), (60, 151)):
                m = (tsec[idx] >= lo) & (tsec[idx] < hi)
                if m.any():
                    lines.append(f"      {lo:3d}-{hi:3d}s {int(m.sum()):5d} real {gains['vH'][m].mean():+6.2f}±"
                                 f"{lsw.se(gains['vH'][m]):.2f} share {gains['vH'][m].sum() / tot if tot else np.nan:+.2f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--tag", required=True, choices=("plain", "search"))
    q.add_argument("--variant", default="o4f_v28r", help="blind_search_eval VARIANTS tag for --tag search")
    q.add_argument("--base", default="p17", choices=("p17", "v29", "v30"), help="our acting policy (v29/v30: tags get _b29/_b30)")
    q.add_argument("--opponent", required=True)
    q.add_argument("--seeds", type=int, default=64)
    q.add_argument("--seed0", type=int, default=2_100_000)
    q.add_argument("--chunk", type=int, default=64)
    q.add_argument("--cap", type=int, default=3750)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/q11_v30"))
    q = sub.add_parser("cf")
    q.add_argument("--opponent", required=True)
    q.add_argument("--n-roots", type=int, default=600)
    q.add_argument("--base", default="p17", choices=("p17", "v29", "v30"))
    q.add_argument("--chunk0", type=int, default=-1, help="only roots of this play chunk (its first task index)")
    q.add_argument("--min-tick", type=int, default=0)
    q.add_argument("--select-seed", type=int, default=0)
    q.add_argument("--part", default=None, help="k/m: every m-th selected root from k")
    q.add_argument("--suffix", default="")
    q.add_argument("--reseeds", type=int, default=2)
    q.add_argument("--horizon", type=int, default=750)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/q11_v30"))
    q = sub.add_parser("report")
    q.add_argument("--opponents", default="v29,v28,v25")
    q.add_argument("--base", default="p17", choices=("p17", "v29", "v30"))
    q.add_argument("--out", default=str(REPO / "reports/q11_v30"))
    args = p.parse_args()
    if getattr(args, "device", None) == "":
        from blackout_env.sim.search import default_device

        args.device = default_device()
    {"play": cmd_play, "cf": cmd_cf, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
