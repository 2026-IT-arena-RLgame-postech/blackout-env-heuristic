"""D1: why do the V29/V30 students fail?  (docs/policy_v2_design_ko.md §48.)  Simulator only.

``play``: ``--ours`` (heuristic descriptor or net alias, CPU) plays ``--opponent`` (heuristic or net alias) on
(seed, side) pairs.  Every decision records both teams' positions, classes, held items, scores, events, our
executed moves and the moves of every ``--shadows`` policy for our units (heuristics: own instance per game fed
every observation = a DAgger label on *this* trajectory; nets: stateless).  For heuristic shadows also the
inspection flags of the instance: ``sentries`` (V22 sanctuary sentry this tick), ``staged`` (V27 reserve),
``joining`` (V29 reserve join).  One npz per (ours, opponent, chunk) under ``<out>/raw``.

Two uses: ours = a student with shadow = the teacher (labels on the student's own states, what DAgger sees at
beta ~ 0), and ours = the teacher with shadows = the students (the students' decisions on teacher states).

``agree``: per (phase window x role x situation) the shadow-vs-executor agreement in the training metric's sense
(K = 32 bin of the teacher's heading +- 1 bin, i.e. within 16.9 deg; STAND = zero move), split by teacher STAND
vs move, sentry / staged / other.  ``outcome``: per executor the margin per window, decision-99 Hunters, enemy
Hunter transforms / Hunter-seconds per window, our Hunters near the sanctuary while we have one Hunter.

    PYTHONPATH=. python examples/d1_why.py play --ours p19 --opponent v28 --shadows 'v30;p17' --seeds 32
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

FROZEN = Path(os.environ.get("D1_FROZEN", str(REPO / "checkpoints/distill/frozen")))
NETS = {k: str(FROZEN / v) for k, v in {
    "p16": "p16_v27.pt", "p17": "p17_v28.pt", "p18": "p18_v29.pt", "p18bp2": "p18bp2_v29.pt",
    "p19": "p19_v30.pt"}.items()}
NETS.update(dict(kv.split("=", 1) for kv in os.environ.get("BLACKOUT_NETS", "").split(",") if "=" in kv))
HEUR = {
    "v23": "strategic_v23", "v25": "strategic_v25",
    "v27": "strategic_v27:reserve_stage=1.5",
    "v28": "strategic_v28:reserve_stage=0.5,trade_hold=advance",
    "v29": "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve",
    "v30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
}
F32 = np.float32
COLLECTOR, HUNTER, CARRIER = 0, 1, 2
SANCT = (11.0, 13.0)
PHASES = ((0, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 150))
AGREE_DEG = 360 / 32 * 1.5  # K = 32, act-arc-bins 1: the student's heading within the teacher's bin +- 1


def is_net(name: str) -> bool:
    return name in NETS or name.startswith("net:")


class NoStand:
    """``nostand:<lo>,<hi>:<policy>``: the policy, but every zero move outside [lo, hi) seconds of play becomes a
    dither (a fixed per-unit direction, sign flipping every decision) -- what a student without a trained STAND
    output does where the teacher stands (§43: STAND only trained in 2.5-6 s)."""

    def __init__(self, inner, lo: float, hi: float):
        self.inner, self.lo, self.hi = inner, lo, hi
        self.needs_inputs = getattr(inner, "needs_inputs", True)
        self.tick = {}

    def reset(self, n):
        self.inner.reset(n)
        self.tick = {}

    def moves(self, bsim, idx, teams, inputs):
        from blackout_env.sim import core

        out = self.inner.moves(bsim, idx, teams, inputs)
        for tm in teams:
            mv = out[tm]
            for j, b in enumerate(idx):
                t = self.tick.get((int(b), tm), 0)
                self.tick[(int(b), tm)] = t + 1
                sec = t / 25.0
                if self.lo <= sec < self.hi:
                    continue
                z = np.linalg.norm(mv[j], axis=-1) < 1e-5
                if z.any():
                    ang = (np.arange(5) * 2.399 + int(b) * 0.7)  # fixed per unit and game
                    d = np.stack([np.cos(ang), np.sin(ang)], -1).astype(F32) * (1 if t % 2 == 0 else -1)
                    mv[j][z] = d[z]
        return out


def make_policy(name: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name.startswith("until:"):  # until:<s>:<A>|<B> = A before s seconds of play, B after (sim.batch.HybridPolicy)
        from blackout_env.sim.batch import HybridPolicy

        _, sec, rest = name.split(":", 2)
        a, b = rest.split("|", 1)
        return HybridPolicy(make_policy(a), make_policy(b), switch_ticks=int(round(float(sec) * 50)))
    if name.startswith("nostand:"):
        _, rng, inner = name.split(":", 2)
        lo, hi = (float(x) for x in rng.split(","))
        return NoStand(make_policy(inner), lo, hi)
    if is_net(name):
        return NetPolicy.from_checkpoint(NETS[name] if name in NETS else name[4:], device="cpu")
    return HeuristicPolicy(HEUR.get(name, name))


def sanct_dist(p):
    dx = np.maximum(np.maximum(SANCT[0] - p[..., 0], p[..., 0] - SANCT[1]), 0)
    dy = np.maximum(np.maximum(SANCT[0] - p[..., 1], p[..., 1] - SANCT[1]), 0)
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------------- play

def play_chunk(job):
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, HeuristicPolicy
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState
    from q7_why import exhausted_after, held_info

    ours_name, opp_name, shadow_names, seeds, sides, cap_s, stop_ex, path = job
    t0 = time.time()
    n, cap = len(seeds), cap_s * 25
    sides = np.asarray(sides, np.int64)
    ours, opp = make_policy(ours_name), make_policy(opp_name)
    shadows = [make_policy(s) for s in shadow_names]
    for p in (ours, opp, *shadows):
        p.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    need = any(getattr(p, "needs_inputs", True) for p in (ours, opp, *shadows))
    moves = np.zeros((n, 10, 2), F32)

    def team_moves(pol, live, team_of_game):
        out = np.zeros((len(live), 5, 2), F32)
        sub = tuple(x[: len(live)] for x in bufs)
        for team in (0, 1):
            sel = team_of_game == team
            if sel.any():
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                out[sel] = pol.moves(bs, live[sel], (team,), s_in)[team]
        return out

    S = len(shadows)
    pos = np.zeros((cap + 1, n, 10, 2), np.float16)
    cls = np.zeros((cap + 1, n, 10), np.int8)
    htyp = np.full((cap + 1, n, 10), -1, np.int8)
    score = np.zeros((cap + 1, n, 2), np.int16)
    ownmv = np.zeros((cap, n, 5, 2), np.float16)
    shmv = np.zeros((S, cap, n, 5, 2), np.float16)
    flags = np.zeros((S, cap, n, 5), np.int8)  # bit 0 sentry, bit 1 staged, bit 2 joining (team flag)
    events = []
    end = np.full(n, cap, np.int64)
    stopped = np.zeros(n, bool)

    def snap(t):
        pos[t], cls[t] = bs.uf, bs.ui[:, :, core.U_CLS]
        htyp[t] = held_info(bs)[0]
        score[t] = bs.scores

    snap(0)
    for t in range(cap):
        live = np.nonzero(~bs.over & ~stopped)[0]
        if not len(live):
            break
        abs_before = bs.gf[live, core.GF_ABS].copy()
        if need:
            bs.render(live, *bufs)
        ot = sides[live]
        own = team_moves(ours, live, ot)
        for k, sp in enumerate(shadows):
            shmv[k, t, live] = team_moves(sp, live, ot)
            if isinstance(sp, HeuristicPolicy):
                for j, b in enumerate(live):
                    inst = sp.instances.get((int(b), int(ot[j])))
                    if inst is None:
                        continue
                    base = 5 * int(ot[j])
                    for nm in getattr(inst, "sentries", []) or []:
                        flags[k, t, b, int(nm.split("_")[1]) - base] |= 1
                    for nm in getattr(inst, "staged", []) or []:
                        flags[k, t, b, int(nm.split("_")[1]) - base] |= 2
                    if getattr(inst, "joining", False):
                        flags[k, t, b] |= 4
        oo = team_moves(opp, live, 1 - ot)
        for team in (0, 1):
            mine = (ot == team)[:, None, None]
            moves[live, 5 * team:5 * team + 5] = np.where(mine, own, oo)
        ownmv[t, live] = own
        bs.step(moves, live)
        if stop_ex:
            stopped[exhausted_after(bs, live, abs_before)] = True
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] in (core.EV_KILL, core.EV_TRANSFORM, core.EV_DEPOSIT, core.EV_STEAL):
                    events.append((int(b), t + 1, *map(int, e[1:])))
        snap(t + 1)
        done = live[bs.over[live] | stopped[live]]
        end[done] = np.minimum(end[done], t + 1)
    np.savez_compressed(path, seed=np.asarray(seeds), side=sides, pos=pos, cls=cls, htyp=htyp, score=score,
                        ownmv=ownmv, shmv=shmv, flags=flags, end=end, final=bs.scores,
                        shadows=np.asarray(shadow_names), events=np.asarray(events, np.int64).reshape(-1, 7))
    sc = bs.scores
    lead = np.where(sides == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])
    return path, float(lead.mean()), time.time() - t0


def cmd_play(args):
    raw = Path(args.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    shadows = [s for s in args.shadows.split(";") if s]
    jobs = []
    for opp in args.opponent.split(";"):
        for ours in args.ours.split(";"):
            tasks = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)]
            for c in range(0, len(tasks), args.chunk):
                path = raw / f"{ours}__{opp}__{args.seed0}_{c:04d}.npz"
                if path.exists():
                    continue
                ch = tasks[c:c + args.chunk]
                jobs.append((ours, opp, shadows, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s,
                             args.stop_exhausted, str(path)))
    print(f"{len(jobs)} chunks", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for path, m, wall in pool.imap_unordered(play_chunk, jobs):
            print(f"{Path(path).name} margin {m:+.1f} {wall:.0f}s", flush=True)


def load(out: Path, ours: str, opp: str) -> dict:
    parts = [dict(np.load(p)) for p in sorted((Path(out) / "raw").glob(f"{ours}__{opp}__*.npz"))]
    if not parts:
        raise FileNotFoundError(f"{ours}__{opp}")
    m, off = {}, 0
    for p in parts:
        p["events"] = p["events"].copy()
        p["events"][:, 0] += off
        off += len(p["seed"])
    T = max(p["ownmv"].shape[0] for p in parts)
    for k in parts[0]:
        if k == "shadows":
            m[k] = [str(x) for x in parts[0][k]]
        elif k in ("seed", "side", "final", "end", "events"):
            m[k] = np.concatenate([p[k] for p in parts])
        else:
            ax = 2 if k in ("shmv", "flags") else 1
            m[k] = np.concatenate([p[k] for p in parts], axis=ax)
    return m


# ---------------------------------------------------------------------------------- analysis

def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def unit_frame(D):
    """Flattened per (tick, game, our unit) arrays inside each game's played span."""
    side = D["side"].astype(int)
    n = len(side)
    T = D["ownmv"].shape[0]
    tt = np.arange(T)[:, None]
    live = tt < D["end"][None, :]  # [T, n]
    us = np.stack([np.arange(5) + 5 * s for s in side])  # [n, 5]
    cls = np.take_along_axis(D["cls"][:T], us[None], axis=2)  # [T, n, 5]
    htyp = np.take_along_axis(D["htyp"][:T], us[None], axis=2)
    pos = np.take_along_axis(D["pos"][:T].astype(F32), us[None, :, :, None].repeat(2, -1), axis=2)
    nh = (cls == HUNTER).sum(-1, keepdims=True).repeat(5, -1)
    m = live[:, :, None].repeat(5, -1)
    return dict(mask=m, t=np.broadcast_to(tt[:, :, None], m.shape), cls=cls, htyp=htyp, pos=pos, nh=nh,
                game=np.broadcast_to(np.arange(n)[None, :, None], m.shape))


def moves_agree(a, b):
    """Training-metric agreement of move a with label b: both stand, or both move within AGREE_DEG."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    sa, sb = na < 1e-3, nb < 1e-3
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-9)
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return np.where(sa | sb, sa & sb, ang <= AGREE_DEG), sa, sb, ang


def role_names(cls, htyp, flg):
    r = np.full(cls.shape, "?", dtype=object)
    r[(cls == COLLECTOR) & (htyp < 0)] = "C-empty"
    r[(cls == COLLECTOR) & (htyp >= 0)] = "C-held"
    r[cls == HUNTER] = "H-other"
    r[(cls == HUNTER) & ((flg & 1) > 0)] = "H-sentry"
    r[(cls == COLLECTOR) & ((flg & 2) > 0)] = "C-staged"
    r[(cls == CARRIER) & (htyp < 0)] = "K-empty"
    r[(cls == CARRIER) & (htyp >= 0)] = "K-loaded"
    return r


def agree_table(D, a_mv, b_mv, flg, title):
    """Agreement of a (actor) with b (label) per phase x role, split by the label standing."""
    U = unit_frame(D)
    ok, sa, sb, ang = moves_agree(a_mv, b_mv)
    roles = role_names(U["cls"], U["htyp"], flg)
    m = U["mask"]
    sec = U["t"] / 25.0
    lines = [title, f"  {'phase':7s} {'role':9s} {'n':>7s} {'share':>6s} {'agree':>6s} | {'lab-stand':>9s} {'agr|st':>6s} "
             f"{'act-stand|mv':>12s} {'agr|mv':>6s} {'deg|mv':>6s}"]
    tot = m.sum()
    allr = sorted(set(roles[m]))
    for a, z in PHASES:
        pm = m & (sec >= a) & (sec < z)
        for r in ["ALL"] + allr:
            mm = pm if r == "ALL" else pm & (roles == r)
            cnt = int(mm.sum())
            if cnt < 50:
                continue
            st = mm & sb
            mv = mm & ~sb
            both = mv & ~sa
            lines.append(f"  {a:>3d}-{z:<3d} {r:9s} {cnt:7d} {cnt / tot:6.3f} {ok[mm].mean():6.3f} | {st.sum() / cnt:9.3f} "
                         f"{(ok[st].mean() if st.any() else float('nan')):6.3f} {(sa[mv].mean() if mv.any() else float('nan')):12.3f} "
                         f"{(ok[mv].mean() if mv.any() else float('nan')):6.3f} {(np.median(ang[both]) if both.any() else float('nan')):6.1f}")
    # the training metric: valid labels only (label moving)
    lines.append(f"  training-style agree (label moving only): {ok[m & ~sb].mean():.3f}; label STAND share of unit-ticks "
                 f"{sb[m].mean():.3f}; agree incl. STAND {ok[m].mean():.3f}")
    return "\n".join(lines)


def cmd_agree(args):
    out = Path(args.out)
    txt = []
    for spec in args.pairs.split(";"):
        ours, opp = spec.split("@")
        D = load(out, ours, opp)
        names = D["shadows"]
        own = D["ownmv"].astype(F32)
        # flags: from the first heuristic shadow (teacher) if any
        hk = [k for k, s in enumerate(names) if not is_net(s)]
        flg = D["flags"][hk[0]] if hk else np.zeros(own.shape[:-1], np.int8)
        if not is_net(ours):  # executor = teacher: a net shadow acting on teacher states
            for k, s in enumerate(names):
                if is_net(s):
                    # flags of the teacher itself are not recorded when it executes; use a heuristic shadow of
                    # the same descriptor if given, else none
                    txt.append(agree_table(D, D["shmv"][k].astype(F32), own, flg,
                                           f"== {s} on {ours}'s states vs {opp} (label = {ours}'s executed move)"))
        else:
            for k, s in enumerate(names):
                if not is_net(s):
                    txt.append(agree_table(D, own, D["shmv"][k].astype(F32), D["flags"][k],
                                           f"== {ours} on its own states vs {opp} (label = shadow {s})"))
            for k, s in enumerate(names):
                if is_net(s):
                    txt.append(agree_table(D, D["shmv"][k].astype(F32), own, flg,
                                           f"== shadow {s} vs {ours}'s executed move ({ours} states vs {opp})"))
    s = "\n\n".join(txt)
    print(s)
    if args.save:
        Path(args.save).write_text(s + "\n")


def outcome_rows(D):
    from blackout_env.sim import core

    side = D["side"].astype(int)
    n = len(side)
    sc = D["score"].astype(np.int64)
    T = sc.shape[0] - 1
    lead = np.where(side[None] == 0, sc[..., 0] - sc[..., 1], sc[..., 1] - sc[..., 0])
    for b in range(n):
        lead[D["end"][b] + 1:, b] = lead[D["end"][b], b]
    F = {"margin": lead[-1].astype(float)}
    for a, z in PHASES:
        F[f"m{a}_{z}"] = (lead[min(z * 25, T)] - lead[min(a * 25, T)]).astype(float)
    us = np.stack([np.arange(5) + 5 * s for s in side])
    op = np.stack([np.arange(5) + 5 * (1 - s) for s in side])
    cls = D["cls"]
    F["h99"] = np.array([(cls[min(99, T), b, us[b]] == HUNTER).sum() for b in range(n)], float)
    F["opp_h99"] = np.array([(cls[min(99, T), b, op[b]] == HUNTER).sum() for b in range(n)], float)
    ev = D["events"]
    for a, z in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 150)):
        for who, grp in (("o", us), ("p", op)):
            for nm in ("trH", "dep", "death", "kill"):
                F[f"{nm}_{who}_{a}_{z}"] = np.zeros(n)
    for row in ev:
        b, t, kind, a_, bb, c, d = map(int, row)
        s = t / 25
        w = next(((a, z) for a, z in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 150)) if a <= s < z), None)
        if w is None:
            continue
        suf = f"_{w[0]}_{w[1]}"
        ours = a_ in set(us[b])
        if kind == core.EV_TRANSFORM and bb == HUNTER:
            F[f"trH_{'o' if ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_DEPOSIT:
            F[f"dep_{'o' if ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_KILL:
            victim_ours = bb in set(us[b])
            F[f"death_{'o' if victim_ours else 'p'}{suf}"][b] += 1
    # sanctuary guard: share of 5-30 s ticks with exactly one Hunter of ours where it is within 3 tiles of the sanctuary
    for nm, (a, z) in (("guard1_5_30", (5, 30)), ("guard_any_5_30", (5, 30))):
        F[nm] = np.zeros(n)
    hs = np.zeros(n)
    ohs = np.zeros(n)
    for b in range(n):
        e = int(D["end"][b])
        lo, hi = 5 * 25, min(30 * 25, e)
        c = cls[lo:hi, b, us[b]]
        d = sanct_dist(D["pos"][lo:hi, b, us[b]].astype(F32))
        one = (c == HUNTER).sum(-1) == 1
        near = ((c == HUNTER) & (d < 3.0)).any(-1)
        F["guard1_5_30"][b] = (one & near).sum() / max(one.sum(), 1)
        F["guard_any_5_30"][b] = near.mean() if len(near) else 0
        hs[b] = (cls[1:e + 1, b, us[b]] == HUNTER).sum() / 25
        ohs[b] = (cls[1:e + 1, b, op[b]] == HUNTER).sum() / 25
    F["hunter_s"], F["opp_hunter_s"] = hs, ohs
    F["ticks"] = D["end"].astype(float)
    return F


def cmd_outcome(args):
    out = Path(args.out)
    lines = []
    for opp in args.opponent.split(";"):
        rows = {}
        for ours in args.ours.split(";"):
            try:
                D = load(out, ours, opp)
            except FileNotFoundError:
                continue
            rows[ours] = (D["seed"], D["side"], outcome_rows(D))
        if not rows:
            continue
        ref = args.ref if args.ref in rows else next(iter(rows))
        keys = list(next(iter(rows.values()))[2].keys())
        lines.append(f"== vs {opp}  (n games: " + ", ".join(f"{k} {len(v[0])}" for k, v in rows.items()) + f"; delta = - {ref})")
        lines.append("  " + f"{'feature':18s}" + "".join(f"{k:>18s}" for k in rows))
        rs, rsd, rF = rows[ref]
        rkey = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(rs, rsd))}
        for f in keys:
            cells = []
            for k, (s_, sd_, F) in rows.items():
                v = F[f]
                if k == ref:
                    cells.append(f"{v.mean():+8.2f}         ")
                    continue
                idx = [(i, rkey[(int(a), int(b))]) for i, (a, b) in enumerate(zip(s_, sd_)) if (int(a), int(b)) in rkey]
                dlt = np.array([v[i] - rF[f][j] for i, j in idx])
                per = defaultdict(list)
                for (i, j), x in zip(idx, dlt):
                    per[int(s_[i])].append(x)
                ps = [np.mean(x) for x in per.values()]
                cells.append(f"{v.mean():+8.2f} ({np.mean(ps):+5.1f}±{se(ps):3.1f})")
            lines.append(f"  {f:18s}" + "".join(f"{c:>18s}" for c in cells))
    s = "\n".join(lines)
    print(s)
    if args.save:
        Path(args.save).write_text(s + "\n")


def cmd_join(args):
    """Race join (decisions 60-99): per executor, how often the teacher shadow stages / holds / joins on those
    states, what the executor does with the staged unit, and Hunters at decision 99."""
    out = Path(args.out)
    lines = [f"{'pair':16s} {'n':>4s} {'h99=4':>6s} {'staged@80':>9s} {'T:join any':>10s} {'T:hold any':>10s} "
             f"{'stagedT-stand':>13s} {'exec stand|Tst':>14s} {'exec move|Tst':>13s} {'exec stand|Tmv':>14s} {'agree 60-99':>11s}"]
    for spec in args.pairs.split(";"):
        ours, opp = spec.split("@")
        D = load(out, ours, opp)
        names = D["shadows"]
        k = next((i for i, s in enumerate(names) if s == args.teacher), None)
        if k is None:
            continue
        side = D["side"].astype(int)
        n = len(side)
        own = D["ownmv"].astype(F32)
        sh = D["shmv"][k].astype(F32)
        fl = D["flags"][k]
        us = np.stack([np.arange(5) + 5 * s for s in side])
        h99 = np.array([(D["cls"][99, b, us[b]] == HUNTER).sum() for b in range(n)])
        lo, hi = 60, 99
        st = (fl[lo:hi] & 2) > 0  # [t, n, 5]
        jn = ((fl[lo:hi] & 4) > 0).any(-1)  # [t, n]
        tstand = np.linalg.norm(sh[lo:hi], axis=-1) < 1e-3
        estand = np.linalg.norm(own[lo:hi], axis=-1) < 1e-3
        ok = moves_agree(own[lo:hi], sh[lo:hi])[0]
        stT = st & tstand
        stM = st & ~tstand
        lines.append(f"{spec:16s} {n:4d} {(h99 == 4).mean():6.2f} {st[80 - lo].any(-1).mean():9.2f} {jn.any(0).mean():10.2f} "
                     f"{stT.any((0, 2)).mean():10.2f} {stT.sum() / max(st.sum(), 1):13.3f} "
                     f"{(estand & stT).sum() / max(stT.sum(), 1):14.3f} {(~estand & stT).sum() / max(stT.sum(), 1):13.3f} "
                     f"{(estand & stM).sum() / max(stM.sum(), 1):14.3f} {ok[st].mean() if st.any() else float('nan'):11.3f}")
    # reserve distance profile: our 4th-nearest Collector to the sanctuary (3 walkers + the reserve)
    lines.append("")
    lines.append(f"{'pair':16s} reserve dist p50 at t=70/80/88/94/98 | first T-join tick p25/50/75 | h99=4 | "
                 "h99=4 given T-join<=90 / >90")
    for spec in args.pairs.split(";"):
        ours, opp = spec.split("@")
        D = load(out, ours, opp)
        names = D["shadows"]
        k = next((i for i, s in enumerate(names) if s == args.teacher), None)
        if k is None:
            continue
        side = D["side"].astype(int)
        n = len(side)
        us = np.stack([np.arange(5) + 5 * s for s in side])
        prof = []
        for t in (70, 80, 88, 94, 98):
            r = []
            for b in range(n):
                c = D["cls"][t, b, us[b]]
                d = np.sort(sanct_dist(D["pos"][t, b, us[b]].astype(F32))[c == COLLECTOR])
                r.append(d[3] if len(d) > 3 else np.nan)
            prof.append(np.nanmedian(r))
        jn = ((D["flags"][k][:99] & 4) > 0).any(-1)  # [99, n]
        first = np.array([np.argmax(jn[:, b]) if jn[:, b].any() else 999 for b in range(n)])
        h4 = np.array([(D["cls"][99, b, us[b]] == HUNTER).sum() == 4 for b in range(n)])
        fj = first[first < 999]
        q = np.percentile(fj, [25, 50, 75]) if len(fj) else [np.nan] * 3
        e, l_ = first <= 90, (first > 90) & (first < 999)
        lines.append(f"{spec:16s} {' '.join(f'{x:.2f}' for x in prof)} | {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} | {h4.mean():.2f} | "
                     f"{(h4[e].mean() if e.any() else float('nan')):.2f} (n={int(e.sum())}) / "
                     f"{(h4[l_].mean() if l_.any() else float('nan')):.2f} (n={int(l_.sum())})")
    s = "\n".join(lines)
    print(s)
    if args.save:
        Path(args.save).write_text(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--ours", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--shadows", default="")
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed0", type=int, default=1100000)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--cap-s", type=int, default=150)
    p.add_argument("--stop-exhausted", action="store_true", default=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="reports/distill_why")
    q = sub.add_parser("agree")
    q.add_argument("--pairs", required=True, help="'ours@opp;...'")
    q.add_argument("--out", default="reports/distill_why")
    q.add_argument("--save", default="")
    r = sub.add_parser("outcome")
    r.add_argument("--ours", required=True)
    r.add_argument("--opponent", required=True)
    r.add_argument("--ref", default="v30")
    r.add_argument("--out", default="reports/distill_why")
    r.add_argument("--save", default="")
    j = sub.add_parser("join")
    j.add_argument("--pairs", required=True)
    j.add_argument("--teacher", default="v30")
    j.add_argument("--out", default="reports/distill_why")
    j.add_argument("--save", default="")
    args = ap.parse_args()
    {"play": cmd_play, "agree": cmd_agree, "outcome": cmd_outcome, "join": cmd_join}[args.cmd](args)


if __name__ == "__main__":
    main()
