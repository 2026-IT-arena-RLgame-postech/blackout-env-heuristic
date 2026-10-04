"""Q7: what does the student P16 do differently from its teacher V27 that wins?
(docs/policy_v2_design_ko.md §36.)  Simulator only, full matches (cap 150 s, ``--stop-exhausted`` as the P16 eval).

``play``: ``--ours`` (heuristic descriptor or a net alias ``p16`` / ``p14b`` / ``p11`` / ``net:<path>``, CPU) plays
``--opponent`` (a heuristic descriptor) on (seed, side) pairs; every decision records both teams' positions,
classes, held item (type, amount), scores, all executed moves, sim events, and the move a *shadow* teacher
(``--shadow``, default V27(1.5)) would make for each of our units (its own instance per game, fed every
observation, i.e. a DAgger label).  One npz per (ours, opponent, chunk) under ``<out>/raw``.

``analyze``: ours vs a reference (``--ref``, default the V27 mirror) on the same opponent and (seed, side):
margin gained per time window, race / second race, reserve distance at decision 99, per-window event counts
(kills / deaths by class / deposits / steals / transforms), Hunter-seconds; and where (phase x role) the
student's move departs from the shadow teacher's.

``cf``: counterfactual swaps (the Q6 way).  Roots (match, tick) are sampled from the recorded ``--ours`` matches;
each match is replayed with the recorded moves of our team (the opponent and the shadow re-act and are checked
against the recording), the state and both heuristic instances are copied at every root, and every root is
continued in parallel arms to the end of the match: ``base`` (the student for the rest), ``base2`` (the same:
a batch-rounding noise control) and ``swap:<units>`` (the shadow teacher's move for those units for ``--window``
decisions, then the student).  Value = final margin(arm) - final margin(base).  -> ``<out>/cf_<ours>_<opp>.jsonl``.

    PYTHONPATH=. python examples/q7_why.py play --ours p16 --opponent 'strategic_v27:reserve_stage=1.5' --seeds 64
    PYTHONPATH=. python examples/q7_why.py analyze --ours p16
    PYTHONPATH=. python examples/q7_why.py cf --ours p16 --n-roots 400
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NETS = {"p11": "checkpoints/distill/frozen/p11_v22.pt", "p14b": "checkpoints/distill/frozen/p14_v25.pt",
        "p16": "checkpoints/distill/frozen/p16_v27.pt"}
NETS.update(dict(kv.split("=", 1) for kv in os.environ.get("BLACKOUT_NETS", "").split(",") if "=" in kv))
V27 = "strategic_v27:reserve_stage=1.5"
SANCT = (11.0, 13.0)
F32 = np.float32
COLLECTOR, HUNTER, CARRIER = 0, 1, 2
WINDOWS = ((0, 5), (5, 10), (10, 30), (30, 60), (60, 150))


def short(name: str) -> str:
    return name.replace("strategic_", "v").replace(":reserve_stage=", "r").replace(":", "_").replace("=", "")[:60]


def _policy(name: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name in NETS or name.startswith("net:"):
        path = NETS[name] if name in NETS else name[4:]
        return NetPolicy.from_checkpoint(str(REPO / path), device="cpu")
    return HeuristicPolicy(name)


def sanct_dist(p):
    dx = np.maximum(np.maximum(SANCT[0] - p[..., 0], p[..., 0] - SANCT[1]), 0)
    dy = np.maximum(np.maximum(SANCT[0] - p[..., 1], p[..., 1] - SANCT[1]), 0)
    return np.hypot(dx, dy)


def held_info(bs):
    from blackout_env.sim import core

    h = bs.ui[:, :, core.U_HELD]
    typ = np.full(h.shape, -1, np.int8)
    amt = np.zeros(h.shape, np.int8)
    b, u = np.nonzero(h >= 0)
    typ[b, u] = bs.it[b, h[b, u], core.I_TYPE]
    amt[b, u] = bs.it[b, h[b, u], core.I_AMT]
    return typ, amt


class Game:
    """Steps a BatchSim with our policy / opponent / shadow; optional own-move override."""

    def __init__(self, bs, sides, ours, opp, shadow):
        from blackout_env.sim.batch import N_AGENT_COLS, N_CH
        from blackout_env.sim.mapdata import H, W

        n = bs.n
        self.bs, self.sides, self.ours, self.opp, self.shadow = bs, np.asarray(sides, np.int64), ours, opp, shadow
        self.bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
        self.moves = np.zeros((n, 10, 2), F32)
        self.need = any(getattr(p, "needs_inputs", True) for p in (ours, opp) if p is not None)

    def _team_moves(self, pol, live, team_of_game):
        """{game j in live: [5, 2]} for the team ``team_of_game[j]``."""
        out = np.zeros((len(live), 5, 2), F32)
        sub = tuple(x[: len(live)] for x in self.bufs)
        for team in (0, 1):
            sel = team_of_game == team
            if sel.any():
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                out[sel] = pol.moves(self.bs, live[sel], (team,), s_in)[team]
        return out

    def step(self, live, own_override=None, want_shadow=True):
        """One decision; ``own_override(live, own, shadow) -> own`` may replace our moves.  Returns (own, shadow)."""
        if self.need:
            self.bs.render(live, *self.bufs)
        ours_team = self.sides[live]
        own = self._team_moves(self.ours, live, ours_team) if self.ours is not None else None
        sh = self._team_moves(self.shadow, live, ours_team) if (self.shadow is not None and want_shadow) else None
        if own_override is not None:
            own = own_override(live, own, sh)
        oo = self._team_moves(self.opp, live, 1 - ours_team)
        for team in (0, 1):
            mine = (ours_team == team)[:, None, None]
            self.moves[live, 5 * team:5 * team + 5] = np.where(mine, own, oo)
        self.bs.step(self.moves, live)
        return own, sh


def exhausted_after(bs, live, abs_before):
    from blackout_env.sim import core
    from blackout_env.sim.batch import batteries_in_play

    fired = live[(bs.gf[live, core.GF_ABS] < abs_before) & ~bs.over[live]]
    if not len(fired):
        return fired
    return fired[batteries_in_play(bs, fired) == 0]


# ---------------------------------------------------------------------------------- play

def play_chunk(job):
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import core
    from blackout_env.sim.batch import BatchSim, HeuristicPolicy
    from blackout_env.sim.state import SimState

    ours_name, opp_name, shadow_name, seeds, sides, cap_s, stop_ex, path = job
    t0 = time.time()
    n, cap = len(seeds), cap_s * 25
    sides = np.asarray(sides, np.int64)
    ours, opp = _policy(ours_name), HeuristicPolicy(opp_name)
    shadow = HeuristicPolicy(shadow_name) if shadow_name else None
    for p in (ours, opp, shadow):
        if p is not None:
            p.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    gm = Game(bs, sides, ours, opp, shadow)
    pos = np.zeros((cap + 1, n, 10, 2), np.float16)
    cls = np.zeros((cap + 1, n, 10), np.int8)
    htyp = np.full((cap + 1, n, 10), -1, np.int8)
    hamt = np.zeros((cap + 1, n, 10), np.int8)
    score = np.zeros((cap + 1, n, 2), np.int16)
    allmv = np.zeros((cap, n, 10, 2), np.float16)
    ownmv = np.zeros((cap, n, 5, 2), F32)  # exact (for the cf replay)
    shmv = np.zeros((cap, n, 5, 2), np.float16)
    events = []
    end = np.full(n, cap, np.int64)
    stopped = np.zeros(n, bool)

    def snap(t):
        pos[t], cls[t] = bs.uf, bs.ui[:, :, core.U_CLS]
        htyp[t], hamt[t] = held_info(bs)
        score[t] = bs.scores

    snap(0)
    for t in range(cap):
        live = np.nonzero(~bs.over & ~stopped)[0]
        if not len(live):
            break
        abs_before = bs.gf[live, core.GF_ABS].copy()
        own, sh = gm.step(live)
        ownmv[t, live] = own
        if sh is not None:
            shmv[t, live] = sh
        allmv[t, live] = gm.moves[live]
        if stop_ex:
            stopped[exhausted_after(bs, live, abs_before)] = True
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] in (core.EV_KILL, core.EV_TRANSFORM, core.EV_PICKUP, core.EV_STEAL, core.EV_DEPOSIT,
                            core.EV_ABSORB, core.EV_LOST):
                    events.append((int(b), t + 1, *map(int, e[1:])))
        snap(t + 1)
        done = live[bs.over[live] | stopped[live]]
        end[done] = np.minimum(end[done], t + 1)
    np.savez_compressed(path, seed=np.asarray(seeds), side=sides, pos=pos, cls=cls, htyp=htyp, hamt=hamt,
                        score=score, allmv=allmv, ownmv=ownmv, shmv=shmv, end=end, final=bs.scores,
                        events=np.asarray(events, np.int64).reshape(-1, 7), exhausted=stopped)
    sc = bs.scores
    lead = np.where(sides == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])
    return path, float(lead.mean()), time.time() - t0


def cmd_play(args):
    raw = Path(args.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for opp in args.opponent.split(";"):
        for ours in args.ours.split(";"):
            tasks = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)]
            for c in range(0, len(tasks), args.chunk):
                path = raw / f"{short(ours)}__{short(opp)}__{args.seed0}_{c:04d}.npz"
                if path.exists():
                    continue
                ch = tasks[c:c + args.chunk]
                jobs.append((ours, opp, args.shadow, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s,
                             args.stop_exhausted, str(path)))
    print(f"{len(jobs)} chunks", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for path, m, wall in pool.imap_unordered(play_chunk, jobs):
            print(f"{Path(path).name} margin {m:+.1f} {wall:.0f}s", flush=True)


def load(out: Path, ours: str, opp: str) -> dict:
    parts = [dict(np.load(p)) for p in sorted((out / "raw").glob(f"{short(ours)}__{short(opp)}__*.npz"))]
    if not parts:
        raise FileNotFoundError(f"{short(ours)}__{short(opp)}")
    m, off = {}, 0
    for p in parts:
        p["events"] = p["events"].copy()
        p["events"][:, 0] += off
        off += len(p["seed"])
    for k in parts[0]:
        if k in ("seed", "side", "final", "end", "events", "exhausted"):
            m[k] = np.concatenate([p[k] for p in parts])
        else:
            m[k] = np.concatenate([p[k] for p in parts], axis=1)
    return m


# ---------------------------------------------------------------------------------- analyze

def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def seed_se(vals, seeds):
    """mean, SE over seeds (sides averaged first)."""
    per = defaultdict(list)
    for v, s in zip(vals, seeds):
        per[int(s)].append(v)
    ps = [np.mean(x) for x in per.values()]
    return float(np.mean(ps)), se(ps)


def match_features(D):
    """Per-match dict of features (our perspective)."""
    from blackout_env.sim import core

    n = len(D["seed"])
    side = D["side"].astype(int)
    sc = D["score"].astype(np.int64)  # [T+1, n, 2]
    T = sc.shape[0] - 1
    end = D["end"]
    F = defaultdict(lambda: np.zeros(n))
    lead = np.where(side[None] == 0, sc[..., 0] - sc[..., 1], sc[..., 1] - sc[..., 0])  # [T+1, n]
    for b in range(n):
        lead[end[b] + 1:, b] = lead[end[b], b]
    F["margin"] = lead[-1].astype(float)
    for a, z in WINDOWS:
        F[f"m{a}_{z}"] = (lead[min(z * 25, T)] - lead[min(a * 25, T)]).astype(float)
    cls = D["cls"]
    ours_u = np.stack([np.arange(5) + 5 * s for s in side])  # [n, 5]
    opp_u = np.stack([np.arange(5) + 5 * (1 - s) for s in side])
    first = np.full((n, 2), -1)
    second = np.full((n, 2), -1)
    for b in range(n):
        for k, us in enumerate((ours_u[b], opp_u[b])):
            h = (cls[:, b, us] == HUNTER).any(axis=1)
            idx = np.nonzero(h)[0]
            first[b, k] = idx[0] if len(idx) else -1
    F["first_ours"], F["first_opp"] = first[:, 0], first[:, 1]
    hs = np.zeros((n, 2))
    for b in range(n):
        e = end[b]
        hs[b, 0] = (cls[1:e + 1, b, ours_u[b]] == HUNTER).sum() / 25
        hs[b, 1] = (cls[1:e + 1, b, opp_u[b]] == HUNTER).sum() / 25
    F["hunter_s"], F["opp_hunter_s"] = hs[:, 0], hs[:, 1]
    # reserve: our nearest unit > 1 tile from the sanctuary at decision 99
    pos = D["pos"].astype(np.float32)
    d99 = sanct_dist(pos[min(99, T)])  # [n, 10]
    F["reserve99"] = np.array([np.min(np.where(d99[b, ours_u[b]] > 1.0, d99[b, ours_u[b]], np.inf)) for b in range(n)])
    F["opp_reserve99"] = np.array([np.min(np.where(d99[b, opp_u[b]] > 1.0, d99[b, opp_u[b]], np.inf)) for b in range(n)])
    F["ticks"] = end.astype(float)
    # events per window
    ev = D["events"]
    keys = []
    for a, z in WINDOWS:
        for nm in ("kill", "death", "death_c", "death_h", "death_k", "dep", "steal", "pick", "tr_h", "tr_k", "lost"):
            for who in ("o", "p"):
                keys.append(f"{nm}_{who}_{a}_{z}")
    for k in keys:
        F[k] = np.zeros(n)
    win_of = np.zeros(T + 2, int)
    for wi, (a, z) in enumerate(WINDOWS):
        win_of[a * 25:min(z * 25, T + 1) + 1] = wi
    for row in ev:
        b, t, kind, a, bb, c, d = map(int, row)
        a_, z_ = WINDOWS[win_of[min(t, T + 1)]]
        suf = f"_{a_}_{z_}"
        ours = set(ours_u[b])
        if kind == core.EV_KILL:
            victim, vcls = bb, c
            who = "o" if victim in ours else "p"
            F[f"death_{who}{suf}"][b] += 1
            F[f"death_{'chk'[vcls]}_{who}{suf}"][b] += 1
            if a >= 0:
                F[f"kill_{'o' if a in ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_DEPOSIT:
            F[f"dep_{'o' if a in ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_STEAL:
            F[f"steal_{'o' if a in ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_PICKUP:
            F[f"pick_{'o' if a in ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_TRANSFORM:
            F[f"tr_{'h' if bb == HUNTER else 'k'}_{'o' if a in ours else 'p'}{suf}"][b] += 1
        elif kind == core.EV_LOST:
            F[f"lost_{'o' if a in ours else 'p'}{suf}"][b] += c
    # second race: first Hunter transform later than our first Hunter + 8 decisions
    for b in range(n):
        for k, us in enumerate((ours_u[b], opp_u[b])):
            if first[b, k] < 0:
                continue
            sel = ev[(ev[:, 0] == b) & (ev[:, 2] == core.EV_TRANSFORM) & (ev[:, 4] == HUNTER)
                     & np.isin(ev[:, 3], us) & (ev[:, 1] > first[b, k] + 8)]
            second[b, k] = sel[:, 1].min() if len(sel) else -1
    F["second_ours"], F["second_opp"] = second[:, 0], second[:, 1]
    return dict(F)


def wtl(a, b):
    x = np.where(np.asarray(a) >= 0, a, 10 ** 9)
    y = np.where(np.asarray(b) >= 0, b, 10 ** 9)
    s = np.sign(y - x)
    return int((s > 0).sum()), int((s == 0).sum()), int((s < 0).sum())


def angle_dev(mv, sh):
    """Angle (deg) between moves; stops: one of them zero."""
    na, nb = np.linalg.norm(mv, axis=-1), np.linalg.norm(sh, axis=-1)
    c = (mv * sh).sum(-1) / np.maximum(na * nb, 1e-9)
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return ang, na < 1e-3, nb < 1e-3


def role_of(cls, htyp):
    """per unit role label: C0 (empty Collector) / Cb (Collector with battery) / Cs (special) / H / K."""
    out = np.full(cls.shape, "?", dtype=object)
    out[(cls == COLLECTOR) & (htyp < 0)] = "C-empty"
    out[(cls == COLLECTOR) & (htyp == 0)] = "C-batt"
    out[(cls == COLLECTOR) & (htyp > 0)] = "C-special"
    out[cls == HUNTER] = "Hunter"
    out[(cls == CARRIER) & (htyp < 0)] = "K-empty"
    out[(cls == CARRIER) & (htyp >= 0)] = "K-loaded"
    return out


def teacher_dev_table(D, max_t=3750):
    """Student vs shadow teacher move per (phase, role): n, mean angle, frac > 30 deg, stop rates."""
    side = D["side"].astype(int)
    n = len(side)
    T = min(D["shmv"].shape[0], max_t)
    rows = defaultdict(lambda: [0, 0.0, 0, 0, 0])
    for b in range(n):
        e = min(int(D["end"][b]), T)
        us = np.arange(5) + 5 * side[b]
        mv = D["ownmv"][:e, b].astype(np.float32)
        sh = D["shmv"][:e, b].astype(np.float32)
        ang, s_st, t_st = angle_dev(mv, sh)
        roles = role_of(D["cls"][:e, b][:, us], D["htyp"][:e, b][:, us])
        tt = np.arange(e)[:, None].repeat(5, 1)
        for wi, (a, z) in enumerate(WINDOWS):
            m = (tt >= a * 25) & (tt < z * 25)
            for r in np.unique(roles[m]):
                mm = m & (roles == r)
                both = mm & ~s_st & ~t_st
                acc = rows[(f"{a}-{z}", r)]
                acc[0] += int(mm.sum())
                acc[1] += float(ang[both].sum())
                acc[2] += int((ang[both] > 30).sum())
                acc[3] += int((mm & s_st & ~t_st).sum())
                acc[4] += int((mm & t_st & ~s_st).sum())
    lines = [f"{'phase':8s} {'role':10s} {'n':>8s} {'mean_deg':>8s} {'>30deg':>7s} {'stu_stop':>8s} {'tea_stop':>8s}"]
    for (ph, r), (cnt, s, big, ss, ts) in sorted(rows.items(), key=lambda kv: (int(kv[0][0].split('-')[0]), kv[0][1])):
        nb = max(cnt - ss - ts, 1)
        lines.append(f"{ph:8s} {r:10s} {cnt:8d} {s / nb:8.1f} {big / nb:7.3f} {ss / max(cnt, 1):8.3f} {ts / max(cnt, 1):8.3f}")
    return "\n".join(lines)


def cmd_analyze(args):
    out = Path(args.out)
    lines = []
    for opp in args.opponent.split(";"):
        A = load(out, args.ours, opp)
        B = load(out, args.ref, opp)
        FA, FB = match_features(A), match_features(B)
        ka = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(A["seed"], A["side"]))}
        kb = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(B["seed"], B["side"]))}
        keys = sorted(set(ka) & set(kb))
        ia, ib = np.array([ka[k] for k in keys]), np.array([kb[k] for k in keys])
        seeds = [k[0] for k in keys]
        lines.append(f"=== {args.ours} vs {opp}  (ref {args.ref}, {len(keys)} matches)")
        for f in ["margin"] + [f"m{a}_{z}" for a, z in WINDOWS] + ["hunter_s", "opp_hunter_s", "ticks"]:
            da = FA[f][ia]
            db = FB[f][ib]
            m, s = seed_se(da - db, seeds)
            lines.append(f"  {f:14s} ours {da.mean():+7.2f}  ref {db.mean():+7.2f}  diff {m:+7.2f} ± {s:5.2f}")
        for nm, a, b in (("race", "first_ours", "first_opp"), ("2nd race", "second_ours", "second_opp")):
            lines.append(f"  {nm:10s} w/t/l ours {wtl(FA[a][ia], FA[b][ia])}  ref {wtl(FB[a][ib], FB[b][ib])}")
        fa, fb = FA["first_ours"][ia], FB["first_ours"][ib]
        lines.append(f"  first Hunter ours: 99 {int((fa == 99).sum())} 100 {int((fa == 100).sum())} other {int(((fa != 99) & (fa != 100)).sum())}"
                     f" | ref 99 {int((fb == 99).sum())}; opp first 99: {int((FA['first_opp'][ia] == 99).sum())} / {int((FB['first_opp'][ib] == 99).sum())}")
        for f in ("reserve99", "opp_reserve99"):
            lines.append(f"  {f:14s} median ours {np.median(FA[f][ia]):.2f} (p10 {np.quantile(FA[f][ia], .1):.2f})  ref {np.median(FB[f][ib]):.2f} (p10 {np.quantile(FB[f][ib], .1):.2f})")
        s2a, s2b = FA["second_ours"][ia], FB["second_ours"][ib]
        lines.append(f"  2nd Hunter decision median ours {np.median(s2a[s2a >= 0]) if (s2a >= 0).any() else -1:.0f} ref {np.median(s2b[s2b >= 0]) if (s2b >= 0).any() else -1:.0f}; "
                     f"opp {np.median(FA['second_opp'][ia][FA['second_opp'][ia] >= 0]) if (FA['second_opp'][ia] >= 0).any() else -1:.0f} / "
                     f"{np.median(FB['second_opp'][ib][FB['second_opp'][ib] >= 0]) if (FB['second_opp'][ib] >= 0).any() else -1:.0f}")
        lines.append("  events per match (ours - ref, by window):")
        hdr = "    " + f"{'event':10s}" + "".join(f"{f'{a}-{z}s':>16s}" for a, z in WINDOWS)
        lines.append(hdr)
        for nm in ("kill", "death", "death_c", "death_h", "death_k", "dep", "steal", "pick", "tr_h", "tr_k", "lost"):
            for who in ("o", "p"):
                cells = []
                for a, z in WINDOWS:
                    k = f"{nm}_{who}_{a}_{z}"
                    m, s = seed_se(FA[k][ia] - FB[k][ib], seeds)
                    cells.append(f"{FA[k][ia].mean():5.2f}{m:+6.2f}±{s:4.2f}")
                lines.append(f"    {nm + '_' + ('our' if who == 'o' else 'opp'):10s}" + "".join(f"{c:>16s}" for c in cells))
        # margin gain split by race outcomes
        r_a = np.sign(np.where(FA["first_opp"][ia] >= 0, FA["first_opp"][ia], 1e9) - np.where(FA["first_ours"][ia] >= 0, FA["first_ours"][ia], 1e9))
        s_a = np.sign(np.where(FA["second_opp"][ia] >= 0, FA["second_opp"][ia], 1e9) - np.where(FA["second_ours"][ia] >= 0, FA["second_ours"][ia], 1e9))
        s_b = np.sign(np.where(FB["second_opp"][ib] >= 0, FB["second_opp"][ib], 1e9) - np.where(FB["second_ours"][ib] >= 0, FB["second_ours"][ib], 1e9))
        dm = FA["margin"][ia] - FB["margin"][ib]
        lines.append("  margin diff by 2nd race (ours, ref) outcome:  [n, mean diff, share of total]")
        tot = dm.sum()
        for x in (1, 0, -1):
            for y in (1, 0, -1):
                m = (s_a == x) & (s_b == y)
                if m.any():
                    lines.append(f"    ours {'WTL'[1 - x]} ref {'WTL'[1 - y]}: n={int(m.sum()):4d} {dm[m].mean():+7.2f} share {dm[m].sum() / tot if tot else 0:+.2f}")
        lines.append(f"  race tied (ours): {int((r_a == 0).sum())}")
        if "shmv" in A and np.abs(A["shmv"]).sum() > 0:
            lines.append("  student vs shadow-teacher moves (ours=" + args.ours + "):")
            lines.append("    " + teacher_dev_table(A).replace("\n", "\n    "))
        if "shmv" in B and np.abs(B["shmv"]).sum() > 0:
            ang, a_st, t_st = angle_dev(B["ownmv"].astype(np.float32), B["shmv"].astype(np.float32))
            lines.append(f"  ref sanity: ref vs shadow max angle {np.nanmax(ang):.2f} deg (should be 0 if ref = shadow)")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    (out / f"analyze_{short(args.ours)}.txt").write_text(txt + "\n")


# ---------------------------------------------------------------------------------- counterfactual

def cmd_cf(args):
    import torch

    torch.set_num_threads(args.threads)
    from blackout_env.sim import core
    from blackout_env.sim.batch import BatchSim, HeuristicPolicy
    from blackout_env.sim.state import SimState

    out = Path(args.out)
    D = load(out, args.ours, args.opponent)
    rng = np.random.default_rng(args.select_seed)
    n = len(D["seed"])
    # roots: uniform over (match, tick) within [tmin, tmax) decisions and before the match end
    roots = []
    lo, hi = args.tmin, args.tmax
    while len(roots) < args.n_roots:
        b = int(rng.integers(n))
        e = int(D["end"][b])
        if e - 1 <= lo:
            continue
        t = int(rng.integers(lo, min(hi, e - 1)))
        roots.append((b, t))
    if args.part is not None:
        k, m = map(int, args.part.split("/"))
        roots = roots[k::m]
    arms = ["base", "base2"] + [f"swap:{u}" for u in args.units.split(";")]
    path = out / f"cf_{short(args.ours)}__{short(args.opponent)}{args.suffix}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["game"], r["tick"]))
    roots = [r for r in roots if r not in done]
    print(f"{len(roots)} roots x {len(arms)} arms -> {path}", flush=True)
    by_game = defaultdict(list)
    for b, t in roots:
        by_game[b].append(t)
    ours_pol = _policy(args.ours)
    games = sorted(by_game)
    t_all = time.time()
    for c0 in range(0, len(games), args.replay_chunk):
        gch = games[c0:c0 + args.replay_chunk]
        sides = D["side"][gch].astype(np.int64)
        seeds = D["seed"][gch]
        bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
        opp, shadow = HeuristicPolicy(args.opponent), HeuristicPolicy(args.shadow)
        opp.reset(len(gch))
        shadow.reset(len(gch))
        gm = Game(bs, sides, None, opp, shadow)
        want = defaultdict(list)
        for j, b in enumerate(gch):
            for t in by_game[b]:
                want[t].append(j)
        snaps = []
        tmax = max(want)
        mism = 0
        stopped = np.zeros(len(gch), bool)
        for t in range(tmax + 1):
            for j in want.get(t, []):
                b = gch[j]
                mism += int(np.abs(bs.uf[j] - D["pos"][t, b].astype(np.float32)).max() > 0.01)
                snaps.append((b, t, bs.state(j), copy.deepcopy(opp.instances.get((j, 1 - int(sides[j])))),
                              copy.deepcopy(shadow.instances.get((j, int(sides[j])))), int(sides[j])))
            if t == tmax:
                break
            live = np.nonzero(~bs.over & ~stopped)[0]
            rec = D["ownmv"][t, np.asarray(gch)[live]]
            abs_before = bs.gf[live, core.GF_ABS].copy()
            gm.step(live, own_override=lambda lv, own, sh, rec=rec: rec, want_shadow=True)
            if args.stop_exhausted:
                stopped[exhausted_after(bs, live, abs_before)] = True
        # ---- arms
        sims, teams, arm_of, root_of, o_inst, s_inst = [], [], [], [], {}, {}
        for r, (b, t, st, oi, si, side) in enumerate(snaps):
            for a in arms:
                q = len(sims)
                sims.append(st)
                teams.append(side)
                arm_of.append(a)
                root_of.append(r)
                o_inst[(q, 1 - side)] = copy.deepcopy(oi)
                s_inst[(q, side)] = copy.deepcopy(si)
        N = len(sims)
        cb = BatchSim(sims)
        teams = np.asarray(teams, np.int64)
        copp, cshadow = HeuristicPolicy(args.opponent), HeuristicPolicy(args.shadow)
        copp.instances, cshadow.instances = o_inst, s_inst
        ours_pol.reset(N)
        cg = Game(cb, teams, ours_pol, copp, cshadow)
        t0s = np.array([snaps[root_of[q]][1] for q in range(N)])
        mask = np.zeros((N, 5), bool)
        for q, a in enumerate(arm_of):
            if a.startswith("swap:"):
                spec = a[5:]
                if spec == "all":
                    mask[q] = True
                elif spec[0].isdigit():  # explicit own unit indices 0-4
                    for u in spec.split(","):
                        mask[q, int(u)] = True
        # resolve role-named unit sets at the root: "c<k>" (our k-th nearest Collector to the sanctuary; in the
        # race c1-c3 are the walkers and c4 the reserve), "hunters", "carriers", "collectors"
        for q, a in enumerate(arm_of):
            if not a.startswith("swap:") or a[5:] == "all" or a[5:][0].isdigit():
                continue
            st = sims[q]
            us = np.arange(5) + 5 * teams[q]
            c = st.ui[us, core.U_CLS]
            d = sanct_dist(st.uf[us])
            sel = np.zeros(5, bool)
            for tok in a[5:].split(","):
                if tok == "hunters":
                    sel |= c == HUNTER
                elif tok == "carriers":
                    sel |= c == CARRIER
                elif tok == "collectors":
                    sel |= c == COLLECTOR
                elif tok[0] == "c" and tok[1:].isdigit():  # k-th nearest Collector to the sanctuary
                    cc = np.nonzero(c == COLLECTOR)[0]
                    order = cc[np.argsort(d[cc], kind="stable")]
                    if len(order) >= int(tok[1:]):
                        sel[order[int(tok[1:]) - 1]] = True
            mask[q] = sel
        window = args.window
        cstop = np.zeros(N, bool)

        def override(lv, own, sh):
            if step_i[0] >= window:
                return own
            return np.where(mask[lv][:, :, None], sh, own)

        step_i = [0]
        t1 = time.time()
        cap = args.cap_s * 25
        for k in range(cap):
            live = np.nonzero(~cb.over & ~cstop & (t0s + k < cap))[0]
            if not len(live):
                break
            abs_before = cb.gf[live, core.GF_ABS].copy()
            step_i[0] = k
            cg.step(live, own_override=override, want_shadow=True)
            if args.stop_exhausted:
                cstop[exhausted_after(cb, live, abs_before)] = True
        sc = cb.scores
        fin = np.where(teams == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])
        with path.open("a") as fh:
            for r, (b, t, st, oi, si, side) in enumerate(snaps):
                qs = [q for q in range(N) if root_of[q] == r]
                base = fin[qs[0]]
                lead0 = int(st.g[core.G_SCORE_A] - st.g[core.G_SCORE_B]) * (1 if side == 0 else -1)
                us = np.arange(5) + 5 * side
                row = {"game": int(b), "tick": int(t), "seed": int(D["seed"][b]), "side": side,
                       "recorded_final": int((D["final"][b, side] - D["final"][b, 1 - side])),
                       "lead0": lead0, "base": int(base),
                       "cls": st.ui[us, core.U_CLS].tolist(), "dsan": [round(float(x), 2) for x in sanct_dist(st.uf[us])],
                       "arms": {arm_of[q]: int(fin[q] - base) for q in qs},
                       "masks": {arm_of[q]: mask[q].astype(int).tolist() for q in qs if arm_of[q].startswith("swap:")}}
                fh.write(json.dumps(row) + "\n")
        print(f"games {c0 + len(gch)}/{len(games)}: {len(snaps)} roots, replay mismatches {mism}, "
              f"arms {time.time() - t1:.0f}s, total {time.time() - t_all:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--ours", default="p16", help="';'-separated")
    q.add_argument("--opponent", default=V27, help="';'-separated heuristic descriptors")
    q.add_argument("--shadow", default=V27, help="teacher labelled for our units ('' = none)")
    q.add_argument("--seeds", type=int, default=64)
    q.add_argument("--seed0", type=int, default=1_300_000)
    q.add_argument("--chunk", type=int, default=16)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--stop-exhausted", action="store_true")
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--out", default=str(REPO / "reports/q7_v28"))
    q = sub.add_parser("analyze")
    q.add_argument("--ours", default="p16")
    q.add_argument("--ref", default=V27)
    q.add_argument("--opponent", default=V27)
    q.add_argument("--out", default=str(REPO / "reports/q7_v28"))
    q = sub.add_parser("cf")
    q.add_argument("--ours", default="p16")
    q.add_argument("--opponent", default=V27)
    q.add_argument("--shadow", default=V27)
    q.add_argument("--n-roots", type=int, default=200)
    q.add_argument("--tmin", type=int, default=0)
    q.add_argument("--tmax", type=int, default=1500)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--units", default="all;c4;hunters;carriers", help="';'-separated unit sets: all, c<k>, hunters, carriers, collectors (comma = union)")
    q.add_argument("--select-seed", type=int, default=0)
    q.add_argument("--part", default=None, help="k/m: this process takes roots k::m")
    q.add_argument("--suffix", default="")
    q.add_argument("--replay-chunk", type=int, default=8)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--stop-exhausted", action="store_true")
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--out", default=str(REPO / "reports/q7_v28"))
    args = p.parse_args()
    {"play": cmd_play, "analyze": cmd_analyze, "cf": cmd_cf}[args.cmd](args)


if __name__ == "__main__":
    main()
