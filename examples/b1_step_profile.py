"""B1: where a batched leaf-phase decision spends its CPU (core.step_kernel by sub-phase, and the render).

``capture``: N mid-match games (P20 x P20 for --warm decisions, then the rollout net v28d256 both sides for
--horizon decisions = a search leaf phase), saving the start states and every decision's moves.
``profile``: replays the recorded moves single-threaded and times
  step         the full ``batch_step_seq`` (the reference)
  move_math    per tick and unit, ``_normalized`` + ``_speed`` + the two ``_clamp_axis`` (pure; on the state at
               each decision -- the positions are not written)
  interact     ``_process_interactions``' neighbour scan + overlap tests without acting (pure)
  stat         ``_speed`` + ``_half_width`` alone (the modifier-list loops)
  world        ``world_update`` on a copy (timers, spawner)
  render_raw   ``render_raw_state`` + ``render_raw_map``
  render_pk    the whole ``batch_render_packed_seq`` (raw + packed two-view map + units)
each per game-decision, best of --reps.

    PYTHONPATH=. python examples/b1_step_profile.py capture --games 1024 --out /tmp/b1_states.npz
    PYTHONPATH=. python examples/b1_step_profile.py profile --data /tmp/b1_states.npz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.batch import _SIM_ARRAYS, BatchSim, _game, _game_S  # noqa: E402
from blackout_env.sim.mapdata import H, W  # noqa: E402


def cmd_capture(a):
    import torch

    import blind_search_eval as bse
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.search import default_device

    torch.set_num_threads(1)
    dev = a.device or default_device()
    p20 = NetPolicy.from_checkpoint(str(REPO / bse.BASES["p20"]), device=dev)
    roll = bse.rollout_policy(bse.V28R, dev)
    res = BatchRollout(p20, p20).run([SimState.from_seed(5_100_000 + k) for k in range(a.games + 64)], a.warm)
    keep = np.nonzero(~res.over)[0][:a.games]
    roots = [res.bsim.state(int(b)) for b in keep]
    br = BatchRollout(roll, roll)
    bs = BatchSim(roots)
    start = {name: getattr(bs, name).copy() for name in _SIM_ARRAYS}
    moves = np.zeros((a.horizon, bs.n, 10, 2), np.float32)
    rec = []

    class Rec:  # record the moves BatchRollout steps with
        def __init__(self, inner):
            self.inner, self.t = inner, 0

        def __call__(self, mv, idx=None, parallel=None):
            moves[self.t, :] = mv
            self.t += 1
            return self.inner(mv, idx, parallel)

    bs.step = Rec(bs.step)
    br.run(None, a.horizon, bsim=bs)
    rec.append(bs.step.t)
    np.savez_compressed(a.out, moves=moves[:rec[0]], seeds=keep + 5_100_000, warm=a.warm,
                        **{f"S_{k}": v for k, v in start.items()})
    print(f"captured {len(roots)} games x {rec[0]} decisions -> {a.out}")


# ---------------------------------------------------------------------------------- probes

@njit(cache=True)
def probe_move_math(Sb, Mb, moves, idx, out):
    acc = 0.0
    for j in range(idx.shape[0]):
        b = idx[j]
        S, M = _game(Sb, Mb, b)
        uf, ui = S[0], S[1]
        for k in range(2):
            for u in range(10):
                team = 0 if u < 5 else 1
                nx, ny = core._normalized(moves[b, u, 0], moves[b, u, 1])
                spd = core._speed(S, u)
                dt = S[6][core.GF_DT]
                dx = nx * dt * spd
                dy = ny * dt * spd
                r = core.CLS_WIDTH[ui[u, core.U_CLS]] / core.F2
                x, y = uf[u, 0], uf[u, 1]
                x, y = core._clamp_axis(M, x, y, dx, core.F0, r, team)
                x, y = core._clamp_axis(M, x, y, core.F0, dy, r, team)
                out[b, k, u, 0] = x
                out[b, k, u, 1] = y
                acc += x + y
    return acc


@njit(cache=True, inline='always')
def _clamp_sq(qx, qy, r, tx, ty, ext):
    """``_clamp_circle_square`` with selects instead of branches (same float32 / float64 expressions)."""
    lo, hi = tx - ext, tx + ext
    cqx = qx
    cqx = lo if qx < lo else cqx
    cqx = hi if (qx >= lo) and (qx > hi) else cqx
    lo, hi = ty - ext, ty + ext
    cqy = qy
    cqy = lo if qy < lo else cqy
    cqy = hi if (qy >= lo) and (qy > hi) else cqy
    dx = qx - cqx
    dy = qy - cqy
    d2 = np.float64(dx) * np.float64(dx) + np.float64(dy) * np.float64(dy)
    hit = d2 < np.float64(r) * np.float64(r)
    zero = d2 <= 0.0
    dx = core.F0 if zero else dx
    dy = core.F1 if zero else dy
    mag = core.F32(math.sqrt(np.float64(dx) * np.float64(dx) + np.float64(dy) * np.float64(dy)))
    ok = mag > core.K_EPS
    mag1 = mag if ok else core.F1
    nx = dx / mag1 if ok else core.F0
    ny = dy / mag1 if ok else core.F0
    return (cqx + nx * r if hit else qx), (cqy + ny * r if hit else qy)


@njit(cache=True)
def soa_move_math(walk, half, px, py, ix, iy, spd, dt, r, team, game, out_x, out_y):
    """Every game's new position of one unit (``move_unit`` up to ``_change_pos``), the game axis innermost and
    branch-free: walk [n, 2, 24, 24] / half [n, 24, 24] per game (``Mb[9]``, ``Mb[10]``), the rest [m] SoA."""
    for g in range(px.shape[0]):
        b = np.uint64(game[g])
        x0, y0 = ix[g], iy[g]
        mag = core.F32(math.sqrt(np.float64(x0) * np.float64(x0) + np.float64(y0) * np.float64(y0)))
        ok = mag > core.K_EPS
        mag1 = mag if ok else core.F1
        nx = x0 / mag1 if ok else core.F0
        ny = y0 / mag1 if ok else core.F0
        dx = nx * dt[g] * spd[g]
        dy = ny * dt[g] * spd[g]
        x, y = px[g], py[g]
        rr = r[g]
        tm = np.uint64(team[g])
        for axis in range(2):
            ddx = dx if axis == 0 else core.F0
            ddy = core.F0 if axis == 0 else dy
            move = np.float64(ddx) * np.float64(ddx) + np.float64(ddy) * np.float64(ddy) >= \
                np.float64(core.K_EPS) * np.float64(core.K_EPS)
            qx = x + ddx
            qy = y + ddy
            cx = int(math.floor(x))
            cy = int(math.floor(y))
            # only the 3 neighbours on the moving side pass ``xx >= 0 and yy >= 0 and xx + yy > 0`` (NEI8 order kept)
            dv = ddx if axis == 0 else ddy
            sg = 1 if dv > core.F0 else -1
            for k in range(3):
                ex = sg if axis == 0 else k - 1
                ey = k - 1 if axis == 0 else sg
                xn, yn = cx + ex, cy + ey
                inm = (xn >= 0) & (xn < 24) & (yn >= 0) & (yn < 24)
                xc = np.uint64(min(max(xn, 0), 23))  # unsigned: no wrap-around branch
                yc = np.uint64(min(max(yn, 0), 23))
                cell = xc * np.uint64(24) + yc
                blocked = ~(inm & walk[b * np.uint64(1152) + tm * np.uint64(576) + cell])
                hv = half[b * np.uint64(576) + cell]
                ext = hv if inm else core.FHALF
                act = move & blocked
                cqx, cqy = _clamp_sq(qx, qy, rr, core.F32(xn) + core.FHALF, core.F32(yn) + core.FHALF, ext)
                qx = cqx if act else qx
                qy = cqy if act else qy
            x = qx if move else x
            y = qy if move else y
        out_x[g] = x
        out_y[g] = y


def np_move_math(walk, half, px, py, ix, iy, spd, dt, r, team, game):
    """``soa_move_math`` as whole-array numpy (SIMD ufuncs, one pass per operation); same values bit for bit."""
    f32, f64 = np.float32, np.float64
    mag = np.sqrt(ix.astype(f64) ** 2 + iy.astype(f64) ** 2).astype(f32)
    ok = mag > core.K_EPS
    mag1 = np.where(ok, mag, f32(1))
    nx = np.where(ok, ix / mag1, f32(0))
    ny = np.where(ok, iy / mag1, f32(0))
    dx = nx * dt * spd
    dy = ny * dt * spd
    x, y = px.copy(), py.copy()
    eps2 = f64(core.K_EPS) * f64(core.K_EPS)
    wflat, hflat = walk.reshape(-1), half.reshape(-1)
    for axis in range(2):
        dv = dx if axis == 0 else dy
        move = dv.astype(f64) ** 2 >= eps2
        qx = x + dv if axis == 0 else x.copy()
        qy = y.copy() if axis == 0 else y + dv
        cx = np.floor(x).astype(np.int64)
        cy = np.floor(y).astype(np.int64)
        sg = np.where(dv > 0, 1, -1)
        for k in range(3):
            xn = cx + (sg if axis == 0 else k - 1)
            yn = cy + (k - 1 if axis == 0 else sg)
            inm = (xn >= 0) & (xn < 24) & (yn >= 0) & (yn < 24)
            cell = np.clip(xn, 0, 23) * 24 + np.clip(yn, 0, 23)
            blocked = ~(inm & wflat[game * 1152 + team * 576 + cell])
            ext = np.where(inm, hflat[game * 576 + cell], core.FHALF)
            act = move & blocked
            tx = xn.astype(f32) + core.FHALF
            ty = yn.astype(f32) + core.FHALF
            lo, hi = tx - ext, tx + ext
            cqx = np.where(qx < lo, lo, np.where(qx > hi, hi, qx))
            lo, hi = ty - ext, ty + ext
            cqy = np.where(qy < lo, lo, np.where(qy > hi, hi, qy))
            ddx, ddy = qx - cqx, qy - cqy
            d2 = ddx.astype(f64) ** 2 + ddy.astype(f64) ** 2
            hit = act & (d2 < r.astype(f64) ** 2)
            zero = d2 <= 0.0
            ddx = np.where(zero, f32(0), ddx)
            ddy = np.where(zero, f32(1), ddy)
            m2 = np.sqrt(ddx.astype(f64) ** 2 + ddy.astype(f64) ** 2).astype(f32)
            okm = m2 > core.K_EPS
            m1 = np.where(okm, m2, f32(1))
            ux = np.where(okm, ddx / m1, f32(0))
            uy = np.where(okm, ddy / m1, f32(0))
            qx = np.where(hit, cqx + ux * r, qx)
            qy = np.where(hit, cqy + uy * r, qy)
        x = np.where(move, qx, x)
        y = np.where(move, qy, y)
    return x, y


@njit(cache=True)
def probe_move_soa(Sb, Mb, moves, idx, bufs, out):
    """``probe_move_math`` reorganised: per (tick, unit) gather the games into SoA, ``soa_move_math``, check."""
    px, py, ix, iy, spd, dt, r, ox, oy = bufs
    n = idx.shape[0]
    team_i = np.empty(n, np.int64)
    acc = 0.0
    for k in range(2):
        for u in range(10):
            for j in range(n):
                b = idx[j]
                S, M = _game(Sb, Mb, b)
                px[j] = S[0][u, 0]
                py[j] = S[0][u, 1]
                ix[j] = moves[b, u, 0]
                iy[j] = moves[b, u, 1]
                spd[j] = core._speed(S, u)
                dt[j] = S[6][core.GF_DT]
                r[j] = core.CLS_WIDTH[S[1][u, core.U_CLS]] / core.F2
                team_i[j] = 0 if u < 5 else 1
            soa_move_math(Mb[9].reshape(-1), Mb[10].reshape(-1), px[:n], py[:n], ix[:n], iy[:n], spd[:n], dt[:n], r[:n], team_i, idx, ox[:n], oy[:n])
            for j in range(n):
                out[idx[j], k, u, 0] = ox[j]
                out[idx[j], k, u, 1] = oy[j]
                acc += ox[j] + oy[j]
    return acc


@njit(cache=True)
def probe_views(Sb, Mb, idx):
    """Only building the per-game (S, M) view tuples, 20 times per game (one per unit move)."""
    acc = 0
    for j in range(idx.shape[0]):
        for q in range(20):
            S, M = _game(Sb, Mb, idx[j])
            acc += S[1][q % 10, 0] + M[8][0]
    return acc


@njit(cache=True)
def probe_stat(Sb, Mb, idx):
    acc = 0.0
    for j in range(idx.shape[0]):
        S, M = _game(Sb, Mb, idx[j])
        for k in range(2):
            for u in range(10):
                acc += core._speed(S, u) + core._half_width(S, u)
    return acc


@njit(cache=True)
def probe_interact(Sb, Mb, idx):
    """The neighbour scan of ``_process_interactions`` at every unit's current position, overlap tests only."""
    acc = 0
    for j in range(idx.shape[0]):
        S, M = _game(Sb, Mb, idx[j])
        uf, it, tl, tn = S[0], S[2], S[3], S[4]
        for k in range(2):
            for u in range(10):
                team = 0 if u < 5 else 1
                px, py = uf[u, 0], uf[u, 1]
                cx, cy = core._floor(px), core._floor(py)
                for q in range(9):
                    x, y = cx + core.NEI9[q, 0], cy + core.NEI9[q, 1]
                    if not core._walkable(M, x, y, team):
                        continue
                    for jj in range(tn[x, y]):
                        code = tl[x, y, jj]
                        if code == u:
                            continue
                        hu = core._half_width(S, u)
                        if code < core.ITEM_BASE:
                            if core._overlap(uf[code, 0], uf[code, 1], core._half_width(S, code), px, py, hu):
                                acc += 1
                        else:
                            i = code - core.ITEM_BASE
                            if it[i, core.I_STATE] == core.POOL:
                                continue
                            if it[i, core.I_STATE] == core.CARRIED:
                                h = it[i, core.I_HOLDER]
                                ix, iy = uf[h, 0], uf[h, 1]
                            else:
                                ix, iy = core.F32(it[i, core.I_TX]) + core.FHALF, core.F32(it[i, core.I_TY]) + core.FHALF
                            if core._overlap(ix, iy, core.ITEM_WIDTH[it[i, core.I_TYPE]] / core.F2, px, py, hu):
                                acc += 1
    return acc


@njit(cache=True)
def probe_world(Sb, Mb, idx):
    for j in range(idx.shape[0]):
        S, M = _game(Sb, Mb, idx[j])
        for k in range(2):
            core.world_update(S, M)


@njit(cache=True)
def probe_render_raw(Sb, bg, idx, raw, packed):
    for j in range(idx.shape[0]):
        b = idx[j]
        S = _game_S(Sb, b)
        core.render_raw_state(S, raw)
        core.render_raw_map(S, bg[b], packed)


def _copy(bs: BatchSim) -> BatchSim:
    from blackout_env.sim.search import take_games

    return take_games(bs, np.arange(bs.n))


def load(path):
    from blackout_env.sim import SimState

    d = np.load(path)
    n = d["S_uf"].shape[0]
    sims = []
    for b in range(n):
        s = SimState.from_seed(int(d["seeds"][b]))  # layout / background of that seed's map
        for name in _SIM_ARRAYS:
            setattr(s, name, d[f"S_{name}"][b].copy())
        sims.append(s)
    return BatchSim(sims), d["moves"]


def timeit(fn, reps):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def cmd_profile(a):
    from blackout_env.sim.batch import batch_render_packed_seq, batch_step_seq

    bs0, moves = load(a.data)
    T = min(a.decisions, len(moves))
    n = bs0.n
    idx = np.arange(n, dtype=np.int64)
    # the states at every decision (for the pure probes) by replaying the reference
    snaps = []
    bs = _copy(bs0)
    for t in range(T):
        snaps.append(_copy(bs))
        live = np.nonzero(~bs.over)[0]
        batch_step_seq(bs.Sb, bs.Mb, moves[t], bs.order, live)
    gd = sum(int((~s.over).sum()) for s in snaps)

    def full():
        b = _copy(bs0)
        for t in range(T):
            live = np.nonzero(~b.over)[0]
            batch_step_seq(b.Sb, b.Mb, moves[t], b.order, live)

    def copy_only():
        _copy(bs0)

    def over(fn):
        def run():
            for t, s in enumerate(snaps):
                fn(s, t, np.nonzero(~s.over)[0])
        return run

    views = np.zeros((n, 2, H, W), np.int16)
    team = np.zeros((n, 2, 4), np.float32)
    agents = np.zeros((n, 2, 10, 12), np.float32)
    raw = np.zeros(44, np.float32)
    packed = np.zeros((H, W), np.uint16)
    soa = tuple(np.zeros(n, np.float32) for _ in range(9))  # px py ix iy spd dt r | ox oy
    o1, o2 = np.zeros((n, 2, 10, 2), np.float32), np.zeros((n, 2, 10, 2), np.float32)
    for t, s_ in enumerate(snaps):  # the SoA move math equals the scalar one bit for bit
        i = np.nonzero(~s_.over)[0]
        probe_move_soa(s_.Sb, s_.Mb, moves[t], i, soa, o1)
        probe_move_math(s_.Sb, s_.Mb, moves[t], i, o2)
        assert np.array_equal(o1.view(np.uint32), o2.view(np.uint32)), t
    print("SoA move math bit-identical on", len(snaps), "decisions")
    probes = {
        "move_math": over(lambda s, t, i: probe_move_math(s.Sb, s.Mb, moves[t], i, o2)),
        "move_soa": over(lambda s, t, i: probe_move_soa(s.Sb, s.Mb, moves[t], i, soa, o1)),
        "views": over(lambda s, t, i: probe_views(s.Sb, s.Mb, i)),
        "stat": over(lambda s, t, i: probe_stat(s.Sb, s.Mb, i)),
        "interact": over(lambda s, t, i: probe_interact(s.Sb, s.Mb, i)),
        "world": over(lambda s, t, i: probe_world(_copy(s).Sb, s.Mb, i)),
        "world_copy": over(lambda s, t, i: _copy(s)),
        "render_raw": over(lambda s, t, i: probe_render_raw(s.Sb, s.bg, i, raw, packed)),
        "render_pk": over(lambda s, t, i: batch_render_packed_seq(s.Sb, s.bg, i, views, team, agents)),
    }
    # warm-up (numba compile)
    full()
    for f in probes.values():
        f()
    res = {"step": (timeit(full, a.reps) - timeit(copy_only, a.reps)) / gd}
    for k, f in probes.items():
        res[k] = timeit(f, a.reps) / gd
    res["world"] -= res.pop("world_copy")
    lines = [f"{n} games x {T} decisions = {gd} game-decisions (leaf phase, rollout net v28d256 both sides)",
             f"{'phase':12s} {'us/game-decision':>18s} {'% of step':>10s}"]
    for k, v in res.items():
        lines.append(f"{k:12s} {v * 1e6:18.2f} {100 * v / res['step']:10.1f}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("capture")
    q.add_argument("--games", type=int, default=1024)
    q.add_argument("--warm", type=int, default=1500)
    q.add_argument("--horizon", type=int, default=125)
    q.add_argument("--device", default="")
    q.add_argument("--out", required=True)
    q = sub.add_parser("profile")
    q.add_argument("--data", required=True)
    q.add_argument("--decisions", type=int, default=125)
    q.add_argument("--reps", type=int, default=3)
    q.add_argument("--out", default="")
    a = p.parse_args()
    {"capture": cmd_capture, "profile": cmd_profile}[a.cmd](a)


if __name__ == "__main__":
    main()
