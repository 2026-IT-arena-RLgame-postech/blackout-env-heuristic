"""Numba kernels: one decision step of BlackOut, a line-by-line port of the Unity C# rules.

Every function names the C# method it mirrors.  Arithmetic on positions, speeds and sizes is
float32, in the same operation order as Unity's Vector2 / Mathf code, so trajectories reproduce
Unity's bit for bit as long as the inputs do (see docs/policy_v2_design_ko.md "S2").

State layout (all arrays owned by ``state.SimState``; kernels take the two tuples)::

  S = (uf, ui, it, tl, tn, g, gf, rng, ev, reg_lock, fs, ml, mn)
    uf   float32 [10, 2]     unit world position (x, y)
    ui   int32   [10, 4]     class, held item slot (-1), owned tile x, y
    it   int32   [MAXI, 9]   item slots: type (-1 free), amount, state, tile x, y, applied team
                             context (-1 none / 0 A / 1 B / 2 neutral), holder, modifier team,
                             modifier sequence number
    tl   int32   [24, 24, K] MapTile.MapObjects in list order: unit u -> u, item i -> 16 + i
    tn   int32   [24, 24]    list lengths
    g    int64   [NG]        scores, flags, counters (G_* below)
    gf   float32 [4]         episode timer, absorption timer, special-item spawn timer, dt
    rng  uint32  [4]         UnityEngine.Random state (xorshift128)
    ev   int32   [MAXEV, 6]  event log of the current step: (tick, kind, a, b, c, d)
    reg_lock int32 [R]       locked unit of a unique-instance sanctuary (-1 = free)
    fs   int32   [MAXF, 4]   forced special spawns (type, x, y, tick), consumed in order (G_FS_*);
                             SPAWN_TIMED spawns each at world-update number ``tick``
    ml   int32   [3, MAXM]   TeamContext.activeModifiers per team (A, B, neutral): item slots
    mn   int32   [3]         list lengths
  M = (tile, region, reg_kind, reg_team, reg_in, reg_out, reg_locked, reg_tiles, reg_ntiles,
       walk, half)           mapdata.Layout arrays
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .mapdata import (COLLECTOR, FLOOR, NEUTRAL, R_SANCTUARY,
                      R_STORAGE, SPAWN_CELL)
from .unity_random import range_int

F32 = np.float32
# Time.fixedDeltaTime as the build sees it: 4 float32 ulps below 0.02f.  Measured from the
# episode / absorption timers and from free-space movement of 40 recorded matches (0.02f itself
# drifts the timers by a tick within 20 s).  Kept per state in gf[GF_DT].
DT = np.nextafter(np.nextafter(np.nextafter(np.nextafter(F32(0.02), F32(0)), F32(0)), F32(0)), F32(0))
EPISODE_SECONDS = F32(420.0)           # GameBalanceConfig.MaxEpisodeTime
ABSORPTION_SECONDS = F32(20.0)         # AbsorptionInterval
SPAWN_COOLDOWN = F32(10.0)             # SpecialItemSpawnCooldown
TARGET_SCORE = 100
K_EPS = F32(1e-5)                      # Vector2.kEpsilon
F0, F1, F2, FHALF, F15 = F32(0.0), F32(1.0), F32(2.0), F32(0.5), F32(1.5)
BOUNDS = F32(24.0)                     # BlackOutEpisodeCoordinator.mapBounds

N_UNITS = 10
ITEM_BASE = 16                         # object code of item slot 0 in tile lists
MAXI = 160
MAXK = 16
MAXEV = 512
MAXM = 32
MAXF = 256

# unit classes: Collector, Hunter, Carrier (UnitData assets)
CLS_WIDTH = np.array([0.55, 0.65, 0.45], dtype=np.float32)
CLS_SPEED = np.array([4.0, 6.0, 6.0], dtype=np.float32)
CLS_COLLECT = np.array([True, False, True])
# BEATS[a, b]: a's UnitData.Beats contains b
BEATS = np.array([[False, False, True],    # Collector beats Carrier
                  [True, True, True],      # Hunter beats all (Hunter vs Hunter: both die)
                  [False, False, True]])   # Carrier beats Carrier (both die)

# items (KnownItems order): Battery, BuffSpeed, DebuffSpeed, BuffSize, DebuffSize
BATTERY = 0
ITEM_WIDTH = np.array([0.6, 0.8, 0.8, 0.8, 0.8], dtype=np.float32)
ITEM_MAX = np.array([10, 1, 1, 1, 1], dtype=np.int64)
# BuffItemEffect per special item: stat (0 speed, 1 size), value, target (0 owner, 1 opponent),
# condition (-1 any class, else the only class it applies to)
MOD_STAT = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
MOD_VALUE = np.array([0.0, 0.5, -0.9, 0.5, -0.3], dtype=np.float32)
MOD_TARGET = np.array([-1, 0, 1, 0, 1], dtype=np.int64)
MOD_COND = np.array([-1, -1, COLLECTOR, -1, -1], dtype=np.int64)

# item state
POOL, GROUND, CARRIED = 0, 1, 2
I_TYPE, I_AMT, I_STATE, I_TX, I_TY, I_CTX, I_HOLDER, I_MODTEAM, I_MODSEQ = range(9)
U_CLS, U_HELD, U_TX, U_TY = range(4)
# exit reasons
DROPPED, DESTROYED, REFRESH, ABSORBED = 0, 1, 2, 3

# globals
(G_SCORE_A, G_SCORE_B, G_SCORE_N, G_OVER, G_WINNER, G_SPECIAL, G_MODSEQ, G_TICK, G_NEV,
 G_SPAWN_MODE, G_FS_N, G_FS_I, G_NSPAWN, G_FIRST, G_TILESEQ) = range(15)
NG = 16
GF_EP, GF_ABS, GF_SPAWN, GF_DT = 0, 1, 2, 3
SPAWN_RNG, SPAWN_FORCED, SPAWN_NONE, SPAWN_TIMED = 0, 1, 2, 3

# events
EV_PICKUP, EV_DEPOSIT, EV_KILL, EV_TRANSFORM, EV_ABSORB, EV_SPAWN, EV_LOST, EV_END, EV_STEAL = range(9)

# InteractionSystem neighbours (x, y), MovementSystem neighbours
NEI9 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)
NEI8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)
SPAWN = np.array(SPAWN_CELL, dtype=np.int64)


# ---------------------------------------------------------------------------------- helpers

@njit(cache=True, inline='always')
def _floor(v) -> int:
    return int(math.floor(v))


@njit(cache=True, inline='always')
def _sq64(x, y):
    """x*x + y*y as Mono evaluates float expressions: in double, rounded only when stored."""
    return np.float64(x) * np.float64(x) + np.float64(y) * np.float64(y)


@njit(cache=True, inline='always')
def _normalized(x, y):
    """Vector2.normalized: magnitude = (float)Math.Sqrt(x*x + y*y) (sum in double); zero below
    kEpsilon; the division result is stored (rounded) into the Vector2 fields."""
    mag = F32(math.sqrt(_sq64(x, y)))
    if mag > K_EPS:
        return x / mag, y / mag
    return F0, F0


@njit(cache=True, inline='always')
def _in_map(x, y) -> bool:
    return 0 <= x < 24 and 0 <= y < 24


@njit(cache=True, inline='always')
def _walkable(M, x, y, team) -> bool:
    if not _in_map(x, y):
        return False
    return M[9][team, x, y]


@njit(cache=True, inline='always')
def _event(S, kind, a, b, c, d):
    g, ev = S[5], S[8]
    n = g[G_NEV]
    if n < ev.shape[0]:
        ev[n, 0] = g[G_TICK]
        ev[n, 1] = kind
        ev[n, 2] = a
        ev[n, 3] = b
        ev[n, 4] = c
        ev[n, 5] = d
        g[G_NEV] = n + 1


@njit(cache=True, inline='always')
def _tile_add(S, x, y, code):
    tl, tn = S[3], S[4]
    n = tn[x, y]
    tl[x, y, n] = code
    tn[x, y] = n + 1


@njit(cache=True, inline='always')
def _tile_remove(S, x, y, code):
    """List.Remove: the first occurrence, later entries shift left."""
    tl, tn = S[3], S[4]
    n = tn[x, y]
    for k in range(n):
        if tl[x, y, k] == code:
            for j in range(k, n - 1):
                tl[x, y, j] = tl[x, y, j + 1]
            tn[x, y] = n - 1
            return


@njit(cache=True, inline='always')
def _first_item_on(S, x, y) -> int:
    """MapObjects.Find(x => x is ItemObject)."""
    tl, tn = S[3], S[4]
    for k in range(tn[x, y]):
        if tl[x, y, k] >= ITEM_BASE:
            return tl[x, y, k] - ITEM_BASE
    return -1


# ---------------------------------------------------------------------------------- stats

@njit(cache=True, inline='always')
def _stat(S, u, stat, base):
    """TeamContext.CalculateStat: Multiply modifiers summed (list order) into one factor."""
    ui, it, ml, mn = S[1], S[2], S[11], S[12]
    team = 0 if u < 5 else 1
    cls = ui[u, U_CLS]
    pct = 0.0  # a float local: Mono keeps it in double
    for k in range(mn[team]):
        t = it[ml[team, k], I_TYPE]
        if MOD_STAT[t] == stat and (MOD_COND[t] < 0 or MOD_COND[t] == cls):
            pct = pct + np.float64(MOD_VALUE[t])
    v = F32(np.float64(base) * (1.0 + pct))  # rounded when passed to Mathf.Max(float, float)
    return v if v > F0 else F0


@njit(cache=True)
def _mod_remove(S, i):
    """TeamContext.RemoveModifier for item i's tracked modifier (if any)."""
    it, ml, mn = S[2], S[11], S[12]
    team = it[i, I_MODTEAM]
    if team < 0:
        return
    n = mn[team]
    for k in range(n):
        if ml[team, k] == i:
            for j in range(k, n - 1):
                ml[team, j] = ml[team, j + 1]
            mn[team] = n - 1
            break
    it[i, I_MODTEAM] = -1


@njit(cache=True, inline='always')
def _speed(S, u):
    return _stat(S, u, 0, CLS_SPEED[S[1][u, U_CLS]])


@njit(cache=True, inline='always')
def _half_width(S, u):
    """Unit.CollisionBound (scaled by UnitSize) .HalfWidth."""
    w = CLS_WIDTH[S[1][u, U_CLS]] * _stat(S, u, 1, F1)
    return w / F2


# ---------------------------------------------------------------------------------- score / effects

@njit(cache=True)
def _add_score(S, ctx, amount, notify):
    """TeamContext.AddScore -> MatchManager.CheckWinCondition on notify."""
    g = S[5]
    g[G_SCORE_A + ctx] += amount
    if notify and ctx != NEUTRAL and g[G_OVER] == 0 and g[G_SCORE_A + ctx] >= TARGET_SCORE:
        g[G_OVER] = 1
        g[G_WINNER] = ctx
        _event(S, EV_END, ctx, 0, 0, 0)


@njit(cache=True)
def _effect_enter(S, i, ctx):
    """ItemObject.ApplyEffectInternal(isEnter): ScoreItemEffect / BuffItemEffect.OnEnterStorage."""
    it, g = S[2], S[5]
    t = it[i, I_TYPE]
    if t == BATTERY:
        _add_score(S, ctx, it[i, I_AMT], True)
    else:
        target = ctx if MOD_TARGET[t] == 0 else (1 - ctx if ctx < 2 else -1)
        if target >= 0:
            ml, mn = S[11], S[12]
            ml[target, mn[target]] = i
            mn[target] += 1
            it[i, I_MODTEAM] = target
            g[G_MODSEQ] += 1
            it[i, I_MODSEQ] = g[G_MODSEQ]


@njit(cache=True)
def _effect_exit(S, i, ctx, reason):
    it = S[2]
    _mod_remove(S, i)  # tracked modifiers are always removed
    if it[i, I_TYPE] == BATTERY:
        if reason == REFRESH:
            _add_score(S, ctx, -it[i, I_AMT], False)
        elif reason != ABSORBED:
            _add_score(S, ctx, -it[i, I_AMT], True)


@njit(cache=True)
def _exit_region(S, i, reason):
    it = S[2]
    ctx = it[i, I_CTX]
    if ctx >= 0:
        _effect_exit(S, i, ctx, reason)
        it[i, I_CTX] = -1


@njit(cache=True)
def _remove_from_map(S, i):
    it = S[2]
    if it[i, I_TX] >= 0:
        _tile_remove(S, it[i, I_TX], it[i, I_TY], ITEM_BASE + i)
        it[i, I_TX] = -1
        it[i, I_TY] = -1


@njit(cache=True)
def _destroy(S, i, skip_effect):
    """ItemObject.OnDestroyed."""
    it = S[2]
    if it[i, I_STATE] == POOL:
        return
    if not skip_effect:
        _exit_region(S, i, DESTROYED)
    _remove_from_map(S, i)
    it[i, I_STATE] = POOL
    it[i, I_HOLDER] = -1
    _mod_remove(S, i)


@njit(cache=True)
def _move_to_tile(S, M, i, x, y):
    """ItemObject.MoveToTile + UpdateRegionState."""
    it = S[2]
    _remove_from_map(S, i)
    it[i, I_TX] = x
    it[i, I_TY] = y
    _tile_add(S, x, y, ITEM_BASE + i)
    new_ctx = M[3][M[1][x, y]]  # region team: 0, 1 or 2 (neutral context)
    if it[i, I_CTX] != new_ctx:
        _exit_region(S, i, DROPPED)
        _effect_enter(S, i, new_ctx)
        it[i, I_CTX] = new_ctx


@njit(cache=True)
def _update_amount(S, i, amount):
    """ItemObject.UpdateAmount: Refresh exit, new amount, re-enter."""
    it = S[2]
    if amount == it[i, I_AMT]:
        return
    ctx = it[i, I_CTX]
    if ctx >= 0:
        _effect_exit(S, i, ctx, REFRESH)
    it[i, I_AMT] = amount
    if ctx >= 0:
        _effect_enter(S, i, ctx)


@njit(cache=True)
def new_item(S, M, t, amount, x, y) -> int:
    """LevelDirector.SpawnItem: pool Get + RegisterToMap."""
    it = S[2]
    for i in range(it.shape[0]):
        if it[i, I_STATE] == POOL:
            it[i, I_TYPE] = t
            it[i, I_AMT] = amount
            it[i, I_STATE] = GROUND
            it[i, I_TX] = -1
            it[i, I_TY] = -1
            it[i, I_CTX] = -1
            it[i, I_HOLDER] = -1
            it[i, I_MODTEAM] = -1
            it[i, I_MODSEQ] = 0
            _move_to_tile(S, M, i, x, y)
            return i
    return -1


# ---------------------------------------------------------------------------------- unit rules

@njit(cache=True)
def _set_class(S, u, cls):
    """Unit.SetUnitClass: a non-collector loses (destroys) its item."""
    ui, it = S[1], S[2]
    ui[u, U_CLS] = cls
    h = ui[u, U_HELD]
    if not CLS_COLLECT[cls] and h >= 0:
        _event(S, EV_LOST, u, it[h, I_TYPE], it[h, I_AMT], 0)
        _destroy(S, h, False)
        ui[u, U_HELD] = -1


@njit(cache=True)
def _try_distribute(S, M, u, reg) -> bool:
    """Storage.TryDistributeItem + ExecutePlan + the OnUnitEnter tail."""
    ui, it, g = S[1], S[2], S[5]
    i = ui[u, U_HELD]
    t = it[i, I_TYPE]
    remaining = it[i, I_AMT]
    start = remaining
    n = M[8][reg]
    plan_k = np.empty(n, dtype=np.int64)
    plan_amt = np.empty(n, dtype=np.int64)
    plan_new = np.empty(n, dtype=np.bool_)
    np_ = 0
    destroy = True
    for k in range(n):
        if remaining <= 0:
            break
        x, y = M[7][reg, k, 0], M[7][reg, k, 1]
        on = _first_item_on(S, x, y)
        if on < 0:
            transfer = min(remaining, ITEM_MAX[t])
        elif it[on, I_TYPE] != t:
            continue
        else:
            space = ITEM_MAX[t] - it[on, I_AMT]
            if space <= 0:
                continue
            transfer = min(space, remaining)
        if transfer > 0:
            if on < 0:
                destroy = False
            plan_k[np_] = k
            plan_amt[np_] = transfer
            plan_new[np_] = on < 0
            np_ += 1
            remaining -= transfer
    if remaining > 0:
        return False
    for p in range(np_):
        if g[G_OVER]:
            return True
        x, y = M[7][reg, plan_k[p], 0], M[7][reg, plan_k[p], 1]
        if plan_new[p]:
            _update_amount(S, i, plan_amt[p])
            it[i, I_STATE] = GROUND
            it[i, I_HOLDER] = -1
            _move_to_tile(S, M, i, x, y)
        else:
            on = _first_item_on(S, x, y)
            _update_amount(S, on, it[on, I_AMT] + plan_amt[p])
    if g[G_OVER]:
        return True
    ui[u, U_HELD] = -1
    if destroy:
        it[i, I_HOLDER] = -1
        it[i, I_STATE] = CARRIED  # still carried until destroyed (no applied context)
        _destroy(S, i, False)
    _event(S, EV_DEPOSIT, u, t, start, reg)
    return True


@njit(cache=True)
def _region_enter(S, M, u, reg):
    """MapRegion.OnUnitEnter: Storage (deposit) / Sanctuary (transform)."""
    ui, reg_lock = S[1], S[9]
    kind = M[2][reg]
    team = 0 if u < 5 else 1
    if kind == R_STORAGE:
        if M[3][reg] == team and ui[u, U_HELD] >= 0:
            _try_distribute(S, M, u, reg)
    elif kind == R_SANCTUARY:
        locked = M[6][reg]
        if locked and reg_lock[reg] >= 0:
            return
        if ui[u, U_CLS] == M[4][reg]:
            _set_class(S, u, M[5][reg])
            _event(S, EV_TRANSFORM, u, M[5][reg], reg, 0)
            if locked:
                reg_lock[reg] = u


@njit(cache=True, inline='always')
def _change_pos(S, M, u, x, y):
    """Unit.ChangePos: tile lists, region enter, then the transform."""
    uf, ui = S[0], S[1]
    cx, cy = _floor(x), _floor(y)
    if _in_map(cx, cy) and (ui[u, U_TX] != cx or ui[u, U_TY] != cy):
        ox, oy = ui[u, U_TX], ui[u, U_TY]
        prev = -1
        if ox >= 0:
            _tile_remove(S, ox, oy, u)
            prev = M[1][ox, oy]
        _tile_add(S, cx, cy, u)
        ui[u, U_TX] = cx
        ui[u, U_TY] = cy
        reg = M[1][cx, cy]
        if prev != reg:
            _region_enter(S, M, u, reg)
    uf[u, 0] = x
    uf[u, 1] = y


@njit(cache=True)
def _die(S, M, u, killer):
    """Unit.Die -> MatchManager.RespawnUnit (Teleport, SetUnitClass(Collector)) -> OnUnitDead.

    Teleport's ProcessInteractions at the spawn is skipped: the 3x3 around either spawn is own
    base floor and walls, where no enemy unit or item can be (bases block enemies, items only
    lie on neutral floor and storages)."""
    ui, it, reg_lock = S[1], S[2], S[9]
    h = ui[u, U_HELD]
    if h >= 0:
        _event(S, EV_LOST, u, it[h, I_TYPE], it[h, I_AMT], 1)
        _destroy(S, h, False)
        ui[u, U_HELD] = -1
    _event(S, EV_KILL, killer, u, ui[u, U_CLS], 0 if killer < 0 else ui[killer, U_CLS])
    team = 0 if u < 5 else 1
    sx, sy = SPAWN[team, 0], SPAWN[team, 1]
    _change_pos(S, M, u, F32(sx) + FHALF, F32(sy) + FHALF)
    _set_class(S, u, COLLECTOR)
    for r in range(reg_lock.shape[0]):
        if reg_lock[r] == u:
            reg_lock[r] = -1


@njit(cache=True, inline='always')
def _overlap(ax, ay, ha, bx, by, hb) -> bool:
    """CollisionUtils.IsOverlapping for two circles."""
    d2 = _sq64(ax - bx, ay - by)  # (posA - posB) is a stored Vector2, sqrMagnitude in double
    reach = np.float64(ha) * 1.5 + np.float64(hb) * 1.5
    if d2 > reach * reach:
        return False
    rs = np.float64(ha) + np.float64(hb)
    return d2 <= rs * rs


@njit(cache=True)
def _pick_up(S, M, u, i):
    """Unit.OnOverlapped(item) -> ItemObject.OnPickedUp."""
    ui, it = S[1], S[2]
    team = 0 if u < 5 else 1
    if not CLS_COLLECT[ui[u, U_CLS]] or ui[u, U_HELD] >= 0 or it[i, I_STATE] != GROUND:
        return
    src = M[3][M[1][it[i, I_TX], it[i, I_TY]]]
    if src == team:  # ObjectInteractionOption.IgnoreFriend: not from own territory
        return
    ui[u, U_HELD] = i
    _event(S, EV_STEAL if src < 2 else EV_PICKUP, u, it[i, I_TYPE], it[i, I_AMT], src)
    _exit_region(S, i, DROPPED)
    _remove_from_map(S, i)
    it[i, I_STATE] = CARRIED
    it[i, I_HOLDER] = u


@njit(cache=True)
def _combat(S, M, a, d):
    """UnitInteractionSystem.ResolveCombat."""
    ui = S[1]
    ca, cd = ui[a, U_CLS], ui[d, U_CLS]
    aw = BEATS[ca, cd]
    dw = BEATS[cd, ca]
    if aw and dw:
        _die(S, M, a, -1)
        _die(S, M, d, -1)
    elif aw:
        _die(S, M, d, a)
    elif dw:
        _die(S, M, a, d)


@njit(cache=True, inline='always')
def _process_interactions(S, M, u, px, py):
    """UnitInteractionSystem.ProcessInteractions(currentWorldPos)."""
    uf, it, tl, tn, g = S[0], S[2], S[3], S[4], S[5]
    team = 0 if u < 5 else 1
    cx, cy = _floor(px), _floor(py)
    snap = np.empty(MAXK, dtype=np.int32)
    for k in range(9):
        x, y = cx + NEI9[k, 0], cy + NEI9[k, 1]
        if not _walkable(M, x, y, team):
            continue
        n = tn[x, y]
        for j in range(n):
            snap[j] = tl[x, y, j]
        for j in range(n):
            code = snap[j]
            if code == u:
                continue
            hu = _half_width(S, u)
            if code < ITEM_BASE:
                if _overlap(uf[code, 0], uf[code, 1], _half_width(S, code), px, py, hu):
                    if (code < 5) != (u < 5):
                        _combat(S, M, u, code)
            else:
                i = code - ITEM_BASE
                if it[i, I_STATE] == POOL:
                    continue
                if it[i, I_STATE] == CARRIED:  # parented to its holder
                    h = it[i, I_HOLDER]
                    ix, iy = uf[h, 0], uf[h, 1]
                else:
                    ix, iy = F32(it[i, I_TX]) + FHALF, F32(it[i, I_TY]) + FHALF
                if _overlap(ix, iy, ITEM_WIDTH[it[i, I_TYPE]] / F2, px, py, hu):
                    _pick_up(S, M, u, i)
            if g[G_OVER]:
                return


@njit(cache=True, inline='always')
def _clamp_circle_square(px, py, r, tx, ty, ext):
    """CollisionUtils.GetClampedPosition (circle mover vs square tile)."""
    qx = px
    if qx < tx - ext:
        qx = tx - ext
    elif qx > tx + ext:
        qx = tx + ext
    qy = py
    if qy < ty - ext:
        qy = ty - ext
    elif qy > ty + ext:
        qy = ty + ext
    dx = px - qx
    dy = py - qy
    d2 = _sq64(dx, dy)
    if d2 < np.float64(r) * np.float64(r):
        if d2 <= 0.0:  # Mathf.Epsilon^2 underflows to 0
            dx, dy = F0, F1
        nx, ny = _normalized(dx, dy)
        return qx + nx * r, qy + ny * r
    return px, py


@njit(cache=True, inline='always')
def _clamp_axis(M, px, py, dx, dy, r, team):
    """UnitMovementSystem.ClampAxis."""
    if _sq64(dx, dy) < np.float64(K_EPS) * np.float64(K_EPS):  # displacement == Vector2.zero
        return px, py
    qx = px + dx
    qy = py + dy
    cx, cy = _floor(px), _floor(py)
    for k in range(8):
        ddx, ddy = NEI8[k, 0], NEI8[k, 1]
        xx = F32(ddx) * dx
        yy = F32(ddy) * dy
        if xx >= F0 and yy >= F0 and xx + yy > F0:
            x, y = cx + ddx, cy + ddy
            if not _walkable(M, x, y, team):
                ext = M[10][x, y] if _in_map(x, y) else FHALF
                qx, qy = _clamp_circle_square(qx, qy, r, F32(x) + FHALF, F32(y) + FHALF, ext)
    return qx, qy


@njit(cache=True, inline='always')
def move_unit(S, M, u, ix, iy):
    """Unit.Move(input, fixedDeltaTime)."""
    uf, ui = S[0], S[1]
    team = 0 if u < 5 else 1
    nx, ny = _normalized(ix, iy)
    spd = _speed(S, u)
    dt = S[6][GF_DT]
    dx = nx * dt * spd
    dy = ny * dt * spd
    r = CLS_WIDTH[ui[u, U_CLS]] / F2  # walls use the unscaled UnitData bound
    x, y = uf[u, 0], uf[u, 1]
    x, y = _clamp_axis(M, x, y, dx, F0, r, team)
    x, y = _clamp_axis(M, x, y, F0, dy, r, team)
    _change_pos(S, M, u, x, y)
    if S[5][G_OVER] == 0:
        _process_interactions(S, M, u, x, y)


# ---------------------------------------------------------------------------------- world update

@njit(cache=True)
def _spawn_special(S, M):
    """LevelDirector.SpawnSpecialItem (UnityEngine.Random, or a forced trace, or nothing)."""
    g, rng, fs = S[5], S[7], S[10]
    mode = g[G_SPAWN_MODE]
    if mode == SPAWN_NONE:
        return
    if mode == SPAWN_FORCED:
        k = g[G_FS_I]
        if k >= g[G_FS_N]:
            return
        g[G_FS_I] = k + 1
        t, x, y = fs[k, 0], fs[k, 1], fs[k, 2]
    else:
        t = 1 + range_int(rng, 0, 4)
        n = 0
        for yy in range(24):          # MapManager.allTilesCache order: y outer, x inner
            for xx in range(24):
                if M[3][M[1][xx, yy]] == NEUTRAL and M[0][xx, yy] == FLOOR and _first_item_on(S, xx, yy) < 0:
                    n += 1
        if n == 0:
            return
        pick = range_int(rng, 0, n)
        x, y = -1, -1
        for yy in range(24):
            for xx in range(24):
                if M[3][M[1][xx, yy]] == NEUTRAL and M[0][xx, yy] == FLOOR and _first_item_on(S, xx, yy) < 0:
                    if pick == 0 and x < 0:
                        x, y = xx, yy
                    pick -= 1
    i = new_item(S, M, t, 1, x, y)
    g[G_SPECIAL] = i
    g[G_NSPAWN] += 1
    _event(S, EV_SPAWN, t, x, y, i)


@njit(cache=True)
def _absorb(S):
    """LevelDirector.OnAbsorption: every item lying in a storage is absorbed."""
    it = S[2]
    for i in range(it.shape[0]):
        if it[i, I_STATE] == GROUND and it[i, I_TX] >= 0:
            ctx = it[i, I_CTX]
            if ctx == 0 or ctx == 1:
                _event(S, EV_ABSORB, ctx, it[i, I_TYPE], it[i, I_AMT], i)
                _exit_region(S, i, ABSORBED)
                _destroy(S, i, True)


@njit(cache=True)
def world_update(S, M):
    """BlackOutEpisodeCoordinator.FixedUpdate -> GameScenario.EpisodeUpdate(fixedDeltaTime)."""
    it, g, gf, fs = S[2], S[5], S[6], S[10]
    # LevelDirector.ManualLateUpdate -> UpdateSpecialItemSpawner
    sp = g[G_SPECIAL]
    if g[G_SPAWN_MODE] == SPAWN_TIMED:  # validation: spawn exactly when the trace did
        while g[G_FS_I] < g[G_FS_N] and fs[g[G_FS_I], 3] <= g[G_TICK] + 1:
            k = g[G_FS_I]
            g[G_FS_I] = k + 1
            if fs[k, 3] == g[G_TICK] + 1:
                i = new_item(S, M, fs[k, 0], 1, fs[k, 1], fs[k, 2])
                g[G_SPECIAL] = i
                g[G_NSPAWN] += 1
                _event(S, EV_SPAWN, fs[k, 0], fs[k, 1], fs[k, 2], i)
    elif not (sp >= 0 and it[sp, I_STATE] != POOL):
        gf[GF_SPAWN] = gf[GF_SPAWN] + gf[GF_DT]
        if gf[GF_SPAWN] >= SPAWN_COOLDOWN:
            _spawn_special(S, M)
            gf[GF_SPAWN] = F0
    # TimerManager.Tick: episode timer, then absorption timer
    gf[GF_EP] = gf[GF_EP] + gf[GF_DT]
    if gf[GF_EP] >= EPISODE_SECONDS:
        if g[G_OVER] == 0:
            g[G_OVER] = 1
            a, b = g[G_SCORE_A], g[G_SCORE_B]
            g[G_WINNER] = 0 if a > b else (1 if b > a else -1)
            _event(S, EV_END, g[G_WINNER], 1, 0, 0)
        return
    gf[GF_ABS] = gf[GF_ABS] + gf[GF_DT]
    if gf[GF_ABS] >= ABSORPTION_SECONDS:
        _absorb(S)
        gf[GF_ABS] = gf[GF_ABS] - ABSORPTION_SECONDS


@njit(cache=True)
def step_kernel(S, M, moves, order):
    """One decision step: Unity runs two FixedUpdate ticks per decision (decisionPeriod 2), each
    = every unit's Move in ML-Agents AgentAct order, then the world update.  The first step of a
    match moves once (the first decision tick after reset is suppressed)."""
    g = S[5]
    g[G_NEV] = 0
    for k in range(2):
        if g[G_OVER]:
            return
        if not (g[G_FIRST] and k == 1):
            for j in range(N_UNITS):
                u = order[j]
                move_unit(S, M, u, moves[u, 0], moves[u, 1])
                if g[G_OVER]:
                    return
        world_update(S, M)
        g[G_TICK] += 1
    g[G_FIRST] = 0


@njit(cache=True)
def run_kernel(S, M, moves_seq, order, out_pos):
    """``step_kernel`` over a [T, 10, 2] move sequence, recording positions; stops at game end.
    Returns the number of steps taken."""
    for t in range(moves_seq.shape[0]):
        if S[5][G_OVER]:
            return t
        step_kernel(S, M, moves_seq[t], order)
        out_pos[t] = S[0]
    return moves_seq.shape[0]


# ---------------------------------------------------------------------------------- observation

@njit(cache=True)
def render_raw_state(S, out):
    """MapObsAgent.CollectObservations -> float32[44]."""
    uf, ui, it, g, gf = S[0], S[1], S[2], S[5], S[6]
    for u in range(N_UNITS):
        out[4 * u] = uf[u, 0] / BOUNDS * F2 - F1     # ((pos - origin) / bounds) * 2 - one
        out[4 * u + 1] = uf[u, 1] / BOUNDS * F2 - F1
        h = ui[u, U_HELD]
        if h < 0:
            out[4 * u + 2] = F0
        elif it[h, I_TYPE] == BATTERY:
            out[4 * u + 2] = -F32(it[h, I_AMT])
        else:
            out[4 * u + 2] = F32(it[h, I_TYPE] + 1)
        out[4 * u + 3] = F32(ui[u, U_CLS])
    target = F32(TARGET_SCORE)
    out[40] = F32(g[G_SCORE_A]) / target
    out[41] = F32(g[G_SCORE_B]) / target
    r = gf[GF_EP] / EPISODE_SECONDS
    out[42] = F1 - min(max(r, F0), F1)
    r = gf[GF_ABS] / ABSORPTION_SECONDS
    out[43] = F1 - min(max(r, F0), F1)


@njit(cache=True)
def render_raw_map(S, background, out):
    """SemanticMapRenderer.Render (team A view, rows top-down as the sensor writes them)."""
    it = S[2]
    out[:, :] = background
    for i in range(it.shape[0]):
        if it[i, I_STATE] != GROUND or it[i, I_TX] < 0:
            continue
        x, y = it[i, I_TX], it[i, I_TY]
        t = it[i, I_TYPE]
        if t == BATTERY:
            bits = min(max(it[i, I_AMT], 0), 15) << 3
        else:
            bits = t << 7
        out[23 - y, x] |= bits
