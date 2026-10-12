"""Official-format observations rendered from a ``SimState`` (validation) and parsed back."""

from __future__ import annotations

import numpy as np

from ..sim import core
from ..sim.mapdata import BLOCK_ALL, H, R_STORAGE, TILE_COLLISION, W

F32 = np.float32
SCALE = 4                               # SemanticMapRenderer.resolutionScale (official)
HO, WO = H * SCALE, W * SCALE
ID_EMPTY, ID_WALL, ID_ALLY_STORAGE, ID_ENEMY_STORAGE, ID_ALLY_UNIT, ID_ENEMY_UNIT, ID_ITEM = range(7)


def official_background(lay) -> np.ndarray:
    """``RenderBackground`` for team A, rows top-down: every pixel samples its tile."""
    tile = np.zeros((W, H), np.uint8)
    for x in range(W):
        for y in range(H):
            if TILE_COLLISION[lay.tile[x, y]] == BLOCK_ALL:
                tile[x, y] = ID_WALL
            elif lay.reg_kind[lay.region[x, y]] == R_STORAGE:
                tile[x, y] = ID_ALLY_STORAGE if lay.reg_team[lay.region[x, y]] == 0 else ID_ENEMY_STORAGE
    up = np.repeat(np.repeat(tile, SCALE, 0), SCALE, 1)   # [x_px, y_px]
    return np.ascontiguousarray(up.T[::-1])                # [row = HO-1-y_px, x_px]


def official_map(sim) -> np.ndarray:
    """``SemanticMapRenderer.Render`` (official): uint8 ids [96, 96], team A view, rows top-down."""
    bg = getattr(sim, "_official_bg", None)
    if bg is None:
        bg = sim._official_bg = official_background(sim.layout)
    out = bg.copy()
    it = sim.it
    for i in np.nonzero((it[:, core.I_STATE] == core.GROUND) & (it[:, core.I_TX] >= 0))[0]:
        px, py = SCALE * it[i, core.I_TX] + SCALE // 2, SCALE * it[i, core.I_TY] + SCALE // 2
        out[HO - 1 - py, px] = ID_ITEM + it[i, core.I_TYPE]
    for u in range(10):
        px = int(np.floor(F32(sim.uf[u, 0] * F32(SCALE))))
        py = int(np.floor(F32(sim.uf[u, 1] * F32(SCALE))))
        if 0 <= px < WO and 0 <= py < HO:
            out[HO - 1 - py, px] = ID_ALLY_UNIT if u < 5 else ID_ENEMY_UNIT
    return out


def official_vector(sim, observer: int) -> np.ndarray:
    """``BlackOutAgent.CollectObservations`` (official) of unit ``observer``: float32[45]."""
    out = np.zeros(45, F32)
    my_team = 0 if observer < 5 else 1
    for u in range(10):
        out[4 * u] = sim.uf[u, 0] / core.BOUNDS
        out[4 * u + 1] = sim.uf[u, 1] / core.BOUNDS
        out[4 * u + 2] = 1.0 if (u < 5) == (my_team == 0) else -1.0
        h = sim.ui[u, core.U_HELD]
        out[4 * u + 3] = 0.0 if h < 0 else F32(sim.it[h, core.I_TYPE] + 1)
    out[40] = sim.ui[observer, core.U_CLS]
    sa, sb = F32(sim.g[core.G_SCORE_A]) / F32(core.TARGET_SCORE), F32(sim.g[core.G_SCORE_B]) / F32(core.TARGET_SCORE)
    out[41], out[42] = (sa, sb) if my_team == 0 else (sb, sa)
    r = sim.gf[core.GF_EP] / core.EPISODE_SECONDS
    out[43] = core.F1 - min(max(r, core.F0), core.F1)
    out[44] = observer
    return out


def parse_vector(vec) -> dict:
    """Official float32[45] -> fork-format pieces (team A / B absolute where it matters).

    ``pos_obs``: float32 [10, 2] in the fork's [-1, 1] normalisation -- bit-identical to what
    MapObsAgent would send, since both start from the same float32 ``pos / 24``."""
    v = np.asarray(vec, dtype=F32)
    me = int(round(float(v[44])))
    my_team = 0 if me < 5 else 1
    blocks = v[:40].reshape(10, 4)
    pos_obs = (blocks[:, :2] * F32(2.0) - F32(1.0)).astype(F32)
    held = np.rint(blocks[:, 3]).astype(np.int64)            # 0 none, 1 battery, 2-5 specials
    own, opp = int(round(float(v[41]) * 100)), int(round(float(v[42]) * 100))
    scores = (own, opp) if my_team == 0 else (opp, own)
    return {"me": me, "team": my_team, "pos_obs": pos_obs, "held": held, "my_class": int(round(float(v[40]))),
            "scores": scores, "time_left": F32(v[43])}


N_ITEMS, N_CLASSES = 5, 3
UNIT_PRE = 2 + 1 + N_ITEMS + 1                     # pos, team sign, holding one-hot
VECTOR_PRE = 10 * UNIT_PRE + N_CLASSES + 3          # 96: what the submitted model receives


def official_preprocessed(sim, observer: int) -> dict:
    """What the official ``BlackOutEnv`` hands the model for ``observer``: {"vector": float32[96],
    "graphic": float32[96, 96, 11]} (ObsPreprocessor + flip_team_perspective for team B)."""
    raw = official_vector(sim, observer)
    vec = np.zeros(VECTOR_PRE, F32)
    for u in range(10):
        b = u * UNIT_PRE
        vec[b:b + 3] = raw[4 * u:4 * u + 3]
        vec[b + 3 + int(round(float(raw[4 * u + 3])))] = 1.0
    vec[10 * UNIT_PRE + int(raw[40])] = 1.0
    vec[10 * UNIT_PRE + N_CLASSES:] = raw[41:44]
    ids = official_map(sim)
    if observer >= 5:
        ids = _flip_ids(ids)
    return {"vector": vec, "graphic": (ids[:, :, None] == np.arange(ID_ITEM + N_ITEMS)).astype(F32)}


def _flip_ids(ids):
    out = ids.copy()
    for a, b in ((ID_ALLY_STORAGE, ID_ENEMY_STORAGE), (ID_ALLY_UNIT, ID_ENEMY_UNIT)):
        out[ids == a], out[ids == b] = b, a
    return out


def from_preprocessed(vector, graphic, me: int):
    """Inverse of ``official_preprocessed``: (official raw float32[45], team A id map uint8 [96, 96]).
    ``me``: the observer's unit index -- not in the preprocessed vector; the runner batches a
    team's agents in agent order (row k = unit 5 * team + k).  ``graphic`` HWC or CHW."""
    v = np.asarray(vector, dtype=F32)
    raw = np.zeros(45, F32)
    for u in range(10):
        b = u * UNIT_PRE
        raw[4 * u:4 * u + 3] = v[b:b + 3]
        oh = v[b + 3:b + UNIT_PRE]
        raw[4 * u + 3] = float(np.argmax(oh)) if oh.max() > 0 else 0.0
    raw[40] = float(np.argmax(v[10 * UNIT_PRE:10 * UNIT_PRE + N_CLASSES]))
    raw[41:44] = v[10 * UNIT_PRE + N_CLASSES:]
    raw[44] = me
    g = np.asarray(graphic)
    if g.shape[0] == ID_ITEM + N_ITEMS and g.shape[-1] != ID_ITEM + N_ITEMS:
        g = np.moveaxis(g, 0, -1)
    ids = np.argmax(g, axis=-1).astype(np.uint8)
    if me >= 5:
        ids = _flip_ids(ids)
    return raw, ids


def team_of_preprocessed(vector) -> int:
    """0 when unit 0 is an ally of the observer (team A), else 1."""
    return 0 if float(np.asarray(vector)[2]) > 0 else 1
