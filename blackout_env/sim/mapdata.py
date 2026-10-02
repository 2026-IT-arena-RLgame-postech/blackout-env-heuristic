"""The static BlackOut map, transcribed from the Unity scene (Prototype.unity) and data assets.

World frame = Unity's: cell (x, y) covers [x, x+1) x [y, y+1), centre (x+0.5, y+0.5), y up.
The semantic map observation stores row ``r = 23 - y``.

What is fixed every match (``ProceduralMapGenerator`` + ``Tilemap_Terrain``):
  * the painted tiles (walls, floor, spawn points, sanctuary sites, storage floor),
  * the fixed regions: the Hunter sanctuary (2x2 centre, neutral), one locked Carrier sanctuary
    per team, the protected base storage A1/B1, the 4x4 team bases (their floor blocks enemies),
  * the default (neutral) region over everything else.
What varies: ``StorageCountPerTeam`` (4) of the 5 storage candidates of team A are chosen by a
seeded shuffle; team B gets their y = x mirror images.  Unchosen candidate tiles keep the painted
``Floor_Storage`` tile data in the neutral default region: walkable for both teams, but neither
a storage nor an item-spawn tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

W = H = 24

# Tilemap_Terrain, top row first (y = 23 ... 0).  # wall . floor S spawn H hunter site
# s storage floor C carrier site.
TERRAIN = (
    "########################",
    "#S.s...##ss..ss##ss....#",
    "#..s...##ss..ss##ss....#",
    "#C.....................#",
    "#......................#",
    "#......##########....ss#",
    "#....................ss#",
    "#####.............#..###",
    "#.......sss#......#..###",
    "#.......s###......#..ss#",
    "#s......s#........#..ss#",
    "#s...#..##.HH.....#....#",
    "#s...#.....HH.##..#....#",
    "#s...#........#s..#..ss#",
    "#s...#......###s..#..ss#",
    "######......#sss..#..###",
    "#ss...............#..###",
    "#ss....................#",
    "#.......#####..........#",
    "#.......#.......#......#",
    "#.......#.......#....ss#",
    "#.....ss#.......#......#",
    "#.....ss#sssss..#...C.S#",
    "########################",
)

# tile data assets
FLOOR, WALL, STORAGE_FLOOR, SITE_HUNTER, SITE_CARRIER, SPAWN = 0, 1, 2, 3, 4, 5
_CHAR = {".": FLOOR, "#": WALL, "s": STORAGE_FLOOR, "H": SITE_HUNTER, "C": SITE_CARRIER, "S": SPAWN}
# The build differs from the scene file here: unchosen storage *candidate* cells behave as Floor
# (they receive batteries; the spawn filter is TileData == Floor), while the two decorative 2x2
# storage-floor patches at (1-2, 6-7) / (6-7, 1-2) never do.  Established from the initial
# battery layouts of 40 recorded matches together with the seeded RNG replay (unity_random.py),
# which only reproduces Unity's battery draws with this tile set.
# TileCollisionOption: 0 Pass, 2 BlockEnemy, 3 BlockAll (MapTileData assets)
PASS, BLOCK_ENEMY, BLOCK_ALL = 0, 2, 3
TILE_COLLISION = np.array([BLOCK_ENEMY, BLOCK_ALL, PASS, BLOCK_ENEMY, BLOCK_ENEMY, BLOCK_ENEMY], dtype=np.int32)
TILE_HALF = np.array([0.5, 0.5, 0.5, 0.45, 0.45, 0.5], dtype=np.float32)  # CollisionBound.Width / 2

TEAM_A, TEAM_B, NEUTRAL = 0, 1, 2
SPAWN_CELL = ((1, 22), (22, 1))  # (x, y) per team

# region kinds
R_DEFAULT, R_SANCTUARY, R_STORAGE = 0, 1, 2
# unit classes (KnownClasses order)
COLLECTOR, HUNTER, CARRIER = 0, 1, 2

# (name, team, rects (x, y, w, h), kind, sanctuary in/out class, locked) -- fixedRegions order
FIXED_REGIONS = (
    ("Hunter Sanctuary", NEUTRAL, ((11, 11, 2, 2),), R_SANCTUARY, COLLECTOR, HUNTER, False),
    ("Carrier Sanctuary A", TEAM_A, ((1, 20, 1, 1),), R_SANCTUARY, COLLECTOR, CARRIER, True),
    ("Carrier Sanctuary B", TEAM_B, ((20, 1, 1, 1),), R_SANCTUARY, COLLECTOR, CARRIER, True),
    ("Storage A1", TEAM_A, ((3, 21, 1, 2),), R_STORAGE, -1, -1, False),
    ("Storage B1", TEAM_B, ((21, 3, 2, 1),), R_STORAGE, -1, -1, False),
    ("Team A Base", TEAM_A, ((1, 19, 4, 4),), R_DEFAULT, -1, -1, False),
    ("Team B Base", TEAM_B, ((19, 1, 4, 4),), R_DEFAULT, -1, -1, False),
)
# storageCandidatesA (team B: the y = x mirror, RectInt(y, x, h, w))
STORAGE_CANDIDATES = (
    ("Storage_A2", ((9, 21, 2, 2),)),
    ("Storage_A3", ((13, 21, 2, 2),)),
    ("Storage_A4", ((17, 21, 2, 2),)),
    ("Storage_A5", ((8, 13, 1, 3), (8, 15, 3, 1))),
    ("Storage_A6", ((9, 1, 5, 1),)),
)
STORAGE_COUNT = 4
MAX_STORAGE_TILES = 8


def terrain() -> np.ndarray:
    """Painted tile kind per cell, indexed [x, y]."""
    out = np.zeros((W, H), dtype=np.int32)
    for r, line in enumerate(TERRAIN):
        y = H - 1 - r
        for x, ch in enumerate(line):
            out[x, y] = _CHAR[ch]
    for _, rects in STORAGE_CANDIDATES:
        for rx, ry, rw, rh in rects + _mirror(rects):
            out[rx:rx + rw, ry:ry + rh] = FLOOR
    return out


def _mirror(rects):
    return tuple((y, x, h, w) for (x, y, w, h) in rects)


@dataclass
class Layout:
    """One match's map: tiles, regions and their ordered storage tiles (numba-ready arrays)."""

    chosen: tuple[int, ...]           # indices into STORAGE_CANDIDATES, in shuffle order
    tile: np.ndarray                  # int32 [W, H] tile data kind
    region: np.ndarray                # int32 [W, H] region id
    reg_kind: np.ndarray              # int32 [R]
    reg_team: np.ndarray              # int32 [R]
    reg_in: np.ndarray                # int32 [R] sanctuary input class
    reg_out: np.ndarray               # int32 [R] sanctuary output class
    reg_locked: np.ndarray            # bool  [R] unique-instance sanctuary
    reg_tiles: np.ndarray             # int32 [R, MAX_STORAGE_TILES, 2] storage fill order (x, y)
    reg_ntiles: np.ndarray            # int32 [R]
    walk: np.ndarray                  # bool  [2, W, H] MapManager.IsWalkable per team
    half: np.ndarray                  # float32 [W, H] tile collision half width
    names: list[str]

    @property
    def n_regions(self) -> int:
        return len(self.reg_kind)


def build_layout(chosen) -> Layout:
    """``GenerateMapData`` for a given storage choice (candidate indices in shuffle order)."""
    chosen = tuple(int(c) for c in chosen)
    paint = terrain()
    tile = np.full((W, H), -1, dtype=np.int32)
    region = np.full((W, H), -1, dtype=np.int32)
    kinds, teams, ins, outs, locked, tiles, names = [], [], [], [], [], [], []

    def add(name, team, rects, kind, cin=-1, cout=-1, lock=False, anchor=None, procedural=False):
        rid = len(kinds)
        kinds.append(kind), teams.append(team), ins.append(cin), outs.append(cout), locked.append(lock)
        names.append(name)
        cells = []
        for (rx, ry, rw, rh) in rects:
            for y in range(ry, ry + rh):
                for x in range(rx, rx + rw):
                    if not (0 <= x < W and 0 <= y < H) or region[x, y] >= 0:
                        continue
                    tile[x, y] = STORAGE_FLOOR if procedural else paint[x, y]
                    region[x, y] = rid
                    cells.append((x, y))
        if kind == R_STORAGE and anchor is not None:  # Storage.Initialize: farthest first, then x, y
            ax, ay = anchor
            cells.sort(key=lambda c: (-(abs(c[0] - ax) + abs(c[1] - ay)), c[0], c[1]))
        tiles.append(cells)

    for name, team, rects, kind, cin, cout, lock in FIXED_REGIONS:
        add(name, team, rects, kind, cin, cout, lock)
    for c in chosen[:STORAGE_COUNT]:
        name, rects = STORAGE_CANDIDATES[c]
        add(name, TEAM_A, rects, R_STORAGE, anchor=SPAWN_CELL[TEAM_A], procedural=True)
        add(name.replace("_A", "_B"), TEAM_B, _mirror(rects), R_STORAGE, anchor=SPAWN_CELL[TEAM_B], procedural=True)
    rid = len(kinds)
    kinds.append(R_DEFAULT), teams.append(NEUTRAL), ins.append(-1), outs.append(-1), locked.append(False)
    names.append("Default")
    tiles.append([])
    for x in range(W):
        for y in range(H):
            if region[x, y] < 0:
                region[x, y] = rid
                tile[x, y] = paint[x, y]

    n = len(kinds)
    reg_tiles = np.full((n, MAX_STORAGE_TILES, 2), -1, dtype=np.int32)
    reg_ntiles = np.zeros(n, dtype=np.int32)
    for r, cells in enumerate(tiles):
        if kinds[r] == R_STORAGE:
            reg_ntiles[r] = len(cells)
            reg_tiles[r, : len(cells)] = cells
    reg_team = np.asarray(teams, dtype=np.int32)
    walk = np.zeros((2, W, H), dtype=bool)
    for team in (TEAM_A, TEAM_B):
        opp = 1 - team
        for x in range(W):
            for y in range(H):
                opt = TILE_COLLISION[tile[x, y]]
                if opt == PASS:
                    walk[team, x, y] = True
                elif opt == BLOCK_ENEMY:
                    walk[team, x, y] = reg_team[region[x, y]] != opp
    return Layout(
        chosen=chosen, tile=tile, region=region, reg_kind=np.asarray(kinds, dtype=np.int32), reg_team=reg_team,
        reg_in=np.asarray(ins, dtype=np.int32), reg_out=np.asarray(outs, dtype=np.int32),
        reg_locked=np.asarray(locked, dtype=bool), reg_tiles=reg_tiles, reg_ntiles=reg_ntiles, walk=walk,
        half=TILE_HALF[tile].astype(np.float32), names=names,
    )


def storage_cells(chosen) -> tuple[set, set]:
    """Cells of team A / team B storage regions for a storage choice."""
    lay = build_layout(chosen)
    a, b = set(), set()
    for r in range(lay.n_regions):
        if lay.reg_kind[r] == R_STORAGE:
            cells = {tuple(c) for c in lay.reg_tiles[r, : lay.reg_ntiles[r]]}
            (a if lay.reg_team[r] == TEAM_A else b).update(cells)
    return a, b


def chosen_from_map(raw_map: np.ndarray) -> tuple[int, ...]:
    """Which storage candidates a packed semantic map (team A view) shows.  The shuffle order
    among them is not observable; it only orders region ids, which the rules never depend on."""
    base = np.asarray(raw_map).astype(np.int64) & 7
    ally = {(x, H - 1 - r) for r, x in zip(*np.nonzero(base == 6))}
    out = []
    for i, (_, rects) in enumerate(STORAGE_CANDIDATES):
        cells = {(x, y) for (rx, ry, rw, rh) in rects for x in range(rx, rx + rw) for y in range(ry, ry + rh)}
        if cells <= ally:
            out.append(i)
    if len(out) != STORAGE_COUNT:
        raise ValueError(f"map shows storage candidates {out}, expected {STORAGE_COUNT}")
    return tuple(out)
