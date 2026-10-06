"""O1 (docs/policy_v2_design_ko.md "O1"): the V22-V30 speed-ups are bit-identical to the code they replace.

Each fast path is checked against the reference it keeps (``_segment_clear_ref``, ``_seg_clear_exact_ref``,
``_progress_moves``, ``_economy_pairs_py``, the float32 ``_to_pixel`` formula, the full ``_node_adjacency``),
and whole sim matches against action hashes recorded with the code before O1 (commit 78e1f72).
"""

import hashlib
import math

import numpy as np
import pytest

from blackout_env.heuristics import _shared
from blackout_env.heuristics import any_angle as A
from blackout_env.heuristics import race_path as R
from blackout_env.heuristics import teach as T
from blackout_env.heuristics.strategic import WALL, StrategicHeuristic

V30 = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28 = "strategic_v28:reserve_stage=0.5,trade_hold=advance"


def _walls(seed=3):
    from blackout_env.sim import SimState

    return np.ascontiguousarray(SimState.from_seed(seed).obs()["unit_0"]["graphic"][..., WALL] >= 0.5)


def _segments(rng, blocked, n):
    free = np.argwhere(~blocked)
    hard = np.argwhere(blocked)
    for k in range(n):
        mode = k % 4
        if mode == 0:
            a, b = rng.uniform(-1.5, 25, 2), rng.uniform(-1.5, 25, 2)
        elif mode == 1:
            a = free[rng.integers(len(free))] + rng.uniform(-.5, .5, 2)
            b = free[rng.integers(len(free))] + rng.uniform(-.5, .5, 2)
        elif mode == 2:  # grazing a blocked square at about the clearance
            c = hard[rng.integers(len(hard))]
            off = 0.5 + rng.choice([0.275, 0.325, 0.375, 0.399, 0.4, 0.025])
            a = c + np.array([off * rng.choice([-1, 1]), -6.0 * rng.random()])
            b = a + np.array([0.0, 12.0 * rng.random()])
            if rng.random() < .5:
                a, b = a[::-1].copy(), b[::-1].copy()
        else:  # cell centres and corners
            a = rng.integers(0, 24, 2) + rng.choice([0, 0.5], 2)
            b = rng.integers(0, 24, 2) + rng.choice([0, 0.5], 2)
        yield float(a[0]), float(a[1]), float(b[0]), float(b[1])


@pytest.mark.parametrize("seed", [0, 1])
def test_segment_clear_matches_reference(seed):
    rng = np.random.default_rng(seed)
    for blocked in (_walls(), np.ascontiguousarray(rng.random((24, 24)) < 0.25)):
        for seg in _segments(rng, blocked, 3000):
            for req in (0.0, 0.024, 0.275, 0.325, 0.399, 0.4, 0.5, 0.6):
                for step in (0.05, 0.1):
                    assert A._segment_clear(blocked, *seg, req, step) == A._segment_clear_ref(blocked, *seg, req, step)


def test_race_segment_matches_reference():
    rng = np.random.default_rng(2)
    for blocked in (_walls(), np.ascontiguousarray(rng.random((24, 24)) < 0.25)):
        for seg in _segments(rng, blocked, 4000):
            for req in (0.0, 0.276, 0.2765, 0.5, 0.9):
                assert R.seg_clear_exact(blocked, *seg, req) == R._seg_clear_exact_ref(blocked, *seg, req)


def test_progress_bits_match_progress_moves():
    rng = np.random.default_rng(3)
    for k in range(3000):
        walkable = rng.random((24, 24)) < 0.8
        distance = (rng.random((24, 24)) * 30).astype(np.float32)
        if k % 3 == 0:
            distance = np.round(distance).astype(np.float32)  # exact ties
        distance[~walkable] = np.inf
        pos = (int(rng.integers(24)), int(rng.integers(24)))
        mask = T.TeachLabeller._progress_moves(distance, pos, walkable, (24, 24))
        assert np.array_equal(mask, T._MASKS[T.TeachLabeller._progress_bits(distance, pos, walkable, (24, 24))])


def test_mask_tables_are_the_expressions():
    for bits in range(1, 256):
        mask = T._MASKS[bits]
        mean = T.DIRECTIONS[mask].mean(axis=0)
        assert np.array_equal(T._MEAN[bits], mean) and T._MEAN_NORM[bits] == float(np.linalg.norm(mean))
        base = mean / max(float(np.linalg.norm(mean)), 1e-6)
        assert np.array_equal(A._BASE[bits], base)
        if T._MEAN_NORM[bits] >= T._CANCELLING:  # every valid label
            label = T.TeachLabel(mask.copy(), True, "x", None)
            assert np.array_equal(T.label_action(label), (mean / float(np.linalg.norm(mean))).astype(np.float32))


def test_to_pixel_matches_float32_formula():
    rng = np.random.default_rng(4)
    half = ((np.arange(-1, 25) + 1.0) / 12.0 - 1.0).astype(np.float32)  # positions on cell boundaries
    points = np.concatenate([rng.uniform(-1.05, 1.05, (50000, 2)).astype(np.float32),
                             np.stack(np.meshgrid(half, half), -1).reshape(-1, 2)])
    for p in points:
        y = float(np.float32((1.0 - float(p[1])) * 0.5 * 24 - 0.5))
        x = float(np.float32((float(p[0]) + 1.0) * 0.5 * 24 - 0.5))
        assert StrategicHeuristic._to_pixel(p, (24, 24)) == (min(23, max(0, int(round(y)))), min(23, max(0, int(round(x)))))


def test_math_dist_on_grid_cells_is_sqrt():
    for dy in range(-30, 31):
        for dx in range(-30, 31):
            assert math.dist((0, 0), (dy, dx)) == math.sqrt(dy * dy + dx * dx)


def test_derived_field_graph_matches_full_adjacency():
    rng = np.random.default_rng(5)
    walkable = ~_walls()
    A._FIELD_GRAPHS.clear()
    A._GRAPH_PARENTS.clear()
    A._field_graph(walkable, A.FIELD_CLEARANCE, A.FIELD_STEP)  # the parent
    rows, cols = np.mgrid[:24, :24]
    for k in range(40):
        child = walkable.copy()
        for _ in range(int(rng.integers(1, 3))):
            r, c = rng.integers(0, 24, 2)
            child &= ~((rows - r) ** 2 + (cols - c) ** 2 <= rng.choice([1, 4, 9]))
        nodes, adj = A._field_graph(child, A.FIELD_CLEARANCE, A.FIELD_STEP)
        full = A._node_adjacency(child, A._corner_nodes(child, A.FIELD_CLEARANCE), A.FIELD_CLEARANCE - 1e-3,
                                 A.FIELD_STEP)
        assert np.array_equal(nodes, A._corner_nodes(child, A.FIELD_CLEARANCE)) and np.array_equal(adj, full)


def _match_hash(a, b, seed, ticks, check=None):
    from blackout_env.sim.state import SimState, obs_from_raw
    from blackout_env.train.distill.opponents import make_opponent

    sim = SimState.from_seed(seed)
    pols = [make_opponent(a, 0), make_opponent(b, 1)]
    h = hashlib.sha256()
    for t in range(ticks):
        if sim.over:
            break
        obs = obs_from_raw(sim.raw_state(), sim.raw_map())
        mv = np.zeros((10, 2), np.float32)
        for team, pol in enumerate(pols):
            for name, act in pol.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * team, 5 * team + 5)}).items():
                mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
            if check is not None:
                check(pol)
        h.update(mv.tobytes())
        sim.step(mv)
    return h.hexdigest()[:16], sim.scores


# Recorded with the pre-O1 code (78e1f72): 700 decisions of each match, both teams' moves.
GOLDEN = [
    (V30, V28, 4_000_003, "1f78feb843e29ed3", (64, 25)),
    (V28, V30, 4_000_005, "394184b9f4608460", (23, 100)),
    ("strategic_v25", "strategic_v27:reserve_stage=1.5", 4_000_007, "55724f95df7ee2ec", (39, 70)),
    ("strategic_v21", "strategic_v19", 4_000_011, "19ece4f9f885c7c2", (57, 23)),
]


@pytest.mark.parametrize("a,b,seed,digest,scores", GOLDEN)
def test_matches_reproduce_pre_o1_moves(a, b, seed, digest, scores):
    assert _match_hash(a, b, seed, 700) == (digest, scores)


def test_process_caches_do_not_change_play():
    """A cold process (empty shared caches) and a warm one play the same match."""
    _shared.clear_all()
    A._FIELD_GRAPHS.clear()
    A._GRAPH_PARENTS.clear()
    cold = _match_hash(V30, V28, 4_000_013, 400)
    warm = _match_hash(V30, V28, 4_000_013, 400)
    assert cold == warm and _shared.GOAL_FIELDS.hits > 0


def test_economy_kernel_matches_reference():
    """``_economy_pairs`` (numba) and ``_economy_pairs_py`` agree on every V30 context of a match."""
    seen = [0]

    def check(pol):
        ctx = getattr(pol, "_teach_ctx", None)
        if ctx is None:
            return
        units = [u for u in ctx["own"] if u.cls != 1 and not u.holding]
        if not units:
            return
        fast = pol._economy_pairs(ctx, units)
        ref = pol._economy_pairs_py(ctx, units)
        assert fast[0] == ref[0] and fast[1] == ref[1] and fast[2] == ref[2]
        seen[0] += 1

    _match_hash(V30, V28, 4_000_017, 500, check)
    assert seen[0] > 100


def test_race_planner_query_cache_is_the_query():
    walkable = ~_walls()
    from blackout_env.sim import SimState
    from blackout_env.heuristics.strategic import SITE_HUNTER

    g = SimState.from_seed(3).obs()["unit_0"]["graphic"]
    tiles = tuple(sorted((int(y), int(x)) for y, x in np.argwhere((g[..., SITE_HUNTER] > 0.5) & walkable)))
    planner = R.RacePlanner(walkable, tiles, 0.275)
    rng = np.random.default_rng(6)
    cells = np.argwhere(walkable)
    for _ in range(300):
        p = tuple(cells[rng.integers(len(cells))] + rng.uniform(-.3, .3, 2))
        raw = R._query(planner.blocked, planner.nodes, planner.node_g, planner.boxes, planner.inset,
                       float(p[0]), float(p[1]), planner.q_required)
        assert planner.distance(p) == float(raw[0]) and planner.distance(p) == float(raw[0])
