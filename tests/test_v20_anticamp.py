import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV17Teach, StrategicHeuristicV20Teach
from blackout_env.heuristics.teach import MOVES

COLLECTOR, HUNTER, CARRIER = 0, 1, 2


def _xy(row: int, col: int) -> tuple[float, float]:
    return (col + 0.5) / 12 - 1, 1 - (row + 0.5) / 12


def _obs(own: dict[int, tuple[int, int]], camper: tuple[int, int] | None, walls=()):
    """Our base is the 4x4 corner at (0,0).  Units not listed stand far away, unit 4 as our Carrier (so
    the listed units are not sent to transform into one)."""
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[0:2, 2:4, 3] = 1       # carrier sanctuary, inside our base as in the game
    graphic[11:13, 11:13, 2] = 1   # hunter sanctuary
    graphic[1:3, 1:3, 6] = 1       # own storage
    graphic[20:23, 20:23, 7] = 1   # enemy storage
    graphic[0, 0, 4] = 1
    graphic[23, 23, 5] = 1
    graphic[18, 6, 8] = graphic[19, 6, 8] = graphic[18, 7, 8] = 10 / 15  # the Carrier's
    graphic[8, 3, 8] = 8 / 15  # unit 0's: straight down from the base, past (5, 5)
    for r, c in walls:
        graphic[r, c, 1] = 1
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        pos = own.get(i, (16, 16 + (i % 5))) if i < 5 else (22, 12 + i)
        cls = CARRIER if i == 4 and 4 not in own else COLLECTOR
        if i == 5 and camper is not None:
            pos, cls = camper, HUNTER
        states[i, :2] = _xy(*pos)
        states[i, 2] = 1 if i < 5 else -1
        states[i, 3] = 1
        states[i, 9 + cls] = 1
    return {f"unit_{i}": {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}
            for i in range(5)}


def _next_tiles(label, pos):
    return {(pos[0] + MOVES[i][0], pos[1] + MOVES[i][1]) for i in np.flatnonzero(label.mask)}


def test_routes_around_a_camper_when_a_way_exists():
    obs = _obs({0: (2, 2)}, camper=(5, 5))
    v17 = StrategicHeuristicV17Teach().teach(obs)["unit_0"]
    v20 = StrategicHeuristicV20Teach().teach(obs)["unit_0"]
    assert v20.valid and v20.kind == v17.kind and not np.array_equal(v20.mask, v17.mask)
    # never steps toward the camper's zone; our own base cells are safe
    for r, c in _next_tiles(v20, (2, 2)):
        assert (r <= 3 and c <= 3) or (r - 5) ** 2 + (c - 5) ** 2 > 9


def test_no_camper_changes_nothing():
    obs = _obs({0: (2, 2)}, camper=None)
    assert np.array_equal(StrategicHeuristicV17Teach().teach(obs)["unit_0"].mask,
                          StrategicHeuristicV20Teach().teach(obs)["unit_0"].mask)


# The base has one way out, at (4, 3), covered by a Hunter at (6, 3).
_WALLS = [(4, c) for c in range(0, 8) if c != 3] + [(r, 4) for r in range(0, 4)]


def test_a_lone_blocked_unit_waits_inside_and_two_break_out_together():
    teacher = StrategicHeuristicV20Teach()
    alone = teacher.teach(_obs({0: (2, 2)}, camper=(6, 3), walls=_WALLS))
    assert teacher.waiting == ["unit_0"] and not alone["unit_0"].valid
    pair = teacher.teach(_obs({0: (2, 2), 1: (1, 1)}, camper=(6, 3), walls=_WALLS))
    assert teacher.waiting == [] and pair["unit_0"].valid and pair["unit_1"].valid


def test_v20_is_a_registered_heuristic_that_plays_its_labels():
    policy = POLICY_REGISTRY["strategic_v20"]()
    actions = policy.act(_obs({0: (2, 2)}, camper=(5, 5)))
    assert set(actions) == {f"unit_{i}" for i in range(5)}
    assert all(np.isfinite(a).all() and a.shape == (2,) for a in actions.values())
