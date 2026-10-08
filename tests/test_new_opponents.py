"""Strategy opponents (raider, cargo hunter, early rush, fortress, duelist) and the perturbation wrapper."""

import numpy as np
import pytest

from blackout_env.heuristics import POLICY_REGISTRY, make_heuristic
from blackout_env.train.distill.opponents import check_opponent, make_opponent, opponent_label, opponent_name
from blackout_env.train.distill.perturb import PerturbedPolicy, split_descriptor

COLLECTOR, HUNTER, CARRIER = 0, 1, 2
NEW_IDS = ("strategic_raider", "strategic_cargo_hunter", "strategic_early_rush", "strategic_fortress", "strategic_duelist")
OWN = [f"unit_{i}" for i in range(5)]


def _xy(row: int, col: int) -> tuple[float, float]:
    return (col + 0.5) / 12 - 1, 1 - (row + 0.5) / 12


def _obs(units: dict[int, tuple[tuple[int, int], int, float]], *, time_left: float = 1.0, enemy_loot: float = 8.0):
    """24x24 map: our base is the corner at (0, 0), the enemy's at (23, 23).  ``units`` maps a unit
    index (0-4 ours, 5-9 theirs) to (tile, class, battery carried); unlisted units stand in their
    own base.  Each team has a protected storage by its spawn and an exposed one mid-map; the enemy's
    exposed storage holds ``enemy_loot``."""
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[11:13, 11:13, 2] = 1   # hunter sanctuary
    graphic[0:2, 2:4, 3] = 1       # carrier sanctuary, inside our base
    graphic[0, 0, 4] = 1
    graphic[23, 23, 5] = 1
    graphic[1:3, 1:3, 6] = 1       # our protected storage
    graphic[7:9, 15:17, 6] = 1     # our exposed storage
    graphic[21:23, 21:23, 7] = 1   # their protected storage
    graphic[15:17, 7:9, 7] = 1     # their exposed storage
    graphic[15, 7, 8] = enemy_loot / 15
    for r, c in [(5, 10), (10, 5), (18, 13), (13, 18)]:
        graphic[r, c, 8] = 5 / 15  # field batteries
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        default = ((1, 1 + i % 3), COLLECTOR, 0.0) if i < 5 else ((22, 20 + i % 3), COLLECTOR, 0.0)
        pos, cls, cargo = units.get(i, default)
        states[i, :2] = _xy(*pos)
        states[i, 2] = 1 if i < 5 else -1
        if cargo:
            states[i, 4] = cargo / 15
        else:
            states[i, 3] = 1
        states[i, 9 + cls] = 1
    team_state = np.array([0, 0, time_left, 0.5], np.float32)
    return {name: {"graphic": graphic, "team_state": team_state, "agent_states": states} for name in OWN}


def _valid(actions, names=OWN):
    assert set(actions) == set(names)
    for action in actions.values():
        assert action.shape == (2,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1.0 + 1e-6)


@pytest.mark.parametrize("policy_id", NEW_IDS)
def test_new_strategies_are_registered_and_play(policy_id):
    policy = make_heuristic(policy_id)
    busy = _obs({0: ((6, 6), HUNTER, 0.0), 5: ((14, 14), HUNTER, 0.0), 6: ((10, 12), COLLECTOR, 6.0), 7: ((9, 9), CARRIER, 0.0)})
    for obs in (_obs({}), busy, _obs({}, time_left=0.5)):
        _valid(policy.act(obs))
    partial = {name: view for name, view in busy.items() if name in ("unit_1", "unit_3")}
    _valid(policy.act(partial), partial)
    policy.reset()
    _valid(policy.act(busy))
    assert policy_id in POLICY_REGISTRY and make_opponent(policy_id, 0).__class__ is POLICY_REGISTRY[policy_id]


def test_new_strategies_stay_out_of_the_default_mixture():
    from blackout_env.heuristics import HeuristicPolicyMixture

    assert not set(NEW_IDS) & set(HeuristicPolicyMixture(seed=0).weights)


def test_raider_squad_skips_the_field_and_its_escort_guards_their_storage():
    policy = make_heuristic("strategic_raider")
    policy.act(_obs({0: ((12, 6), HUNTER, 0.0), 1: ((6, 6), COLLECTOR, 0.0), 2: ((8, 4), COLLECTOR, 0.0)}))
    assert len(policy.squad) == 2
    assert all(policy.plan[name][1] != "battery" for name in policy.squad)
    target, kind = policy.plan["unit_0"]
    assert kind == "patrol_defend" and 14 <= target[0] <= 17 and 6 <= target[1] <= 9  # their exposed storage


def test_cargo_hunter_ignores_empty_collectors_and_intercepts_cargo():
    policy = make_heuristic("strategic_cargo_hunter")
    empty = _obs({0: ((10, 10), HUNTER, 0.0), 5: ((11, 13), COLLECTOR, 0.0)})
    policy.act(empty)
    assert policy.plan["unit_0"][1] == "patrol_defend"
    loaded = _obs({0: ((10, 10), HUNTER, 0.0), 5: ((13, 13), COLLECTOR, 8.0)})
    policy.act(loaded)
    target, kind = policy.plan["unit_0"]
    assert kind == "hunt" and target is not None


def test_early_rush_sweeps_with_four_hunters_then_gives_hunters_back():
    policy = make_heuristic("strategic_early_rush")
    policy.act(_obs({i: ((1, 1 + i % 3), COLLECTOR, 0.0) for i in range(5)}))
    assert policy.phase == "rush" and list(policy.role_assignments.values()).count("hunter") == 4
    policy.act(_obs({0: ((10, 10), HUNTER, 0.0), 5: ((12, 8), COLLECTOR, 0.0)}))
    assert policy.plan["unit_0"] == ((12, 8), "hunt")  # an empty Collector is worth a sweep
    late = _obs({0: ((10, 10), HUNTER, 0.0), 1: ((6, 6), HUNTER, 0.0), 5: ((14, 14), HUNTER, 0.0)}, time_left=0.5)
    policy.act(late)
    assert policy.phase == "economy" and policy.hunter_quota == 1
    assert "hunter" not in {policy.role_assignments[f"unit_{i}"] for i in range(2, 5)}  # no new transforms
    assert policy.plan["unit_1"] == ((14, 14), "hunt")  # the surplus Hunter goes for a trade


def test_fortress_drops_tasks_near_enemy_hunters_and_holds_its_posts():
    policy = make_heuristic("strategic_fortress")
    policy.act(_obs({0: ((6, 6), HUNTER, 0.0), 2: ((6, 9), COLLECTOR, 0.0), 5: ((5, 11), HUNTER, 0.0), 6: ((20, 3), COLLECTOR, 0.0)}))
    assert policy.plan["unit_2"][0] != (5, 10)  # the battery beside their Hunter
    target, kind = policy.plan["unit_0"]  # their Hunter is 7 tiles from our exit post: out of the zone
    assert kind == "patrol_defend" and max(target) <= 5
    far = _obs({0: ((5, 5), HUNTER, 0.0), 6: ((20, 3), COLLECTOR, 6.0)})
    policy.act(far)
    assert policy.plan["unit_0"][1] == "patrol_defend"  # cargo far from every post is left alone


def test_duelist_hunts_hunters_and_escorts_cargo_otherwise():
    policy = make_heuristic("strategic_duelist")
    policy.act(_obs({0: ((6, 6), HUNTER, 0.0), 5: ((15, 18), HUNTER, 0.0), 6: ((8, 8), COLLECTOR, 5.0)}))
    assert policy.plan["unit_0"] == ((15, 18), "hunt")
    policy.act(_obs({0: ((6, 6), HUNTER, 0.0), 1: ((10, 14), COLLECTOR, 6.0)}))
    target, kind = policy.plan["unit_0"]
    assert kind == "patrol_defend" and abs(target[0] - 10) + abs(target[1] - 14) <= 3


# ---------------------------------------------------------------------- perturbation wrapper


class _Constant:
    """Every unit walks east; counts how often it is asked and what it saw."""

    def __init__(self):
        self.seen, self.resets = [], 0

    def reset(self):
        self.resets += 1

    def act(self, obs):
        self.seen.append(float(next(iter(obs.values()))["team_state"][2]))
        return {name: np.array([1.0, 0.0], np.float32) for name in obs}


def _run(policy, ticks, names=OWN):
    out = []
    for t in range(ticks):
        view = {"team_state": np.array([0, 0, 1.0 - t * 1e-4, 1], np.float32)}
        out.append(policy.act({name: view for name in names}))
    return np.array([[a[name] for name in names] for a in out])  # (ticks, units, 2)


def test_descriptor_syntax_builds_nested_policies_and_fails_early():
    assert split_descriptor("perturb:eps=0.2,lag=3:teacher:v21-teach") == ({"eps": 0.2, "lag": 3}, "teacher:v21-teach")
    policy = make_opponent("perturb:sticky=4,seed=7:strategic_v19:hunter_quota=4", 11)
    assert isinstance(policy, PerturbedPolicy) and policy.sticky == 4 and policy.seed == 18
    assert policy.policy.hunter_quota == 4
    nested = make_opponent("perturb:eps=0.1:perturb:freeze=0.2:random", 0)
    assert isinstance(nested.policy, PerturbedPolicy)
    _valid(make_opponent("perturb:eps=0.3,dither=0.2,sticky=3,lag=2,freeze=0.1:strategic_fortress", 0).act(_obs({})))
    assert opponent_label("perturb:eps=0.2:mixture", make_opponent("perturb:eps=0.2:mixture", 0)).startswith("perturb:eps=0.2:mixture:")
    assert opponent_name("perturb:eps=0.2:strategic_v21") == "perturb-eps=0.2-strategic_v21"
    for bad in ("perturb:strategic_v21", "perturb::strategic_v21", "perturb:eps=0.2", "perturb:wobble=1:random",
                "perturb:eps=0.2:self", "perturb:freeze=1.0:random", "perturb:eps=0.2:ckpt:missing.pt"):
        with pytest.raises(ValueError):
            check_opponent(bad)


def test_eps_noise_rate_and_seeded_determinism():
    policy = PerturbedPolicy(_Constant(), seed=3, eps=0.2)
    first = _run(policy, 2000)
    replaced = ~np.all(np.isclose(first, [1.0, 0.0]), axis=-1)
    # a random compass move is east 1/8 of the time, so the visible rate is 0.2 * 7/8
    assert abs(replaced.mean() - 0.2 * 7 / 8) < 0.02
    assert np.allclose(np.linalg.norm(first, axis=-1), 1.0)
    policy.reset()
    assert np.array_equal(_run(policy, 2000), first)
    assert not np.array_equal(_run(PerturbedPolicy(_Constant(), seed=4, eps=0.2), 2000), first)


def test_freeze_and_dither_spend_their_share_of_ticks_in_stretches():
    frozen = _run(PerturbedPolicy(_Constant(), seed=1, freeze=0.25, freeze_ticks=20), 20000)
    still = np.all(frozen == 0, axis=-1)
    assert abs(still.mean() - 0.25) < 0.04
    runs = [len(r) for unit in still.T for r in "".join("1" if s else "0" for s in unit).split("0") if r]
    assert np.median(runs) >= 20  # whole stretches, not scattered ticks
    dithered = _run(PerturbedPolicy(_Constant(), seed=1, dither=0.3, dither_ticks=12, dither_period=3), 20000)
    reversed_ = np.all(np.isclose(dithered, [-1.0, 0.0]), axis=-1)
    assert abs(reversed_.mean() - 0.15) < 0.03  # half of every burst runs backwards
    flips = (np.abs(np.diff(dithered[:, :, 0], axis=0)) > 1).sum()
    blocks = reversed_.sum() / 3  # each 3-tick reversed block turns the unit around twice
    assert abs(flips - 2 * blocks) < 0.1 * 2 * blocks


def test_sticky_holds_each_unit_for_k_ticks_and_lag_delays_the_observation():
    class Turning:
        def __init__(self):
            self.t = 0

        def act(self, obs):
            self.t += 1
            angle = self.t * 0.3
            return {name: np.array([np.cos(angle), np.sin(angle)], np.float32) for name in obs}

    held = _run(PerturbedPolicy(Turning(), seed=0, sticky=4), 400)
    changes = np.any(np.abs(np.diff(held, axis=0)) > 1e-6, axis=-1)
    assert abs(changes.mean() - 1 / 4) < 0.02
    base = _Constant()
    lagged = PerturbedPolicy(base, lag=3)
    _run(lagged, 10)
    assert base.seen[:4] == [base.seen[0]] * 4  # the first obs is replayed until the lag fills
    assert np.allclose(np.diff(base.seen[3:]), -1e-4, atol=1e-6)
    assert base.resets == 1
    lagged.reset()
    assert base.resets == 2 and len(lagged._history) == 0


def test_lagged_policy_covers_units_missing_from_the_old_observation():
    policy = PerturbedPolicy(_Constant(), lag=2)
    view = {"team_state": np.array([0, 0, 1, 1], np.float32)}
    policy.act({"unit_0": view})
    actions = policy.act({"unit_0": view, "unit_1": view})
    assert set(actions) == {"unit_0", "unit_1"} and np.array_equal(actions["unit_1"], np.zeros(2))
