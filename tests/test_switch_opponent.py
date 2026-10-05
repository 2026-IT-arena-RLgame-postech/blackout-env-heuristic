"""switch:<seconds>:<A>|<B> opponents / act teachers (P18b, docs §43)."""
import numpy as np
import pytest

from blackout_env.train.distill.opponents import SwitchPolicy, check_opponent, make_opponent, split_switch


def _obs(elapsed_s: float) -> dict:
    ts = np.zeros(4, np.float32)
    ts[2] = 1.0 - elapsed_s / 420.0
    return {f"unit_{i}": {"team_state": ts} for i in range(5)}


def test_switch_plays_a_then_b_and_steps_both():
    pol = make_opponent("switch:6:random|idle", 3)
    assert isinstance(pol, SwitchPolicy)
    early = pol.act(_obs(5.9))
    assert all(np.linalg.norm(m) > 0.5 for m in early.values())  # the random walker
    late = pol.act(_obs(6.0))
    assert all(not m.any() for m in late.values())  # idle
    assert split_switch("switch:6:strategic_v29:reserve_stage=0.5,reserve_join=reserve|ckpt:/x.pt") == (
        6.0, "strategic_v29:reserve_stage=0.5,reserve_join=reserve", "ckpt:/x.pt")


def test_switch_is_checked():
    check_opponent("switch:6:strategic_v25|idle")
    with pytest.raises(ValueError):
        check_opponent("switch:6:strategic_v25|self")
    with pytest.raises(ValueError):
        check_opponent("switch:6:strategic_v25")
    with pytest.raises(ValueError):
        check_opponent("switch:6:strategic_v25|ckpt:/no/such.pt")


def _obs_h(elapsed_s: float, enemy_h: int) -> dict:
    obs = _obs(elapsed_s)
    a = np.zeros((10, 12), np.float32)
    a[:5, 2], a[5:, 2] = 1, -1
    a[:, 9] = 1
    a[5:5 + enemy_h, 9], a[5:5 + enemy_h, 10] = 0, 1
    a[0, 9], a[0, 10] = 0, 1  # an own Hunter never counts
    for v in obs.values():
        v["agent_states"] = a
    return obs


@pytest.mark.parametrize("enemy_h,late", [(4, "a"), (3, "b")])
def test_joinswitch_keeps_a_only_after_a_four_hunter_enemy_race(enemy_h, late):
    pol = make_opponent("joinswitch:3.6,4.4,4.4:random|idle", 3)
    assert all(np.linalg.norm(m) > 0.5 for m in pol.act(_obs_h(1.0, 0)).values())
    pol.act(_obs_h(4.0, enemy_h))
    moved = [np.linalg.norm(m) > 0.5 for m in pol.act(_obs_h(5.0, 0)).values()]
    assert all(moved) if late == "a" else not any(moved)
    pol.act(_obs_h(0.1, 0))  # a new match: forgets the race
    assert not any(np.linalg.norm(m) > 0.5 for m in pol.act(_obs_h(5.0, 0)).values())
    check_opponent("joinswitch:3.6,4.4,4.4:strategic_v25|idle")
