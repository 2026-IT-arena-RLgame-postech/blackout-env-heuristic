"""OfficialTracker on scripted edge cases: truth ticks vs reconstruction from official obs."""

import numpy as np
import pytest

from blackout_env.official.render import official_map, official_vector
from blackout_env.official.tracker import MultiTracker, OfficialTracker, _Fixed
from blackout_env.official.validate import true_tick
from blackout_env.sim import core
from blackout_env.sim.mapdata import HUNTER
from blackout_env.sim.state import SimState

F32 = np.float32


def _play(truth: SimState, script, blind: bool):
    """script: list of [10, 2] moves, one per tick.  Returns per-tick (raw equal, map equal)."""
    finder = _Fixed(truth.rng.copy(), "test")
    tr = MultiTracker(finder) if blind else OfficialTracker()
    see = (lambda s: None) if blind else official_map
    # the tracker must start from this exact scenario, not a generated match
    tr.observe(official_vector(truth, 0), see(truth))
    tgt = tr.best if blind else tr
    tgt.sim = truth.copy()
    if blind:
        tr.children = [tgt]
    out, tr.truth_events = [], []
    for mv in script:
        true_tick(truth.S, truth.M, np.asarray(mv, F32), truth.order)
        tr.truth_events += [tuple(e[1:4]) for e in truth.events()]
        tr.observe(official_vector(truth, 0), see(truth))
        out.append((np.array_equal(tr.raw_state(), truth.raw_state()), np.array_equal(tr.raw_map(), truth.raw_map())))
    return out, tr


def _scenario_die_after_pickup():
    """Unit 3 (A, Collector) steps onto a battery; unit 6 (B, Hunter) acts next in AgentAct order
    and kills it in the same tick: the battery is destroyed with the carrier."""
    sim = SimState.empty((0, 1, 2, 3))
    sim.gf[core.GF_EP] = F32(10.0)
    sim.g[core.G_TICK] = 500
    sim.add_item(core.BATTERY, 5, 10, 6)
    sim.place_unit(3, 9.77, 6.5)
    sim.place_unit(6, 9.93, 7.32, HUNTER)
    mv = np.zeros((10, 2), F32)
    mv[3] = (1, 0)
    mv[6] = (0, -1)
    return sim, [mv, mv] + [np.zeros((10, 2), F32)] * 5


def test_die_after_pickup_map():
    sim, script = _scenario_die_after_pickup()
    out, tr = _play(sim, script, blind=False)
    ev = tr.truth_events
    assert (core.EV_PICKUP, 3, core.BATTERY) in ev and (core.EV_KILL, 6, 3) in ev, ev
    assert all(a and b for a, b in out), out


def test_die_after_pickup_blind():
    """Blind: the destroyed battery cannot be seen to go; the dying unit's last move (actions are
    held for 2 ticks) is extrapolated and its pick-up replayed before the death."""
    sim, script = _scenario_die_after_pickup()
    out, tr = _play(sim, script, blind=True)
    assert all(a and b for a, b in out), out  # the extrapolated last move replays the pick-up


def test_spawn_under_standing_collector():
    sim = SimState.empty((0, 1, 2, 3))
    sim.gf[core.GF_EP] = F32(10.0)
    sim.g[core.G_TICK] = 500
    sim.place_unit(2, 10.5, 6.5)
    sim.force_timed_spawns([(1, 10, 6, 503)])
    out, tr = _play(sim, [np.zeros((10, 2), F32)] * 6, blind=False)
    assert any(e[0] == core.EV_PICKUP and e[1] == 2 for e in tr.truth_events), tr.truth_events
    assert all(a and b for a, b in out), out
    assert tr.sim.ui[2, core.U_HELD] >= 0


def test_mutual_hunter_kill_with_bystander():
    """Hunters 2 (A) and 8 (B) collide and both die; hunter 9 (B) passes their old spots."""
    sim = SimState.empty((0, 1, 2, 3))
    sim.gf[core.GF_EP] = F32(10.0)
    sim.g[core.G_TICK] = 500
    sim.place_unit(2, 10.5, 6.5, HUNTER)
    sim.place_unit(8, 11.2, 6.5, HUNTER)
    sim.place_unit(9, 11.9, 6.0, HUNTER)
    mv = np.zeros((10, 2), F32)
    mv[2], mv[8], mv[9] = (1, 0), (-1, 0), (-1, 0.3)
    out, tr = _play(sim, [mv] * 4, blind=False)
    victims = {e[2] for e in tr.truth_events if e[0] == core.EV_KILL}
    assert victims == {2, 8}, tr.truth_events
    assert all(a and b for a, b in out), out


@pytest.mark.parametrize("blind", [False, True])
def test_sanctuary_lock_held_until_carrier_dies_later_in_tick(blind):
    """Carrier 8 (B) holds its team's sanctuary lock; Collector 6 (B, earlier in AgentAct order)
    walks into the locked sanctuary, then Hunter 1 (A, later) kills 8 -- the lock still held
    when 6 entered, so 6 must stay a Collector (a death applied at tick start would free it)."""
    from blackout_env.sim.mapdata import CARRIER, COLLECTOR

    sim = SimState.empty((0, 1, 2, 3))
    sim.gf[core.GF_EP] = F32(10.0)
    sim.g[core.G_TICK] = 500
    sim.place_unit(8, 22.7, 6.4, CARRIER)
    sim.reg_lock[2] = 8                       # FIXED_REGIONS[2] = Carrier Sanctuary B
    sim.place_unit(6, 21.05, 1.27)
    sim.place_unit(1, 22.05, 6.4, HUNTER)
    mv = np.zeros((10, 2), F32)
    mv[6], mv[1] = (-1, 0), (1, 0)
    out, tr = _play(sim, [mv, mv] + [np.zeros((10, 2), F32)] * 3, blind=blind)
    assert (core.EV_KILL, 1, 8) in tr.truth_events, tr.truth_events
    assert int(sim.ui[6, core.U_CLS]) == COLLECTOR
    assert all(a and b for a, b in out), out
