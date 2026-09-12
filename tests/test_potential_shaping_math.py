"""
Pure-math validation of the potential-based shaping formula used for both Psi_k (team, §6/§14.2)
and Phi_i (individual nav, §15.3) in reward_proposal.md:

    shaped_t = eta * (gamma * potential(s_{t+1}) - potential(s_t))

This does not need Unity at all -- the formula's safety properties (§11.2, §15.5: repeated
approach/retreat or pickup/drop cycles must not let an agent accumulate unbounded reward just by
repeating an action more times) are a property of the formula itself, independent of how
potential(s) happens to be computed ( Psi_k's tanh(...) or Phi_i's 1-tanh(d/L)). Verifying them
here with synthetic potential trajectories is exact and immediate; the separate empirical check
(examples/evaluate_reward_model.py's JitterPolicy/run_cycle_probe) exercises the real game and
confirms this formula is actually what got wired into BlackOutEpisodeCoordinator.

Run directly (no pytest required):
    ./.venv/bin/python tests/test_potential_shaping_math.py
Or, if pytest is installed:
    ./.venv/bin/python -m pytest tests/test_potential_shaping_math.py -v
"""

from __future__ import annotations

import math
import random

GAMMA = 0.99995   # reward_proposal.md §6 / reward_config.json potentialGamma
ETA = 0.25        # potentialEta
ETA_NAV = 0.08    # navPotentialEta


def shaped_reward(psi_t: float, psi_t1: float, eta: float = ETA, gamma: float = GAMMA) -> float:
    """r_t = eta * (gamma * Psi(s_{t+1}) - Psi(s_t)) -- reward_proposal.md §6's boxed formula."""
    return eta * (gamma * psi_t1 - psi_t)


def discounted_return(potentials: list[float], eta: float = ETA, gamma: float = GAMMA) -> float:
    """Sum_{t=0}^{T-1} gamma^t * shaped_reward(psi_t, psi_{t+1}) -- the quantity that actually
    enters an RL agent's return, as opposed to the raw un-discounted reward sum."""
    total = 0.0
    discount = 1.0
    for t in range(len(potentials) - 1):
        total += discount * shaped_reward(potentials[t], potentials[t + 1], eta, gamma)
        discount *= gamma
    return total


def raw_reward_sum(potentials: list[float], eta: float = ETA, gamma: float = GAMMA) -> float:
    """Sum_{t=0}^{T-1} shaped_reward(psi_t, psi_{t+1}) with NO outer discount -- this is what
    accumulates in a single episode's total reward (e.g. TensorBoard's episode_return), and what
    examples/evaluate_reward_model.py's cumulative-shaped-return diagnostic actually measures."""
    return sum(shaped_reward(potentials[t], potentials[t + 1], eta, gamma) for t in range(len(potentials) - 1))


def random_walk(n: int, seed: int, bound: float = 1.0) -> list[float]:
    rng = random.Random(seed)
    vals = [rng.uniform(-bound, bound)]
    for _ in range(n - 1):
        vals.append(max(-bound, min(bound, vals[-1] + rng.uniform(-0.2, 0.2))))
    return vals


def cyclic_trajectory(period: int, n_cycles: int, shape_fn) -> list[float]:
    """A potential trajectory that returns to its starting value every `period` steps, repeated
    `n_cycles` times -- reward_proposal.md's scenario for repeated pickup/drop, approach/retreat,
    or buff insert/remove (§9 scenarios 8 & 21, §15.5)."""
    out = []
    for c in range(n_cycles):
        out.extend(shape_fn(t / period) for t in range(period))
    out.append(shape_fn(0.0))  # close the final cycle back to the start value
    return out


# ---------------------------------------------------------------------------
# Test 1: exact telescoping identity (reward_proposal.md §15.5's boxed identity)
# ---------------------------------------------------------------------------

def test_discounted_return_telescopes_exactly():
    """Sum_t gamma^t * eta*(gamma*Psi_{t+1} - Psi_t) == eta*(gamma^T * Psi_T - Psi_0), for ANY
    potential sequence -- not just cyclic ones. This is what makes PBRS provably not change the
    optimal policy (Ng, Harada & Russell 1999): it is an algebraic identity of the formula, not
    an empirical property of any particular Psi/Phi implementation."""
    for seed in range(20):
        potentials = random_walk(n=200, seed=seed)
        T = len(potentials) - 1
        lhs = discounted_return(potentials, ETA, GAMMA)
        rhs = ETA * (GAMMA ** T * potentials[-1] - potentials[0])
        assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9), (seed, lhs, rhs)


# ---------------------------------------------------------------------------
# Test 2: a trajectory that returns exactly to its start (s_T == s_0) cannot bank reward
# ---------------------------------------------------------------------------

def test_full_return_to_start_has_bounded_discounted_return():
    """If s_T == s_0 (potential identical at start and end -- e.g. picked an item up and put it
    back, or approached and fully retreated), the discounted return collapses to
    eta*Psi_0*(gamma^T - 1), which is bounded by eta*|Psi_0| regardless of how large T is or how
    many times the underlying cycle repeated inside that span (reward_proposal.md §15.5)."""
    for period, n_cycles in [(10, 1), (10, 5), (10, 50), (3, 200), (60, 30)]:
        traj = cyclic_trajectory(period, n_cycles, lambda phase: math.sin(2 * math.pi * phase))
        T = len(traj) - 1
        assert math.isclose(traj[0], traj[-1], abs_tol=1e-9)

        ret = discounted_return(traj, ETA_NAV, GAMMA)
        predicted = ETA_NAV * traj[0] * (GAMMA ** T - 1)
        assert math.isclose(ret, predicted, rel_tol=1e-6, abs_tol=1e-9)
        assert abs(ret) <= ETA_NAV * 1.0 + 1e-9  # Phi in [0, 1] in the real implementation


def test_discounted_return_does_not_scale_with_repetition_count_at_fixed_horizon():
    """The core anti-farming property (reward_proposal.md §9 scenario 8/21): repeating the SAME
    cycle shape more times within a FIXED total horizon T must not increase accumulated reward.
    Compare N cycles of period P against 4N cycles of period P/4 over the same total length T --
    they must land within numerical noise of each other, not scale with cycle count."""
    total_len = 240
    shape_fn = lambda phase: math.sin(2 * math.pi * phase)
    for base_period in (60, 20, 8):
        n_cycles = total_len // base_period
        traj_slow = cyclic_trajectory(base_period, n_cycles, shape_fn)[:total_len + 1]
        traj_fast = cyclic_trajectory(base_period // 4, n_cycles * 4, shape_fn)[:total_len + 1]

        ret_slow = discounted_return(traj_slow, ETA_NAV, GAMMA)
        ret_fast = discounted_return(traj_fast, ETA_NAV, GAMMA)
        # Both should be tiny and close to each other -- cycling 4x more often in the same
        # window does not multiply the reward by 4.
        assert abs(ret_slow - ret_fast) < 1e-3, (base_period, ret_slow, ret_fast)


def test_raw_undiscounted_sum_leakage_is_bounded_by_episode_length_not_cycle_count():
    """The raw (non-return-discounted) reward sum -- what a TensorBoard episode-return counter
    or evaluate_reward_model.py's cumulative-shaped-return diagnostic actually accumulates -- is
    NOT an exact telescoping identity when gamma < 1 (only the gamma^t-weighted RETURN is, per
    test_discounted_return_telescopes_exactly). It carries a small leakage term proportional to
    (1-gamma) * T * avg(Psi), which is what an agent could theoretically exploit by staying at a
    high potential for the whole episode. This test pins down that the leakage is bounded by
    episode length (fixed by the game's 420s / DecisionPeriod cap) and independent of how many
    times a cycle repeats within that fixed length -- i.e. cycling faster doesn't let an agent
    extract more of this leakage, only staying at a high potential for longer does, and that is
    already capped by the episode ending."""
    total_len = 4000  # comparable in order of magnitude to one absorption interval's ticks
    max_leak_bound = ETA_NAV * (1 - GAMMA) * total_len * 1.0  # |Psi|<=1 in the real implementation

    for base_period in (200, 40, 8, 2):
        n_cycles = total_len // base_period
        traj = cyclic_trajectory(base_period, n_cycles, lambda phase: math.sin(2 * math.pi * phase))[:total_len + 1]
        leak = raw_reward_sum(traj, ETA_NAV, GAMMA)
        assert abs(leak) <= max_leak_bound + 1e-9, (base_period, leak, max_leak_bound)

    # And the bound itself is small in absolute terms for realistic episode lengths/eta.
    assert max_leak_bound < 0.05


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)} checks passed.")
