"""End-to-end: the policy must play the same game on either side of the map.

Before the canonical team frame, Team B got the same world grid with only its ally/enemy channel
labels swapped, so the network was learning two differently-oriented versions of the task. These
tests take a Team B observation, build the Team A view of the same situation by canonicalizing it,
and require the two chosen world actions to be mirror images -- which they can only be if
canonicalization, the unit-row mapping and the action mapping all agree.

Q-values are sampled (IQN draws tau ~ U(0,1) per call), so each comparison seeds torch first and
uses matching batch shapes: otherwise two calls on the same state need not even agree with
themselves.
"""

from __future__ import annotations

import numpy as np
import torch

from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.env.team_frame import canonical_obs, is_team_b_view, mirror_direction_idx
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy, direction_vector_to_idx

H = W = 24
N_UNITS, N_COLS = 10, 12
SEED = 7


def _team_b_observation(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A Team B view as the env produces one: world grid, world unit rows, team column flipped."""
    states = rng.uniform(-1.0, 1.0, size=(N_UNITS, N_COLS)).astype(np.float32)
    states[:5, 2] = -1.0  # physical team A units are the enemy from here
    states[5:, 2] = 1.0
    view = {
        "graphic": rng.random((H, W, 13)).astype(np.float32),
        "team_state": rng.random(4).astype(np.float32),
        "agent_states": states,
    }
    assert is_team_b_view(view["agent_states"])
    return view


def _act(policy: MyPolicy, view: dict[str, np.ndarray], agents) -> dict[str, int]:
    torch.manual_seed(SEED)  # same tau draws on both sides
    actions = policy.act({a: view for a in agents})
    return {a: int(direction_vector_to_idx(actions[a][None, :])[0]) for a in actions}


def test_policy_plays_mirror_images_on_the_two_sides():
    rng = np.random.default_rng(0)
    policy = MyPolicy(MyModel().eval(), device="cpu")

    for _ in range(5):
        view_b = _team_b_observation(rng)
        view_a = canonical_obs(view_b)  # the same situation, as Team A would see it
        assert not is_team_b_view(view_a["agent_states"])

        idx_b = _act(policy, view_b, team_b_agents())
        idx_a = _act(policy, view_a, team_a_agents())

        # canonical row k is physical unit 5 + k, so team B's slot k pairs with team A's slot k
        for slot, (agent_a, agent_b) in enumerate(zip(team_a_agents(), team_b_agents())):
            assert idx_b[agent_b] == mirror_direction_idx(idx_a[agent_a]), (
                f"slot {slot}: team A chose {idx_a[agent_a]}, team B chose {idx_b[agent_b]}"
            )


def test_the_two_views_are_genuinely_different_inputs():
    """Guards the test above: it would pass vacuously if both sides saw identical arrays."""
    view_b = _team_b_observation(np.random.default_rng(1))
    view_a = canonical_obs(view_b)
    assert not np.array_equal(view_a["graphic"], view_b["graphic"])
    assert not np.allclose(view_a["agent_states"], view_b["agent_states"])


def test_a_team_a_view_is_passed_through_untouched():
    rng = np.random.default_rng(2)
    view_b = _team_b_observation(rng)
    view_a = canonical_obs(view_b)
    assert canonical_obs(view_a) is view_a  # already canonical: no second mirror


def test_collect_step_uses_the_same_frame_as_the_policy():
    """QMIXTrainer.select_actions duplicates MyPolicy's frame handling -- keep them in step."""
    from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

    rng = np.random.default_rng(3)
    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device="cpu", tb_log_dir=None))
    trainer.net.eval()
    trainer.ema_net.load_state_dict(trainer.net.state_dict())
    trainer._online_is_team_a = True
    trainer._opponent_is_heuristic = False

    view_b = _team_b_observation(rng)
    view_a = canonical_obs(view_b)  # treat it as team A's own view of the same situation
    obs = {a: view_a for a in team_a_agents()} | {a: view_b for a in team_b_agents()}

    torch.manual_seed(SEED)
    env_actions, full_idx = trainer.select_actions(obs, epsilon=0.0)

    # Replay the same two forwards the trainer runs, in the same order so the tau draws match:
    # net for the online side, ema_net for this episode's opponent side (team B here).
    torch.manual_seed(SEED)
    graphic, team_state, agent_states = trainer._to_batch(view_a, canonical_obs(view_b))
    with torch.no_grad():
        q_online, *_ = trainer.net(graphic, team_state, agent_states, n_quantiles=trainer.cfg.n_quantiles)
        q_ema, *_ = trainer.ema_net(graphic, team_state, agent_states, n_quantiles=trainer.cfg.n_quantiles)
    q = torch.stack([q_online[0], q_ema[1]])
    greedy = q[:, :5].argmax(dim=-1).numpy()  # own team is rows 0-4 in both canonical views

    for slot, agent in enumerate(team_a_agents()):
        assert full_idx[unit_index(agent)] == greedy[0, slot]
        assert np.allclose(env_actions[agent], DIRECTION_VECTORS[greedy[0, slot]])
    for slot, agent in enumerate(team_b_agents()):
        expected = mirror_direction_idx(int(greedy[1, slot]))  # canonical -> world
        assert full_idx[unit_index(agent)] == expected, f"{agent}: {full_idx[unit_index(agent)]} vs {expected}"
        assert np.allclose(env_actions[agent], DIRECTION_VECTORS[expected])


def test_action_values_are_in_the_world_frame():
    """act() is argmax over action_values, and both sides agree on which direction that is."""
    rng = np.random.default_rng(4)
    policy = MyPolicy(MyModel().eval(), device="cpu")

    view_b = _team_b_observation(rng)
    view_a = canonical_obs(view_b)

    torch.manual_seed(SEED)
    q_b = policy.action_values({a: view_b for a in team_b_agents()})
    torch.manual_seed(SEED)
    q_a = policy.action_values({a: view_a for a in team_a_agents()})

    for agent_a, agent_b in zip(team_a_agents(), team_b_agents()):
        # the same underlying canonical row, read out in each side's own world frame
        assert np.allclose(q_b[agent_b], q_a[agent_a][[mirror_direction_idx(i) for i in range(8)]])

    torch.manual_seed(SEED)
    acted = policy.act({a: view_b for a in team_b_agents()})
    for agent, row in q_b.items():
        assert int(direction_vector_to_idx(acted[agent][None, :])[0]) == int(np.argmax(row))


def test_wall_masking_only_removes_blocked_directions():
    rng = np.random.default_rng(5)
    view = _team_b_observation(rng)
    view["graphic"] = np.zeros((H, W, 13), dtype=np.float32)
    view["graphic"][:, :, 1] = 1.0  # wall everywhere...
    view["graphic"][10:13, 10:13, 1] = 0.0  # ...except a 3x3 pocket
    view["agent_states"][:, 0] = (11 + 0.5) * 2.0 / W - 1.0  # every unit in the middle of it
    view["agent_states"][:, 1] = 1.0 - (11 + 0.5) * 2.0 / H

    net = MyModel().eval()
    free = MyPolicy(net, device="cpu", mask_walls=False)
    masked = MyPolicy(net, device="cpu", mask_walls=True)
    obs = {a: view for a in team_b_agents()}

    torch.manual_seed(SEED)
    q_rows = free.action_values(obs)
    torch.manual_seed(SEED)
    chosen = masked.act(obs)
    for agent, row in q_rows.items():
        idx = int(direction_vector_to_idx(chosen[agent][None, :])[0])
        # the pocket is 3x3 with the unit at its centre, so every direction stays walkable
        assert idx == int(np.argmax(row))
