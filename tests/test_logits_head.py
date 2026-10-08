"""P5: MyModel(policy_head="logits") -- plain per-unit logits instead of the IQN quantile head.

Checks: the fold of an IQN head at one tau is exact, the default fold uses the mean tau embedding
at inference's fixed taus; state-dict inference and round trips; the learner and every inference
mode (traced, stateless and temporal); K bins + STAND from an 8-way IQN checkpoint; team-B
mirroring; and head_fit moves a folded head toward the IQN outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from blackout_env.env.constants import all_agents, team_a_agents, team_b_agents
from blackout_env.env.team_frame import canonical_obs, mirror_direction_idx
from blackout_env.model.direction_bins import bins_of_width
from blackout_env.model.modules.logits_head import iqn_tau_embedding, state_from_iqn
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import MyPolicy, direction_vector_to_idx
from blackout_env.train.distill.inference import INFERENCE, fixed_tau, load_net, make_student

from tests.test_distill import _view
from tests.test_policy_team_equivariance import _team_b_observation


def _frames(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 13, 24, 24, generator=g).round(), torch.rand(n, 4, generator=g),
            torch.rand(n, 10, 12, generator=g) * 2 - 1)


def _iqn(seed=0, **kwargs) -> MyModel:
    torch.manual_seed(seed)
    net = MyModel(**kwargs).eval()
    with torch.no_grad():  # a trained-looking head: logits spread over several units
        net.q_head.value_head[1].weight.mul_(20.0)
    return net


def test_logits_head_has_no_tau_keys_and_is_detected_from_the_state():
    net = MyModel(policy_head="logits")
    state = net.state_dict()
    assert "q_head.in_scale" in state and not any(k.startswith("q_head.tau_proj") for k in state)
    assert MyModel.config_from_state_dict(state)["policy_head"] == "logits"
    assert MyModel.config_from_state_dict(MyModel().state_dict())["policy_head"] == "iqn"
    with pytest.raises(ValueError):
        MyModel(policy_head="c51")


def test_default_model_is_still_the_iqn_head():
    torch.manual_seed(0)
    a = MyModel().eval()
    torch.manual_seed(0)
    b = MyModel(policy_head="iqn").eval()
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_fold_at_one_tau_is_exact(t):
    iqn = _iqn()
    tau = torch.tensor([t])
    net = MyModel(policy_head="logits").eval()
    net.load_state_dict(state_from_iqn(iqn.state_dict(), tau, prefix="q_head."))
    g, ts, ag = _frames()
    with torch.no_grad():
        want = iqn(g, ts, ag, tau=tau.expand(8, 1))[0]
        got, quantiles, got_tau, *_ = net(g, ts, ag)
    assert torch.allclose(got, want, atol=1e-5)
    assert quantiles.shape == (8, 10, 1, 8) and torch.equal(quantiles[:, :, 0], got) and got_tau is None


def test_default_fold_is_the_mean_tau_embedding_at_the_fixed_taus():
    iqn = _iqn()
    state = MyModel.iqn_to_logits_state(iqn.state_dict())
    with torch.no_grad():
        emb = iqn.q_head.tau_proj(torch.cos(torch.pi * iqn.q_head.cos_ids * fixed_tau().unsqueeze(-1)))[0]
    assert torch.allclose(state["q_head.in_scale"], emb.mean(0), atol=1e-6)
    assert torch.allclose(iqn_tau_embedding(iqn.state_dict(), fixed_tau()[0], "q_head."), emb, atol=1e-6)
    assert MyModel.iqn_to_logits_state(state) is state  # already logits: unchanged
    net = load_net(iqn.state_dict(), policy_head="logits")
    assert net.policy_head == "logits" and torch.equal(net.q_head.in_scale.detach(), state["q_head.in_scale"])


def test_round_trips_and_refusals():
    from blackout_env.train.distill.run import load_weights

    torch.manual_seed(1)
    net = MyModel(n_actions=33, policy_head="logits").eval()
    again = load_net(net.state_dict())
    assert again.policy_head == "logits" and bins_of_width(again.q_head.value_head[1].out_features) == (32, True)
    g, ts, ag = _frames()
    with torch.no_grad():
        assert torch.equal(net(g, ts, ag)[0], again(g, ts, ag)[0])
    with pytest.raises(ValueError):
        MyModel.from_state_dict(net.state_dict(), policy_head="iqn")
    with pytest.raises(ValueError):
        load_weights(MyModel(n_actions=33), net.state_dict())


@pytest.mark.parametrize("k, stand", [(8, False), (32, False), (32, True)])
def test_k_bin_logits_net_from_an_eight_way_iqn_checkpoint(k, stand):
    """run.load_weights: fold the IQN head, then interpolate its 8 direction rows to K bins."""
    from blackout_env.train.distill.run import load_weights

    iqn = _iqn()
    folded = load_net(iqn.state_dict(), policy_head="logits")
    net = MyModel(n_actions=k + int(stand), policy_head="logits").eval()
    load_weights(net, iqn.state_dict())
    step = k // 8
    g, ts, ag = _frames(16)
    with torch.no_grad():
        q8, qk = folded(g, ts, ag)[0], net(g, ts, ag)[0]
    assert torch.allclose(qk[..., :k:step], q8, atol=1e-5)
    assert torch.equal(qk[..., :k].argmax(-1), q8.argmax(-1) * step)
    if stand:
        assert (qk[..., k] < q8.min(-1).values).all()


@pytest.mark.parametrize("team_b", [False, True])
def test_logits_student_traced_matches_eager_and_every_mode_moves(team_b):
    torch.manual_seed(0)
    net = MyModel(n_actions=33, policy_head="logits").eval()
    names = all_agents()[5:] if team_b else all_agents()[:5]
    team = {name: _view(team_b) for name in names}
    eager = MyPolicy(net).action_values(team)
    fast = make_student(net, "iqn").action_values(team)
    assert all(np.allclose(fast[n], eager[n], atol=1e-5) for n in names)
    iqn_moves, fixed_moves = make_student(net, "iqn").act(team), make_student(net, "fixed_tau").act(team)
    assert all(np.array_equal(iqn_moves[n], fixed_moves[n]) for n in names)  # no tau: the same argmax
    for inference in INFERENCE:
        for move in make_student(net, inference).act(team).values():
            assert np.isclose(np.linalg.norm(move), 1.0, atol=1e-5) or not move.any()


def test_logits_policy_plays_mirror_images_on_the_two_sides():
    torch.manual_seed(3)
    policy = make_student(MyModel(policy_head="logits"), "iqn")
    rng = np.random.default_rng(0)
    for _ in range(5):
        view_b = _team_b_observation(rng)
        view_a = canonical_obs(view_b)
        moves_b = policy.act({a: view_b for a in team_b_agents()})
        moves_a = policy.act({a: view_a for a in team_a_agents()})
        for agent_a, agent_b in zip(team_a_agents(), team_b_agents()):
            idx_a = int(direction_vector_to_idx(moves_a[agent_a][None])[0])
            idx_b = int(direction_vector_to_idx(moves_b[agent_b][None])[0])
            assert idx_b == mirror_direction_idx(idx_a)


def test_temporal_logits_sequence_equals_steps_and_the_traced_student():
    torch.manual_seed(0)
    net = MyModel(temporal=True, policy_head="logits", n_actions=9).eval()
    torch.nn.init.normal_(net.temporal.out_proj.weight, std=0.05)
    B, T = 2, 6
    g, ts, ag = (x.reshape(B, T, *x.shape[1:]) for x in _frames(B * T))
    with torch.no_grad():
        q_seq, quantiles, tau, memory = net.forward_sequence(g, ts, ag)
        assert tau is None and quantiles.shape == (B, T, 10, 1, 9)
        mem, steps = None, []
        for t in range(T):
            out = net(g[:, t], ts[:, t], ag[:, t], memory=mem)
            steps.append(out[0])
            mem = out[-1]
    assert torch.allclose(q_seq, torch.stack(steps, 1), atol=1e-4)

    view = _view(False)
    team = {name: view for name in all_agents()[:5]}
    fast, eager = make_student(net, "fixed_tau"), MyPolicy(net)
    for _ in range(3):  # memory carried across ticks on both
        a, b = fast.action_values(team), eager.action_values(team)
        assert all(np.allclose(a[n], b[n], atol=1e-4) for n in team)


def test_temporal_logits_learner_warm_starts_from_a_stateless_iqn_state():
    from blackout_env.train.distill.run import Learner, load_weights, parse_args

    args = parse_args(["--build", "x", "--device", "cpu", "--temporal", "--policy-head", "logits",
                       "--windows", "2", "--window", "3", "--burn-in", "2"])
    learner = Learner(args)
    iqn = _iqn()
    load_weights(learner.net, iqn.state_dict())
    folded = load_net(iqn.state_dict(), policy_head="logits")
    g, ts, ag = _frames(2)
    with torch.no_grad():
        learner.net.eval()
        assert torch.allclose(learner.net(g, ts, ag)[0], folded(g, ts, ag)[0], atol=1e-5)


def test_logits_learner_steps():
    from blackout_env.train.distill.run import Learner, parse_args

    learner = Learner(parse_args(["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32"]))
    rng = np.random.default_rng(0)
    rows = {"graphic": rng.integers(0, 16, (4, 24, 24, 13)).astype(np.uint8), "team_state": rng.random((4, 4), dtype=np.float32),
            "agent_states": rng.random((4, 10, 12), dtype=np.float32),
            "labels": rng.integers(1, 2**32, (4, 5)).astype(np.uint64), "kinds": np.ones((4, 5), np.uint8)}
    rows["agent_states"][..., 9:12] = [1, 0, 0]
    learner.step(rows)
    assert np.isfinite(learner.pop_stats()["train/ce"])


def test_head_fit_moves_the_folded_head_toward_the_iqn_outputs():
    from blackout_env.train.distill.head_fit import compare, fit_head, iqn_logits

    iqn = _iqn()
    net = load_net(iqn.state_dict(), policy_head="logits")
    features = torch.randn(4096, 256)
    reference = iqn_logits(iqn, features)
    before = compare(net.q_head, features, reference)
    fit_head(net, features, reference, steps=200, batch=512, lr=1e-3)
    after = compare(net.q_head, features, reference)
    assert after["mean_abs_diff"] < before["mean_abs_diff"]
