"""MyModel(temporal=True): the per-slot KDA memory between trunk and Q head (design option b2,
docs/kda_temporal_design_ko.md).

Checks: a warm start from a stateless net gives identical Q; forward_sequence == T successive
single-step calls carrying memory (with resets mid-sequence and burn-in); KDA per-position
resets are exact; the stateful policy carries memory per team and reset() restores it; traced ==
eager; STAND (9 actions) + temporal; load_net round trips; and a small model learns a task only
memory can solve.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.model.modules.kda import KDA, kda_chunk, kda_recurrent_reference
from blackout_env.model.my_model import TEMPORAL_PREFIX, MyModel
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.distill.inference import fixed_tau, load_net, make_student

H = W = 24
C_GRAPHIC = 13
N_UNITS, N_COLS = 10, 12


def _frames(*lead, seed=0):
    g = torch.Generator().manual_seed(seed)
    graphic = torch.rand(*lead, C_GRAPHIC, H, W, generator=g)
    team_state = torch.rand(*lead, 4, generator=g)
    agent_states = torch.rand(*lead, N_UNITS, N_COLS, generator=g) * 2 - 1
    return graphic, team_state, agent_states


def _temporal_net(seed=0, **kwargs) -> MyModel:
    """A temporal net whose memory branch is switched on (out_proj is zero at init)."""
    torch.manual_seed(seed)
    net = MyModel(temporal=True, **kwargs).eval()
    torch.nn.init.normal_(net.temporal.out_proj.weight, std=0.05)
    return net


def _view(team_b: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    states = rng.uniform(-1.0, 1.0, size=(N_UNITS, N_COLS)).astype(np.float32)
    states[:5, 2], states[5:, 2] = (-1.0, 1.0) if team_b else (1.0, -1.0)
    return {"graphic": rng.random((H, W, C_GRAPHIC)).astype(np.float32),
            "team_state": rng.random(4).astype(np.float32), "agent_states": states}


# ------------------------------------------------------------------------------ KDA resets


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
def test_kda_chunk_reset_matches_reference(chunk):
    g = torch.Generator().manual_seed(3)
    B, Hh, T, dk, dv = 3, 2, 37, 8, 5
    q = torch.randn(B, Hh, T, dk, generator=g, dtype=torch.float64)
    k = F.normalize(torch.randn(B, Hh, T, dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, Hh, T, dv, generator=g, dtype=torch.float64)
    la = -torch.rand(B, Hh, T, dk, generator=g, dtype=torch.float64)
    beta = torch.rand(B, Hh, T, generator=g, dtype=torch.float64)
    s0 = torch.randn(B, Hh, dk, dv, generator=g, dtype=torch.float64)
    reset = torch.rand(B, T, generator=g) < 0.15
    reset[0, 0] = reset[1, 8] = reset[1, 9] = reset[2, T - 1] = True  # start, adjacent, last
    o_ref, s_ref = kda_recurrent_reference(q, k, v, la, beta, s0, reset=reset)
    o, s = kda_chunk(q, k, v, la, beta, s0, chunk_size=chunk, reset=reset)
    torch.testing.assert_close(o, o_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(s, s_ref, rtol=1e-10, atol=1e-10)


def test_kda_layer_reset_equals_fresh_start():
    """A reset at t makes the output from t on equal a fresh run from t (state and short conv)."""
    torch.manual_seed(0)
    layer = KDA(16, n_heads=2, head_dim=8, chunk_size=8).double()
    x = torch.randn(2, 30, 16, dtype=torch.float64)
    state = tuple(torch.randn_like(t) for t in layer.initial_state(2, dtype=torch.float64))
    reset = torch.zeros(2, 30, dtype=torch.bool)
    reset[0, 11] = reset[1, 5] = reset[1, 28] = True
    y, (s, conv) = layer(x, state, reset=reset)
    y_step, (s_step, conv_step) = layer(x, state, mode="step", reset=reset)
    torch.testing.assert_close(y, y_step)
    torch.testing.assert_close(s, s_step)
    torch.testing.assert_close(conv, conv_step)
    fresh0, _ = layer(x[:1, 11:])
    torch.testing.assert_close(y[:1, 11:], fresh0)
    fresh1, (s1, conv1) = layer(x[1:, 28:])
    torch.testing.assert_close(y[1:, 28:], fresh1)
    torch.testing.assert_close(s[1:], s1)
    torch.testing.assert_close(conv[1:], conv1)


# ------------------------------------------------------------------------------ model


def test_stateless_model_is_unchanged():
    net = MyModel()
    assert not net.is_temporal and net.initial_memory() is None
    assert not any(key.startswith(TEMPORAL_PREFIX) for key in net.state_dict())
    assert len(net(*_frames(2))) == 5
    with pytest.raises(ValueError):
        net(*_frames(1), memory=MyModel(temporal=True).initial_memory(1))


@pytest.mark.parametrize("n_actions", [8, 9])
def test_warm_start_from_stateless_gives_identical_q(n_actions):
    torch.manual_seed(0)
    stateless = MyModel(n_actions=n_actions).eval()
    state = stateless.state_dict()
    temporal = load_net(state, temporal=True).eval()
    assert temporal.is_temporal
    assert set(temporal.state_dict()) - set(state) == {k for k in temporal.state_dict() if k.startswith(TEMPORAL_PREFIX)}
    tau = torch.rand(2, 32)
    memory = None
    with torch.no_grad():
        for t in range(3):  # several ticks: the memory builds up but the branch is still zero
            frame = _frames(2, seed=t)
            ref = stateless(*frame, tau=tau)
            out = temporal(*frame, tau=tau, memory=memory)
            memory = out[-1]
            for a, b in zip(ref, out[:5]):
                assert torch.equal(a, b)
        seq = temporal.forward_sequence(*_frames(2, 4, seed=9), tau=tau)
        ref = stateless(*(x.flatten(0, 1) for x in _frames(2, 4, seed=9)), tau=tau.repeat_interleave(4, 0))[0]
    torch.testing.assert_close(seq[0].flatten(0, 1), ref, rtol=0, atol=1e-5)  # batched vs not: fp32 order
    # temporal=None keeps whatever the state is; a stateless state stays stateless
    assert not load_net(state).is_temporal
    with pytest.raises(ValueError):
        load_net(temporal.state_dict(), temporal=False)


def test_load_net_round_trips_a_temporal_state():
    net = _temporal_net(seed=1, n_actions=9, temporal_heads=2, temporal_head_dim=32)
    again = load_net(net.state_dict()).eval()
    assert again.is_temporal and again.temporal.kda.n_heads == 2 and again.temporal.kda.dk == 32
    tau = fixed_tau().expand(2, -1)
    frame = _frames(2)
    with torch.no_grad():
        assert torch.equal(net(*frame, tau=tau)[0], again(*frame, tau=tau)[0])
    partial = {k: v for k, v in net.state_dict().items() if k != "temporal.out_proj.weight"}
    with pytest.raises(RuntimeError):  # a temporal state must be complete
        MyModel(temporal=True, n_actions=9, temporal_heads=2, temporal_head_dim=32).load_warm_start(partial)


@pytest.mark.parametrize("n_actions", [8, 9])
def test_forward_sequence_equals_successive_steps_with_resets(n_actions):
    net = _temporal_net(seed=2, n_actions=n_actions, temporal_chunk=4)  # chunk 4: T spans several chunks
    B, T = 2, 11
    frames = _frames(B, T, seed=5)
    tau = torch.rand(B, T, 32)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 0] = reset[0, 6] = reset[1, 3] = True
    memory0 = tuple(torch.randn_like(m) for m in net.initial_memory(B))  # a warm start state
    with torch.no_grad():
        q_seq, quant_seq, tau_seq, mem_seq = net.forward_sequence(*frames, memory=memory0, reset=reset, tau=tau, q_units=5)
        memory, qs = memory0, []
        for t in range(T):
            keep = (~reset[:, t]).float()
            memory = tuple(m * keep.view(B, *[1] * (m.dim() - 1)) for m in memory)
            out = net(*(x[:, t] for x in frames), tau=tau[:, t], q_units=5, memory=memory)
            qs.append(out[0])
            memory = out[-1]
    assert q_seq.shape == (B, T, 5, n_actions) and quant_seq.shape == (B, T, 5, 32, n_actions)
    assert torch.equal(tau_seq, tau)
    torch.testing.assert_close(q_seq, torch.stack(qs, 1), rtol=1e-4, atol=1e-5)
    for a, b in zip(mem_seq, memory):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    # the memory matters: without it (every frame a reset) the Q-values differ
    with torch.no_grad():
        q_nomem = net.forward_sequence(*frames, reset=torch.ones(B, T, dtype=torch.bool), tau=tau, q_units=5)[0]
    assert (q_nomem - q_seq)[:, 1:].abs().max() > 1e-4


def test_forward_sequence_burn_in():
    net = _temporal_net(seed=3)
    B, T, burn = 2, 9, 4
    frames = _frames(B, T, seed=6)
    tau = torch.rand(B, T, 32)
    with torch.no_grad():
        full = net.forward_sequence(*frames, tau=tau)
    net.train(False)
    part = net.forward_sequence(*frames, tau=tau, burn_in=burn)
    assert part[0].shape[1] == T - burn
    torch.testing.assert_close(part[0].detach(), full[0][:, burn:], rtol=1e-4, atol=1e-5)
    part[0].sum().backward()  # the gradient reaches the trunk and the memory
    assert net.token_type_emb.weight.grad.abs().sum() > 0
    assert net.temporal.out_proj.weight.grad.abs().sum() > 0
    assert net.temporal.kda.q_proj.weight.grad.abs().sum() > 0


# ------------------------------------------------------------------------------ policy / inference


@pytest.mark.parametrize("inference", ["fixed_tau", "tie_mean"])
def test_stateful_policy_reset_restores_initial_behaviour(inference):
    net = _temporal_net(seed=4)
    rng = np.random.default_rng(0)
    views = [_view(False, rng) for _ in range(6)]
    policy = make_student(net, inference)

    def play():
        return [policy.action_values({a: v for a in team_a_agents()}) for v in views]

    first = play()
    policy.reset()
    again = play()
    for x, y in zip(first, again):
        for agent in x:
            np.testing.assert_array_equal(x[agent], y[agent])
    policy.reset()
    policy.action_values({a: views[3] for a in team_a_agents()})  # a different history
    later = policy.action_values({a: views[1] for a in team_a_agents()})
    assert any(not np.allclose(later[a], first[1][a], atol=1e-6) for a in later)  # the memory is used


@pytest.mark.parametrize("n_actions", [8, 9])
def test_traced_student_matches_eager_step_per_team(n_actions):
    """make_student's batch-1 traced step == eager MyModel carrying memory, each team with its own
    memory (a mirror match: one policy object on both sides), STAND column mirrored."""
    net = _temporal_net(seed=5, n_actions=n_actions)
    rng = np.random.default_rng(1)
    fast = make_student(net, "fixed_tau")

    class _Fixed(torch.nn.Module):  # eager, full rows, fixed tau
        is_temporal = True

        def forward(self, graphic, team_state, agent_states, memory=None):
            return net(graphic, team_state, agent_states, tau=fixed_tau(), memory=memory)

    eager_a, eager_b = MyPolicy(_Fixed()), MyPolicy(_Fixed())
    for _ in range(5):
        va, vb = _view(False, rng), _view(True, rng)
        fa = fast.action_values({a: va for a in team_a_agents()})
        fb = fast.action_values({a: vb for a in team_b_agents()})
        ea = eager_a.action_values({a: va for a in team_a_agents()})
        eb = eager_b.action_values({a: vb for a in team_b_agents()})
        for f, e in ((fa, ea), (fb, eb)):
            for agent in f:
                assert f[agent].shape == (n_actions,)
                np.testing.assert_allclose(f[agent], e[agent], rtol=1e-5, atol=1e-5)


def test_checkpoint_style_student_resets_between_matches():
    """collect.py resets every policy with hasattr(policy, 'reset') at match start; a cached
    checkpoint opponent (opponents._checkpoint_policy -> make_student) must then replay a match
    exactly."""
    net = _temporal_net(seed=6)
    student = make_student(load_net(net.state_dict()), "iqn")
    assert hasattr(student, "reset")
    rng = np.random.default_rng(2)
    views = [_view(True, rng) for _ in range(4)]
    runs = []
    for _ in range(2):
        student.reset()
        torch.manual_seed(0)  # iqn draws tau per call
        runs.append([student.act({a: v for a in team_b_agents()}) for v in views])
    for x, y in zip(*runs):
        for agent in x:
            np.testing.assert_array_equal(x[agent], y[agent])


# ------------------------------------------------------------------------------ learning


def test_temporal_model_learns_a_history_dependent_task():
    """A cue shown only at frame 0 (team_state[0] = +-1, zero afterwards) decides the right action
    at frames 1..T-1 (direction 0 or 4 for the own units).  Every later frame is independent of
    the cue, so a stateless net is stuck at chance (trained the same way as the control); the
    temporal net must remember it -- and loses it again when its memory is reset every frame."""
    T, B, steps = 5, 16, 60
    started = time.time()
    tau = fixed_tau(8).expand(B, -1)

    def batch(gen):
        cue = torch.randint(0, 2, (B,), generator=gen)
        graphic = torch.rand(B, T, C_GRAPHIC, H, W, generator=gen)
        team_state = torch.rand(B, T, 4, generator=gen)
        team_state[:, :, 0] = 0.0
        team_state[:, 0, 0] = cue.float() * 2 - 1
        agent_states = torch.rand(B, T, N_UNITS, N_COLS, generator=gen) * 2 - 1
        target = (cue * 4).view(B, 1, 1).expand(B, T - 1, 5)
        return (graphic, team_state, agent_states), target

    def q_of(net, frames, reset=None):
        if net.is_temporal:
            return net.forward_sequence(*frames, reset=reset, tau=tau, n_quantiles=8, q_units=5)[0]
        q = net(*(x.flatten(0, 1) for x in frames), tau=tau.repeat_interleave(T, 0), q_units=5)[0]
        return q.reshape(B, T, 5, -1)

    def train(temporal: bool) -> MyModel:
        torch.manual_seed(0)
        kwargs = dict(temporal_heads=2, temporal_head_dim=16, temporal_chunk=8) if temporal else {}
        net = MyModel(hidden_size=64, temporal=temporal, **kwargs)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(1)
        for _ in range(steps):
            frames, target = batch(gen)
            q = q_of(net, frames)
            loss = F.cross_entropy(q[:, 1:].reshape(-1, q.shape[-1]), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net.eval()

    def accuracy(net, reset=None) -> float:
        frames, target = batch(torch.Generator().manual_seed(99))
        with torch.no_grad():
            q = q_of(net, frames, reset)
        return (q[:, 1:].argmax(-1) == target).float().mean().item()

    temporal = train(True)
    temporal_acc = accuracy(temporal)
    forgetful_acc = accuracy(temporal, reset=torch.ones(B, T, dtype=torch.bool))
    stateless_acc = accuracy(train(False))
    assert temporal_acc > 0.95, temporal_acc
    assert stateless_acc < 0.8 and forgetful_acc < 0.8, (stateless_acc, forgetful_acc)
    assert time.time() - started < 90
