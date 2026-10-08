"""Batched simulator rollouts (blackout_env/sim/batch.py) against the single-game reference path.

* ``BatchSim.step`` (prange and sequential) == ``SimState.step`` per game, every state array;
* ``render`` == ``canonical_obs(sim.obs()[unit])`` tensors, team A *and* team B's mirrored frame;
* a batched student rollout == ``MyPolicy`` (``make_student``) on ``sim.obs()``, bit for bit on CPU,
  when the net sees batch 1 (``per_sample``; batched CPU GEMMs round differently, see NetPolicy);
* action repeat and the hybrid policy call the policies when they should.
The recorded Unity trace (tests/data/sim_trace_short.npz) supplies mid-match states with items,
classes and scores on both sides."""

from pathlib import Path

import numpy as np
import pytest
import torch

from blackout_env.env.team_frame import canonical_obs
from blackout_env.sim import SimState
from blackout_env.sim.batch import (N_AGENT_COLS, N_CH, BatchPolicy, BatchRollout, BatchSim, HybridPolicy,
                                    NetPolicy, decode_moves)
from blackout_env.sim.trace import Trace, init_sim

FIXTURE = Path(__file__).resolve().parent / "data" / "sim_trace_short.npz"


def _trace_states(every=40):
    tr = Trace.load(FIXTURE)
    sim = init_sim(tr, 0, None, True, "trace", 0)  # the recorded match, special spawns as recorded
    states = []
    for k in range(min(len(tr.moves), 560)):
        if k % every == 0:
            states.append(sim.copy())
        sim.step(tr.moves[k])
        if sim.over:
            break
    return states


def _student(seed=0, n_actions=32):
    from blackout_env.model.my_model import MyModel

    torch.manual_seed(seed)
    net = MyModel(hidden_size=64, n_actions=n_actions, policy_head="logits", angle_head=True, angle_arc_bins=1).eval()
    with torch.no_grad():  # the offsets start at 0: make them matter
        net.q_head.angle_out.weight.normal_(0, 0.5)
        net.q_head.angle_out.bias.normal_(0, 0.5)
    return net


def test_batched_step_matches_single_states():
    states = _trace_states(40) + [SimState.from_seed(s) for s in (11, 12)]
    rng = np.random.default_rng(0)
    for parallel in (True, False):
        ref = [s.copy() for s in states]
        bs = BatchSim(states)
        for _ in range(60):
            mv = rng.normal(size=(len(states), 10, 2)).astype(np.float32)
            mv[rng.random((len(states), 10)) < 0.2] = 0
            live = np.nonzero(~bs.over)[0]
            bs.step(mv, live, parallel=parallel)
            for b in live:
                ref[b].step(mv[b])
        for b, r in enumerate(ref):
            got = bs.state(b)
            for name in SimState._ARRAYS:
                if name != "ev":  # the event log keeps stale rows past its count
                    assert np.array_equal(getattr(got, name), getattr(r, name)), (parallel, b, name)


@pytest.mark.parametrize("parallel", [True, False])
def test_render_matches_canonical_obs_both_teams(parallel):
    states = _trace_states(20)
    bs = BatchSim(states)
    n = len(states)
    g = np.zeros((n, 2, N_CH, 24, 24), np.float32)
    t = np.zeros((n, 2, 4), np.float32)
    a = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
    bs.render(bs.all, g, t, a, parallel=parallel)
    held = 0
    for b, s in enumerate(states):
        obs = s.obs()
        held += int((s.ui[:, 1] >= 0).sum())
        for team, name in ((0, "unit_0"), (1, "unit_5")):
            c = canonical_obs(obs[name])
            assert np.asarray(c["graphic"], np.float32).transpose(2, 0, 1).tobytes() == g[b, team].tobytes()
            assert np.asarray(c["team_state"], np.float32).tobytes() == t[b, team].tobytes()
            assert np.asarray(c["agent_states"], np.float32).tobytes() == a[b, team].tobytes()
    assert held > 0  # the fixture's states do exercise held items


@pytest.mark.parametrize("n_actions", [32, 33])  # 33 = 32 bins + STAND (P18b, §43)
def test_decode_matches_mypolicy_team_b(n_actions):
    from blackout_env.model.my_policy import MyPolicy

    net = _student(1, n_actions)
    if n_actions % 2:  # make STAND win for some units and lose for others
        with torch.no_grad():
            row = [m for m in net.q_head.value_head if isinstance(m, torch.nn.Linear)][-1]
            row.weight[-1].normal_(0, 3.0)
    pol = MyPolicy(net)
    states = _trace_states(60)
    stood = moved = 0
    for s in states:
        obs = s.obs()
        for team in (0, 1):
            names = [f"unit_{i}" for i in range(5 * team, 5 * team + 5)]
            acts = pol.act({k: obs[k] for k in names})
            c = canonical_obs(obs[names[0]])
            gr = torch.tensor(c["graphic"]).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                out = net(gr, torch.tensor(c["team_state"]).unsqueeze(0), torch.tensor(c["agent_states"]).unsqueeze(0),
                          angle=True)  # all 10 rows, as MyPolicy around a bare MyModel computes them
            mv = decode_moves(out[0][:, :5].numpy(), out[-1][:, :5].numpy(), np.array([team == 1]))[0]
            for i, k in enumerate(names):
                assert np.clip(acts[k], -1, 1).tobytes() == mv[i].tobytes(), (team, k)
                stood += int(not mv[i].any()); moved += int(mv[i].any())
    assert moved > 0 and (stood > 0) == bool(n_actions % 2)


def test_batched_student_rollout_equals_reference():
    from blackout_env.train.distill.inference import make_student

    torch.set_num_threads(1)
    net = _student(2)
    roots = _trace_states(120) + [SimState.from_seed(21)]
    steps = 40
    stu = make_student(net)
    ref = []
    for i, r in enumerate(roots):
        sim = r.copy()
        sim.reseed(100 + i)
        traj = []
        for _ in range(steps):
            if sim.over:
                break
            obs = sim.obs()
            acts = {**stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5)}),
                    **stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5, 10)})}
            mv = np.zeros((10, 2), np.float32)
            for k, v in acts.items():
                mv[int(k.split("_")[1])] = np.clip(np.asarray(v, np.float32), -1, 1)
            sim.step(mv)
            traj.append(sim.uf.copy())
        ref.append((np.array(traj), sim.scores))
    pol = NetPolicy(net, device="cpu", per_sample=True)
    res = BatchRollout(pol, pol).run(roots, steps, reseeds=[100 + i for i in range(len(roots))], record=True)
    for b, (traj, scores) in enumerate(ref):
        assert np.array_equal(res.positions[: len(traj), b], traj), b
        assert tuple(res.scores[b]) == scores


class _Counter(BatchPolicy):
    """Moves every unit along +x on its own team's call, counting calls per game."""

    needs_inputs = False

    def reset(self, n):
        self.calls = np.zeros(n, np.int64)

    def moves(self, bsim, idx, teams, inputs):
        self.calls[idx] += 1
        return {tm: np.tile(np.array([1, 0], np.float32), (len(idx), 5, 1)) for tm in teams}


def test_action_repeat_and_hybrid_schedule():
    late = _trace_states(260)[-1]  # past the 10 s opening
    assert late.g[7] >= 500
    early = SimState.from_seed(5)
    pol = _Counter()
    roll = BatchRollout(pol, pol, repeat=4)
    roll.run([early, late], 12)
    # the opening calls every decision; past it every 4th (decisions 0, 4, 8)
    assert list(pol.calls) == [12, 3]
    first, then = _Counter(), _Counter()
    BatchRollout(HybridPolicy(first, then), HybridPolicy(first, then)).run([early, late], 5)
    # one HybridPolicy object per team would be called twice per game; here the two are distinct
    assert first.calls[0] > 0 and first.calls[1] == 0 and then.calls[1] > 0 and then.calls[0] == 0


def test_heuristic_policy_both_teams_move():
    """One HeuristicPolicy for both teams keeps an instance per team (a shared V23 instance
    returns zero moves for the second team it sees)."""
    from blackout_env.sim.batch import HeuristicPolicy

    pol = HeuristicPolicy("strategic_v23")
    res = BatchRollout(pol, pol).run([SimState.from_seed(7)], 10, record=True)
    start = SimState.from_seed(7).uf
    moved = np.abs(res.positions[-1, 0] - start).sum(axis=1) > 0
    assert moved[:5].any() and moved[5:].any()


def _rollout_net(seed=0, **kw):
    from blackout_env.sim.rollout_net import RolloutNet

    torch.manual_seed(seed)
    return RolloutNet(arch="attn", **{"d": 32, "layers": 1, "heads": 2, "hidden": 32, **kw}).eval()


# r5's shape, and the map-token variants of docs §22 (f3 6x6 tokens as cross-attention K/V, with and
# without a narrower map_dim, d256_f3_x32_d2 / f3_x_d2 in miniature)
STUDENT_NETS = {"plain": {}, "f3_cross": {"map_tokens": "f3", "map_attn": "cross", "layers": 2},
                "f3_cross_dm16": {"map_tokens": "f3", "map_attn": "cross", "map_dim": 16, "d": 64, "hidden": 64, "layers": 2}}


@pytest.mark.parametrize("variant", sorted(STUDENT_NETS))
def test_rollout_net_student_matches_batched_policy(tmp_path, variant):
    """A RolloutNet checkpoint (config + state; with or without map tokens) loads through the
    distill student path (inference.load_checkpoint_net + make_student: the Unity evaluations'
    MyPolicy, batch-1 jit trace) and plays the same trajectory as the simulator's NetPolicy at batch 1."""
    from blackout_env.train.distill.inference import checkpoint_meta, load_checkpoint_net, make_student

    torch.set_num_threads(1)
    net = _rollout_net(3, **STUDENT_NETS[variant])
    path = tmp_path / "rnet.pt"
    torch.save({"config": net.config, "state": net.state_dict()}, path)
    assert checkpoint_meta(path)["action_bins"] == 32
    stu = make_student(load_checkpoint_net(path))
    roots = _trace_states(120)[:2] + [SimState.from_seed(22)]
    steps = 30
    ref = []
    for i, r in enumerate(roots):
        sim = r.copy()
        sim.reseed(200 + i)
        traj = []
        for _ in range(steps):
            if sim.over:
                break
            obs = sim.obs()
            acts = {**stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5)}),
                    **stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5, 10)})}
            mv = np.zeros((10, 2), np.float32)
            for k, v in acts.items():
                mv[int(k.split("_")[1])] = np.clip(np.asarray(v, np.float32), -1, 1)
            sim.step(mv)
            traj.append(sim.uf.copy())
        ref.append(np.array(traj))
    pol = NetPolicy(load_checkpoint_net(path), device="cpu", per_sample=True)
    res = BatchRollout(pol, pol).run(roots, steps, reseeds=[200 + i for i in range(len(roots))], record=True)
    for b, traj in enumerate(ref):
        assert np.abs(res.positions[: len(traj), b] - traj).max() < 1e-4, b
    loaded = load_checkpoint_net(path)
    assert loaded.config == net.config and loaded.map_tokens == STUDENT_NETS[variant].get("map_tokens", "none")


ARCH_CKPTS = Path(__file__).resolve().parents[1] / "checkpoints/rollout_net/arch"


@pytest.mark.parametrize("name", ["d256_f3_x32_d2", "f3_x_d2"])
def test_map_token_checkpoint_student_equals_net(name):
    """The real §22 map-token checkpoints (skipped where they are not synced): the student path's
    traced batch-1 forward gives the net's own logits and offsets, on states the trace never saw."""
    from blackout_env.env.team_frame import canonical_obs
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.distill.inference import load_checkpoint_net, make_student

    path = ARCH_CKPTS / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"{path} not here")
    torch.set_num_threads(1)
    net = load_checkpoint_net(path)
    assert isinstance(net, RolloutNet) and net.config["map_tokens"] == "f3" and net.config["map_attn"] == "cross"
    stu = make_student(load_checkpoint_net(path))
    states = [SimState.from_seed(31)] + _trace_states(150)[:2]
    for sim in states:
        obs = sim.obs()
        stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5)})  # the first call traces
        c = canonical_obs(obs["unit_0"])
        x = (torch.tensor(c["graphic"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
             torch.tensor(c["team_state"], dtype=torch.float32).unsqueeze(0),
             torch.tensor(c["agent_states"], dtype=torch.float32).unsqueeze(0))
        with torch.inference_mode():
            want = net(*x, q_units=5, angle=True)
            got = stu._net(*x)
        for w, g in zip(want, got):
            torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-5)


def test_stop_when_exhausted_and_lead_trace():
    """A game stops at an absorption that leaves no battery in play; the lead trace holds the
    final lead past a game's end."""
    from blackout_env.sim import core
    from blackout_env.sim.batch import batteries_in_play

    s = SimState.from_seed(9)
    assert batteries_in_play(BatchSim([s]), np.array([0]))[0] > 0
    empty = s.copy()
    for i in range(empty.it.shape[0]):  # every battery back to the pool
        if empty.it[i, core.I_TYPE] == 0 and empty.it[i, core.I_STATE] != core.POOL:
            empty.it[i, core.I_STATE] = core.POOL
    assert batteries_in_play(BatchSim([empty]), np.array([0]))[0] == 0
    pol = _Counter()
    res = BatchRollout(pol, pol).run([s, empty], 30 * 25, stop_when_exhausted=True, trace_steps=[10, 2000])
    assert not res.exhausted[0] and res.exhausted[1]
    assert res.steps[1] <= 20 * 25 + 1 and res.steps[0] == 30 * 25
    assert res.lead_trace.shape == (2, 2) and res.lead_trace[1, 1] == res.lead(0)[1]


def test_net_device_from_env(monkeypatch, tmp_path):
    """$BLACKOUT_CUDA_NETS picks which checkpoints the student path puts on the GPU (by file name,
    or ``all``); without CUDA everything stays on the CPU.  Where CUDA exists, the GPU student
    gives the CPU student's moves."""
    from blackout_env.train.distill import inference

    monkeypatch.delenv(inference.CUDA_NETS_ENV, raising=False)
    assert inference.net_device("x/p11_v22.pt") == "cpu"
    monkeypatch.setenv(inference.CUDA_NETS_ENV, "p11_v22.pt, p5a_v22.pt")
    cuda = torch.cuda.is_available()
    assert inference.net_device("x/p11_v22.pt") == ("cuda" if cuda else "cpu")
    assert inference.net_device("x/f3_x_d2.pt") == "cpu"
    monkeypatch.setenv(inference.CUDA_NETS_ENV, "all")
    assert inference.net_device("x/f3_x_d2.pt") == ("cuda" if cuda else "cpu")
    if cuda:
        net = _rollout_net(4, **STUDENT_NETS["f3_cross"])
        obs = SimState.from_seed(41).obs()
        team = {f"unit_{u}": obs[f"unit_{u}"] for u in range(5, 10)}
        cpu = inference.make_student(net).act(team)
        gpu = inference.make_student(_rollout_net(4, **STUDENT_NETS["f3_cross"]), device="cuda").act(team)
        for k in cpu:
            assert np.abs(cpu[k] - gpu[k]).max() < 1e-3
