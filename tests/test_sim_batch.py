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


def _student(seed=0):
    from blackout_env.model.my_model import MyModel

    torch.manual_seed(seed)
    net = MyModel(hidden_size=64, n_actions=32, policy_head="logits", angle_head=True, angle_arc_bins=1).eval()
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


def test_decode_matches_mypolicy_team_b():
    from blackout_env.model.my_policy import MyPolicy

    net = _student(1)
    pol = MyPolicy(net)
    states = _trace_states(60)
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


def _rollout_net(seed=0):
    from blackout_env.sim.rollout_net import RolloutNet

    torch.manual_seed(seed)
    return RolloutNet(arch="attn", d=32, layers=1, heads=2, hidden=32).eval()


def test_rollout_net_student_matches_batched_policy(tmp_path):
    """A RolloutNet checkpoint (config + state) loads through the distill student path
    (inference.load_checkpoint_net + make_student: the Unity evaluations' MyPolicy) and plays the
    same trajectory as the simulator's NetPolicy at batch 1."""
    from blackout_env.train.distill.inference import checkpoint_meta, load_checkpoint_net, make_student

    torch.set_num_threads(1)
    net = _rollout_net(3)
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
