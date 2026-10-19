"""GRPO pieces (blackout_env/rl/grpo.py): greedy LearnerPolicy = NetPolicy, Seats routing, branch variance."""
import numpy as np
import pytest
import torch

from blackout_env.rl.grpo import GRPOConfig, GRPOTrainer, LearnerPolicy, Seats
from blackout_env.sim.batch import BatchRollout, NetPolicy
from blackout_env.sim.state import SimState
from blackout_env.train.distill.inference import _OwnQ, load_net

CKPT = "checkpoints/distill/frozen/p24c_v321.pt"
VALUE = "checkpoints/value/s2v_f3xd2/iter2.pt"
pytestmark = pytest.mark.skipif(not __import__("os").path.exists(CKPT), reason="needs the P24c checkpoint")


def _net():
    return load_net(torch.load(CKPT, map_location="cpu", weights_only=True)["policy_state"]).eval()


def test_greedy_learner_matches_netpolicy():
    """Same inputs, same batch: the greedy learner's one-hot decode = NetPolicy's argmax decode."""
    from blackout_env.sim.batch import BatchSim, N_AGENT_COLS, H, W

    net = _net()
    roots = [SimState.from_seed(s) for s in (3, 4)]
    res = BatchRollout(NetPolicy(net), NetPolicy(net), packed=True).run(roots, 700)
    bsim = res.bsim
    idx = np.arange(bsim.n)
    bufs = (np.zeros((2, 2, H, W), np.int16), np.zeros((2, 2, 4), np.float32), np.zeros((2, 2, 10, N_AGENT_COLS), np.float32))
    bsim.render_packed(idx, *bufs)
    lp = LearnerPolicy(_OwnQ(net), "cpu", 0, fp16=False)
    lp.reset(2)
    for tm in (0, 1):
        want = NetPolicy(net).moves(bsim, idx, (tm,), bufs)[tm]
        got = lp.moves(bsim, idx, (tm,), bufs)[tm]
        assert np.array_equal(want, got)


def test_sampling_records_only_window():
    net = _net()
    lp = LearnerPolicy(_OwnQ(net), "cpu", 5, fp16=False)
    side = np.array([0, 1, 0])
    opp = NetPolicy(net)
    roots = [SimState.from_seed(s) for s in (1, 2, 3)]
    BatchRollout(Seats(0, lp, [opp], np.zeros(3, int), side), Seats(1, lp, [opp], np.zeros(3, int), side),
                 packed=True).run(roots, 20)
    game = np.concatenate([r[0] for r in lp.records])
    assert np.bincount(game, minlength=3).tolist() == [5, 5, 5]
    assert all(r[4].shape[1:] == (5,) for r in lp.records)


def test_trainer_step_cpu():
    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=2, k=3, trunk_step=400, sample_steps=5,
                     leaf_steps=10, minibatch=16, epochs=1)
    s = GRPOTrainer(cfg).step()
    assert s["views"] == 2 * 3 * 5 and np.isfinite(s["loss"])


def test_ppo_step_cpu():
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=3, segment=8, minibatch=8, epochs=1, critic_warmup=1)
    tr = PPOTrainer(cfg)
    s0, s1 = tr.step(), tr.step()
    assert s0["views"] == 24 and "pg" not in s0 and np.isfinite(s1["pg"]) and np.isfinite(s1["vf_loss"])
