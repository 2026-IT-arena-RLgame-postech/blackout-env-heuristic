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


def test_pfsp_weights():
    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=2, k=2,
                     opponents=[f"net:{CKPT}", "random"], opp_weights=[1.0, 3.0], pfsp="hard")
    tr = GRPOTrainer(cfg)
    assert np.allclose(tr.opp_w, [0.25, 0.75])            # equal scores: base weights
    tr.opp_score[:] = [0.2, 0.9]                            # we lose to opponent 0
    w = tr._pfsp_weights()
    f = cfg.pfsp_floor + (1 - np.array([0.2, 0.9])) ** 2
    assert np.allclose(w, np.array([1, 3]) * f / (np.array([1, 3]) * f).sum()) and w[0] > 0.25


def test_heuristic_opponent_step_cpu():
    """A heuristic opponent in the pool: branch instances are separate from the trunk's and reset."""
    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=2, k=2, trunk_step=300, sample_steps=3,
                     leaf_steps=4, minibatch=16, epochs=1, opponents=["random"], pfsp="hard")
    tr = GRPOTrainer(cfg)
    assert tr.branch_opps[0] is not tr.opps[0]
    s = tr.step()
    assert s["views"] == 2 * 2 * 3 and np.isfinite(s["loss"]) and len(s["opp_score"]) == 1
    assert len(tr.opps[0].instances) == 2 and len(tr.branch_opps[0].instances) == 4


def test_drop_flat_and_ref_reset():
    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=2, k=2, trunk_step=300, sample_steps=3,
                     leaf_steps=4, minibatch=16, epochs=1, drop_flat=1e9, ref_reset_every=1)
    tr = GRPOTrainer(cfg)
    s = tr.step()                                   # every group is "flat": no update, no crash
    assert s["kept_frac"] == 0.0 and s["loss"] == 0.0
    tr.cfg.drop_flat = 0.0
    s = tr.step()
    assert s["kept_frac"] == 1.0
    for p, q in zip(tr.net.parameters(), tr.ref.net.parameters()):
        assert torch.equal(p, q)                    # reset after the step


def test_value_retraining_cpu(tmp_path):
    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=2, k=2, trunk_step=5000, sample_steps=2,
                     leaf_steps=2, minibatch=16, epochs=1, value_train=True, value_snaps=4, value_every=1,
                     value_steps=2, value_batch=8, value_min_rows=1)
    tr = GRPOTrainer(cfg)
    before = [p.clone() for p in tr.leaf.net.parameters()]
    s = tr.step(); s = tr.step()                       # 2 x 5000 decisions: every trunk game has ended once
    assert tr.vbuf.size > 0 and np.isfinite(s["v_loss"])
    assert set(np.unique(tr.vbuf.result[:tr.vbuf.size])) <= {-1, 0, 1}
    # both views of a game carry opposite labels
    assert np.isclose(tr.vbuf.final[:tr.vbuf.size].sum(), 0.0, atol=1e-4)
    assert any(not torch.equal(p, q) for p, q in zip(before, tr.leaf.net.parameters()))
    tr.save(str(tmp_path / "x.pt"))
    from blackout_env.value.model import load_value_net
    load_value_net(str(tmp_path / "x_value.pt"))
