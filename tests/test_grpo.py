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


def test_ppo_mixed_pool_pfsp_cpu():
    """PPO with the r3b-style pool (net + heuristic) and PFSP: steps run and report the opponent weights."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=4, segment=6, minibatch=8, epochs=1, critic_warmup=0,
                    opponents=[f"net:{CKPT}", "random"], opp_weights=[0.5, 1.0], pfsp="hard")
    tr = PPOTrainer(cfg)
    s = tr.step()
    assert np.isfinite(s["pg"]) and len(s["opp_w"]) == 2 and abs(sum(s["opp_w"]) - 1) < 1e-3


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


def test_value_retraining_cpu(tmp_path, monkeypatch):
    # the old seed range: with it these two trunk games end within the 2 x 5000 decisions below (seed luck, not logic)
    from blackout_env.sim.state import SimState
    monkeypatch.setattr(GRPOTrainer, "_fresh_seed", lambda self: SimState.from_seed(int(self.rng.integers(0, 2**31 - 1))))
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


def test_ppo_overlap_heur_is_bit_identical_cpu():
    """overlap_heur (split JointSeats, heuristics of both teams started before the net forwards) collects exactly the
    batch of the per-team Seats path, in-process heuristics and worker processes alike."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    def collect(overlap, workers):
        cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=6, segment=6, seed=3, critic_warmup=0, overlap_heur=overlap,
                        heur_workers=workers, opponents=[f"net:{CKPT}", "random", "net:" + CKPT], pfsp="hard")
        tr = PPOTrainer(cfg)
        return tr.collect()[0]

    ref = collect(False, 0)
    for overlap, workers in ((True, 0), (True, 2)):
        got = collect(overlap, workers)
        for k in ref:
            assert np.array_equal(ref[k], got[k]), (k, overlap, workers)


def test_ppo_hlgauss_critic_cpu():
    """HL-Gauss critic: pre-fit on the first batch, CE steps run, V within [vmin, vmax], diagnostics finite."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=3, segment=8, minibatch=8, epochs=1, critic_warmup=1,
                    critic_head="hlgauss", critic_bins=21)
    tr = PPOTrainer(cfg)
    s0, s1 = tr.step(), tr.step()
    assert "pg" not in s0 and np.isfinite(s1["pg"]) and np.isfinite(s0["vf_loss"]) and np.isfinite(s1["vf_loss"])
    for s in (s0, s1):
        assert np.isfinite(s["expl_var"]) and np.isfinite(s["abs_adv"]) and 0 < s["pmf_entropy"] <= np.log(21) + 1e-4
    assert np.isfinite(s0["prefit_ce_last"]) and s0["prefit_ce_last"] <= s0["prefit_ce_first"] + 1e-6
    g = np.random.default_rng(0)
    batch, _ = tr.collect()
    v = tr._values_np(batch["g"], batch["t"], batch["a"])
    assert (v >= cfg.critic_vmin - 1e-5).all() and (v <= cfg.critic_vmax + 1e-5).all()
    # critic round-trips through the _critic.pt mechanism incl. the new head
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        tr.save(os.path.join(d, "x.pt"))
        sd = torch.load(os.path.join(d, "x_critic.pt"), weights_only=True)
        assert "bin_out.weight" in sd and sd["bin_out.weight"].shape[0] == 21


def test_ppo_reg_critic_diagnostics_cpu():
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    s = PPOTrainer(PPOConfig(init=CKPT, value=VALUE, device="cpu", games=3, segment=8, minibatch=8, epochs=1, critic_warmup=0)).step()
    assert np.isfinite(s["expl_var"]) and np.isfinite(s["abs_adv"]) and np.isnan(s["pmf_entropy"])


def test_ppo_async_collect_matches_sync_when_frozen_cpu():
    """async_collect with lr = critic_lr = 0 (weights never change, so the 1-iteration lag is invisible) collects exactly
    the batches of the synchronous trainer, and its PPO old log-probs come from the behaviour copy."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    def run(async_):
        cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=4, segment=6, minibatch=8, epochs=1, critic_warmup=0, seed=5,
                        lr=0.0, critic_lr=0.0, opponents=[f"net:{CKPT}", "random"], pfsp="hard", async_collect=async_)
        tr = PPOTrainer(cfg)
        got = []
        orig = tr.collect
        def rec():
            b, s = orig()
            got.append(b)
            return b, s
        tr.collect = rec
        stats = [tr.step() for _ in range(3)]
        return got, stats

    ref, s_ref = run(False)
    got, s_got = run(True)
    assert len(got) == 4 and len(ref) == 3   # async collected one batch ahead
    for a, b in zip(ref, got[:3]):
        for k in a:
            assert np.array_equal(a[k], b[k]), k
    for a, b in zip(s_ref, s_got):
        assert abs(a["kl"] - b["kl"]) < 1e-6 and abs(a["clipfrac"] - b["clipfrac"]) < 1e-9 and np.isfinite(b["actor_wait_s"])


def test_ppo_async_collect_learns_cpu():
    """async_collect with a real learning rate: steps run, stats finite, behaviour copy lags the learner by one update."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=3, segment=8, minibatch=8, epochs=1, critic_warmup=0,
                    opponents=["random"], async_collect=True)
    tr = PPOTrainer(cfg)
    for _ in range(3):
        s = tr.step()
        assert np.isfinite(s["pg"]) and np.isfinite(s["vf_loss"]) and np.isfinite(s["kl"])
        assert np.isfinite(s["isw_mean"]) and 0.0 <= s["isw_trunc"] <= 1.0   # decoupled PPO (default on)
    # the pending batch was played by the weights after update 2 (= the actor copy), the learner has moved on by update 3
    w_act = next(iter(tr._act_net.parameters()))
    w_now = next(iter(tr.net.parameters()))
    assert not torch.equal(w_act, w_now)


def test_ppo_async_collect_auto_resolution():
    """async_collect="auto" is off without 2 CUDA GPUs; string/bool overrides resolve to bools."""
    from blackout_env.rl.ppo import PPOConfig, _resolve_async
    c = PPOConfig(device="cpu"); _resolve_async(c)
    assert c.async_collect is False and c.actor_device == ""
    for v, want in (("true", True), ("false", False), (True, True), (False, False)):
        c = PPOConfig(device="cpu", async_collect=v); _resolve_async(c)
        assert c.async_collect is want
    c = PPOConfig(device="cpu", async_collect="true", critic_head="hlgauss"); _resolve_async(c)
    assert c.async_collect is True   # forced: the trainer's own assert then rejects the unsupported recipe


def test_teach_moves_to_bins_roundtrip():
    """moves_to_bins inverts decode_moves' bin decode (bin centres, both teams, STAND)."""
    from blackout_env.rl.teach import moves_to_bins
    from blackout_env.sim.batch import decode_moves

    rng = np.random.default_rng(0)
    for width in (16, 17):
        q = rng.normal(size=(6, 5, width)).astype(np.float32)
        team_b = np.array([0, 1, 0, 1, 1, 0], bool)
        mv = decode_moves(q, None, team_b)
        got = moves_to_bins(mv, team_b, 16, stand=width == 17)
        assert np.array_equal(got, q.argmax(-1).astype(np.int8))


def test_ppo_teach_cpu():
    """teach on: labelled batch, finite CE, coef decays to 0, then the teacher is dropped (batch without labels)."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=3, segment=8, minibatch=8, epochs=1, critic_warmup=0,
                    opponents=["random"], teach="B5", teach_coef=0.5, teach_iters=2, async_collect="false", heur_workers=0)
    tr = PPOTrainer(cfg)
    s0, s1, s2 = tr.step(), tr.step(), tr.step()
    assert np.isfinite(s0["teach_ce"]) and 0 <= s0["teach_acc"] <= 1 and s0["teach_coef"] == 0.5
    assert s1["teach_coef"] == 0.25
    assert "teach_ce" not in s2 and tr._teacher is None


def test_ppo_teach_opps_cpu():
    """teach_opps: only games against the listed opponents carry teacher labels (others -1)."""
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(init=CKPT, value=VALUE, device="cpu", games=4, segment=8, minibatch=8, epochs=1, critic_warmup=0,
                    opponents=["random", "idle"], teach="B5", teach_coef=0.5, teach_iters=10, teach_opps="1",
                    async_collect="false", heur_workers=0)
    tr = PPOTrainer(cfg)
    tr.trunk_opp[:] = np.array([0, 1, 0, 1])
    batch, _ = tr.collect()
    lab = batch["teach"]
    rows = lab.reshape(len(lab), -1)
    labelled = (rows >= 0).all(1)
    unl = (rows < 0).all(1)
    assert labelled.any() and unl.any() and (labelled | unl).all()


def test_ppo_critic_opp_cpu():
    """critic_opp: zero-init opponent embedding -> V identical to the plain critic at start; it trains; async works;
    the embedding round-trips through _critic.pt and critic_init."""
    import os
    import tempfile

    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    kw = dict(init=CKPT, value=VALUE, device="cpu", games=4, segment=6, minibatch=8, epochs=1, critic_warmup=0, seed=5,
              opponents=[f"net:{CKPT}", "random"], pfsp="hard")
    base = PPOTrainer(PPOConfig(**kw))
    tr = PPOTrainer(PPOConfig(**kw, critic_opp=True, critic_opp_slots=16))
    assert tr.critic.opp_emb is not None and tr.critic.opp_emb.weight.shape[0] == 16
    b, _ = base.collect()
    v0 = base._values_np(b["g"], b["t"], b["a"])
    v1 = tr._values_np(b["g"], b["t"], b["a"], o=np.zeros(len(v0), np.int64) + 1)
    assert np.allclose(v0, v1, atol=1e-6)
    s = tr.step()
    assert np.isfinite(s["vf_loss"]) and tr.critic.opp_emb.weight.abs().sum() > 0
    with tempfile.TemporaryDirectory() as d:
        tr.save(os.path.join(d, "x.pt"))
        cp = os.path.join(d, "x_critic.pt")
        assert torch.load(cp, weights_only=True)["opp_emb.weight"].shape[0] == 16
        tr2 = PPOTrainer(PPOConfig(**kw, critic_opp=True, critic_opp_slots=16, critic_init=cp, async_collect="true"))
        assert torch.equal(tr2.critic.opp_emb.weight, tr.critic.opp_emb.weight)
        assert np.isfinite(tr2.step()["vf_loss"]) and np.isfinite(tr2.step()["vf_loss"])


def test_grpo_root_max_s_cpu():
    """Opening-only GRPO: trunk games restart once past their threshold, so every root stays in the opening."""
    from blackout_env.sim import core

    cfg = GRPOConfig(init=CKPT, value=VALUE, device="cpu", groups=3, k=2, trunk_step=60, sample_steps=3, leaf_steps=4,
                     minibatch=16, epochs=1, root_max_s=2.0)
    tr = GRPOTrainer(cfg)
    assert ((tr.root_thr >= 50) & (tr.root_thr <= 100)).all()
    for _ in range(4):
        tr.collect()
        ticks = np.array([s.g[core.G_TICK] for s in tr.trunk])
        assert (ticks <= 100 + 2 * 60).all()
    s = tr.step()
    assert np.isfinite(s["loss"])


def test_ppo_hlgauss_async_and_critic_init_cpu():
    """hlgauss now runs with async_collect; a critic_init that carries a trained bin head skips the pre-fit and is
    loaded exactly; a reg critic_init still gets a fresh (pre-fitted) bin head."""
    import os
    import tempfile

    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    kw = dict(init=CKPT, value=VALUE, device="cpu", games=4, segment=6, minibatch=8, epochs=1, critic_warmup=0, seed=5,
              opponents=[f"net:{CKPT}", "random"], pfsp="hard", critic_head="hlgauss", critic_bins=21)
    tr = PPOTrainer(PPOConfig(**kw, async_collect="true"))
    s0, s1 = tr.step(), tr.step()
    assert np.isfinite(s1["vf_loss"]) and np.isfinite(s1["pg"]) and 0 < s1["pmf_entropy"] <= np.log(21) + 1e-4
    with tempfile.TemporaryDirectory() as d:
        tr.save(os.path.join(d, "x.pt"))
        cp = os.path.join(d, "x_critic.pt")
        tr2 = PPOTrainer(PPOConfig(**kw, critic_init=cp, async_collect="true"))
        assert tr2._prefit_done
        assert torch.equal(tr2.critic.bin_out.weight, tr.critic.bin_out.weight)
        assert np.isfinite(tr2.step()["vf_loss"])
        sd = {k: v for k, v in torch.load(cp, weights_only=True).items() if not k.startswith("bin_out")}
        rp = os.path.join(d, "reg_critic.pt")
        torch.save(sd, rp)
        tr3 = PPOTrainer(PPOConfig(**kw, critic_init=rp))
        assert not tr3._prefit_done
