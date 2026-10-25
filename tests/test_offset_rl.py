"""offset_rl: the chosen bin's angle offset as a stochastic, trained part of the action (grpo.LearnerPolicy / GRPOTrainer.update)."""
import math

import numpy as np
import pytest
import torch

from tests.test_league import _cfg, needs_ckpt

SIG = 0.05


def _on(**kw):
    base = dict(offset_rl=True, offset_std_init=SIG, games=8, segment=8, minibatch=1024, critic_warmup=0)
    base.update(kw)
    return _cfg(**base)


@needs_ckpt
def test_flag_off_unchanged():
    """Default: no offset in the batch / records, and two runs from the same seed are identical."""
    from blackout_env.rl.ppo import PPOTrainer

    outs = []
    for kw in ({}, {"offset_rl": False}):
        tr = PPOTrainer(_cfg(**kw))
        assert tr.log_sigma is None and len(tr.opt.param_groups) == 1
        batch, _ = tr.collect()
        assert "off" not in batch
        outs.append(batch)
        s = PPOTrainer(_cfg(**kw)).step()
        assert np.isfinite(s["pg"]) and "offset_sigma" not in s
    for k in outs[0]:
        assert np.array_equal(outs[0][k], outs[1][k])


@needs_ckpt
@pytest.mark.parametrize("joint", [False, True])
def test_sampled_offsets_statistics(joint):
    """Recorded offsets = mu + N(0, sigma^2) for non-STAND units (mu = the net's offset of the chosen bin), 0 for STAND."""
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_on(joint=joint))
    batch, _ = tr.collect()
    off, act = batch["off"], batch["act"].astype(np.int64)
    assert off.shape == act.shape and off.dtype == np.float32
    with torch.no_grad():
        to = lambda x: torch.from_numpy(x)
        _, mu_all = tr._logits_off(tr.own, to(batch["g"]), to(batch["t"]), to(batch["a"]))
        mu, nonstand = tr._chosen_mu(mu_all, to(act))
    nonstand = nonstand.numpy().astype(bool)
    assert nonstand.sum() > 200
    assert np.all(off[~nonstand] == 0)
    z = (off[nonstand] - mu.numpy()[nonstand]) / SIG
    assert abs(z.mean()) < 0.25 and 0.8 < z.std() < 1.2


@needs_ckpt
def test_logprob_matches_manual_and_trains_sigma_and_offsets():
    """The first-minibatch policy gradient (ratio = 1, beta = 0) equals d/dtheta of -mean(A * [log pi(bin) + Normal.log_prob]),
    with no Gaussian term on STAND units; log_sigma and the angle head get gradients."""
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_on(beta=0.0, epochs=1))
    batch, _ = tr.collect()
    batch["act"][::3] = 32                      # some STAND units
    batch["off"][::3] = 0.0
    to = lambda x: torch.from_numpy(x)
    g, t, a, act, adv, o = (to(batch[k]) for k in ("g", "t", "a", "act", "adv", "off"))
    act = act.long()
    lg, off = tr._logits_off(tr.own, g, t, a)
    lp = torch.log_softmax(lg, -1).gather(-1, act[..., None])[..., 0]
    extra = torch.zeros_like(lp)
    for i in range(act.shape[0]):               # explicit loop over units: independent of the vectorised implementation
        for u in range(5):
            if act[i, u] < 32:
                extra[i, u] = torch.distributions.Normal(off[i, u, act[i, u]], tr.sigma()).log_prob(o[i, u])
    loss = -(adv[:, None] * (lp + extra)).mean()
    params = [p for p in tr.net.parameters() if p.requires_grad] + [tr.log_sigma]
    want = torch.autograd.grad(loss, params, allow_unused=True)
    got = []
    tr.opt.step = lambda *a_, **k_: got.append([None if p.grad is None else p.grad.clone() for p in params])
    tr.cfg.max_grad_norm = 1e9                  # the clip rescales net grads only: compare unclipped
    st = tr.update(batch)
    assert len(got) == 1 and st["clipfrac"] == 0.0 and st["kl_off"] == pytest.approx(0.0, abs=1e-6)
    assert st["offset_sigma"] == pytest.approx(SIG, rel=1e-5) and st["off_mu_shift"] == pytest.approx(0.0, abs=1e-8)
    for w, gt in zip(want, got[0]):
        assert (w is None) == (gt is None)
        if w is not None:
            assert torch.allclose(w, gt, atol=1e-6, rtol=2e-3), float((w - gt).abs().max())
    assert got[0][-1] is not None and got[0][-1].abs() > 0                            # log_sigma gets a gradient
    names = [k for k, p in tr.net.named_parameters() if p.requires_grad]
    ao = [i for i, k in enumerate(names) if "angle_out" in k]
    assert ao and any(got[0][i] is not None and got[0][i].abs().max() > 0 for i in ao)   # offsets get a gradient


@needs_ckpt
def test_step_and_kl_term():
    """A few real steps: sigma moves (own lr), offsets drift from the reference, KL terms are finite and positive."""
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_on(lr=1e-3, offset_std_lr=1e-2, epochs=2, minibatch=16))
    ls0 = float(tr.log_sigma)
    for _ in range(2):
        s = tr.step()
    assert float(tr.log_sigma) != ls0 and s["offset_sigma"] != SIG
    assert s["off_mu_shift"] > 0 and s["kl_off"] > 0 and np.isfinite(s["pg"]) and np.isfinite(s["kl"])
    tr2 = PPOTrainer(_on(offset_std_min=0.2))     # clamp
    assert float(tr2.sigma()) == pytest.approx(0.2)
    tr3 = PPOTrainer(_on(offset_std_learn=False))  # fixed sigma: not in the optimiser
    assert len(tr3.opt.param_groups) == 1 and not tr3.log_sigma.requires_grad


@needs_ckpt
def test_save_resume_roundtrip(tmp_path):
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_on())
    with torch.no_grad():
        tr.log_sigma.fill_(math.log(0.0123))
    tr.save_resume(str(tmp_path / "r.pt"))
    tr.save(str(tmp_path / "m.pt"))
    assert torch.load(str(tmp_path / "m.pt"), weights_only=False)["offset_log_sigma"] == pytest.approx(math.log(0.0123), rel=1e-6)
    tr2 = PPOTrainer(_on())
    assert float(tr2.sigma()) == pytest.approx(SIG)
    tr2.load_resume(str(tmp_path / "r.pt"))
    assert float(tr2.log_sigma) == pytest.approx(float(tr.log_sigma)) and float(tr2.sigma()) == pytest.approx(0.0123, rel=1e-5)
    tr2.step()


@needs_ckpt
def test_offset_detach_trunk_gets_no_offset_gradient():
    """offset_detach: a loss on the offsets alone gives the angle head a gradient and the trunk/bin logits none;
    offset_detach=False lets it reach the trunk.  Separate clipping logs grad_norm_off."""
    from blackout_env.rl.ppo import PPOTrainer
    from blackout_env.sim.batch import unpack_graphic

    for detach in (True, False):
        tr = PPOTrainer(_cfg(offset_rl=True, offset_detach=detach))
        batch, _ = tr.collect()
        g = torch.from_numpy(batch["g"][:8]); t = torch.from_numpy(batch["t"][:8]); a = torch.from_numpy(batch["a"][:8])
        tr.net.zero_grad(set_to_none=True)
        _, off = tr._logits_off(tr.own, g, t, a)
        off.sum().backward()
        ang = {id(q) for n_, m in tr.net.named_modules() if n_.endswith("angle_out") for q in m.parameters()}
        body = [q for q in tr.net.parameters() if id(q) not in ang and q.grad is not None and q.grad.abs().sum() > 0]
        assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in tr.net.parameters() if id(q) in ang)
        assert (len(body) == 0) == detach
    tr = PPOTrainer(_cfg(offset_rl=True, critic_warmup=0))
    s = tr.step()
    assert "grad_norm_off" in s and np.isfinite(s["grad_norm_off"]) and "kl_bin" in s
