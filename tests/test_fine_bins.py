"""fine_bins: hierarchical 32 x 32 direction head (blackout_env/rl/finehead.py; grpo.LearnerPolicy / GRPOTrainer.update; NetPolicy / eval_batched)."""
import hashlib
import math

import numpy as np
import pytest
import torch

from tests.test_league import _cfg, needs_ckpt


def _fcfg(**kw):
    base = dict(fine_bins=32, games=8, segment=8, minibatch=1024, critic_warmup=0)
    base.update(kw)
    return _cfg(**base)


def _randomise(tr, std=0.3, seed=0):
    """Non-zero residual (a trained head): the fine distribution is no longer just the prior."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        tr.fine.fc2.weight.copy_(torch.randn(tr.fine.fc2.weight.shape, generator=gen) * std)
        tr.fine.fc2.bias.copy_(torch.randn(tr.fine.fc2.bias.shape, generator=gen) * std)
        tr.fine_ref.load_state_dict(tr.fine.state_dict())   # the KL anchor = the randomised head: kl_fine starts at 0


def _hash(batch, stats):
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(k.encode()); h.update(np.ascontiguousarray(batch[k]).tobytes())
    for k in sorted(stats):
        if isinstance(stats[k], float) and not k.endswith("_s"):
            h.update(k.encode()); h.update(repr(stats[k]).encode())
    return h.hexdigest()


@needs_ckpt
def test_flag_off_identical():
    """fine_bins=0 (default): no fine head / tap / param group / batch key, and collect + update are the same as with the flag absent."""
    from blackout_env.rl.ppo import PPOTrainer

    hs, nets = [], []
    for kw in ({}, {"fine_bins": 0}):
        tr = PPOTrainer(_cfg(**kw))
        assert tr.fine is None and tr.tap is None and len(tr.opt.param_groups) == 1
        batch, st = tr.collect()
        assert "fine" not in batch
        torch.manual_seed(0)   # update() draws its minibatch permutation from the global RNG
        us = tr.update(batch)
        assert "kl_fine" not in us
        hs.append(_hash(batch, {**st, **us}))
        nets.append(torch.cat([p.detach().flatten() for p in tr.net.parameters()]))
    assert hs[0] == hs[1] and torch.equal(nets[0], nets[1])
    with pytest.raises(AssertionError):
        PPOTrainer(_cfg(fine_bins=32, offset_rl=True))


@needs_ckpt
def test_init_greedy_within_half_subbin():
    """At init (zero residual) the greedy fine centre is the sub-bin nearest the distilled offset: |c_j - mu| <= w / 2; via the
    NetPolicy path and the trainer's learner path; STAND untouched."""
    from blackout_env.rl.ppo import PPOTrainer
    from blackout_env.sim.batch import NetPolicy

    tr = PPOTrainer(_fcfg())
    batch, _ = tr.collect()
    w = tr.fine.width
    assert w == pytest.approx(2 * tr.tap.scale / 32)
    g, t, a = batch["g"], batch["t"], batch["a"]
    pol = NetPolicy(tr.net, device="cpu", fine=tr.fine)
    q, off2 = pol.forward(g, t, a)
    q0, off0 = NetPolicy(tr.net, device="cpu").forward(g, t, a)
    assert np.array_equal(q, q0)
    k = q.argmax(-1)
    ns = k < 32
    kc = np.minimum(k, 31)[..., None]
    mu = np.take_along_axis(off0, kc, -1)[..., 0]
    c = np.take_along_axis(off2, kc, -1)[..., 0]
    assert ns.sum() > 100
    assert np.abs(c - mu)[ns].max() <= w / 2 + 1e-6
    assert np.abs(c - mu)[ns].max() > 0                      # actually quantised
    other = np.ones(off0.shape, bool)
    np.put_along_axis(other, kc, False, -1)
    assert np.array_equal(off2[other], off0[other])           # only the chosen bin's entry changes
    # learner path (rollout): greedy rows give the same centre
    from blackout_env.rl.grpo import LearnerPolicy

    lp = LearnerPolicy(tr.own, tr.dev, 0, fine=tr.fine, tap=tr.tap)
    lp.reset(len(g))
    qq, oo = lp._forward(g, t, a)
    o3, j = lp._fine_apply(oo, qq.argmax(-1), None)
    assert np.allclose(o3.numpy(), off2, atol=1e-6)
    assert (j.numpy()[~ns] == 0).all()


@needs_ckpt
def test_sampled_fine_follows_softmax():
    """Recorded fine indices ~ softmax(fine logits / T) of the chosen coarse bin (random residual so it is not a bare prior);
    STAND units record 0."""
    from blackout_env.rl.ppo import PPOTrainer

    for joint in (False, True):
        tr = PPOTrainer(_fcfg(games=16, segment=16, joint=joint))
        _randomise(tr, 0.5)
        batch, _ = tr.collect()
        fj, act = batch["fine"], batch["act"].astype(np.int64)
        assert fj.dtype == np.int8 and fj.shape == act.shape
        with torch.no_grad():
            to = torch.from_numpy
            q, off, ft = tr._fine_fwd(tr.own, tr.tap, unpack(to(batch["g"])), to(batch["t"]), to(batch["a"]))
            flg, ns = tr._fine_logits(tr.fine, ft, to(act), off)
            p = torch.softmax(flg, -1).double().numpy()
        ns = ns.numpy().astype(bool)
        assert ns.sum() > 800 and np.all(fj[~ns] == 0)
        idx = np.arange(p.shape[-1])
        mean, var = (p * idx).sum(-1), (p * idx ** 2).sum(-1) - (p * idx).sum(-1) ** 2
        z = (fj[ns] - mean[ns]).sum() / math.sqrt(var[ns].sum())
        assert abs(z) < 4.0
        r = ((fj[ns] - mean[ns]) ** 2).sum() / var[ns].sum()
        assert 0.85 < r < 1.15
        # probability integral transform: the observed index's cdf is ~uniform
        cdf = np.cumsum(p, -1)
        u = cdf[ns][np.arange(ns.sum()), fj[ns]] - p[ns][np.arange(ns.sum()), fj[ns]] * np.random.default_rng(0).uniform(size=ns.sum())
        assert abs(u.mean() - 0.5) < 0.03
    # greedy rows (sample_steps exhausted) take the argmax
    from blackout_env.rl.grpo import LearnerPolicy

    lp = LearnerPolicy(tr.own, tr.dev, 0, fine=tr.fine, tap=tr.tap)
    lp.reset(len(batch["g"]))
    qq, oo = lp._forward(batch["g"], batch["t"], batch["a"])
    ch = qq.argmax(-1)
    o3, j = lp._fine_apply(oo, ch, None)
    kc = ch.clamp(max=31)
    flg = tr.fine(lp._feat, kc, oo.gather(-1, kc[..., None])[..., 0])
    assert torch.equal(j[ch < 32], flg.argmax(-1)[ch < 32])


def unpack(g):
    from blackout_env.sim.batch import unpack_graphic

    return unpack_graphic(g) if g.dim() == 3 else g


@needs_ckpt
def test_logprob_matches_manual_and_gradients():
    """First-minibatch gradient (ratio = 1, beta = 0) == d/dtheta of -mean(A * [log pi(k) + [non-STAND] log pi(j | k)]) computed by
    an explicit per-unit loop (incl. STAND units); fine head trained, angle head not; with fine_detach the fine term gives the trunk nothing."""
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_fcfg(beta=0.0, epochs=1))
    _randomise(tr, 0.3)
    batch, _ = tr.collect()
    batch["act"][::3] = 32                      # some STAND units
    batch["fine"][::3] = 0
    to = torch.from_numpy
    g, t, a, act, adv, fj = (to(batch[k]) for k in ("g", "t", "a", "act", "adv", "fine"))
    act, fj = act.long(), fj.long()
    lgt, off, ft = tr._fine_fwd(tr.own, tr.tap, unpack(g), t, a)
    lp = torch.log_softmax(lgt, -1).gather(-1, act[..., None])[..., 0]
    extra = torch.zeros_like(lp)
    for i in range(act.shape[0]):               # explicit loop over units: independent of the vectorised implementation
        for u in range(5):
            if act[i, u] < 32:
                k = act[i, u]
                lf = tr.fine(ft[i, u].detach(), k, off[i, u, k].detach()) / tr.cfg.temperature
                extra[i, u] = torch.log_softmax(lf, -1)[fj[i, u]]
    loss = -(adv[:, None] * (lp + extra)).mean()
    params = [p for p in tr.net.parameters()] + list(tr.fine.parameters())
    want = torch.autograd.grad(loss, params, allow_unused=True)
    got = []
    tr.opt.step = lambda *a_, **k_: got.append([None if p.grad is None else p.grad.clone() for p in params])
    tr.cfg.max_grad_norm = 1e9
    st = tr.update(batch)
    assert len(got) == 1 and st["clipfrac"] == 0.0 and st["kl_fine"] == pytest.approx(0.0, abs=1e-6)
    assert st["grad_norm_fine"] > 0
    for w_, gt in zip(want, got[0]):
        assert (w_ is None) == (gt is None)
        if w_ is not None:
            assert torch.allclose(w_, gt, atol=1e-6, rtol=2e-3), float((w_ - gt).abs().max())
    names = [k for k, _ in tr.net.named_parameters()]
    assert all(got[0][i] is None or got[0][i].abs().max() == 0 for i, n_ in enumerate(names) if "angle_out" in n_)
    assert any(gt is not None and gt.abs().max() > 0 for gt in got[0][len(names):])    # fine head trained

    # the fine term alone: trunk gradient iff not fine_detach
    for detach in (True, False):
        tr = PPOTrainer(_fcfg(fine_detach=detach))
        _randomise(tr, 0.3)
        batch, _ = tr.collect()
        tr.net.zero_grad(set_to_none=True)
        tr.fine.zero_grad(set_to_none=True)
        _, off, ft = tr._fine_fwd(tr.own, tr.tap, unpack(to(batch["g"])), to(batch["t"]), to(batch["a"]))
        flg, ns = tr._fine_logits(tr.fine, ft, to(batch["act"]).long(), off)
        (torch.log_softmax(flg, -1)[..., 5] * ns).sum().backward()
        body = [p for n_, p in tr.net.named_parameters() if "angle_out" not in n_ and p.grad is not None and p.grad.abs().sum() > 0]
        assert (len(body) == 0) == detach
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tr.fine.parameters())
        assert all(p.grad is None for n_, p in tr.net.named_parameters() if "angle_out" in n_)   # mu is detached


@needs_ckpt
def test_step_moves_fine_head_and_kl():
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_fcfg(lr=1e-4, minibatch=16, epochs=2))
    assert len(tr.opt.param_groups) == 2 and tr.opt.param_groups[1]["lr"] == tr.cfg.fine_lr
    w0 = tr.fine.fc2.weight.detach().clone()
    for _ in range(2):
        s = tr.step()
    assert not torch.equal(tr.fine.fc2.weight, w0)
    assert s["kl_fine"] > 0 and np.isfinite(s["pg"]) and np.isfinite(s["kl"]) and s["ent_fine"] > 0
    assert all(not p.requires_grad for p in tr.fine_ref.parameters())
    # ref sync copies the fine head too
    tr._sync_ref()
    assert torch.equal(tr.fine_ref.fc2.weight, tr.fine.fc2.weight)


@needs_ckpt
def test_save_resume_and_snapshot(tmp_path):
    from blackout_env.rl.finehead import FineHead
    from blackout_env.rl.grpo import make_opponent_policy
    from blackout_env.rl.ppo import PPOTrainer
    from blackout_env.sim.batch import NetPolicy

    tr = PPOTrainer(_fcfg(league_dir=str(tmp_path)))
    _randomise(tr, 0.2, 1)
    with torch.no_grad():
        tr.fine_ref.fc2.weight.fill_(0.01)
    tr.save(str(tmp_path / "m.pt"))
    ck = torch.load(str(tmp_path / "m.pt"), weights_only=True)
    assert "fine_head_state" in ck
    h = FineHead.from_state(ck["fine_head_state"])
    assert all(torch.equal(h.state_dict()[k], v) for k, v in tr.fine.state_dict().items())
    tr.save_resume(str(tmp_path / "r.pt"))
    tr2 = PPOTrainer(_fcfg(league_dir=str(tmp_path)))
    assert not torch.equal(tr2.fine.fc2.weight, tr.fine.fc2.weight)
    tr2.load_resume(str(tmp_path / "r.pt"))
    for k, v in tr.fine.state_dict().items():
        assert torch.equal(tr2.fine.state_dict()[k], v)
    for k, v in tr.fine_ref.state_dict().items():
        assert torch.equal(tr2.fine_ref.state_dict()[k], v)
    tr2.step()
    # league snapshot carries the fine head; make_opponent_policy plays with it
    sn = tr._add_snapshot()
    assert "fine_head_state" in torch.load(sn["path"], weights_only=True)
    pol = tr.opps[-1]
    assert isinstance(pol, NetPolicy) and pol.fine is not None
    assert torch.equal(pol.fine.fc2.weight, tr.fine.fc2.weight)
    assert make_opponent_policy("net:" + sn["path"], "cpu").fine is not None
    # a plain checkpoint (no fine head) is unaffected
    assert make_opponent_policy("net:checkpoints/distill/frozen/p24c_v321.pt", "cpu").fine is None


@needs_ckpt
def test_eval_batched_with_fine_checkpoint(tmp_path):
    from blackout_env.rl.evalfast import eval_batched
    from blackout_env.rl.grpo import eval_matches
    from blackout_env.rl.ppo import PPOTrainer
    from blackout_env.sim.batch import NetPolicy

    tr = PPOTrainer(_fcfg())
    _randomise(tr, 0.4, 3)
    path = str(tmp_path / "f.pt")
    tr.save(path)
    seeds, hz = [21, 22], 120
    res = eval_batched(path, ["random"], seeds, "cpu", horizon=hz, graphs=False, fp16=False)
    res_obj = eval_batched(tr.net, ["random"], seeds, "cpu", horizon=hz, graphs=False, fp16=False, fine=tr.fine)
    plain = eval_batched(tr.net, ["random"], seeds, "cpu", horizon=hz, graphs=False, fp16=False)
    assert res["random"]["margin_side0"] == res_obj["random"]["margin_side0"]
    assert res["random"]["margin_side1"] == res_obj["random"]["margin_side1"]
    wr, m, se = eval_matches(path, "random", seeds, "cpu", horizon=hz)
    assert m == pytest.approx(res["random"]["margin"], abs=1e-9)
    assert np.isfinite(plain["random"]["margin"])
    # greedy = argmax fine of the argmax coarse bin
    batch, _ = tr.collect()
    pol = NetPolicy.from_checkpoint(path, device="cpu")
    q, off2 = pol.forward(batch["g"], batch["t"], batch["a"])
    _, off0 = NetPolicy(tr.net, device="cpu").forward(batch["g"], batch["t"], batch["a"])
    to = torch.from_numpy
    k = to(q.argmax(-1))
    kc = k.clamp(max=31)
    with torch.no_grad():
        _, _, ft = tr._fine_fwd(tr.own, tr.tap, unpack(to(batch["g"])), to(batch["t"]), to(batch["a"]))
        lg = tr.fine(ft, kc, to(off0).gather(-1, kc[..., None])[..., 0])
        want = tr.fine.centres[lg.argmax(-1)].numpy()
    got = np.take_along_axis(off2, kc.numpy()[..., None], -1)[..., 0]
    ns = (k < 32).numpy()
    assert np.allclose(got[ns], want[ns], atol=1e-6)


@needs_ckpt
def test_graphforward_fine_paths_cpu_eager():
    """GraphForward's fine options (in-graph greedy decode, exposed features) via the eager fallback (CPU): same values as NetPolicy / the tap."""
    from blackout_env.rl.ppo import PPOTrainer
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.graphfwd import GraphForward

    tr = PPOTrainer(_fcfg())
    _randomise(tr, 0.4, 5)
    batch, _ = tr.collect()
    g, t, a = batch["g"], batch["t"], batch["a"]
    gf = GraphForward(tr.own, "cpu", angle=True, autocast=False, tap=tr.tap, fine=tr.fine)
    out = gf(g, t, a).numpy()
    q, off = NetPolicy(tr.net, device="cpu", fine=tr.fine).forward(g, t, a)
    assert np.allclose(out[..., :gf.kq], q, atol=1e-6) and np.allclose(out[..., gf.kq:], off, atol=1e-6)
    gf2 = GraphForward(tr.own, "cpu", angle=True, autocast=False, tap=tr.tap, expose_feat=True)
    out2 = gf2(g, t, a).numpy()
    _, off0 = NetPolicy(tr.net, device="cpu").forward(g, t, a)
    assert np.allclose(out2[..., gf2.kq:], off0, atol=1e-6)         # exposing features leaves the output untouched
    assert gf2.feat.shape == (len(g), 5, tr.tap.hidden)
