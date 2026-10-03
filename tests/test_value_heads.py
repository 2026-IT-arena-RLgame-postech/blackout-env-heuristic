"""Final-head variants of the S1.6 value net (docs §27): MSE / quantile losses, quantile outputs,
antisymmetric reading, checkpoint round trip."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from value_branch_train import antisym_outputs, final_head_loss  # noqa: E402

from blackout_env.value.model import ValueNet, load_value_net, save_value_net  # noqa: E402


def _fit_constant(kind, target, steps=3000, init=7.7, **kw):
    """Minimise the loss over a constant prediction (one scalar, or Q quantiles)."""
    q = kw.pop("n_quantiles", 0)
    p = torch.full((q,) if q else (1,), init, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.05)
    t = torch.as_tensor(target, dtype=torch.float32)
    for _ in range(steps):
        if q:
            loss = final_head_loss(None, t, "quantile", quantiles=p[None].expand(len(t), q))
        else:
            loss = final_head_loss(p.expand(len(t)), t, kind, **kw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p.detach().numpy()


def test_bimodal_minimisers():
    # a symmetric opening: +80 or -80 with equal odds (tenths of points: +-8); start at the biased +7.7
    tgt = np.array([8.0, -8.0] * 50)
    assert abs(_fit_constant("mse", tgt)[0]) < 0.05  # MSE -> the mean
    l1 = _fit_constant("huber", tgt, beta=1.0)[0]
    assert l1 > 5  # near-L1 Huber: any point between the modes is a minimiser; it barely moves
    q = _fit_constant("quantile", tgt, n_quantiles=8, steps=4000)
    assert abs(q.mean()) < 0.3 and q.min() < -6 and q.max() > 6


def test_quantile_mean_on_skewed_target():
    rng = np.random.default_rng(0)
    tgt = np.concatenate([rng.normal(-2, 0.3, 300), rng.normal(6, 0.3, 100)])  # mean 0, median ~-1.9
    q = _fit_constant("quantile", tgt, n_quantiles=16, steps=4000, init=0.0)
    assert q.mean() == pytest.approx(tgt.mean(), abs=0.35)


def test_antisym_outputs():
    rng = np.random.default_rng(1)
    a = torch.as_tensor(rng.normal(size=(5, 8)), dtype=torch.float32)
    b = torch.as_tensor(rng.normal(size=(5, 8)), dtype=torch.float32)
    ab, ba = antisym_outputs(a, b), antisym_outputs(b, a)
    torch.testing.assert_close(ab[:, :5], -ba[:, :5])  # regressions antisymmetric
    torch.testing.assert_close(ab[:, 5:], torch.flip(ba[:, 5:], dims=[1]))  # result logits mirrored
    same = antisym_outputs(a, a)
    assert torch.all(same[:, :5] == 0)


def _inputs(n=3):
    g = torch.randint(0, 16, (n, 24, 24, 13), dtype=torch.uint8)
    ts = torch.rand(n, 4)
    ag = torch.randn(n, 10, 12)
    return g, ts, ag


def test_quantile_net_forward_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    base = ValueNet(hidden_size=64).eval()
    save_value_net(tmp_path / "base.pt", base)
    net = load_value_net(tmp_path / "base.pt", n_quantiles=8)
    g, ts, ag = _inputs()
    with torch.no_grad():
        out, q = net.forward_full_u8(g, ts, ag)
        ref = base.forward_u8(g, ts, ag)
    h = net.n_horizons
    assert q.shape == (3, 8)
    # quantiles start as copies of the scalar final output: the whole output equals the base net's
    torch.testing.assert_close(q, ref[:, h:h + 1].expand(3, 8), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        net.q_out.bias.add_(torch.linspace(-1, 1, 8))
        out, q = net.forward_full_u8(g, ts, ag)
    torch.testing.assert_close(out[:, h], q.mean(1))
    torch.testing.assert_close(net.forward_u8(g, ts, ag), out)
    save_value_net(tmp_path / "q.pt", net)
    back = load_value_net(tmp_path / "q.pt")
    assert back.n_quantiles == 8 and not back.antisym
    with torch.no_grad():
        torch.testing.assert_close(back.forward_u8(g, ts, ag), out)
    flag = load_value_net(tmp_path / "base.pt", antisym=True)
    save_value_net(tmp_path / "a.pt", flag)
    assert load_value_net(tmp_path / "a.pt").antisym
