"""HL-Gauss distributional final head of the value net (model.py): target, decode, loss, checkpoint round trip."""

import numpy as np
import pytest
import torch

from blackout_env.value.model import (DEFAULT_BIN_RANGE, ValueNet, bin_centers, decode_final_dist, hlgauss_target,
                                      load_value_net, save_value_net, value_loss)

N, R = 112, DEFAULT_BIN_RANGE


def _inputs(n=3):
    return torch.randint(0, 16, (n, 24, 24, 13), dtype=torch.uint8), torch.rand(n, 4), torch.randn(n, 10, 12)


def test_hlgauss_target_is_pmf_with_mean_x():
    x = torch.tensor([-9.0, -0.3, 0.0, 2.2, 6.1])
    t = hlgauss_target(x, N, R)
    assert t.shape == (5, N)
    torch.testing.assert_close(t.sum(1), torch.ones(5))
    mean = (t * bin_centers(N, R)).sum(1)
    torch.testing.assert_close(mean, x, atol=0.02, rtol=0)
    # out-of-range values clip to the edge bins instead of producing NaN
    edge = hlgauss_target(torch.tensor([50.0, -50.0]), N, R)
    assert torch.isfinite(edge).all() and edge[0, -1] > 0.4 and edge[1, 0] > 0.4


def test_default_net_unchanged_and_old_checkpoints_load(tmp_path):
    torch.manual_seed(0)
    net = ValueNet(hidden_size=64).eval()
    assert net.n_bins == 0 and net.bin_out is None
    save_value_net(tmp_path / "old.pt", net)
    ck = torch.load(tmp_path / "old.pt", weights_only=True)
    for k in ("n_bins", "bin_range", "sigma_ratio"):  # simulate a checkpoint written before the head existed
        ck.pop(k)
    torch.save(ck, tmp_path / "older.pt")
    g, ts, ag = _inputs()
    loaded = load_value_net(tmp_path / "older.pt")
    assert loaded.n_bins == 0
    with torch.no_grad():
        torch.testing.assert_close(loaded.forward_u8(g, ts, ag), net.forward_u8(g, ts, ag))


def test_hlgauss_net_forward_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = ValueNet(hidden_size=64, n_bins=N).eval()
    g, ts, ag = _inputs()
    with torch.no_grad():
        out, logits = net.forward_full_u8(g, ts, ag)
    h = net.n_horizons
    assert logits.shape == (3, N) and out.shape == (3, h + 4)
    mean = (torch.softmax(logits, -1) * bin_centers(N, R)).sum(1)
    torch.testing.assert_close(out[:, h], mean)  # scalar column = pmf mean: every consumer reads it unchanged
    save_value_net(tmp_path / "d.pt", net)
    again = load_value_net(tmp_path / "d.pt")
    assert again.n_bins == N
    with torch.no_grad():
        out2, logits2 = again.forward_full_u8(g, ts, ag)
    torch.testing.assert_close(logits2, logits)
    torch.testing.assert_close(out2, out)


def test_value_loss_dist_vs_reg_and_gradients():
    torch.manual_seed(1)
    b, h = 6, 4
    lead = torch.rand(b) - 0.5
    t = {"delta": torch.randn(b, h) * 0.1, "delta_mask": torch.ones(b, h, dtype=torch.bool),
         "final": lead + torch.randn(b), "result": torch.randint(-1, 2, (b,))}
    out = torch.randn(b, h + 4, requires_grad=True)
    logits = torch.randn(b, N, requires_grad=True)
    loss, stats = value_loss(out, t, lead, final_logits=logits)
    loss.backward()
    assert logits.grad.abs().sum() > 0 and out.grad[:, h].abs().sum() == 0  # final column unused by the dist loss
    assert float(stats["final"]) > 1.0  # a cross-entropy over bins, not a Huber
    out.grad = None
    loss_reg, st_reg = value_loss(out, t, lead)  # default behaviour: Huber beta 1
    assert float(st_reg["final"]) == pytest.approx(float(torch.nn.functional.smooth_l1_loss(
        out[:, h].detach(), (t["final"] - lead) * 10.0, beta=1.0)), rel=1e-5)


def test_constant_fit_recovers_bimodal_distribution_and_jensen_gap():
    # remaining change +8 or -8 (+-80 points) with equal odds from lead 0: the pmf has two modes and mean ~0
    logits = torch.zeros(1, N, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    x = torch.tensor([8.0, -8.0] * 50)
    tgt = hlgauss_target(x, N, R)
    for _ in range(800):
        loss = -(tgt * torch.log_softmax(logits.expand(len(x), N), -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    d = decode_final_dist(logits.detach(), torch.zeros(1), N, R, tanh_scale=30.0)
    pmf = d["pmf"][0]
    c = bin_centers(N, R).numpy() * 10
    assert abs(d["mean"][0]) < 0.01
    assert pmf[np.abs(c - 80) < 6].sum() == pytest.approx(0.5, abs=0.03) and pmf[np.abs(c + 80) < 6].sum() == pytest.approx(0.5, abs=0.03)
    assert abs(d["tanh_of_mean"][0]) < 0.01
    assert d["tanh_mean"][0] == pytest.approx(0.0, abs=0.02)  # symmetric: both readings agree
    # a skewed 25/75 mix: E[tanh] < tanh(E) (concave side) is the Jensen direction for a positive mean
    w = torch.tensor([0.9] * 1)
    pm = torch.zeros(1, N)
    pm[0, int(np.argmin(np.abs(c - 90)))] = 0.8
    pm[0, int(np.argmin(np.abs(c + 80)))] = 0.2
    skew = decode_final_dist(torch.log(pm.clamp_min(1e-9)), torch.zeros(1), N, R)
    assert skew["mean"][0] > 0 and skew["tanh_mean"][0] < skew["tanh_of_mean"][0]


def test_decode_final_dist_shifts_by_lead():
    logits = torch.randn(2, N)
    lead = torch.tensor([0.30, -0.2])
    d0 = decode_final_dist(logits, torch.zeros(2), N, R)
    d1 = decode_final_dist(logits, lead, N, R)
    np.testing.assert_allclose(d1["mean"] - d0["mean"], lead.numpy(), atol=1e-5)
    np.testing.assert_allclose(d1["pmf"], d0["pmf"])
