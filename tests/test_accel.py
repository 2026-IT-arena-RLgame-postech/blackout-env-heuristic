"""train/accel.py helpers and the CUDA training paths (docs/cuda_training_ko.md).

The CPU tests run everywhere.  The ``cuda`` tests (skipped without a GPU) check that
torch.compile of the whole model gives the eager gradients (MPS needs COMPILED_MODULES because of
an inductor embedding-gradient bug; CUDA compiles the whole net), and that fp16 autocast stays
close to fp32.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from blackout_env.train.accel import Dist, Prefetcher, adamw, load_npz, pick_device, to_device, unwrap
from blackout_env.train.distill.loss import angle_offset_loss, set_valued_cross_entropy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_load_npz_mmap_matches_np_load(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"a": rng.random((7, 3)).astype(np.float32), "b": rng.integers(0, 2**16, (5, 4, 4)).astype(np.uint16),
              "c": rng.integers(0, 2**60, (6, 5), dtype=np.uint64), "s": np.array("meta")}
    np.savez(tmp_path / "x.npz", **arrays)
    np.savez_compressed(tmp_path / "z.npz", **arrays)
    for name in ("x.npz", "z.npz"):
        got = load_npz(tmp_path / name)
        assert set(got) == set(arrays)
        for k, v in arrays.items():
            assert got[k].dtype == v.dtype and np.array_equal(np.asarray(got[k]), v), (name, k)
    assert isinstance(load_npz(tmp_path / "x.npz")["a"], np.memmap)


def test_prefetcher_keeps_the_inline_order():
    def source(seed):
        rng = np.random.default_rng(seed)
        return lambda: {"x": rng.integers(0, 1000, 8)}

    inline = source(3)
    expected = [inline()["x"] for _ in range(10)]
    pre = Prefetcher(source(3), torch.device("cpu"), depth=3)
    got = [pre.next()["x"].numpy() for _ in range(10)]
    pre.stop()
    assert all(np.array_equal(a, b) for a, b in zip(expected, got))


def test_dist_single_process_is_a_no_op():
    dist = Dist(torch.device("cpu"))
    assert dist.world == 1 and dist.main and not dist.enabled
    net = torch.nn.Linear(2, 2)
    assert dist.wrap(net) is net
    t = torch.ones(3)
    assert torch.equal(dist.all_reduce_sum(t), torch.ones(3))
    assert unwrap(torch.compile(net)) is net if hasattr(torch, "compile") else True


def test_pick_device_auto():
    d = pick_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert d.type == expected
    assert pick_device("cpu").type == "cpu"


def test_loss_total_override():
    """total = the batch's own weight sum reproduces the default; total = 2x halves the loss."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 5, 32, generator=g)
    masks = torch.rand(6, 5, 32, generator=g) > 0.8
    w = torch.rand(6, 5, generator=g)
    loss, _ = set_valued_cross_entropy(logits, masks, w)
    total = (w * masks.any(-1)).sum()
    loss2, _ = set_valued_cross_entropy(logits, masks, w, total=total)
    assert torch.allclose(loss, loss2)
    loss3, _ = set_valued_cross_entropy(logits, masks, w, total=2 * total)
    assert torch.allclose(loss, 2 * loss3)
    off = torch.rand(6, 5, 32, generator=g) * 0.1
    ang = torch.rand(6, 5, generator=g) * 6.0
    ang[0, 0] = float("nan")
    a1, _ = angle_offset_loss(off, ang, w, arc_bins=1)
    a2, _ = angle_offset_loss(off, ang, w, arc_bins=1, total=(w * torch.isfinite(ang)).sum())
    assert torch.allclose(a1, a2)


def _packed_split(n=64, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 8, (n, 24, 24))
    bat = rng.integers(0, 16, (n, 24, 24)) * (rng.random((n, 24, 24)) < 0.1)
    item = rng.integers(0, 5, (n, 24, 24)) * (rng.random((n, 24, 24)) < 0.05)
    agents = rng.uniform(-1, 1, (n, 10, 12)).astype(np.float32)
    agents[..., 9:12] = np.eye(3)[rng.integers(0, 3, (n, 10))]
    agents[..., 2] = np.where(np.arange(10) < 5, 1.0, -1.0)
    labels = np.zeros((n, 5), np.uint64)
    for b in range(3):
        labels |= (rng.random((n, 5)) < 0.7).astype(np.uint64) << rng.integers(0, 32, (n, 5)).astype(np.uint64)
    labels[:4] = 0  # unlabelled rows
    angles = rng.uniform(-np.pi, np.pi, (n, 5)).astype(np.float32)
    angles[rng.random((n, 5)) < 0.2] = np.nan
    return {"map": (base | (bat << 3) | (item << 7)).astype(np.uint16), "team_state": rng.random((n, 4)).astype(np.float32),
            "agent_states": agents, "labels": labels, "angles": angles,
            "tick": rng.integers(-1, 2000, n).astype(np.int32),
            "p11_q": rng.normal(size=(n, 5, 32)).astype(np.float16),
            "p11_off": (rng.normal(size=(n, 5, 32)) * 0.05).astype(np.float16)}


def test_batch_source_matches_the_original_sampling():
    """BatchSource draws the rows the pre-CUDA loop drew (np.sort(rng.choice(...)) on the
    concatenated train + extra arrays)."""
    import rollout_net_train as r

    train, extra = _packed_split(64, 0), _packed_split(40, 1)
    pool, epool = r.labelled_rows(train["labels"]), r.labelled_rows(extra["labels"])
    assert np.array_equal(pool, np.nonzero(r.unpack_labels(train["labels"]).any(-1).any(-1))[0])
    keys = list(train)
    src = r.BatchSource(train, extra, pool, epool, 16, 0.5, 7, keys)
    rng = np.random.default_rng(7)
    cat = {k: np.concatenate([train[k], extra[k]]) for k in keys}
    for _ in range(3):
        got = src()
        idx = np.sort(np.concatenate([rng.choice(pool, 8), rng.choice(len(train["tick"]) + epool, 8)]))
        for k in keys:
            want = cat[k][idx]
            v = got[k].view(want.dtype) if k in ("map", "labels") else got[k]
            assert np.array_equal(v, want, equal_nan=True), k


def _train_args(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(opening_weight=1.5, angle_weight=1.0, p11_kl=0.5, channels_last=False, **kw)


def test_train_step_matches_the_original_loss():
    """TrainStep (device-side decode + losses) = the loss the pre-CUDA loop computed."""
    import math

    import rollout_net_train as r
    from blackout_env.sim.rollout_net import RolloutNet

    torch.manual_seed(0)
    data = _packed_split(32, 2)
    net = RolloutNet(arch="attn").train()
    args = _train_args()
    step = r.train_step_module(torch)(net, args, "none", torch.device("cpu"))
    idx = np.arange(32)
    b = to_device({k: v.view(np.int16) if k == "map" else v.view(np.int64) if k == "labels" else v
                   for k, v in data.items()}, torch.device("cpu"))
    loss, stats = step(b["map"], b["team_state"], b["agent_states"], b["labels"], b["angles"], b["tick"], b["p11_q"], b["p11_off"])
    # the original loop
    g, ts, ag = r.batch_tensors(data, idx, "cpu", torch)
    labels = torch.from_numpy(r.unpack_labels(data["labels"][idx]))
    ang = torch.from_numpy(data["angles"][idx])
    tick = data["tick"][idx]
    w = torch.from_numpy(np.where((tick >= 0) & (tick < r.OPENING_TICKS), 1.5, 1.0).astype(np.float32))[:, None].expand(-1, 5)
    logits, off = net(g, ts, ag)
    ce, _ = set_valued_cross_entropy(logits, labels, w)
    al, _ = angle_offset_loss(off, ang, w, arc_bins=net.config["angle_arc_bins"])
    ref = ce + al
    pq, po = torch.from_numpy(data["p11_q"].astype(np.float32)), torch.from_numpy(data["p11_off"].astype(np.float32))
    kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(pq, -1), log_target=True,
                                    reduction="none").sum(-1).mean()
    j = pq.argmax(-1, keepdim=True)
    oreg = (((off.gather(-1, j) - po.gather(-1, j)) / (math.pi / 32)) ** 2).mean()
    ref = ref + 0.5 * (kl + oreg)
    assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-6), (float(loss), float(ref))


def _grads(module, loss_fn):
    module.zero_grad(set_to_none=True)
    loss_fn().backward()
    return {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}


def _assert_close_grads(a, b, rtol):
    assert a.keys() == b.keys()
    for k in a:  # relative to the gradient's norm: cuDNN's conv weight-gradient is itself
        # nondeterministic at ~1e-3 (eager vs eager, and eager vs an fp64 reference alike)
        err = (a[k] - b[k]).norm() / a[k].norm().clamp_min(1e-12)
        assert err < rtol, f"{k}: rel err {float(err):.2e}"


@cuda
def test_compile_rollout_net_gradients_match_eager_cuda():
    import rollout_net_train as r
    from blackout_env.sim.rollout_net import RolloutNet

    device = torch.device("cuda")
    torch.manual_seed(0)
    data = _packed_split(256, 3)
    b = to_device({k: v.view(np.int16) if k == "map" else v.view(np.int64) if k == "labels" else v
                   for k, v in data.items()}, device)
    net = RolloutNet(arch="attn").train().to(device)
    step = r.train_step_module(torch)(net, _train_args(), "none", device)
    call = lambda f: f(b["map"], b["team_state"], b["agent_states"], b["labels"], b["angles"], b["tick"], b["p11_q"], b["p11_off"])[0]
    eager = _grads(net, lambda: call(step))
    compiled = _grads(net, lambda: call(torch.compile(step)))
    _assert_close_grads(eager, compiled, 1e-3)


@cuda
@pytest.mark.parametrize("head", ["logits", "iqn"])
def test_compile_whole_mymodel_gradients_match_eager_cuda(head):
    """The MPS inductor bug (embedding gradients x batch size) must not appear on CUDA: the whole
    MyModel (embeddings included) compiled gives the eager gradients."""
    from blackout_env.model.my_model import MyModel

    device = torch.device("cuda")
    torch.manual_seed(0)
    kw = dict(n_actions=32, policy_head=head, angle_head=head == "logits", angle_arc_bins=1 if head == "logits" else 0)
    net = MyModel(**kw).to(device).train()
    graphic, team, agents = _mymodel_batch(64, device)
    tau = torch.rand(64, 8, device=device)

    def loss(model):
        out = model(graphic, team, agents, q_units=5, tau=tau, **({"angle": True} if head == "logits" else {}))
        return out[0].float().square().mean() + (out[5].float().square().mean() if head == "logits" else 0) \
            + out[3].float().square().mean()

    eager = _grads(net, lambda: loss(net))
    compiled_net = torch.compile(net)
    compiled = _grads(net, lambda: loss(compiled_net))
    _assert_close_grads(eager, compiled, 5e-3)
    for k in ("token_type_emb.weight", "spr_cls_emb.weight", "vec_slot_emb.weight"):  # the MPS bug: x batch size
        assert torch.allclose(compiled[k].norm(), eager[k].norm(), rtol=1e-3), k


def _mymodel_batch(n, device):
    g = torch.Generator().manual_seed(1)
    graphic = torch.zeros(n, 13, 24, 24)
    base = torch.randint(0, 8, (n, 24, 24), generator=g)
    graphic[:, :8] = torch.nn.functional.one_hot(base, 8).permute(0, 3, 1, 2).float()
    graphic[:, 8] = (torch.rand(n, 24, 24, generator=g) < 0.1).float() * 0.4
    team = torch.rand(n, 4, generator=g)
    agents = torch.rand(n, 10, 12, generator=g) * 2 - 1
    agents[..., 2] = torch.where(torch.arange(10) < 5, 1.0, -1.0)
    agents[..., 9:12] = torch.nn.functional.one_hot(torch.randint(0, 3, (n, 10), generator=g), 3).float()
    return graphic.to(device), team.to(device), agents.to(device)


@cuda
def test_fp16_autocast_close_to_fp32_cuda():
    from blackout_env.model.my_model import MyModel

    device = torch.device("cuda")
    torch.manual_seed(0)
    net = MyModel(n_actions=32, policy_head="logits", angle_head=True, angle_arc_bins=1).to(device).eval()
    graphic, team, agents = _mymodel_batch(64, device)
    with torch.no_grad():
        ref = net(graphic, team, agents, q_units=5, angle=True)
        with torch.autocast("cuda", dtype=torch.float16):
            half = net(graphic, team, agents, q_units=5, angle=True)
    err = (ref[0] - half[0].float()).abs().max() / ref[0].abs().max()
    assert err < 2e-2, float(err)


@cuda
def test_fused_adamw_matches_foreach_cuda():
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.nn.Linear(16, 8).to(device)
    b = torch.nn.Linear(16, 8).to(device)
    b.load_state_dict(a.state_dict())
    oa = adamw(a.parameters(), device, lr=1e-2, weight_decay=1e-2)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=1e-2, foreach=True)
    x = torch.randn(32, 16, device=device)
    for _ in range(5):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()
    assert torch.allclose(a.weight, b.weight, atol=1e-6)
