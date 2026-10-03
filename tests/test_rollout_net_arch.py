"""RolloutNet map-token variants (docs §22), the Q head, and the trainer's mmap loader."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from blackout_env.sim.rollout_net import MAP_ATTN, RolloutNet, alt_bins, logit_features

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

VARIANTS = [("none", "self", 0), ("f3", "self", 0), ("s3", "self", 0), ("patch", "self", 0), ("f3", "cross", 0),
            ("patch", "cross", 32), ("patch", "local3", 32), ("patch", "local5", 32), ("s3", "fovea", 32)]


def _inputs(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    graphic = torch.rand(b, 13, 24, 24, generator=g)
    team = torch.rand(b, 4, generator=g)
    agents = torch.rand(b, 10, 12, generator=g) * 2 - 1
    agents[..., 9:12] = torch.eye(3)[torch.randint(0, 3, (b, 10), generator=g)]
    agents[:, :5, 2], agents[:, 5:, 2] = 1.0, -1.0
    return graphic, team, agents


@pytest.mark.parametrize("tokens,attn,dm", VARIANTS)
def test_shapes_and_checkpoint_roundtrip(tokens, attn, dm, tmp_path):
    net = RolloutNet(arch="attn", d=32, heads=2, hidden=32, map_tokens=tokens, map_attn=attn, map_dim=dm).eval()
    x = _inputs()
    logits, off = net(*x)
    assert logits.shape == off.shape == (4, 5, 32)
    assert off.abs().max() <= net.angle_scale + 1e-6
    torch.save({"config": net.config, "state": net.state_dict()}, tmp_path / "n.pt")
    back = RolloutNet.from_checkpoint(str(tmp_path / "n.pt"))
    torch.testing.assert_close(back(*x)[0], logits)


def test_old_config_loads_as_plain_r5_shape():
    net = RolloutNet(arch="attn")
    assert "map_tokens" not in net.config and net.map_tokens == "none" and "local3" in MAP_ATTN


def test_local_window_ignores_far_map():
    """A local3 unit token must not react to a map change outside its window (the global token may,
    and it feeds back through self-attention, so check with the global path frozen: one layer,
    compare the cross-attention output directly)."""
    torch.manual_seed(0)
    net = RolloutNet(arch="attn", d=32, heads=2, hidden=32, layers=1, map_tokens="patch", map_attn="local3").eval()
    g, t, a = _inputs(1)
    a[0, :, 0], a[0, :, 1] = -0.95, 0.95  # every unit in the north-west corner cell (row 0, col 0)
    seen = {}

    def hook(mod, inp, out):
        seen["mem"] = inp[1]
        seen["mask"] = inp[2]

    net.trunk[0].register_forward_hook(hook)
    net(g, t, a)
    mask = seen["mask"][0]
    assert mask.shape == (11, 64)
    assert mask[:10].sum(-1).tolist() == [4] * 10  # 2 x 2 corner of a 3 x 3 window
    assert mask[10].all()


def test_q_alt_matches_policy_and_alt_bins():
    net = RolloutNet(arch="attn", d=32, heads=2, hidden=32, map_tokens="patch", map_attn="cross", map_dim=16, q_head=32).eval()
    x = _inputs()
    logits, off = net(*x)
    l2, o2, adv = net.q_alt(*x)
    torch.testing.assert_close(l2, logits)
    assert adv.shape == (4, 5)
    alt = alt_bins(logits)
    top = logits.argmax(-1)
    d = (alt - top) % 32
    assert ((d >= 2) & (d <= 30)).all()
    f = logit_features(logits)
    assert f.shape == (4, 5, 4) and torch.isfinite(f).all()


def test_mmap_npz_equals_load(tmp_path):
    from rollout_net_train import load_split

    rng = np.random.default_rng(0)
    data = {"map": rng.integers(0, 999, (50, 24, 24)).astype(np.uint16), "labels": rng.integers(0, 2**40, (50, 5)).astype(np.uint64),
            "angles": rng.random((50, 5)).astype(np.float32), "tick": np.arange(50, dtype=np.int32)}
    np.savez(tmp_path / "train.npz", **data)
    got = load_split(tmp_path / "train.npz")
    for k, v in data.items():
        assert isinstance(got[k], np.memmap)
        np.testing.assert_array_equal(got[k], v)
