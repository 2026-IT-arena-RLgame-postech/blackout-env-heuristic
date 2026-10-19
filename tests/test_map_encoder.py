"""A/B of the map encoder (docs/rl_finetune_plan_ko.md): ViT-style patch embeddings give the same 36 tokens
as the CNN, and the default CNN net is untouched (same keys, same outputs as before the option existed)."""

from pathlib import Path

import pytest
import torch

from blackout_env.model.modules import GraphicEncoder
from blackout_env.model.my_model import MyModel
from blackout_env.train.distill.run import load_weights

BASELINE = Path(__file__).parent / "data" / "cnn_forward_baseline.pt"


def _frames(n=4):
    g = torch.Generator().manual_seed(7)
    return (torch.rand(n, 13, 24, 24, generator=g).round(), torch.rand(n, 4, generator=g),
            torch.rand(n, 10, 12, generator=g) * 2 - 1)


@pytest.mark.parametrize("kind", ["cnn", "patch4", "patch6"])
def test_every_encoder_gives_36_tokens(kind):
    enc = GraphicEncoder(256, map_encoder=kind).eval()
    tokens = enc(torch.rand(3, 19, 24, 24))
    assert tokens.shape == (3, 36, 256)


def test_patch_geometry():
    assert GraphicEncoder(256, map_encoder="patch4").patch_embed.weight.shape == (256, 19, 4, 4)
    assert GraphicEncoder(256, map_encoder="patch6").patch_embed.weight.shape == (256, 19, 6, 6)
    # patch4 tokens are non-overlapping: a perturbation of one 4x4 block changes only its own token
    enc = GraphicEncoder(256, map_encoder="patch4").eval()
    x = torch.rand(1, 19, 24, 24)
    y = x.clone()
    y[:, :, 8:12, 4:8] += 1.0  # block (row 2, col 1) -> token 2 * 6 + 1
    changed = ((enc(x) - enc(y)).abs().amax(-1) > 0)[0].nonzero().flatten().tolist()
    assert changed == [13]
    # patch6 / stride 4 / pad 1: that cell block reaches the neighbours (2-cell overlap)
    enc6 = GraphicEncoder(256, map_encoder="patch6").eval()
    assert ((enc6(x) - enc6(y)).abs().amax(-1) > 0).sum() > 1


@pytest.mark.parametrize("kind", ["patch4", "patch6"])
def test_myset_forward_state_roundtrip_and_grad(kind):
    torch.manual_seed(0)
    net = MyModel(n_actions=33, policy_head="logits", angle_head=True, angle_arc_bins=1, map_encoder=kind)
    assert not any(k.startswith("graphic_encoder.pre_conv") for k in net.state_dict())
    out = net(*_frames(), q_units=5, angle=True)
    assert out[0].shape == (4, 5, 33)
    out[0].square().mean().backward()
    assert net.graphic_encoder.patch_embed.weight.grad.abs().sum() > 0
    again = MyModel.from_state_dict(net.state_dict())  # the encoder is read back from the keys
    assert again.graphic_encoder.map_encoder == kind
    net.eval(); again.eval()
    with torch.no_grad():
        assert torch.equal(net(*_frames(), q_units=5)[0], again(*_frames(), q_units=5)[0])


def test_patch_encoder_rejects_map_xattn():
    with pytest.raises(ValueError):
        MyModel(map_xattn="f3", map_encoder="patch4")


def test_default_path_is_bit_identical_to_before():
    base = torch.load(BASELINE)
    torch.manual_seed(1234)
    net = MyModel(n_actions=33, policy_head="logits", angle_head=True, angle_arc_bins=1).eval()
    assert list(net.state_dict().keys()) == base["keys"]
    assert sum(p.numel() for p in net.parameters()) == base["nparams"]
    with torch.no_grad():
        out = [o for o in net(*_frames(), q_units=5, angle=True) if torch.is_tensor(o)]
    assert len(out) == len(base["out"])
    for a, b in zip(out, base["out"]):
        assert torch.equal(a, b)
    assert MyModel.config_from_state_dict(net.state_dict()).get("map_encoder") is None
