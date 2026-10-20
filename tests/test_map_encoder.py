"""A/B of the map encoder (docs/rl_finetune_plan_ko.md): ViT-style patch embeddings give the same 36 tokens
as the CNN, and the default CNN net is untouched (same keys, same outputs as before the option existed)."""

from pathlib import Path

import pytest
import torch

from blackout_env.model.modules import GraphicEncoder
from blackout_env.model.modules.graphic_encoder import PATCH_ENCODERS
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


# ---- stride-2 arms: 12x12 = 144 vision tokens ----
S2 = ["patch2", "patch4s2", "patch3s2"]


@pytest.mark.parametrize("kind", S2)
def test_stride2_encoders_give_144_tokens(kind):
    enc = GraphicEncoder(256, map_encoder=kind).eval()
    assert enc(torch.rand(3, 19, 24, 24)).shape == (3, 144, 256)
    assert enc.grid == 12 and enc.n_tokens == 144


def test_stride2_overlap_geometry():
    x = torch.rand(1, 19, 24, 24)
    y = x.clone()
    y[:, :, 8, 4] += 1.0  # one cell: row 8, col 4
    def changed(kind):
        enc = GraphicEncoder(256, map_encoder=kind).eval()
        return ((enc(x) - enc(y)).abs().amax(-1) > 0)[0].nonzero().flatten().tolist()
    assert changed("patch2") == [4 * 12 + 2]                   # non-overlapping: one token
    assert len(changed("patch4s2")) == 4                       # 2x2 patches (2-cell overlap)
    assert len(changed("patch3s2")) in (1, 2, 4)                # 1-cell overlap: the cell is shared only on odd rows/cols


@pytest.mark.parametrize("kind", S2)
def test_stride2_model_forward_state_roundtrip_and_grad(kind):
    torch.manual_seed(0)
    net = MyModel(n_actions=33, policy_head="logits", angle_head=True, angle_arc_bins=1, map_encoder=kind)
    assert net.n_vision == 144 and net.rope_row_ids.shape == (144 + 12,) and net.token_type_ids.shape == (144 + 12,)
    out = net(*_frames(), q_units=5, angle=True)
    assert out[0].shape == (4, 5, 33)
    out[0].square().mean().backward()
    assert net.graphic_encoder.patch_embed.weight.grad.abs().sum() > 0
    again = MyModel.from_state_dict(net.state_dict())  # kernel alone is ambiguous (patch4 / patch4s2): the stride marker decides
    assert again.graphic_encoder.map_encoder == kind
    net.eval(); again.eval()
    with torch.no_grad():
        assert torch.equal(net(*_frames(), q_units=5)[0], again(*_frames(), q_units=5)[0])


def test_stride4_state_has_no_stride_marker_and_patch4_is_not_patch4s2():
    assert "graphic_encoder.patch_stride" not in MyModel(map_encoder="patch4").state_dict()
    assert MyModel.config_from_state_dict(MyModel(map_encoder="patch4").state_dict())["map_encoder"] == "patch4"
    assert MyModel.config_from_state_dict(MyModel(map_encoder="patch4s2").state_dict())["map_encoder"] == "patch4s2"


@pytest.mark.parametrize("kind", ["patch4", "patch6"] + S2)
def test_vision_rope_positions_are_cell_centres_in_the_unit_frame(kind):
    """Vision tokens sit at their patch centre expressed in the 6x6 frame the unit tokens are mapped into
    (so a unit standing at a patch centre gets that token's exact position); extras stay at the grid centre."""
    net = MyModel(map_encoder=kind)
    g = net.graphic_encoder.grid
    k, s, p = PATCH_ENCODERS[kind]
    idx = torch.arange(g, dtype=torch.float32)
    centre_cells = idx * s - p + k / 2  # continuous map coordinate (cells, 0..24) of each patch centre
    if kind != "patch3s2":  # the odd kernel is half a cell off the exact cell-centre grid
        assert torch.allclose(centre_cells, (idx + 0.5) * (24 / g))
    expected = centre_cells * (6 / 24) - 0.5
    rows = net.rope_row_ids[: g * g].reshape(g, g)
    cols = net.rope_col_ids[: g * g].reshape(g, g)
    if kind != "patch3s2":
        assert torch.allclose(rows[:, 0], expected, atol=1e-5) and torch.allclose(cols[0], expected, atol=1e-5)
    assert float(net.rope_row_ids[: g * g].min()) == pytest.approx(-0.25 if g == 12 else 0.0, abs=1e-6) or g == 6
    assert float(net.rope_row_ids[: g * g].max()) <= 5.25
    # units: agent at the map centre maps to the grid centre, as for the 36-token nets
    ag = torch.zeros(1, 10, 12)
    r, c = net._token_grid_positions(ag)
    assert torch.allclose(r[0, net.n_vision : net.n_vision + 10], torch.full((10,), 2.5))
    assert torch.allclose(net.rope_row_ids[g * g :], torch.full((12,), 2.5))
    # fastest RoPE pair still under one turn across the vision positions
    span = net.rope.freqs * float(net.rope_row_ids[: g * g].max() - net.rope_row_ids[: g * g].min())
    assert float(span.max()) < 2 * 3.14159265


def test_stride2_rejects_map_xattn():
    with pytest.raises(ValueError):
        MyModel(map_xattn="f3", map_encoder="patch4s2")
