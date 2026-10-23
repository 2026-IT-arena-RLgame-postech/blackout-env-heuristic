"""Value-net velocity (Delta-pos) inputs and attention pooling (model.py, velocity.py, vector_encoder.py)."""

import numpy as np
import torch

from blackout_env.value.model import N_VISION_TOKENS, AttnPool, ValueNet, load_value_net, save_value_net
from blackout_env.value.velocity import MAX_GAP, shard_velocity

H = 64


def _inputs(n=3):
    g = torch.randint(0, 16, (n, 24, 24, 13), dtype=torch.uint8)
    return g, torch.rand(n, 4), torch.randn(n, 10, 12) * 0.3


def _vel(n=3):
    v = torch.randn(n, 10, 3)
    v[..., 2] = 1.0
    return v


def test_flags_off_is_old_net_and_state_dict_keys():
    torch.manual_seed(0)
    base = ValueNet(hidden_size=H)
    assert not base.vel and base.attn_pool is None
    assert not any("extra_proj" in k or "attn_pool" in k for k in base.state_dict())
    assert base.trunk.vector_encoder.extra_proj is None


def test_zero_init_vel_equals_base_at_init():
    torch.manual_seed(0)
    base = ValueNet(hidden_size=H).eval()
    vel = ValueNet(hidden_size=H, vel=True).eval()
    missing, unexpected = vel.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("extra_proj" in k for k in missing)
    g, ts, ag = _inputs()
    with torch.no_grad():
        ref = base.forward_u8(g, ts, ag)
        torch.testing.assert_close(vel.forward_u8(g, ts, ag, _vel()), ref)  # extra weights are zero
        torch.testing.assert_close(vel.forward_u8(g, ts, ag), ref)  # no velocity supplied = zeros
    # once the weights are non-zero the velocity changes the output (and only through the unit tokens)
    torch.nn.init.normal_(vel.trunk.vector_encoder.extra_proj.weight, std=0.5)
    with torch.no_grad():
        assert (vel.forward_u8(g, ts, ag, _vel()) - ref).abs().max() > 1e-3
        torch.testing.assert_close(vel.forward_u8(g, ts, ag, torch.zeros(3, 10, 3)), ref)  # zero + invalid flag = no input


def test_vel_param_groups():
    net = ValueNet(hidden_size=H, vel=True, pool="attn")
    trunk = {id(p) for p in net.trunk_parameters()}
    head = {id(p) for p in net.head_parameters()}
    assert not trunk & head
    assert id(net.trunk.vector_encoder.extra_proj.weight) in head
    assert all(id(p) in head for p in net.attn_pool.parameters())
    assert len(trunk) + len(head) == len(list(net.parameters()))


def test_attn_pool_shapes_and_permutation_invariance():
    torch.manual_seed(0)
    pool = AttnPool(H, 4).eval()
    x = torch.randn(2, 48, H)
    out = pool(x)
    assert out.shape == (2, 4 * H)
    perm = torch.randperm(48)
    torch.testing.assert_close(pool(x[:, perm]), out, atol=1e-5, rtol=1e-5)  # token order does not matter
    # but the queries are distinct: the K outputs are not copies of each other
    assert (out[:, :H] - out[:, H:2 * H]).abs().max() > 1e-3


def test_attn_net_forward_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = ValueNet(hidden_size=H, vel=True, pool="attn", n_queries=4).eval()
    g, ts, ag = _inputs()
    with torch.no_grad():
        out = net.forward_u8(g, ts, ag, _vel())
    assert out.shape == (3, net.n_horizons + 4) and torch.isfinite(out).all()
    assert net.head_in.normalized_shape == (6 * H,)
    save_value_net(tmp_path / "a.pt", net)
    again = load_value_net(tmp_path / "a.pt")
    assert again.vel and again.pool == "attn" and again.n_queries == 4
    with torch.no_grad():
        torch.testing.assert_close(again.forward_u8(g, ts, ag, _vel()), out)


def test_old_checkpoint_without_flags_loads(tmp_path):
    torch.manual_seed(0)
    net = ValueNet(hidden_size=H).eval()
    save_value_net(tmp_path / "o.pt", net)
    ck = torch.load(tmp_path / "o.pt", weights_only=True)
    for k in ("vel", "pool", "n_queries"):
        ck.pop(k)
    torch.save(ck, tmp_path / "older.pt")
    loaded = load_value_net(tmp_path / "older.pt")
    assert not loaded.vel and loaded.pool == "mean"
    g, ts, ag = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(loaded.forward_u8(g, ts, ag), net.forward_u8(g, ts, ag))


def test_load_trunk_accepts_vel_net():
    base = ValueNet(hidden_size=H)
    vel = ValueNet(hidden_size=H, vel=True)
    vel.load_trunk({k: v for k, v in base.trunk.state_dict().items()})  # warm start from a stateless-input trunk


def test_shard_velocity():
    t = 8
    agents = np.zeros((t, 10, 12), np.float32)
    agents[:, 0, 0] = 0.02 * np.arange(t)  # unit 0 walks at normal speed along x
    agents[4:, 1, 0] = np.linspace(0, 0.0, t - 4)
    agents[5:, 2, :2] = 0.9  # unit 2 teleports (respawn) at row 5
    ticks = np.arange(t)
    v = shard_velocity(agents, ticks)
    assert v.shape == (t, 10, 3) and v.dtype == np.float32
    assert (v[0] == 0).all()  # first row of a match: zero + invalid
    np.testing.assert_allclose(v[1:, 0, 0], 1.0, atol=1e-5)
    assert (v[1:, 0, 2] == 1).all()
    assert (v[5, 2] == 0).all() and v[4, 2, 2] == 1 and v[6, 2, 2] == 1  # teleport row invalid, neighbours fine
    # a tick gap: velocity is per tick; a gap beyond MAX_GAP or an unknown tick is invalid
    gapped = np.array([0, 2, 4, 6 + MAX_GAP, 6 + MAX_GAP + 1, -1, -1, -1])
    a2 = np.zeros((t, 10, 12), np.float32)
    a2[:, 0, 0] = 0.02 * gapped
    v2 = shard_velocity(a2, gapped)
    np.testing.assert_allclose(v2[1:3, 0, 0], 1.0, atol=1e-5)
    assert v2[3, 0, 2] == 0 and v2[4, 0, 2] == 1 and v2[5, 0, 2] == 0
    assert (shard_velocity(agents[:1], ticks[:1]) == 0).all()
