"""§25: MyModel's optional map-token cross-attention path -- off = the old net (keys, outputs),
zero-initialised warm start = the old net exactly, the config read back from a state, gradients
reach the path, the distill learner loads and compiles it, and inference loads it from a checkpoint."""

from pathlib import Path

import pytest
import torch

from blackout_env.model.my_model import MAP_XATTN_PREFIX, MyModel
from blackout_env.train.distill.run import load_weights

K = 32
P11 = Path("/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p11_v22.pt")


def _frames(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 13, 24, 24, generator=g).round(), torch.rand(n, 4, generator=g),
            torch.rand(n, 10, 12, generator=g) * 2 - 1)


def _p11_like(seed=0) -> MyModel:
    torch.manual_seed(seed)
    net = MyModel(n_actions=K, policy_head="logits", angle_head=True, angle_arc_bins=1).eval()
    with torch.no_grad():
        net.q_head.value_head[1].weight.mul_(20.0)
        net.q_head.angle_out.bias.uniform_(-1, 1)  # non-zero offsets, so they are compared too
    return net


def _out(net, frames):
    with torch.no_grad():
        out = net(*frames, q_units=5, angle=True)
    return out[0], out[5]


def test_off_by_default_no_keys_same_forward():
    net = _p11_like()
    assert net.map_xattn is None and not any(k.startswith(MAP_XATTN_PREFIX) for k in net.state_dict())
    again = MyModel.from_state_dict(net.state_dict()).eval()
    assert again.map_xattn is None
    for a, b in zip(_out(net, _frames()), _out(again, _frames())):
        assert torch.equal(a, b)


@pytest.mark.parametrize("source,layers", [("f3", 2), ("f2", 2), ("f3", 4), ("f2", 1)])
def test_zero_init_warm_start_is_exactly_the_old_net(source, layers):
    old = _p11_like()
    new = MyModel.from_state_dict(old.state_dict(), map_xattn=source, map_xattn_layers=layers).eval()
    assert new.map_xattn is not None and new.map_xattn.source == source and len(new.map_xattn.blocks) == layers
    for a, b in zip(_out(old, _frames()), _out(new, _frames())):
        assert torch.equal(a, b)
    # the distill learner's loader does the same
    net = MyModel(n_actions=K, policy_head="logits", angle_head=True, angle_arc_bins=1, map_xattn=source,
                  map_xattn_layers=layers).eval()
    load_weights(net, old.state_dict())
    for a, b in zip(_out(old, _frames()), _out(net, _frames())):
        assert torch.equal(a, b)


@pytest.mark.parametrize("source,layers", [("f3", 2), ("f2", 3)])
def test_config_round_trips_through_the_state(source, layers):
    net = MyModel(n_actions=K, policy_head="logits", angle_head=True, map_xattn=source, map_xattn_layers=layers)
    with torch.no_grad():
        for block in net.map_xattn.blocks:
            block.o_proj.weight.normal_(0, 0.05)  # trained: the path changes outputs
    net.eval()
    config = MyModel.config_from_state_dict(net.state_dict())
    assert config["map_xattn"] == source and config["map_xattn_layers"] == layers
    again = MyModel.from_state_dict(net.state_dict()).eval()
    for a, b in zip(_out(net, _frames()), _out(again, _frames())):
        assert torch.equal(a, b)
    plain = MyModel.from_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith(MAP_XATTN_PREFIX)})
    assert not torch.allclose(_out(net, _frames())[0], _out(plain.eval(), _frames())[0])
    with pytest.raises(ValueError):
        MyModel.from_state_dict(net.state_dict(), map_xattn=False)
    with pytest.raises(RuntimeError):  # a path the net does not have
        load_weights(MyModel(n_actions=K, policy_head="logits", angle_head=True), net.state_dict())


def test_gradients_reach_the_path_and_it_learns():
    net = MyModel.from_state_dict(_p11_like().state_dict(), map_xattn="f3").train()
    frames = _frames(8)
    logits = net(*frames, q_units=5)[0]
    logits.square().mean().backward()
    o = net.map_xattn.blocks[0].o_proj.weight.grad
    assert o is not None and o.abs().sum() > 0  # zero output, non-zero gradient
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    opt.step()
    opt.zero_grad()
    net(*frames, q_units=5)[0].square().mean().backward()
    assert net.map_xattn.blocks[0].q_proj.weight.grad.abs().sum() > 0  # q/k/v learn once o_proj moved


def test_unit_query_positions_are_their_cells():
    net = MyModel(n_actions=K, policy_head="logits", map_xattn="f3")
    mx = net.map_xattn
    agent = torch.zeros(1, 10, 12)
    agent[0, 0, :2] = torch.tensor([-0.99, 0.99])  # north-west corner -> cell (0, 0)
    agent[0, 1, :2] = torch.tensor([0.99, -0.99])  # south-east -> (5, 5)
    agent[0, 2, :2] = torch.tensor([0.01, 0.01])  # just right of / above the centre -> (2, 3)
    feat = torch.zeros(1, 128, 6, 6)
    mem, q_pos = mx.memory(feat, agent, 11)
    pos = (mx.pos_row[:, None] + mx.pos_col[None]).reshape(36, -1)
    assert torch.equal(q_pos[0, 0], pos[0]) and torch.equal(q_pos[0, 1], pos[35]) and torch.equal(q_pos[0, 2], pos[2 * 6 + 3])
    assert torch.equal(q_pos[0, 10], torch.zeros_like(pos[0]))  # team_state: no position
    assert torch.allclose(mem[0], mx.in_proj.bias + pos)


def test_temporal_sequence_path_runs():
    net = MyModel(n_actions=K, policy_head="logits", temporal=True, map_xattn="f2").eval()
    g, t, a = _frames(6)
    out = net.forward_sequence(g.view(2, 3, *g.shape[1:]), t.view(2, 3, -1), a.view(2, 3, 10, 12), q_units=5)
    assert out[0].shape == (2, 3, 5, K)


@pytest.mark.skipif(not P11.exists(), reason="needs the frozen P11 checkpoint")
def test_real_p11_warm_start_identical_and_inference_loads_it(tmp_path):
    from blackout_env.train.distill.inference import load_checkpoint_net

    state = torch.load(P11, map_location="cpu", weights_only=True)["policy_state"]
    old = MyModel.from_state_dict(state).eval()
    new = MyModel.from_state_dict(state, map_xattn="f3").eval()
    for a, b in zip(_out(old, _frames()), _out(new, _frames())):
        assert torch.equal(a, b)
    path = tmp_path / "m.pt"
    torch.save({"policy_state": new.state_dict()}, path)
    net = load_checkpoint_net(str(path))
    assert net.map_xattn is not None and net.map_xattn.source == "f3"
