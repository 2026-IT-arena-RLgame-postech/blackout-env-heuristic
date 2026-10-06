"""QZ: NetPolicy's opt-in fast precisions (blackout_env/sim/fastnet.py)."""

import numpy as np
import pytest
import torch

from blackout_env.sim.batch import BatchSim, NetPolicy
from blackout_env.sim.fastnet import FastCore, fast_checkpoint
from blackout_env.sim.state import SimState

from tests.test_sim_batch import _student


def _inputs(n=6):
    bs = BatchSim([SimState.from_seed(s) for s in range(n)])
    from blackout_env.sim.batch import N_AGENT_COLS
    from blackout_env.sim.mapdata import H, W

    bufs = (np.zeros((n, 2, H, W), np.int16), np.zeros((n, 2, 4), np.float32), np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    bs.render_packed(np.arange(n), *bufs)
    return tuple(x.reshape(2 * n, *x.shape[2:]) for x in bufs)


def test_default_precision_is_the_plain_path():
    pol = NetPolicy(_student(), device="cpu")
    assert pol.precision == "fp32" and pol._fast is None


def test_pad_buckets_and_slicing():
    calls = []

    def own(g, t, a):
        calls.append(g.shape[0])
        return g.flatten(1)[:, :3] * 2, t * 3

    core = FastCore.__new__(FastCore)
    core.torch, core.device, core.core = torch, torch.device("cpu"), own
    core.buckets = (4, 8)
    g, t, a = torch.randn(5, 2, 2), torch.randn(5, 4), torch.randn(5, 10, 12)
    q, o = core(g, t, a)
    assert calls == [8] and q.shape == (5, 3) and torch.equal(o, t * 3)
    core(torch.randn(9, 2, 2), torch.randn(9, 4), torch.randn(9, 10, 12))
    assert calls[-1] == 9  # past the largest bucket: as is


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_fp16_close_to_fp32(device):
    if device == "cpu":
        pytest.importorskip("torch")  # CPU autocast fp16 needs a recent torch; it is what the Mac has
    net = _student()
    g, t, a = _inputs()
    q0, o0 = NetPolicy(net, device=device).forward(g, t, a)
    q1, o1 = NetPolicy(net, device=device, precision="fp16").forward(g, t, a)
    assert q1.dtype == np.float32 and q1.shape == q0.shape and o1.shape == o0.shape
    assert np.abs(q1 - q0).max() < 0.05 and np.abs(o1 - o0).max() < 0.05


def test_checkpoint_precision_key(tmp_path):
    net = _student()
    src = tmp_path / "p.pt"
    torch.save({"policy_state": net.state_dict()}, src)
    fast_checkpoint(str(src), str(tmp_path / "f.pt"), "fp16")
    pol = NetPolicy.from_checkpoint(str(tmp_path / "f.pt"), device="cpu")
    assert pol.precision == "fp16" and pol._fast is not None and not pol.fp16
    assert NetPolicy.from_checkpoint(str(tmp_path / "f.pt"), device="cpu", precision="fp32")._fast is None
    assert NetPolicy.from_checkpoint(str(src), device="cpu").precision == "fp32"
