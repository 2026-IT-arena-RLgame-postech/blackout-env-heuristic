"""examples/opponent_id_probe.py: labels, time grid, metrics, probe shapes (no shards, no Unity)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import opponent_id_probe as P  # noqa: E402


def test_family():
    assert P.family("ckpt:20260917-200803_75984/step_5000") == "checkpoint"
    assert P.family("mixture:strategic_v4_near") == "v4fam"
    assert P.family("mixture:strategic_v13") == "v4fam"
    assert P.family("strategic_v3") == "v4fam"
    assert P.family("mixture:strategic_v5") == "v5"
    assert P.family("mixture:strategic_v19") == "v19"
    assert P.family("strategic_raider") == "strategic_raider"
    assert P.family("") is None
    assert P.coarse("v5") == "other" and P.coarse("v18") == "v18"


def _shard(tmp_path, ticks, end_extra=()):
    ticks = np.array(list(ticks) + list(end_extra))
    T = len(ticks)
    ts = np.zeros((T, 4), np.float32)
    ts[:, 2] = 1.0 - ticks / P.MATCH_TICKS
    g = np.zeros((T, 24, 24, 13), np.uint8)
    g[:, 0, 0, 0] = np.arange(T)  # row id, to check which row each grid point used
    a = np.zeros((T, 10, 12), np.float32)
    path = tmp_path / "s.npz"
    np.savez(path, graphic=g, team_state=ts, agent_states=a)
    return str(path)


def test_load_grid_forward_fills_gaps_and_stops_at_match_end(tmp_path):
    ticks = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]  # 9-11 dropped (no label)
    g, ts, a, remap, filled = P.load_grid(_shard(tmp_path, ticks), stride=4, max_ticks=100)
    grid_rows = g[remap, 0, 0, 0]
    # grid ticks 4, 8, 12, 16 -> rows with ticks 4, 8, 12, 16 (indices 3, 7, 8, 12); match ends at 17
    assert grid_rows.tolist() == [3, 7, 8, 12]
    assert filled == 0
    g, ts, a, remap, filled = P.load_grid(_shard(tmp_path, [1, 2, 3, 5, 6, 7, 9]), stride=2, max_ticks=100)
    assert g[remap, 0, 0, 0].tolist() == [1, 2, 4, 5]  # ticks 2, 3(<-4), 6, 7(<-8)
    assert filled == 2


def test_load_grid_caps_at_max_ticks_and_drops_repeated_ticks(tmp_path):
    g, ts, a, remap, _ = P.load_grid(_shard(tmp_path, [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10]), stride=2, max_ticks=6)
    assert g[remap, 0, 0, 0].tolist() == [1, 4, 6]  # ticks 2, 4, 6; the repeated 3 is skipped


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert P.macro_f1(y, y, 2) == 1.0
    assert P.macro_f1(y, np.array([0, 0, 0, 0]), 2) == pytest.approx((2 / 3 + 0) / 2)
    probs = np.zeros((1, 3, 2), np.float32)
    probs[0, :, 0] = [0.9, 0.2, 0.2]
    probs[0, :, 1] = 1 - probs[0, :, 0]
    avg = P.cumulative_average(probs, np.array([3]))
    assert avg[0, 0].argmax() == 0 and avg[0, 1].argmax() == 0 and avg[0, 2].argmax() == 1


@pytest.mark.parametrize("kind", ["stateless", "gru", "kda", "raw_stateless", "raw_kda"])
def test_probe_shapes_and_causality(kind):
    torch.manual_seed(0)
    D = 124 if kind.startswith("raw_") else 32
    net = P.Probe(kind, D, 5, H=16).eval()
    x = torch.randn(2, 10, D) if kind.startswith("raw_") else torch.randn(2, 10, 11, D)
    y = net(x)
    assert y.shape == (2, 10, 5)
    x2 = x.clone()
    x2[:, 6:] = 0  # the future must not change the past
    assert torch.allclose(net(x2)[:, :6], y[:, :6], atol=1e-5)
