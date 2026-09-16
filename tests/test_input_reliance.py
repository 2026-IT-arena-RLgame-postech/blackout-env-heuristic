"""Input-reliance probe: a map-blind net, a map-reading net, and a collapsed SPR latent."""

from __future__ import annotations

import torch
from torch import nn

from blackout_env.model.my_model import MyModel
from blackout_env.train.input_reliance import input_reliance_probe
from blackout_env.train.qmix_trainer import _own_team_rows

B, H, W, C = 16, 24, 24, 13


class _StubNet(nn.Module):
    """MyModel's forward signature; Q built from whichever inputs the test wants read."""

    def __init__(self, reads_graphic: bool, reads_vector: bool, collapsed_spr: bool) -> None:
        super().__init__()
        self.reads_graphic, self.reads_vector, self.collapsed_spr = reads_graphic, reads_vector, collapsed_spr
        self.proj = nn.Linear(4, 8, bias=False)  # no bias: the greedy action must depend on the input

    def forward(self, graphic, team_state, agent_states, n_quantiles=8, tau=None):
        g = graphic.flatten(1)[:, :4] - 0.5 if self.reads_graphic else torch.zeros(graphic.shape[0], 4)
        v = agent_states[:, :, :4].mean(1) if self.reads_vector else torch.zeros(graphic.shape[0], 4)
        per_sample = self.proj(g + v)  # [B, 8]
        q = per_sample[:, None, :] * torch.linspace(0.5, 1.5, 10)[None, :, None]  # [B, 10, 8]
        spr = torch.ones(graphic.shape[0], 6) if self.collapsed_spr else torch.randn(graphic.shape[0], 6)
        return q, q[:, :, None, :], tau, spr, per_sample


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    graphic = torch.rand(B, C, H, W)
    team_state = torch.rand(B, 4)
    agent_states = torch.rand(B, 10, 12) * 2 - 1
    agent_states[:, :5, 2], agent_states[:, 5:, 2] = 1.0, -1.0
    return graphic, team_state, agent_states


def test_a_map_blind_net_reads_as_blind():
    stats = input_reliance_probe(_StubNet(False, True, True), *_batch(), _own_team_rows)
    assert stats["graphic_shuffle_dq"] == 0.0
    assert stats["graphic_shuffle_argmax_kept"] == 1.0
    assert stats["vector_shuffle_dq"] > 0.1
    assert abs(stats["spr_latent_cos"] - 1.0) < 1e-6


def test_a_map_reading_net_reads_as_reading():
    stats = input_reliance_probe(_StubNet(True, False, False), *_batch(), _own_team_rows)
    assert stats["graphic_shuffle_dq"] > 0.1
    assert stats["graphic_shuffle_argmax_kept"] < 1.0
    assert stats["vector_shuffle_dq"] == 0.0
    assert stats["spr_latent_cos"] < 0.5


def test_runs_on_the_real_model():
    graphic, team_state, agent_states = _batch()
    stats = input_reliance_probe(MyModel(hidden_size=128), graphic, team_state, agent_states, _own_team_rows)
    assert set(stats) == {"graphic_shuffle_dq", "vector_shuffle_dq", "graphic_shuffle_argmax_kept", "spr_latent_cos"}
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
