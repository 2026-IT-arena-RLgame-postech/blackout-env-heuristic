"""A small, fast rollout policy for batched simulator rollouts (S3): ``RolloutNet``.

Same inputs and outputs as a P11 student (``MyModel(policy_head="logits", angle_head=True)``,
K = 32 bins + per-bin angle offsets, own units 0-4 of a canonical-frame view), so
``batch.NetPolicy`` drives it unchanged.  Architecture (~30x fewer multiply-adds than P11):

* map: the 13 env channels + 6 unit rasters (own / enemy x Collector / Hunter / Carrier, on the
  unit's tile) -> conv 3x3 (C1, 24x24) -> conv 3x3 s2 (C2, 12x12) -> conv 3x3 s2 (C3, 6x6) ->
  flatten -> Linear -> a global vector;
* per own unit: its agent_states row, its tile's features from the 24x24 and 12x12 maps, the
  bilinear walkability patch around its exact position (derived_obs.local_wall_features, as
  P11), the global vector, team_state, a slot one-hot and every unit's (x, y, class) relative
  to it -> MLP -> hidden; one mixing round (hidden + mean/max over the own team) -> MLP ->
  K logits + K offsets (tanh x angle_scale, as modules/logits_head.py).

``forward(graphic, team_state, agent_states, q_units=5, angle=True)`` returns
``(logits [B, 5, K], offsets [B, 5, K])`` like ``inference._OwnQ`` -- wrap with
``RolloutOwnQ`` for ``NetPolicy``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from blackout_env.model.derived_obs import N_PATCH_FEATURES, local_wall_features

N_UNITS, N_TEAM, K_BINS = 10, 5, 32
N_CLS = 3


class RolloutNet(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 32, c3: int = 64, glob: int = 128, hidden: int = 256,
                 k: int = K_BINS, angle_arc_bins: int = 1, arch: str = "mlp", d: int = 128, layers: int = 2,
                 heads: int = 4):
        super().__init__()
        self.config = dict(c1=c1, c2=c2, c3=c3, glob=glob, hidden=hidden, k=k, angle_arc_bins=angle_arc_bins,
                           arch=arch, d=d, layers=layers, heads=heads)
        self.k, self.arch = k, arch
        self.conv1 = nn.Conv2d(13 + 2 * N_CLS, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.glob = nn.Linear(c3 * 36, glob)
        self.register_buffer("angle_scale", torch.tensor((angle_arc_bins + 0.5) * 2 * math.pi / k))
        self.register_buffer("slot", torch.eye(N_TEAM))
        self.register_buffer("slot10", torch.eye(N_UNITS), persistent=False)
        self.angle_head = True
        if arch == "attn":  # 10 unit tokens + 1 global token, a small transformer, own tokens -> head
            self.unit_proj = nn.Linear(12 + c1 + c2 + N_PATCH_FEATURES + N_UNITS, d)
            self.glob_proj = nn.Linear(glob + 4, d)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 2 * k))
            return
        unit_in = 12 + c1 + c2 + glob + 4 + N_TEAM + N_UNITS * 5 + N_PATCH_FEATURES
        self.unit_in = unit_in
        self.mlp1 = nn.Sequential(nn.Linear(unit_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mlp2 = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, 2 * k)

    def _rasters(self, agent_states, height, width):
        """[B, 6, H, W]: own / enemy unit counts per class on their tiles."""
        b = agent_states.shape[0]
        x, y = agent_states[..., 0], agent_states[..., 1]
        col = ((x + 1.0) * 0.5 * width).floor().clamp(0, width - 1).long()
        row = ((1.0 - y) * 0.5 * height).floor().clamp(0, height - 1).long()
        cls = agent_states[..., 9:12].argmax(-1)
        own = (agent_states[..., 2] > 0).long()
        chan = (1 - own) * N_CLS + cls  # own team first
        idx = (chan * height + row) * width + col  # [B, 10]
        out = torch.zeros(b, 2 * N_CLS * height * width, device=agent_states.device, dtype=agent_states.dtype)
        out.scatter_add_(1, idx, torch.ones_like(x))
        return out.view(b, 2 * N_CLS, height, width), row, col

    def forward(self, graphic, team_state, agent_states, q_units: int = N_TEAM, angle: bool = True):
        if graphic.device.type == "cpu":  # NNPACK's small-channel convs are ~2.5x slower than the default path
            with torch.backends.nnpack.flags(enabled=False):
                return self._forward(graphic, team_state, agent_states, q_units, angle)
        return self._forward(graphic, team_state, agent_states, q_units, angle)

    def _forward(self, graphic, team_state, agent_states, q_units, angle):
        b, _, height, width = graphic.shape
        raster, row, col = self._rasters(agent_states, height, width)
        x = torch.cat([graphic, raster], 1)
        f1 = F.silu(self.conv1(x))                    # [B, c1, 24, 24]
        f2 = F.silu(self.conv2(f1))                   # [B, c2, 12, 12]
        f3 = F.silu(self.conv3(f2))                   # [B, c3, 6, 6]
        g = F.silu(self.glob(f3.flatten(1)))          # [B, glob]
        if self.arch == "attn":
            return self._attn(graphic, team_state, agent_states, q_units, angle, f1, f2, g, row, col)
        r, c = row[:, :q_units], col[:, :q_units]
        t1 = f1.flatten(2).gather(2, (r * width + c).unsqueeze(1).expand(-1, f1.shape[1], -1)).transpose(1, 2)
        t2 = f2.flatten(2).gather(2, ((r // 2) * (width // 2) + c // 2).unsqueeze(1).expand(-1, f2.shape[1], -1)).transpose(1, 2)
        walls = local_wall_features(graphic, agent_states)[:, :q_units]  # [B, q, 25]
        own = agent_states[:, :q_units]
        # every unit relative to this one (dx, dy, class one-hot; fixed slot order, self = 0 offset)
        d = agent_states[:, None, :, :2] - own[:, :, None, :2]                       # [B, q, 10, 2]
        cls = agent_states[:, None, :, 9:12].expand(-1, q_units, -1, -1)            # [B, q, 10, 3]
        rel = torch.cat([d, cls], -1).flatten(2)                                     # [B, q, 50]
        feats = torch.cat([own, t1, t2, g.unsqueeze(1).expand(-1, q_units, -1),
                           team_state.unsqueeze(1).expand(-1, q_units, -1),
                           self.slot[:q_units].unsqueeze(0).expand(b, -1, -1), rel, walls], -1)
        h = self.mlp1(feats)
        ctx = torch.cat([h, h.mean(1, keepdim=True).expand_as(h), h.amax(1, keepdim=True).expand_as(h)], -1)
        h = self.mlp2(ctx) + h
        out = self.out(h)
        logits, off = out[..., : self.k], torch.tanh(out[..., self.k:]) * self.angle_scale
        return (logits, off) if angle else logits

    def _attn(self, graphic, team_state, agent_states, q_units, angle, f1, f2, g, row, col):
        b, width = graphic.shape[0], graphic.shape[-1]
        t1 = f1.flatten(2).gather(2, (row * width + col).unsqueeze(1).expand(-1, f1.shape[1], -1)).transpose(1, 2)
        t2 = f2.flatten(2).gather(2, ((row // 2) * (width // 2) + col // 2).unsqueeze(1).expand(-1, f2.shape[1], -1)).transpose(1, 2)
        walls = local_wall_features(graphic, agent_states)  # [B, 10, 25]
        units = self.unit_proj(torch.cat([agent_states, t1, t2, walls, self.slot10.unsqueeze(0).expand(b, -1, -1)], -1))
        glob = self.glob_proj(torch.cat([g, team_state], -1)).unsqueeze(1)
        h = self.norm(self.trunk(torch.cat([units, glob], 1)))
        out = self.head(h[:, :q_units])
        logits, off = out[..., : self.k], torch.tanh(out[..., self.k:]) * self.angle_scale
        return (logits, off) if angle else logits

    @classmethod
    def from_checkpoint(cls, path: str) -> "RolloutNet":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(**ck["config"])
        net.load_state_dict(ck["state"])
        return net.eval()


class RolloutOwnQ(nn.Module):
    """``inference._OwnQ``'s interface for a RolloutNet: (logits, offsets) of own rows 0-4."""

    def __init__(self, net: RolloutNet):
        super().__init__()
        self.net = net

    def forward(self, graphic, team_state, agent_states):
        return self.net(graphic, team_state, agent_states, q_units=N_TEAM, angle=True)
