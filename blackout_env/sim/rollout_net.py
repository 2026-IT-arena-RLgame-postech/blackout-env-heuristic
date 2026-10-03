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

``arch="attn"`` instead: 10 unit tokens (agent row, own-tile 24x24 / 12x12 features, wall patch, slot)
+ 1 global token (flattened 6x6 map + team_state) -> a pre-norm transformer -> own unit tokens ->
MLP head.  ``map_tokens`` adds spatial map tokens to that sequence (docs §22):

* ``"f3"``    6 x 6 = 36 tokens, Linear(c3 -> d) of the stride-4 map features;
* ``"s3"``    8 x 8 = 64 tokens, conv 3x3 stride 3 (c1 -> d) of the 24 x 24 features;
* ``"patch"`` 8 x 8 = 64 tokens, ViT-style overlapping 5 x 5 patches (stride 3, padding 1) of the
              raw 19-channel input, linear patch embedding.

Every map token gets a learned 2-D position embedding (row + column) of its cell, and each unit token
gets the embedding of the cell it stands in, so units can locate themselves relative to the map.

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
N_QFEAT = 8  # logit_features
MAP_TOKENS = {"none": 0, "f3": 6, "s3": 8, "patch": 8}  # map-token grid side
# how the 11 unit/global tokens use the map tokens: "self" = one joint self-attention sequence (map tokens
# updated every layer); "cross" = K/V only, all tokens see the whole map; "local3"/"local5" = each unit token
# sees only the w x w window of map tokens around its cell (the global token sees all); "fovea" = local 3x3
# window of fine tokens + all 36 coarse f3 tokens (6 x 6) for every unit.
MAP_ATTN = ("self", "cross", "local3", "local5", "fovea")


class RolloutNet(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 32, c3: int = 64, glob: int = 128, hidden: int = 256,
                 k: int = K_BINS, angle_arc_bins: int = 1, arch: str = "mlp", d: int = 128, layers: int = 2,
                 heads: int = 4, map_tokens: str = "none", map_attn: str = "self", map_dim: int = 0, q_head: int = 0):
        super().__init__()
        self.config = dict(c1=c1, c2=c2, c3=c3, glob=glob, hidden=hidden, k=k, angle_arc_bins=angle_arc_bins,
                           arch=arch, d=d, layers=layers, heads=heads)
        if map_tokens != "none":
            self.config["map_tokens"] = map_tokens
            self.config["map_attn"] = map_attn
            if map_dim:
                self.config["map_dim"] = map_dim
        if q_head:
            self.config["q_head"] = q_head
        self.k, self.arch, self.map_tokens = k, arch, map_tokens
        self.cross = map_tokens != "none" and map_attn != "self"
        self.map_attn = map_attn
        assert map_attn in MAP_ATTN, map_attn
        assert map_tokens in MAP_TOKENS and (map_tokens == "none" or arch == "attn"), map_tokens
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
            if self.cross:  # unit/global tokens self-attend + cross-attend to static map tokens (keys/values only)
                self.trunk = nn.ModuleList(CrossLayer(d, heads, map_dim or d) for _ in range(layers))
                self.map_norm = nn.LayerNorm(map_dim or d)
            else:
                self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 2 * k))
            if map_tokens != "none":
                self.grid = MAP_TOKENS[map_tokens]
                # map-token width: d, or (cross trunks only) a narrower map_dim -- cheaper patch embedding and K/V
                dm = self.dm = map_dim or d
                assert dm == d or map_attn != "self"
                if map_tokens == "f3":
                    self.map_proj = nn.Linear(c3, dm)
                elif map_tokens == "s3":
                    self.map_proj = nn.Conv2d(c1, dm, 3, stride=3)
                else:  # "patch"
                    self.map_proj = nn.Conv2d(13 + 2 * N_CLS, dm, 5, stride=3, padding=1)
                self.pos_row = nn.Parameter(torch.randn(self.grid, d) * 0.02)   # unit tokens (and map tokens if dm == d)
                self.pos_col = nn.Parameter(torch.randn(self.grid, d) * 0.02)
                if dm != d:
                    self.mpos_row = nn.Parameter(torch.randn(self.grid, dm) * 0.02)
                    self.mpos_col = nn.Parameter(torch.randn(self.grid, dm) * 0.02)
                if map_attn == "fovea":  # coarse context: the 6 x 6 f3 map, its own position embedding
                    assert map_tokens != "f3"
                    self.coarse_proj = nn.Linear(c3, dm)
                    self.cpos_row = nn.Parameter(torch.randn(6, dm) * 0.02)
                    self.cpos_col = nn.Parameter(torch.randn(6, dm) * 0.02)
            if q_head:  # per own unit: advantage of the S3 "alt" branch over "base" (docs §22 C)
                self.q_net = nn.Sequential(nn.Linear(d + N_QFEAT, q_head), nn.SiLU(), nn.Linear(q_head, q_head), nn.SiLU(),
                                           nn.Linear(q_head, 1))
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
            return self._attn(graphic, team_state, agent_states, q_units, angle, x, f1, f2, f3, g, row, col)
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

    def _attn(self, graphic, team_state, agent_states, q_units, angle, x, f1, f2, f3, g, row, col, trunk_out=False):
        b, width = graphic.shape[0], graphic.shape[-1]
        t1 = f1.flatten(2).gather(2, (row * width + col).unsqueeze(1).expand(-1, f1.shape[1], -1)).transpose(1, 2)
        t2 = f2.flatten(2).gather(2, ((row // 2) * (width // 2) + col // 2).unsqueeze(1).expand(-1, f2.shape[1], -1)).transpose(1, 2)
        walls = local_wall_features(graphic, agent_states)  # [B, 10, 25]
        units = self.unit_proj(torch.cat([agent_states, t1, t2, walls, self.slot10.unsqueeze(0).expand(b, -1, -1)], -1))
        glob = self.glob_proj(torch.cat([g, team_state], -1)).unsqueeze(1)
        seq = [units, glob]
        if self.map_tokens != "none":
            gs = self.grid
            if self.map_tokens == "f3":
                m = self.map_proj(f3.flatten(2).transpose(1, 2))           # [B, 36, d]
            elif self.map_tokens == "s3":
                m = self.map_proj(f1).flatten(2).transpose(1, 2)           # [B, 64, d]
            else:
                m = self.map_proj(x).flatten(2).transpose(1, 2)            # [B, 64, d]
            pos = (self.pos_row[:, None, :] + self.pos_col[None, :, :]).reshape(gs * gs, -1)
            ur = (row * gs) // graphic.shape[-2]
            uc = (col * gs) // width
            seq[0] = units + pos[ur * gs + uc]
            mpos = pos if self.dm == pos.shape[-1] else (self.mpos_row[:, None, :] + self.mpos_col[None, :, :]).reshape(gs * gs, -1)
            seq.append(m + mpos.unsqueeze(0))
        if self.cross:
            h = torch.cat(seq[:2], 1)
            mem, mask = seq[2], None
            if self.map_attn != "cross":
                w = 5 if self.map_attn == "local5" else 3
                gr = torch.arange(gs, device=row.device)
                near_r = (gr[None, None, :] - ur[..., None]).abs() <= w // 2          # [B, 10, gs]
                near_c = (gr[None, None, :] - uc[..., None]).abs() <= w // 2
                win = (near_r[..., :, None] & near_c[..., None, :]).flatten(2)       # [B, 10, gs*gs]
                mask = torch.cat([win, torch.ones_like(win[:, :1])], 1)               # global token: all
                if self.map_attn == "fovea":
                    cpos = (self.cpos_row[:, None, :] + self.cpos_col[None, :, :]).reshape(36, -1)
                    coarse = self.coarse_proj(f3.flatten(2).transpose(1, 2)) + cpos.unsqueeze(0)
                    mem = torch.cat([mem, coarse], 1)
                    mask = torch.cat([mask, torch.ones_like(mask[..., :36])], -1)
            mem = self.map_norm(mem)
            for layer in self.trunk:
                h = layer(h, mem, mask)
            h = self.norm(h)
        else:
            h = self.norm(self.trunk(torch.cat(seq, 1)))
        out = self.head(h[:, :q_units])
        logits, off = out[..., : self.k], torch.tanh(out[..., self.k:]) * self.angle_scale
        if trunk_out:
            return logits, off, h[:, :q_units]
        return (logits, off) if angle else logits

    def q_alt(self, graphic, team_state, agent_states, detach_trunk: bool = False):
        """(logits, offsets, adv) for own units 0-4: adv [B, 5] = predicted leaf-value gain (points)
        of the S3 ``alt u`` branch (unit u's best bin outside its top bin +- 1) over ``base``."""
        b, _, height, width = graphic.shape
        raster, row, col = self._rasters(agent_states, height, width)
        x = torch.cat([graphic, raster], 1)
        f1 = F.silu(self.conv1(x))
        f2 = F.silu(self.conv2(f1))
        f3 = F.silu(self.conv3(f2))
        g = F.silu(self.glob(f3.flatten(1)))
        logits, off, h = self._attn(graphic, team_state, agent_states, N_TEAM, True, x, f1, f2, f3, g, row, col, trunk_out=True)
        q_in = torch.cat([h.detach() if detach_trunk else h, logit_features(logits.detach() if detach_trunk else logits)], -1)
        return logits, off, self.q_net(q_in).squeeze(-1)

    @classmethod
    def from_checkpoint(cls, path: str) -> "RolloutNet":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(**ck["config"])
        net.load_state_dict(ck["state"])
        return net.eval()


class CrossLayer(nn.Module):
    """Pre-norm block for the cross-attention trunks (``map_attn`` != "self"): self-attention among
    the 11 unit/global tokens, cross-attention from them to the map tokens (keys/values only -- the
    map tokens are never updated), GELU FFN (2d).  ``mask`` [B, 11, S] (True = may attend) restricts
    what each token sees (local windows, docs §22)."""

    def __init__(self, d: int, heads: int, dm: int | None = None):
        super().__init__()
        self.h = heads
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(dm or d, 2 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mem, mask=None):
        y = self.n1(x)
        x = x + self.sa(y, y, y, need_weights=False)[0]
        b, n, d = x.shape
        q = self.q(self.n2(x)).view(b, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(b, mem.shape[1], 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.unsqueeze(1))
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.n3(x))


def alt_bins(logits: torch.Tensor) -> torch.Tensor:
    """[..., K] -> the S3 ``alt`` bin: arg max outside the top bin +- 1 (search.ModNetPolicy)."""
    k = logits.shape[-1]
    top = logits.argmax(-1, keepdim=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for dlt in (-1, 0, 1):
        mask.scatter_(-1, (top + dlt) % k, True)
    return logits.masked_fill(mask, float("-inf")).argmax(-1)


def logit_features(logits: torch.Tensor) -> torch.Tensor:
    """[B, 5, K] -> [B, 5, N_QFEAT]: top logit - alt logit, top-arc mass, alt-arc mass, entropy, and the
    canonical-frame directions (cos, sin) of the top and the alt bin (what the alternative *is*)."""
    k = logits.shape[-1]
    p = torch.softmax(logits.float(), -1)
    top = logits.argmax(-1, keepdim=True)
    alt = alt_bins(logits).unsqueeze(-1)
    arc = lambda j: sum(p.gather(-1, (j + dlt) % k) for dlt in (-1, 0, 1)).squeeze(-1)  # noqa: E731
    gap = (logits.gather(-1, top) - logits.gather(-1, alt)).squeeze(-1).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)
    ta = top.squeeze(-1).float() * (2 * math.pi / k)
    aa = alt.squeeze(-1).float() * (2 * math.pi / k)
    return torch.stack([gap.clamp(max=20.0) / 10.0, arc(top), arc(alt), ent, ta.cos(), ta.sin(), aa.cos(), aa.sin()],
                       -1).to(logits.dtype)


class RolloutOwnQ(nn.Module):
    """``inference._OwnQ``'s interface for a RolloutNet: (logits, offsets) of own rows 0-4."""

    def __init__(self, net: RolloutNet):
        super().__init__()
        self.net = net

    def forward(self, graphic, team_state, agent_states):
        return self.net(graphic, team_state, agent_states, q_units=N_TEAM, angle=True)
