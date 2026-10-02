"""S1 value net: the MyModel trunk with a multi-head value head.

The trunk (graphic + unit + team_state tokens through the 4-layer attention stack,
MyModel._trunk) is shared with the policy and can be warm-started from any MyModel checkpoint
(``policy_state``; the policy head is ignored).  The value head reads the post-attention
team_state (global) token, the mean of the own and of the enemy unit tokens, and the SPR CLS
token, and outputs

  [0, H)    lead change over each horizon (targets.HORIZONS_S)
  H         remaining lead change to the match end (final lead - lead now)
  H+1..H+3  result logits: loss, draw, win

Regressions are trained in tenths of points (``REG_SCALE`` x the score/100 targets) with a Huber
loss; the result with cross-entropy.  ``decode_outputs`` turns the raw outputs back into score/100
units, a final lead and P(win).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blackout_env.model.my_model import N_UNITS, N_VISION_TOKENS, MyModel
from blackout_env.model.modules import SwiGLUBlock
from blackout_env.train.distill.dataset import GRAPHIC_SCALE
from blackout_env.value.targets import HORIZONS_S

REG_SCALE = 10.0  # score/100 -> tenths of points
HUBER_BETA = 1.0  # = 10 points
N_TEAM = 5
TRUNK_MODULES = ("graphic_encoder", "vector_encoder", "attention")


class ValueNet(nn.Module):
    def __init__(self, hidden_size: int = 256, n_horizons: int = len(HORIZONS_S)):
        super().__init__()
        self.n_horizons = n_horizons
        self.trunk = MyModel(hidden_size=hidden_size, policy_head="logits")
        # the policy / SPR heads are not used here: drop them so they take no optimizer state
        self.trunk.q_head = None
        self.trunk.spr_head = None
        self.head_in = nn.LayerNorm(4 * hidden_size)
        self.head = nn.Sequential(SwiGLUBlock(4 * hidden_size, 2 * hidden_size, hidden_size), nn.GELU(),
                                  nn.Linear(hidden_size, n_horizons + 1 + 3))

    @property
    def n_outputs(self) -> int:
        return self.n_horizons + 4

    def load_trunk(self, policy_state: dict) -> None:
        """Warm start from a MyModel state dict (either policy head; head keys ignored)."""
        keep = {k: v for k, v in policy_state.items()
                if not k.startswith(("q_head.", "spr_head.", "temporal."))}
        missing, unexpected = self.trunk.load_state_dict(keep, strict=False)
        missing = [k for k in missing if not k.startswith(("q_head.", "spr_head."))]
        if missing or unexpected:
            raise RuntimeError(f"trunk warm start: missing {missing}, unexpected {unexpected}")

    def trunk_parameters(self):
        return self.trunk.parameters()

    def head_parameters(self):
        return list(self.head_in.parameters()) + list(self.head.parameters())

    def forward(self, graphic: torch.Tensor, team_state: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """graphic [B, C, H, W] float (decoded), team_state [B, 4], agent_states [B, 10, D] -> [B, H + 4]."""
        tokens = self.trunk._trunk(graphic, team_state, agent_states)
        units = tokens[:, N_VISION_TOKENS: N_VISION_TOKENS + N_UNITS]
        glob = tokens[:, N_VISION_TOKENS + N_UNITS]
        feats = torch.cat((glob, units[:, :N_TEAM].mean(1), units[:, N_TEAM:].mean(1), tokens[:, -1]), dim=-1)
        return self.head(self.head_in(feats))

    def forward_u8(self, graphic_u8: torch.Tensor, team_state: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """``graphic_u8``: the shard encoding [B, H, W, C] uint8 (x15)."""
        graphic = graphic_u8.permute(0, 3, 1, 2).float() / GRAPHIC_SCALE
        return self(graphic, team_state, agent_states)

    def predict(self, graphic_u8, team_state, agent_states) -> dict:
        """Decoded predictions (``decode_outputs``) for a batch in the shard encoding."""
        return decode_outputs(self.forward_u8(graphic_u8, team_state, agent_states).float(),
                              team_state[:, 0] - team_state[:, 1])


def decode_outputs(out: torch.Tensor, lead: torch.Tensor) -> dict:
    """Raw outputs -> {delta [B, H], final [B], p [B, 3] (loss, draw, win), p_win [B]} as numpy."""
    out, lead = out.detach(), lead.detach()
    h = out.shape[1] - 4
    p = torch.softmax(out[:, h + 1:].float(), dim=-1)
    return {"delta": (out[:, :h].float() / REG_SCALE).cpu().numpy(),
            "final": (lead.float() + out[:, h].float() / REG_SCALE).cpu().numpy(),
            "p": p.cpu().numpy(), "p_win": p[:, 2].cpu().numpy()}


def value_loss(out: torch.Tensor, targets: dict, lead: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Huber on the masked lead changes (mean over horizons) + Huber on the remaining final change
    + cross-entropy on the result.  ``targets``: delta [B, H], delta_mask [B, H], final [B],
    result [B] (-1/0/1)."""
    out = out.float()
    h = out.shape[1] - 4
    mask = targets["delta_mask"].float()
    reg = F.smooth_l1_loss(out[:, :h], targets["delta"].float() * REG_SCALE, beta=HUBER_BETA, reduction="none")
    per_h = (reg * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    delta_loss = per_h.mean()
    final_loss = F.smooth_l1_loss(out[:, h], (targets["final"].float() - lead.float()) * REG_SCALE, beta=HUBER_BETA)
    ce = F.cross_entropy(out[:, h + 1:], targets["result"].long() + 1)
    loss = delta_loss + final_loss + ce
    stats = {"loss": loss.detach(), "delta": delta_loss.detach(), "final": final_loss.detach(), "ce": ce.detach()}
    for i in range(h):
        stats[f"delta_{i}"] = per_h[i].detach()
    return loss, stats


def save_value_net(path, net: ValueNet, extra: dict | None = None) -> None:
    state = {k: v for k, v in net.state_dict().items()}
    torch.save({"value_state": state, "hidden_size": net.trunk.hidden_size, "n_horizons": net.n_horizons,
                "horizons_s": list(HORIZONS_S), **(extra or {})}, path)


def load_value_net(path, device="cpu") -> ValueNet:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = ValueNet(hidden_size=ckpt["hidden_size"], n_horizons=ckpt["n_horizons"])
    net.load_state_dict(_strip_compile(ckpt["value_state"]))
    return net.to(device).eval()


def _strip_compile(state: dict) -> dict:
    return {k.replace("._orig_mod", ""): v for k, v in state.items()}
