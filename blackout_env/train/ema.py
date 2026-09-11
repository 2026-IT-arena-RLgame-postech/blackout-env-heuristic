"""Exponential moving average parameter update, shared by the SPR target encoder (EMA copy
of the whole MyModel) and SPRPredictor's target_projector (EMA copy of its online projector).
Plain manual param loop rather than torch.optim.swa_utils.AveragedModel — that utility's
default averaging convention (equal-weight running average, unless you supply a custom
avg_fn) isn't the EMA update these two need, and a custom avg_fn would end up being this
exact loop anyway."""

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- tau * target + (1 - tau) * source, applied to every parameter AND buffer
    (buffers matter here too — e.g. RMSNorm running stats, if any ever appear — so the EMA
    copy doesn't silently drift out of sync with the online net's non-parameter state)."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)
    for t_buf, s_buf in zip(target.buffers(), source.buffers()):
        if t_buf.dtype.is_floating_point:
            t_buf.data.mul_(tau).add_(s_buf.data, alpha=1.0 - tau)
        else:
            t_buf.data.copy_(s_buf.data)
