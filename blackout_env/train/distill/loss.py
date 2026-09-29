"""Set-valued cross-entropy: the teacher marks every compass move that is equally right.

The loss maximises the probability mass the net puts on the whole set, log sum_{i in set} p_i,
so straight/diagonal ties are not forced to one side (forcing one would be label noise, see
heuristics/teach.py).  The net's 8 "Q-values" are read as logits; a small z-loss keeps their
log-normaliser near zero so the logits cannot drift off without bound (Run 14's CE-on-Q
divergence came from bootstrapped Q targets, which distillation does not have, but nothing else
anchors their scale here).
"""

from __future__ import annotations

import torch


def set_valued_cross_entropy(
    logits: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor | None = None,
    z_loss: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """logits [..., 8], masks bool [..., 8] (all False = unlabelled, ignored), weights [...].

    Returns (loss, stats) with stats ``ce``, ``z`` and ``accuracy`` (argmax inside the set),
    all averaged over labelled entries with ``weights``."""
    valid = masks.any(dim=-1)
    w = valid.to(logits.dtype) if weights is None else weights * valid.to(logits.dtype)
    total = w.sum().clamp_min(1e-8)

    log_norm = torch.logsumexp(logits, dim=-1)
    # Unlabelled entries take the full set (nll 0): an empty set would give -inf, and its NaN
    # gradient survives a torch.where.
    safe = masks | ~valid.unsqueeze(-1)
    nll = log_norm - torch.logsumexp(logits.masked_fill(~safe, float("-inf")), dim=-1)
    ce = (nll * w).sum() / total
    z = (torch.where(valid, log_norm, torch.zeros_like(log_norm)) ** 2 * w).sum() / total

    with torch.no_grad():
        hit = masks.gather(-1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        accuracy = (hit.to(logits.dtype) * w).sum() / total
    return ce + z_loss * z, {"ce": ce.detach(), "z": z.detach(), "accuracy": accuracy}
