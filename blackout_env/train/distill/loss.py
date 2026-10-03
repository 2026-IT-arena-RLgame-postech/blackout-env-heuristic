"""Set-valued cross-entropy: the teacher marks every compass move that is equally right.

The loss maximises the probability mass the net puts on the whole set, log sum_{i in set} p_i,
so straight/diagonal ties are not forced to one side (forcing one would be label noise, see
heuristics/teach.py).  The net's 8 "Q-values" are read as logits; a small z-loss keeps their
log-normaliser near zero so the logits cannot drift off without bound (Run 14's CE-on-Q
divergence came from bootstrapped Q targets, which distillation does not have, but nothing else
anchors their scale here).
"""

from __future__ import annotations

import math

import torch


def set_valued_cross_entropy(
    logits: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor | None = None,
    z_loss: float = 1e-4,
    total: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """logits [..., 8], masks bool [..., 8] (all False = unlabelled, ignored), weights [...].

    Returns (loss, stats) with stats ``ce``, ``z`` and ``accuracy`` (argmax inside the set),
    all averaged over labelled entries with ``weights``.  ``total``: the normaliser to use instead
    of this batch's weight sum (multi-GPU: the global sum / world size, accel.Dist)."""
    valid = masks.any(dim=-1)
    w = valid.to(logits.dtype) if weights is None else weights * valid.to(logits.dtype)
    total = w.sum().clamp_min(1e-8) if total is None else total

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


def angle_offset_loss(
    offsets: torch.Tensor,
    angles: torch.Tensor,
    weights: torch.Tensor | None = None,
    arc_bins: int = 0,
    total: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """P11's angle head: offsets [..., K] radians (each in [-pi/K, pi/K], or +-(arc_bins + 1/2)
    bins), angles [...] the teacher's raw headings in the bins' frame (NaN = none: ignored),
    weights [...].

    The target is the bin whose centre is nearest the angle (the act label's centre bin) and
    the residual ``wrap(angle - centre)``; the loss is the squared error of that bin's offset in
    units of half a bin width, ``((offset - residual) / (pi / K))^2``, weighted mean over rows
    with an angle.  Normalised so it does not depend on K: an untrained (zero) offset scores the
    residual's mean square, 1/3 for a uniform residual, the same order as the cross-entropy; 2
    degrees at K = 32 is (2 / 5.625)^2 = 0.13.

    ``arc_bins`` N > 0 (``--angle-arc-bins``, with an arc label ``--act-arc-bins``): the bins
    ``centre +- 1..N`` are trained too, each to point at the same angle (their residual
    ``wrap(angle - their centre)``), so whichever bin of the label's arc wins the arg max executes
    the teacher's angle; the loss is the mean over the 2N + 1 bins.  Returns (loss, stats
    ``angle_loss``, ``angle_loss_centre`` (the teacher's bin alone), ``angle_rows``)."""
    k = offsets.shape[-1]
    half = math.pi / k
    valid = torch.isfinite(angles)
    w = valid.to(offsets.dtype) if weights is None else weights * valid.to(offsets.dtype)
    total = w.sum().clamp_min(1e-8) if total is None else total  # ``total``: as set_valued_cross_entropy
    angle = torch.where(valid, angles.to(offsets.dtype), torch.zeros_like(offsets[..., 0]))
    centre = torch.round(angle / (2 * half)).long() % k
    losses = []
    for shift in [0] + [s for j in range(1, arc_bins + 1) for s in (j, -j)]:
        target = (centre + shift) % k
        residual = torch.remainder(angle - target.to(offsets.dtype) * 2 * half + math.pi, 2 * math.pi) - math.pi
        pred = offsets.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        err = ((pred - residual) / half) ** 2
        losses.append((torch.where(valid, err, torch.zeros_like(err)) * w).sum() / total)
    loss = torch.stack(losses).mean()
    return loss, {"angle_loss": loss.detach(), "angle_loss_centre": losses[0].detach(),
                  "angle_rows": valid.float().sum().detach()}
