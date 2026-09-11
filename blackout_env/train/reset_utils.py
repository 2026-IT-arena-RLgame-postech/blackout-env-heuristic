"""
Periodic parameter reset ("shrink and perturb", Ash & Adams 2020; used in BBF to fight
primacy-bias/plasticity loss over long training runs). Applied conservatively for now: only
to MyModel.graphic_encoder, on the trainer's schedule — see QMIXTrainer. Extending this to the
attention trunk's FFN blocks is a later, separate decision (not implemented here).

theta <- alpha * theta + (1 - alpha) * theta_reinit, where theta_reinit comes from a freshly
constructed instance of the SAME module (built via `reinit_fn`, e.g. `lambda: GraphicEncoder
(hidden_size)`) — i.e. shrink the existing weights toward (rather than fully replace with) a
random reinitialization, which is gentler than a hard reset while still perturbing away from
whatever the optimizer had converged to.

Deliberately takes a constructor callback rather than guessing a reinit scheme from each
parameter's shape: a shape-based heuristic (e.g. "1-D params are biases, zero them") is wrong
for things like GroupNorm's weight, which is 1-D but should default-init to 1s, not 0 —
reusing the module's own __init__/reset_parameters via reinit_fn sidesteps that entirely.
"""

from typing import Callable, TypeVar

import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)


@torch.no_grad()
def shrink_and_perturb(module: M, reinit_fn: Callable[[], M], alpha: float) -> None:
    """
    module    : the submodule to reset in place (e.g. net.graphic_encoder).
    reinit_fn : zero-arg callable building a fresh instance of the same architecture; only
                its randomly-initialized weights are used.
    alpha     : how much of the OLD weights to keep (1.0 = no-op, 0.0 = full reinit).
    """
    assert 0.0 <= alpha <= 1.0
    reinit = reinit_fn()

    for p, p_reinit in zip(module.parameters(), reinit.parameters()):
        assert p.shape == p_reinit.shape, "reinit_fn must build the same architecture as module"
        p.data.mul_(alpha).add_(p_reinit.data, alpha=1.0 - alpha)
    for b, b_reinit in zip(module.buffers(), reinit.buffers()):
        if b.dtype.is_floating_point:
            b.data.mul_(alpha).add_(b_reinit.data, alpha=1.0 - alpha)
