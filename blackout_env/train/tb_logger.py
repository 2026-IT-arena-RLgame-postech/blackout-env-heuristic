"""
Thin TensorBoard writer wrapper for QMIXTrainer.

Kept as a separate, self-contained module so that adding or changing what gets logged never
requires touching forward/backward/training math itself -- QMIXTrainer only ever passes in
already-computed values (losses, td-errors, rewards, schedule outputs) or named submodules (for
weight/gradient norms); this module owns all SummaryWriter calls and the norm-computation
helpers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from torch import nn


class TBLogger:
    """
    No-op-safe wrapper: every method is a no-op when `enabled=False` (or when
    `torch.utils.tensorboard` / the `tensorboard` package isn't installed), so callers don't
    need to guard every call site with an `if` -- just construct with `enabled=False` to fully
    disable logging.

    Auto-disables on any write failure instead of raising. SummaryWriter's actual file I/O
    happens on a background thread (EventFileWriter); if its target directory disappears mid-run
    (e.g. another process/run sharing the same log_dir wipes it, or a filesystem hiccup) that
    thread dies and *every subsequent* add_scalar call re-raises the same exception from the
    dead thread -- left unhandled, that takes down the whole training process over a logging
    side-channel that was never supposed to be load-bearing. One failure here prints a warning
    once and turns TB logging off for the rest of the run instead.
    """

    def __init__(self, log_dir: str | Path | None, enabled: bool = True) -> None:
        self.enabled = enabled and log_dir is not None
        self._writer = None
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter  # local import: optional dependency

                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(log_dir))
            except Exception as e:  # noqa: BLE001 -- logging setup must never block training
                print(f"[TBLogger] failed to initialize (log_dir={log_dir!r}): {e!r} -- disabling TB logging.", file=sys.stderr)
                self.enabled = False
                self._writer = None

    def _guard(self, fn) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- see class docstring
            print(f"[TBLogger] write failed ({e!r}) -- disabling TB logging for the rest of this run.", file=sys.stderr)
            self.enabled = False
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._guard(lambda: self._writer.add_scalar(tag, value, step))

    def scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        """Logs each entry of `values` as its own scalar under `{prefix}/{name}`."""
        for name, value in values.items():
            if value is not None:
                self.scalar(f"{prefix}/{name}", value, step)

    # ------------------------------------------------------------------
    # Per-part weight / gradient norms
    # ------------------------------------------------------------------

    @staticmethod
    def _module_norm(module: nn.Module) -> float:
        total = 0.0
        for p in module.parameters():
            total += float(p.detach().pow(2).sum())
        return total**0.5

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float | None:
        total = 0.0
        found = False
        for p in module.parameters():
            if p.grad is None:
                continue
            found = True
            total += float(p.grad.detach().pow(2).sum())
        return total**0.5 if found else None

    def weight_norms(self, prefix: str, named_modules: dict[str, nn.Module], step: int) -> None:
        """L2 norm of every parameter in each named module -- call any time (no grad needed)."""
        for name, module in named_modules.items():
            self.scalar(f"{prefix}/{name}", self._module_norm(module), step)

    def grad_norms(self, prefix: str, named_modules: dict[str, nn.Module], step: int) -> None:
        """
        L2 norm of every parameter's `.grad` in each named module. Call this AFTER
        `loss.backward()` but BEFORE `clip_grad_norm_` (which mutates grads in place) so the
        logged values reflect the raw, pre-clip gradient magnitude -- the useful diagnostic for
        "is this part's gradient exploding/vanishing", not "what did clipping leave behind".
        Modules with no grad yet (e.g. never touched by this loss) are silently skipped.
        """
        for name, module in named_modules.items():
            norm = self._module_grad_norm(module)
            if norm is not None:
                self.scalar(f"{prefix}/{name}", norm, step)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        # Deferred via lambda, not `self._writer.flush` directly: the latter evaluates
        # `self._writer` (which may already be None from a prior failure) BEFORE _guard's
        # try/except runs, defeating the guard entirely.
        self._guard(lambda: self._writer.flush())

    def close(self) -> None:
        if self.enabled:
            self._guard(lambda: self._writer.close())


def default_run_dir(base: str = "runs") -> str:
    """A fresh timestamped subdirectory under `base`, unique per process.

    Used as QMIXConfig.tb_log_dir's default so two training runs (e.g. two sessions/processes
    launched around the same time) never share the exact same log_dir -- writing into a shared
    top-level `runs/` let one run's startup/cleanup step delete the directory out from under
    another run's already-open SummaryWriter, which is what crashed training in the first place.
    `tensorboard --logdir runs` still shows every subdirectory as its own comparable run.
    """
    import os

    return f"{base}/{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"
