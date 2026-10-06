"""QZ: faster MyModel student forwards (P20) for search and evaluation, docs/policy_v2_design_ko.md "QZ".

``NetPolicy(..., precision=...)`` (opt-in; the default ``"fp32"`` is the unchanged, bit-exact path) routes its
``_own`` -- the device part every caller goes through: ``NetPolicy.forward`` and B1's ``PipelinedRollout``
(``_net_forward``) -- through a ``FastCore``:

* ``fp16``   autocast fp16 (eager);
* ``fp16c``  autocast fp16 under ``torch.compile`` (static shapes: a batch is padded up to the next of
             ``FAST_BUCKETS`` with copies of its row 0, the pad outputs dropped, so a handful of shapes compile once).
             3.5x the fp32 forward on a 2080 Ti at 2048-8192 views (reports/qz_quant).

The policy keeps ``fp16=False`` (callers would otherwise wrap a second autocast around it).  A checkpoint may carry
a ``"precision"`` key (``fast_checkpoint``): ``NetPolicy.from_checkpoint`` then builds that precision, so a
``BLACKOUT_NETS`` alias or a search base path selects the fast net without code changes.
"""

from __future__ import annotations

PRECISIONS = ("fp32", "fp16", "fp16c")
FAST_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


class FastCore:
    def __init__(self, own, precision: str, device, buckets=FAST_BUCKETS):
        import torch

        if precision not in PRECISIONS or precision == "fp32":
            raise ValueError(f"precision must be one of {PRECISIONS[1:]}, got {precision!r}")
        self.torch, self.precision, self.device = torch, precision, device
        self.compiled = precision.endswith("c")
        self.buckets = tuple(sorted(int(b) for b in buckets)) if (buckets and self.compiled) else ()
        self.core = torch.compile(own, dynamic=False) if self.compiled else own

    def _pad(self, m: int) -> int:
        for b in self.buckets:
            if b >= m:
                return b
        return m

    def __call__(self, g, t, a):
        """Unpacked float maps [m, 13, H, W], team [m, 4], agents [m, 10, 12] (device) -> float32 logits
        (, offsets) [m, ...]."""
        torch = self.torch
        m = g.shape[0]
        size = self._pad(m)
        if size != m:
            idx = torch.zeros(size, dtype=torch.long, device=g.device)
            idx[:m] = torch.arange(m, device=g.device)
            g, t, a = g[idx], t[idx], a[idx]
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self.core(g, t, a)
        if isinstance(out, tuple):
            return tuple(x[:m].float() for x in out)
        return out[:m].float()


def fast_checkpoint(src: str, dst: str, precision: str, **extra) -> None:
    """``src`` (a distill checkpoint) re-saved with ``precision`` (+ ``extra`` keys) for ``from_checkpoint``."""
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=True)
    ck = {"policy_state": ck["policy_state"], "distill": ck.get("distill", {}), "precision": precision, **extra}
    torch.save(ck, dst)
