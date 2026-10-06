"""QZ: faster MyModel student forwards (P20) for search and evaluation, docs/policy_v2_design_ko.md "QZ".

``NetPolicy(..., precision=...)`` (opt-in; the default ``"fp32"`` is the unchanged, bit-exact path) routes its
``_own`` -- the device part every caller goes through: ``NetPolicy.forward`` and B1's ``PipelinedRollout``
(``_net_forward``) -- through a ``FastCore``:

* ``fp16``   autocast fp16 (eager);
* ``fp16c``  autocast fp16 with the net's parts under ``torch.compile`` (``compile_parts``; static shapes: a batch is
             padded up to the next of ``FAST_BUCKETS`` with copies of its row 0, the pad outputs dropped, so a handful
             of shapes compile once).  ~3x the fp32 forward on a 2080 Ti at 2048-8192 views (reports/qz_quant).

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
        self.core = compile_parts(own) if self.compiled else own

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


COMPILED_PARTS = ("graphic_encoder", "vector_encoder", "attention", "q_head")


def compile_parts(own):
    """A copy of ``own`` (an ``_OwnQ`` over a MyModel) with ``COMPILED_PARTS`` compiled one by one (static shapes).

    Not the whole forward: inductor (torch 2.11, fp16 autocast) miscompiles MyModel's trunk when the attention
    stack shares a graph with the token assembly before it -- arg max agreement with fp32 0.75-0.90, NaN logits on
    some rows -- while each part compiled alone matches the eager fp16 forward (agreement 0.997, reports/qz_quant).
    The parts cost ~7 ms of the whole-graph 37.6 ms at 2048 views."""
    import copy

    import torch

    from blackout_env.model.my_model import MyModel

    own = copy.deepcopy(own)
    net = getattr(own, "net", None)
    if not isinstance(net, MyModel):
        raise ValueError("fp16c compiles a MyModel student's parts; got " + type(net).__name__)
    for name in COMPILED_PARTS:
        setattr(net, name, torch.compile(getattr(net, name), dynamic=False))
    return own


def fast_checkpoint(src: str, dst: str, precision: str, **extra) -> None:
    """``src`` (a distill checkpoint) re-saved with ``precision`` (+ ``extra`` keys) for ``from_checkpoint``."""
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=True)
    ck = {"policy_state": ck["policy_state"], "distill": ck.get("distill", {}), "precision": precision, **extra}
    torch.save(ck, dst)
