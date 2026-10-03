"""Device, precision, data-feeding and multi-GPU helpers shared by the offline trainers.

The trainers were tuned on Apple MPS; this module adds the CUDA paths (the poscat servers'
RTX 2080 Ti: Turing sm_75, fp16 tensor cores, no bf16 / TF32) without changing what MPS does.

* ``pick_device("auto")``: cuda > mps > cpu.  Under ``torchrun`` a cuda device is the rank's
  ``LOCAL_RANK`` GPU.
* ``autocast(device, amp)`` / ``grad_scaler(device, amp)``: ``amp`` in none / fp16 / bf16 (fp16
  needs the scaler; bf16 is refused on a GPU without it, it would be emulated and slow).
* ``adamw(...)``: the fused CUDA kernel on cuda (one launch for all parameters), the default
  elsewhere -- identical maths.
* ``load_npz(path)``: an *uncompressed* .npz (np.savez) as read-only memory maps, so a few
  ranks of one machine share one page-cache copy instead of each holding the arrays in RAM.
* ``Prefetcher``: builds batches in a background thread (numpy fancy indexing releases the
  GIL), pins them, and ``to_device`` copies with ``non_blocking``.
* ``pick_device`` on cuda also caps torch's CPU threads (``cuda_host_threads``).
* ``Dist``: torchrun's environment -> process group (nccl on cuda, gloo otherwise), rank /
  world, ``all_reduce_sum``, ``barrier``, ``wrap`` in DistributedDataParallel; a single
  process when not launched by torchrun (world 1, every call a no-op).
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import zipfile
from typing import Callable

import numpy as np
import torch

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def pick_device(name: str | None = "auto") -> torch.device:
    """``auto``/None: cuda (this rank's GPU under torchrun) > mps > cpu; otherwise ``name``."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", local_rank())
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True  # fixed input shapes: pick the fastest conv algorithm once
        cuda_host_threads()
    return device


CUDA_HOST_THREADS = 4


def cuda_host_threads() -> None:
    """Intra-op CPU threads for a CUDA trainer: CUDA_HOST_THREADS unless OMP_NUM_THREADS was set by
    hand (torchrun's own OMP_NUM_THREADS=1 does not count).  Measured on the shared 28-core servers
    (RolloutNet step, 2080 Ti): 1 thread halved the step rate on an idle host (host-side copies and
    CPU ops around the optimizer), while torch's default (one per core) made it 3-4x slower once
    the host was oversubscribed (load 60+: OpenMP teams spinning against everyone else); 2-4 was
    fast in both cases."""
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp is None or (omp == "1" and "TORCHELASTIC_RUN_ID" in os.environ):
        torch.set_num_threads(CUDA_HOST_THREADS)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def check_amp(device: torch.device, amp: str | None) -> str:
    amp = amp or "none"
    if amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError(f"{torch.cuda.get_device_name(device)} has no bf16: use --amp fp16")
    return amp


def autocast(device: torch.device, amp: str | None):
    dtype = AMP_DTYPES.get(amp or "none")
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def grad_scaler(device: torch.device, amp: str | None) -> torch.amp.GradScaler:
    """fp16's narrow exponent underflows small gradients: scale the loss (disabled = a no-op)."""
    return torch.amp.GradScaler(device.type, enabled=amp == "fp16")


def adamw(params, device: torch.device, **kwargs) -> torch.optim.AdamW:
    if device.type == "cuda":
        kwargs.setdefault("fused", True)
    return torch.optim.AdamW(params, **kwargs)


# ------------------------------------------------------------------------------ data


def load_npz(path, mmap: bool = True) -> dict[str, np.ndarray]:
    """Every array of an .npz; members stored uncompressed (np.savez) come back as read-only
    memory maps (``mmap``), compressed ones are read into memory."""
    path = str(path)
    if not mmap:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        for info in zf.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            if info.compress_type != zipfile.ZIP_STORED:
                with zf.open(info) as member:
                    out[name] = np.lib.format.read_array(member, allow_pickle=False)
                continue
            # local file header: 30 bytes + name + extra, then the stored .npy
            fh.seek(info.header_offset)
            header = fh.read(30)
            start = info.header_offset + 30 + int.from_bytes(header[26:28], "little") + int.from_bytes(header[28:30], "little")
            fh.seek(start)
            version = np.lib.format.read_magic(fh)
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read_header(fh)
            if dtype.hasobject:
                raise ValueError(f"{path}:{name} holds objects; cannot memory-map it")
            out[name] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape,
                                  order="F" if fortran else "C")
    return out


def pinned(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """A host tensor of ``array`` (pinned when the device is cuda, for an async copy)."""
    t = torch.from_numpy(np.ascontiguousarray(array))
    return t.pin_memory() if device.type == "cuda" else t


def to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """numpy arrays or (pinned) host tensors -> device tensors: asynchronous only from pinned
    memory (cuda); a pageable source is copied synchronously (MPS / CPU behave as before)."""
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v
        out[k] = t.to(device, non_blocking=device.type == "cuda" and t.is_pinned())
    return out


class Prefetcher:
    """``make()`` -> a dict of numpy arrays, called in a background thread ``depth`` batches ahead;
    ``next()`` returns it as host tensors (pinned on cuda: ``to_device`` then copies async).
    ``make`` must be thread-confined (own RNG); batches come out in the order they were made,
    so a seeded ``make`` gives the same batch sequence as calling it inline."""

    def __init__(self, make: Callable[[], dict], device: torch.device, depth: int = 4):
        self.make, self.device = make, device
        self.q: queue.Queue = queue.Queue(depth)
        self._stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                batch = {k: pinned(v, self.device) for k, v in self.make().items()}
                while not self._stop.is_set():
                    try:
                        self.q.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # surfaced in next()
            self.error = exc
            self.q.put(None)

    def next(self) -> dict[str, torch.Tensor]:
        batch = self.q.get()
        if batch is None:
            raise RuntimeError("prefetch thread failed") from self.error
        return batch

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------------------------ multi-GPU


class Dist:
    """torchrun's environment (RANK / WORLD_SIZE / LOCAL_RANK) or a single process."""

    def __init__(self, device: torch.device):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.device = device
        self.enabled = self.world > 1
        if self.enabled and not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            torch.distributed.init_process_group(backend, device_id=device if device.type == "cuda" else None)

    @property
    def main(self) -> bool:
        return self.rank == 0

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.enabled:
            return module
        ids = [self.device.index] if self.device.type == "cuda" else None
        return torch.nn.parallel.DistributedDataParallel(module, device_ids=ids, broadcast_buffers=False)

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            torch.distributed.all_reduce(t)
        return t

    def barrier(self):
        if self.enabled:
            torch.distributed.barrier()

    def close(self):
        if self.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """The model under DistributedDataParallel and/or torch.compile wrappers."""
    while True:
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        elif hasattr(module, "_orig_mod"):
            module = module._orig_mod
        else:
            return module
