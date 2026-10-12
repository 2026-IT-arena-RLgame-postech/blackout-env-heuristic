"""BlackOut submission (ist-tech-AI-games/blackout-env ``policy.py`` format): the P24c student
playing from the official observations alone.

    model = load_checkpoint(BlackoutPolicy, "checkpoint.pt", vector_size=96, n_channels=11)

The official obs leave out what the student reads (battery amounts, the other units' classes,
the absorption timer, site / spawn tiles), so each call runs two stages:

1. **tracker** (stateful, Python + numba when installed): the rule-exact BlackOut simulator,
   driven by the observed positions, with the match's RNG stream recovered (no map needed:
   InitState(0) chain) -- renders the fork broadcast (raw_state [44] + packed map [24, 24]);
2. **graph** (stateless, ONNX): fork obs decoding, Team B mirroring, the P24c net, the arg max
   heading + angle offset, back to world (dx, dy) for the 5 own units.

3. **search** (only when the checkpoint carries the search nets, docs §61 ``o4f_r24c``): from 5 s,
   every 25 decisions the r24cd256 rollout net searches 8 branches x 4 opponent variants from the
   tracker's state (leaf V s2v_f3xd2 iter2); the chosen branch's units follow it for 25 decisions,
   the others P24c.  ``BLACKOUT_POLICY_SEARCH=0`` turns it off; ``BLACKOUT_SEARCH_DEVICE`` picks
   the device (default cuda when available).

``checkpoint.pt`` = {"policy_state": {onnx, torchscript, sim_src[, base_net, rollout_net,
value_net]}} -- uint8 buffers holding the ONNX graph, the same graph as TorchScript (used when
onnxruntime is missing), the zipped fork code (tracker + search; used when no ``bo_fork/`` sits
next to this file) and, for the search version, the three torch checkpoints.  Only numpy + torch
are required; onnxruntime and numba make it faster (numba: much faster for the search).

Decisions: every second tick of a match (the cadence the student was trained at), each move held
for the tick in between.  Rows are the team's units in order (run_match batches ``env.agents``,
unit_0..4 / unit_5..9).  One instance may play both sides, also in the same match (one tracker, fed once per tick).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BUFFERS = ("onnx", "torchscript", "sim_src", "base_net", "rollout_net", "value_net")
DECIDE_EVERY = 2


class BlackoutPolicy(nn.Module):
    def __init__(self, vector_size: int = 96, n_channels: int = 11, **_):
        super().__init__()
        if vector_size != 96 or n_channels != 11:
            raise ValueError(f"built for vector_size 96 / n_channels 11, got {vector_size} / {n_channels}")
        for name in BUFFERS:
            self.register_buffer(name, torch.zeros(0, dtype=torch.uint8))
        self._graph = None
        self._sim = None
        self._search = None
        self.reset()

    # the buffers' sizes are only known from the checkpoint
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in BUFFERS:
            if prefix + name not in state_dict:   # a plain checkpoint has no search nets
                state_dict[prefix + name] = torch.zeros(0, dtype=torch.uint8)
            setattr(self, name, torch.zeros_like(state_dict[prefix + name]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    # ------------------------------------------------------------------ lazy runtime
    def _runtime(self):
        if self._sim is None:
            self._sim = _import_sim(bytes(self.sim_src.cpu().numpy()))
        if self._graph is None:
            self._graph = _Graph(bytes(self.onnx.cpu().numpy()), bytes(self.torchscript.cpu().numpy()))
        if self._search is None and self.rollout_net.numel() and os.environ.get("BLACKOUT_POLICY_SEARCH", "1") == "1":
            dev = os.environ.get("BLACKOUT_SEARCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            nets = [io.BytesIO(bytes(b.cpu().numpy())) for b in (self.base_net, self.rollout_net, self.value_net)]
            self._search = self._sim.search_player(*nets, device=dev)
        return self._sim, self._graph

    def reset(self) -> None:
        """Forget everything (not needed between matches: a new one is detected from the timer;
        the tracker must keep its RNG chain across them)."""
        self._tracker, self._elapsed, self._start, self._moves = None, None, 0, {}
        if getattr(self, "_search", None) is not None:
            self._search.new_match()

    # ------------------------------------------------------------------ official interface
    def forward(self, vector: torch.Tensor, graphic: torch.Tensor) -> torch.Tensor:
        """vector (B, 96), graphic (B, 11, 96, 96) of one team's agents -> actions (B, 2)."""
        sim, graph = self._runtime()
        v = vector.detach().float().cpu().numpy()
        team = sim.team_of_preprocessed(v[0])
        try:
            raw, ids = sim.from_preprocessed(v[0], graphic[0].detach().cpu().numpy(), 5 * team)
            el = sim.elapsed(raw)
            if self._tracker is None:
                self._tracker = sim.new_tracker()
            if self._elapsed is None or el < self._elapsed:   # a new match
                self._start, self._moves = el, {}
                if self._search is not None:
                    self._search.new_match()
            if el != self._elapsed:   # once per tick (one instance playing both sides sees each tick twice)
                self._tracker.observe(raw, ids)
                self._elapsed = el
            if (el - self._start) % DECIDE_EVERY == 0 or team not in self._moves:
                moves = graph(self._tracker.raw_state(), self._tracker.raw_map(), team)
                if self._search is not None:
                    decision = (el - self._start) // DECIDE_EVERY
                    seed = int(self._tracker.sim.rng[0]) * 7919 + decision * 104729 + 12345 + team * 31337
                    moves = self._search.act(self._tracker.sim, team, decision, moves, seed)
                self._moves[team] = moves
        except Exception as e:  # never lose a match to a tracker fault: restart it, keep the last moves
            warnings.warn(f"BlackoutPolicy: tracker reset after {type(e).__name__}: {e}")
            self._tracker, self._elapsed = None, None
        out = self._moves.get(team, np.zeros((5, 2), np.float32))[: v.shape[0]]
        return torch.as_tensor(out, dtype=torch.float32, device=vector.device)


class _Graph:
    """The stateless half: ONNX (onnxruntime) or the identical TorchScript trace."""

    def __init__(self, onnx_bytes: bytes, ts_bytes: bytes):
        self.session = None
        if onnx_bytes and os.environ.get("BLACKOUT_POLICY_BACKEND", "onnx") == "onnx":
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.environ.get("BLACKOUT_POLICY_THREADS", "1"))
                self.session = ort.InferenceSession(onnx_bytes, opts, providers=["CPUExecutionProvider"])
            except ImportError:
                pass
        self.script = None if self.session is not None else torch.jit.load(io.BytesIO(ts_bytes), map_location="cpu")
        self.backend = "onnx" if self.session is not None else "torchscript"

    def __call__(self, raw_state, raw_map, team: int) -> np.ndarray:
        rs = np.ascontiguousarray(raw_state, dtype=np.float32)
        rm = np.ascontiguousarray(raw_map, dtype=np.int64)
        tm = np.array(team, dtype=np.int64)
        if self.session is not None:
            return self.session.run(["moves"], {"raw_state": rs, "raw_map": rm, "team": tm})[0]
        with torch.no_grad():
            return self.script(torch.from_numpy(rs), torch.from_numpy(rm), torch.from_numpy(tm))[0].numpy()


def _import_sim(zipped: bytes):
    """``bo_fork`` next to this file, else the copy zipped in the checkpoint (extracted once)."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = None
    if here is not None and (here / "bo_fork" / "__init__.py").is_file():
        root = here
    else:
        if not zipped:
            raise RuntimeError("no bo_fork/ next to policy.py and none in the checkpoint")
        root = Path(tempfile.gettempdir()) / f"bo_fork_{hashlib.sha1(zipped).hexdigest()[:12]}"
        if not (root / "bo_fork" / "__init__.py").is_file():
            tmp = Path(tempfile.mkdtemp(dir=tempfile.gettempdir()))
            zipfile.ZipFile(io.BytesIO(zipped)).extractall(tmp)
            try:
                os.replace(tmp, root)
            except OSError:  # another process won the race
                pass
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import bo_fork

    return bo_fork
