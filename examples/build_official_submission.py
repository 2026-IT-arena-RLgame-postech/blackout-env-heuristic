"""Build the official-format submission (ist-tech-AI-games/blackout-env ``policy.py`` + ``checkpoint.pt``).

    PYTHONPATH=. <venv with torch, onnx, onnxscript, onnxruntime>/python examples/build_official_submission.py \\
        --checkpoint checkpoints/distill/frozen/p24c_v321.pt --states states.npz --out dist/official_submission

Writes to ``--out``:

* ``policy.py``      -- submission/official/policy.py (``BlackoutPolicy``);
* ``checkpoint.pt``  -- {"policy_state": {onnx, torchscript, sim_src[, base_net, rollout_net, value_net]}}
  (uint8 buffers; ``--search`` adds the search's nets);
* ``bo_fork/``       -- the vendored fork code: tracker + search (also zipped inside the checkpoint);
* ``policy.onnx``    -- the stateless graph alone (blackout_env/official/policy_graph.py).

``--states`` (npz with ``raw`` [N, 44] / ``map`` [N, 24, 24] fork broadcasts) checks the ONNX and
TorchScript graphs against the numpy ``MyPolicy`` path for both teams before anything is written.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "bo_fork"   # the vendored fork code's name (the official package is also called blackout_env)
STUBS = ("blackout_env", "blackout_env.model", "blackout_env.official", "blackout_env.env", "blackout_env.heuristics")   # package __init__s not vendored
ROOTS = ("blackout_env.official.tracker", "blackout_env.official.seed_search", "blackout_env.official.render",
         "blackout_env.official.search_player")
JIT_SHIM = '''"""numba when installed, else plain Python (same results, slower)."""
import os

try:
    if os.environ.get("BO_SIM_NO_NUMBA") == "1":
        raise ImportError
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)
'''
INIT = '''"""Vendored blackout-env fork (branch feat/official-obs): the official-obs tracker + the R6 search."""
import os
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
if not os.access(_here, os.W_OK):  # numba's on-disk cache needs a writable place
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "bo_fork_numba_cache"))

from ._jit import HAVE_NUMBA  # noqa: E402
from .official.render import from_preprocessed, parse_vector, team_of_preprocessed  # noqa: E402
from .official.seed_search import RngFinder  # noqa: E402
from .official.tracker import MultiTracker, elapsed_from_time_left  # noqa: E402


def new_tracker():
    """Blind-capable tracker: RNG chain from InitState(0), small-seed search when a map is shown."""
    return MultiTracker(RngFinder(full_scan=False))


def elapsed(raw) -> int:
    """Ticks played in the match, from the official raw vector's time left."""
    return int(elapsed_from_time_left(parse_vector(raw)["time_left"]))


def search_player(base, rollout, value, device="cpu"):
    """The R6 search player (official/search_player.py); file-like checkpoints."""
    from .official.search_player import SearchPlayer

    return SearchPlayer(base, rollout, value, device=device)
'''


def closure(search_nets=None) -> list[str]:
    """Every blackout_env module the tracker + search import (lazy imports included: a SearchPlayer is
    built from ``search_nets`` (base, rollout, value paths) and searches once)."""
    _bare_packages()
    import importlib

    for m in ROOTS:
        importlib.import_module(m)
    from blackout_env.official.search_player import SearchPlayer  # noqa: F401  (lazy imports below)
    from blackout_env.sim.state import SimState
    from blackout_env.value.model import load_value_net  # noqa: F401
    import blackout_env.model.my_model  # noqa: F401
    import blackout_env.sim.pipeline  # noqa: F401

    sim = SimState.from_seed(1)
    if search_nets:
        from blackout_env.official.search_player import SearchPlayer as _SP

        player = _SP(*search_nets)
        player.act(sim, 0, player.search_from, np.zeros((5, 2), np.float32), 1)
        player.act(sim, 0, player.search_from + 1, np.zeros((5, 2), np.float32), 1)
    return sorted(k for k in sys.modules if k.startswith("blackout_env.") or k == "blackout_env")


def vendor(dst: Path, search_nets=None) -> bytes:
    """Write bo_fork/ (the closure, imports renamed, numba optional) to ``dst``; return it zipped."""
    pkg = dst / PACKAGE
    if pkg.exists():
        shutil.rmtree(pkg)
    for name in closure(search_nets):
        mod = sys.modules[name]
        rel = name.split(".")[1:]
        is_pkg = hasattr(mod, "__path__")
        out = pkg.joinpath(*rel, "__init__.py") if is_pkg else pkg.joinpath(*rel[:-1], rel[-1] + ".py")
        out.parent.mkdir(parents=True, exist_ok=True)
        if name in STUBS or getattr(mod, "__file__", None) is None:   # stubbed / namespace packages
            s = INIT if name == "blackout_env" else ""
        else:
            s = Path(mod.__file__).read_text()
            s = re.sub(r"\bblackout_env\b(?=[.\s])", PACKAGE, s)
            s = re.sub(r"^(\s*)from numba import ([\w, ]+)$", r"\1from %s._jit import \2" % PACKAGE, s, flags=re.M)
            s = s.replace(f"try:\n    from {PACKAGE}._jit import njit\nexcept ImportError:  # pragma: no cover\n"
                          "    def njit(*a, **k):\n        return (lambda f: f) if not a or not callable(a[0]) else a[0]\n",
                          f"from {PACKAGE}._jit import njit\n")
        out.write_text(s)
    (pkg / "_jit.py").write_text(JIT_SHIM)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*.py")):
            z.write(f, str(f.relative_to(dst)))
    return buf.getvalue()


def _bare_packages():
    """Import blackout_env.* modules without the package __init__s (they pull in mlagents / gymnasium)."""
    for name in STUBS:
        m = types.ModuleType(name)
        m.__path__ = [str(ROOT / name.replace(".", "/"))]
        sys.modules.setdefault(name, m)


def build_graph(checkpoint: str):
    import torch

    _bare_packages()
    from blackout_env.model.my_model import MyModel
    from blackout_env.official.policy_graph import ForkPolicyGraph

    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["policy_state"]
    net = MyModel.from_state_dict(sd)
    net.load_state_dict(sd)
    return ForkPolicyGraph(net.eval()).eval()


def export(graph, example, onnx_path: Path) -> tuple[bytes, bytes]:
    import torch

    torch.onnx.export(graph, example, str(onnx_path), input_names=["raw_state", "raw_map", "team"],
                      output_names=["moves", "logits"], opset_version=18, dynamo=True, external_data=False)
    with torch.no_grad():
        traced = torch.jit.trace(graph, example, check_trace=False)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return onnx_path.read_bytes(), buf.getvalue()


def check(graph, onnx_bytes: bytes, ts_bytes: bytes, states: str, n: int) -> dict:
    """max |move - torch graph| per backend and the arg max flips (> 1e-3) over ``n`` states x 2 teams."""
    import onnxruntime as ort
    import torch

    d = np.load(states)
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    script = torch.jit.load(io.BytesIO(ts_bytes))
    idx = np.linspace(0, len(d["raw"]) - 1, min(n, len(d["raw"]))).astype(int)
    out = {"states": int(len(idx)), "onnx_err": 0.0, "onnx_flips": 0, "ts_err": 0.0, "ts_flips": 0}
    for i in idx:
        for team in (0, 1):
            rs, rm = d["raw"][i].astype(np.float32), d["map"][i].astype(np.int64)
            args = (torch.from_numpy(rs), torch.from_numpy(rm), torch.tensor(team))
            with torch.no_grad():
                ref = graph(*args)[0].numpy()
                ts = script(*args)[0].numpy()
            ox = sess.run(["moves"], {"raw_state": rs, "raw_map": rm, "team": np.array(team, np.int64)})[0]
            for key, got in (("onnx", ox), ("ts", ts)):
                e = np.abs(got - ref).max(axis=1)
                out[f"{key}_flips"] += int((e > 1e-3).sum())
                out[f"{key}_err"] = max(out[f"{key}_err"], float(e[e <= 1e-3].max(initial=0.0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/distill/frozen/p24c_v321.pt")
    ap.add_argument("--states", required=True)
    ap.add_argument("--n-check", type=int, default=400)
    ap.add_argument("--out", default="dist/official_submission")
    ap.add_argument("--search", action="store_true", help="also pack the R6 search nets (rollout r24cd256, leaf V iter2)")
    ap.add_argument("--rollout", default="checkpoints/rollout_net/r6/r24cd256.pt")
    ap.add_argument("--value", default="checkpoints/value/s2v_f3xd2/iter2.pt")
    args = ap.parse_args()

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph = build_graph(args.checkpoint)
    d = np.load(args.states)
    example = (torch.from_numpy(d["raw"][0].astype(np.float32)), torch.from_numpy(d["map"][0].astype(np.int64)),
               torch.tensor(1))
    onnx_bytes, ts_bytes = export(graph, example, out / "policy.onnx")
    report = check(graph, onnx_bytes, ts_bytes, args.states, args.n_check)
    print("graph check:", report)
    if report["onnx_flips"] or report["ts_flips"] or max(report["onnx_err"], report["ts_err"]) > 1e-4:
        raise SystemExit("exported graph disagrees with the torch graph")
    sim_zip = vendor(out, (args.checkpoint, args.rollout, args.value))
    shutil.copy(ROOT / "submission/official/policy.py", out / "policy.py")
    files = [("onnx", onnx_bytes), ("torchscript", ts_bytes), ("sim_src", sim_zip)]
    if args.search:  # the search's own copies of the nets (torch: batched rollouts)
        files += [(k, Path(f).read_bytes()) for k, f in (("base_net", args.checkpoint), ("rollout_net", args.rollout),
                                                          ("value_net", args.value))]
    state = {name: torch.frombuffer(bytearray(b), dtype=torch.uint8).clone() for name, b in files}
    torch.save({"policy_state": state, "source": {"checkpoint": str(args.checkpoint),
                                                  **({"rollout": args.rollout, "value": args.value} if args.search else {})}},
               out / "checkpoint.pt")
    print({k: f"{v.numel() / 1e6:.2f} MB" for k, v in state.items()}, "->", out)


if __name__ == "__main__":
    main()
