"""Build a competition submission (official blackout-env docs/competition: ONNX, team mode, stateless).

    PYTHONPATH=. <venv with torch + onnx + onnxruntime>/python examples/build_competition_submission.py \\
        --checkpoint <distill checkpoint> --out dist/competition/<name> [--check-states 40]

Writes ``<out>/meta.json`` ({"mode": "team"}) and ``<out>/model.onnx``: official ``vector (5, 96)`` +
``graphic (5, 11, 96, 96)`` -> ``action (5, 2)`` (blackout_env/official/obs_mask.py ``OfficialTeamGraph``:
4x4 tile pooling, team B mirrored into the fork's canonical frame, enemy classes / battery amounts hidden,
the student, its greedy any-angle move back in the world frame).

A checkpoint trained without ``--official-obs`` is masked at export (it then sees less than it was trained
on); one trained with it is exported as it is.  ``--check-states N`` renders N sim states (both teams) in
the official format and compares onnxruntime against the torch graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _stub_missing(prefix: str) -> None:
    """The export venv has no mlagents_envs (its protobuf clashes with onnx's); the export never touches the env,
    so any ``prefix.*`` import resolves to an empty module whose attributes are dummy classes."""
    import importlib.abc
    import importlib.machinery
    import types

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {"__init__": lambda self, *a, **k: None})

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == prefix or name.startswith(prefix + "."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = _Mod(spec.name)
            m.__path__ = []
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.append(_Finder())


try:
    import mlagents_envs  # noqa: F401
except ImportError:
    _stub_missing("mlagents_envs")


class _PlainRMSNorm(torch.nn.Module):
    """nn.RMSNorm in basic ops (the TorchScript ONNX exporter has no aten::rms_norm at opset 17)."""

    def __init__(self, norm: torch.nn.RMSNorm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y


def _plain_rms_norms(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.RMSNorm):
            setattr(module, name, _PlainRMSNorm(child))
        else:
            _plain_rms_norms(child)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mode", type=int, default=2, help="mask for a full-information checkpoint: 1 hide classes too, 2 tracked")
    ap.add_argument("--check-states", type=int, default=40)
    ap.add_argument("--static", default=None, help="saved static_channels() tensor (export venv without the sim)")
    ap.add_argument("--check-only", action="store_true", help="only compare an existing <out>/model.onnx (sim venv)")
    a = ap.parse_args()

    from blackout_env.model.my_model import OFFICIAL_KEY, MyModel
    from blackout_env.official.obs_mask import OfficialTeamGraph, static_channels

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    state = dict(ck["policy_state"])
    if OFFICIAL_KEY not in state:
        print(f"[build] full-information checkpoint: masked at export (mode {a.mode})")
        state[OFFICIAL_KEY] = torch.full((1,), float(a.mode))
    net = MyModel.from_state_dict(state).eval()
    static = torch.load(a.static) if a.static else static_channels()
    graph = OfficialTeamGraph(net, static).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.check_only:
        _check(graph, out, a.check_states)
        return 0
    _plain_rms_norms(graph)
    for m in graph.modules():  # SDPA with enable_gqa has no opset-17 export: repeat K/V heads instead (same math)
        if hasattr(m, "_flash_gqa") and hasattr(m, "_repeat_kv_sdpa"):
            m._flash_gqa = m._repeat_kv_sdpa
    vec, gr = torch.zeros(5, 96), torch.zeros(5, 11, 96, 96)
    vec[:, 2] = 1.0
    with torch.no_grad():
        if graph.tracked:  # enemy classes tracked in the recurrent state (official state_in / state_out)
            torch.onnx.export(graph, (vec, gr, torch.zeros(5, graph.state_size)), str(out / "model.onnx"),
                              input_names=["vector", "graphic", "state_in"], output_names=["action", "state_out"],
                              opset_version=a.opset, dynamo=False)
        else:
            torch.onnx.export(graph, (vec, gr), str(out / "model.onnx"), input_names=["vector", "graphic"],
                              output_names=["action"], opset_version=a.opset, dynamo=False)
    (out / "meta.json").write_text(json.dumps({"mode": "team"}) + "\n")
    print(f"[build] wrote {out}/model.onnx ({(out / 'model.onnx').stat().st_size / 1e6:.1f} MB)")

    if a.check_states:
        _check(graph, out, a.check_states)
    return 0


def _check(graph, out, check_states):
    if check_states:
        import onnxruntime as ort

        from blackout_env.sim.batch import BatchSim, HeuristicPolicy
        from blackout_env.sim.state import SimState
        from blackout_env.official.render import official_preprocessed

        sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
        from blackout_env.official.obs_mask import STATE_SIZE
        heur = HeuristicPolicy("V34c")
        worst, n = 0.0, 0
        names = [i.name for i in sess.get_inputs()]
        for seed in range(3):
            heur.reset(1)
            bs = BatchSim([SimState.from_seed(100 + seed)])
            from blackout_env.sim.batch import N_AGENT_COLS, N_CH
            from blackout_env.sim.mapdata import H, W
            g = np.zeros((1, 2, N_CH, H, W), np.float32)
            t = np.zeros((1, 2, 4), np.float32)
            ag = np.zeros((1, 2, 10, N_AGENT_COLS), np.float32)
            mv = np.zeros((1, 10, 2), np.float32)
            ss = getattr(graph, "state_size", STATE_SIZE)
            st_onnx = [np.zeros((5, ss), np.float32) for _ in range(2)]
            st_torch = [torch.zeros(5, ss) for _ in range(2)]
            for step in range(1200):
                if bs.over[0] or n >= check_states:
                    break
                bs.render(np.array([0]), g, t, ag)
                stateful = "state_in" in names
                if stateful or step % 30 == 0:
                    st = bs.state(0)
                    for tm in (0, 1):
                        rows = [official_preprocessed(st, 5 * tm + k) for k in range(5)]
                        v = np.stack([r["vector"] for r in rows]).astype(np.float32)
                        gg = np.stack([np.moveaxis(r["graphic"], -1, 0) for r in rows]).astype(np.float32)
                        if stateful:
                            onnx_out, st_onnx[tm] = sess.run(None, {"vector": v, "graphic": gg, "state_in": st_onnx[tm]})[:2]
                            with torch.no_grad():
                                ref, st_torch[tm] = graph(torch.from_numpy(v), torch.from_numpy(gg), st_torch[tm])
                            ref = ref.numpy()
                        else:
                            onnx_out = sess.run(None, {"vector": v, "graphic": gg})[0]
                            with torch.no_grad():
                                ref = graph(torch.from_numpy(v), torch.from_numpy(gg)).numpy()
                        worst = max(worst, float(np.abs(onnx_out - ref).max()))
                    if step % 30 == 0:
                        n += 1
                mv[0] = 0
                for tm in (0, 1):
                    mv[0, 5 * tm:5 * tm + 5] = heur.moves(bs, np.array([0]), (tm,), (g, t, ag))[tm]
                bs.step(mv, np.array([0]))
        print(f"[build] onnxruntime vs torch on {n} states x 2 teams: max |diff| {worst:.2e}")


if __name__ == "__main__":
    sys.exit(main())
