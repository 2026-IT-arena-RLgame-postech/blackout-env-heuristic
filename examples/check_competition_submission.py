"""Our re-implementation of the official check_submission.py checks (team mode): meta.json, input names /
dtypes / shapes, output (5, 2), finite outputs, padded rows (an absent unit), time per step.

    PYTHONPATH=<onnxruntime> python examples/check_competition_submission.py <submission dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sub = Path(sys.argv[1])
meta = json.loads((sub / "meta.json").read_text())
assert meta == {"mode": "team"}, meta
sess = ort.InferenceSession(str(sub / "model.onnx"), providers=["CPUExecutionProvider"])
ins = {i.name: i for i in sess.get_inputs()}
states = {n: i for n, i in ins.items() if n.startswith("state_in")}
ins = {n: i for n, i in ins.items() if n not in states}
outs = {o.name: o for o in sess.get_outputs()}
for n, i in states.items():  # every state_in* needs its state_out* of the same shape, float32 (B, fixed...)
    o = outs.get("state_out" + n[len("state_in"):])
    assert o is not None and list(o.shape) == list(i.shape) and i.type == "tensor(float)" and i.shape[0] == 5, (n, i.shape, o)
assert set(ins) <= {"vector", "graphic", "slot"} and ins, ins
want = {"vector": ("tensor(float)", [5, 96]), "graphic": ("tensor(float)", [5, 11, 96, 96]), "slot": ("tensor(int64)", [5])}
for n, i in ins.items():
    assert i.type == want[n][0] and list(i.shape) == want[n][1], (n, i.type, i.shape)
out = next(o for o in sess.get_outputs() if not o.name.startswith("state_out"))
print("inputs", {n: (i.type, i.shape) for n, i in ins.items()}, "states", {n: i.shape for n, i in states.items()},
      "action output", out.name, out.shape)
state = {n: np.zeros(i.shape, np.float32) for n, i in states.items()}
names = [o.name for o in sess.get_outputs()]


def run(v, g):
    feed = {k: x for k, x in (("vector", v), ("graphic", g)) if k in ins}
    feed.update(state)
    res = dict(zip(names, sess.run(None, feed)))
    for n in states:
        state[n] = res["state_out" + n[len("state_in"):]]
    return res[out.name]
rng = np.random.default_rng(0)
times = []
for _ in range(200):
    v = rng.uniform(-1, 1, (5, 96)).astype(np.float32)
    g = np.repeat((rng.random((1, 11, 96, 96)) < 0.1).astype(np.float32), 5, 0)
    t0 = time.perf_counter()
    a = run(v, g)
    times.append((time.perf_counter() - t0) * 1000)
    assert a.shape == (5, 2) and np.isfinite(a).all(), a
v[0] = 0; g[0] = 0  # slot 0 absent (zero-padded by the runner)
a = run(v, g)
assert a.shape == (5, 2) and np.isfinite(a).all()
t = np.array(times)
print(f"time per step (CPU): mean {t.mean():.2f} ms p99 {np.percentile(t, 99):.2f} ms max {t.max():.2f} ms (limit 100)")
print("OK")
