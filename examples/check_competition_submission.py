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
assert set(ins) <= {"vector", "graphic", "slot"} and ins, ins
want = {"vector": ("tensor(float)", [5, 96]), "graphic": ("tensor(float)", [5, 11, 96, 96]), "slot": ("tensor(int64)", [5])}
for n, i in ins.items():
    assert i.type == want[n][0] and list(i.shape) == want[n][1], (n, i.type, i.shape)
out = sess.get_outputs()[0]
print("inputs", {n: (i.type, i.shape) for n, i in ins.items()}, "output", out.name, out.shape)
rng = np.random.default_rng(0)
times = []
for _ in range(200):
    v = rng.uniform(-1, 1, (5, 96)).astype(np.float32)
    g = np.repeat((rng.random((1, 11, 96, 96)) < 0.1).astype(np.float32), 5, 0)
    t0 = time.perf_counter()
    a = sess.run(None, {k: x for k, x in (("vector", v), ("graphic", g)) if k in ins})[0]
    times.append((time.perf_counter() - t0) * 1000)
    assert a.shape == (5, 2) and np.isfinite(a).all(), a
v[0] = 0; g[0] = 0  # slot 0 absent (zero-padded by the runner)
a = sess.run(None, {k: x for k, x in (("vector", v), ("graphic", g)) if k in ins})[0]
assert a.shape == (5, 2) and np.isfinite(a).all()
t = np.array(times)
print(f"time per step (CPU): mean {t.mean():.2f} ms p99 {np.percentile(t, 99):.2f} ms max {t.max():.2f} ms (limit 100)")
print("OK")
