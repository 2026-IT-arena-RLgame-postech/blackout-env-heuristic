"""Numerics gate for the GRPO forward-pass speedups: eager fp16 vs CUDA-graph replay vs different batch compositions.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/perf_numerics_check.py
"""
import numpy as np
import torch

from blackout_env.sim.batch import BatchRollout, NetPolicy, N_AGENT_COLS, H, W
from blackout_env.sim.graphfwd import GraphForward
from blackout_env.sim.state import SimState

CK = "checkpoints/distill/frozen/p22_v31.pt"
pol = NetPolicy.from_checkpoint(CK, device="cuda", fp16=True)
roots = [SimState.from_seed(s) for s in range(100, 164)]
res = BatchRollout(pol, pol, packed=True).run(roots, 400)
bs = res.bsim
n = bs.n
g = np.zeros((n, 2, H, W), np.int16); t = np.zeros((n, 2, 4), np.float32); a = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
bs.render_packed(np.arange(n), g, t, a)
G, T, A = g.reshape(-1, H, W), t.reshape(-1, 4), a.reshape(-1, 10, N_AGENT_COLS)   # 128 views
gf = GraphForward(pol.own, "cuda", angle=pol.angle)


def eager(sl):
    return gf.eager(G[sl], T[sl], A[sl]).cpu().numpy()


def graph(sl):
    return gf(G[sl], T[sl], A[sl]).cpu().numpy()


def rep(name, x, y):
    d = np.abs(x - y)
    kq = gf.kq
    ax, ay = x[..., :kq].argmax(-1), y[..., :kq].argmax(-1)
    print(f"{name:42s} max|d|={d.max():.3e}  bitwise_equal={np.array_equal(x, y)}  argmax_agree={(ax == ay).mean():.5f}")


s = slice(0, 40)
rep("eager vs eager (determinism), m=40", eager(s), eager(s))
rep("graph vs eager, m=40", graph(s), eager(s))
for m in (1, 3, 17, 40, 77, 128):
    rep(f"graph vs eager, m={m}", graph(slice(0, m)), eager(slice(0, m)))
rep("graph replay twice", graph(s), graph(s))
for m in (40, 64, 100):
    big = eager(slice(0, 128))[:m]
    rep(f"eager rows[:{m}]: batch {m} vs batch 128", eager(slice(0, m)), big)
bk = GraphForward(pol.own, "cuda", angle=pol.angle, bucket=64)
rep("bucket-64 graph vs eager, m=40", bk(G[s], T[s], A[s]).cpu().numpy(), eager(s))
rep("bucket-64 graph vs eager, m=100", bk(G[:100], T[:100], A[:100]).cpu().numpy(), eager(slice(0, 100)))
print("graphs:", len(gf._ents), "mem MB", torch.cuda.memory_allocated() / 1e6)
