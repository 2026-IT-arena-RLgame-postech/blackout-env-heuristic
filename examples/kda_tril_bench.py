"""(I + A)^{-1} rhs for strictly lower-triangular A: the KDA chunk-solve candidates, their fp32 error
on (nearly) repeated keys and MPS speed.  Standalone copies of the methods compared on 2026-10-01
(blackout_env/model/modules/kda.py keeps the chosen ones).

    .venv/bin/python examples/kda_tril_bench.py
"""
import time, torch, torch.nn.functional as F

def doubling(A, rhs):
    C = A.shape[-1]; eye = torch.eye(C, dtype=A.dtype, device=A.device)
    power = -A; inv = eye + power
    for _ in range(max(0, (C - 1).bit_length() - 1)):
        power = power @ power; inv = inv + inv @ power
    return inv @ rhs

def blocked(b):
    def f(A, rhs):
        C = A.shape[-1]; parts = []
        for i in range(0, C, b):
            j = min(C, i + b); r = rhs[..., i:j, :]
            if i: r = r - A[..., i:j, :i] @ torch.cat(parts, -2)
            parts.append(doubling(A[..., i:j, i:j], r))
        return torch.cat(parts, -2)
    return f

def rowsub(A, rhs):
    rows = []
    for i in range(A.shape[-1]):
        r = rhs[..., i, :]
        if i: r = r - (A[..., i, :i].unsqueeze(-2) @ torch.stack(rows, -2)).squeeze(-2)
        rows.append(r)
    return torch.stack(rows, -2)

def _inv_rowsub(A):
    """(I + A)^{-1} of a small block by row forward substitution (fla's 16x16 kernel)."""
    C = A.shape[-1]; eye = torch.eye(C, dtype=A.dtype, device=A.device).expand(A.shape).clone()
    rows = [eye[..., 0, :]]
    for i in range(1, C):
        rows.append(eye[..., i, :] - (A[..., i, :i].unsqueeze(-2) @ torch.stack(rows, -2)).squeeze(-2))
    return torch.stack(rows, -2)

def fla16(A, rhs, b=16):
    """Diagonal 16x16 inverses by row substitution, merged block-wise:
    X_ij = -X_ii * sum_{k<i} A_ik X_kj (block forward substitution on the inverse)."""
    C = A.shape[-1]; n = C // b
    Ab = A.reshape(*A.shape[:-2], n, b, n, b).transpose(-3, -2)  # [..., n, n, b, b]
    diag = _inv_rowsub(torch.stack([Ab[..., i, i, :, :] for i in range(n)], -3))  # [..., n, b, b]
    rb = rhs.reshape(*rhs.shape[:-2], n, b, rhs.shape[-1])
    out = []
    for i in range(n):
        r = rb[..., i, :, :]
        for k in range(i): r = r - Ab[..., i, k, :, :] @ out[k]
        out.append(diag[..., i, :, :] @ r)
    return torch.cat(out, -2)

def tree(b):
    def f(A, rhs):
        """Inverse of the b x b diagonal blocks by doubling (all at once), then merged pairwise
        b -> 2b -> ... -> C with [[X11, 0], [-X22 A21 X11, X22]] -- log2(C/b) batched steps."""
        C = A.shape[-1]; lead = A.shape[:-2]
        n = C // b
        blocks = A.reshape(*lead, n, b, n, b).diagonal(dim1=-4, dim2=-2).permute(*range(len(lead)), -1, -3, -2)  # [..., n, b, b]
        eye = torch.eye(b, dtype=A.dtype, device=A.device)
        power = -blocks; inv = eye + power
        for _ in range(max(0, (b - 1).bit_length() - 1)):
            power = power @ power; inv = inv + inv @ power
        size = b
        while size < C:  # merge pairs of size x size inverses
            m = inv.shape[-3] // 2
            x11, x22 = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
            starts = torch.arange(m, device=A.device) * 2 * size
            a21 = torch.stack([A[..., s + size:s + 2 * size, s:s + size] for s in starts.tolist()], -3)
            x21 = -(x22 @ a21 @ x11)
            top = torch.cat((x11, torch.zeros_like(x11)), -1)
            bottom = torch.cat((x21, x22), -1)
            inv = torch.cat((top, bottom), -2)
            size *= 2
        return inv[..., 0, :, :] @ rhs
    return f

def solve(A, rhs):
    C = A.shape[-1]; eye = torch.eye(C, dtype=A.dtype, device=A.device)
    return torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)

CANDS = {"solve": solve, "doubling": doubling, "blocked8": blocked(8), "blocked16": blocked(16), "fla16": fla16, "tree8": tree(8), "tree16": tree(16)}

def make(kind, beta, n=352, C=64, dk=64, dv=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "random": k = F.normalize(torch.randn(n, C, dk, generator=g), dim=-1)
    elif kind == "repeat": k = F.normalize(torch.ones(n, C, dk) + 1e-3 * torch.randn(n, C, dk, generator=g), dim=-1)
    else:  # "mixed": runs of repeated keys (units standing then moving)
        base = F.normalize(torch.randn(n, C // 8, dk, generator=g), dim=-1).repeat_interleave(8, 1)
        k = F.normalize(base + 0.02 * torch.randn(n, C, dk, generator=g), dim=-1)
    b = torch.full((n, C, 1), beta)
    A = torch.tril((b * k) @ k.transpose(-1, -2), -1)
    rhs = torch.randn(n, C, dv, generator=g)
    return A, rhs

print(f"{'data':14s} " + " ".join(f"{c:>10s}" for c in CANDS))
for kind, beta in (("random", 0.5), ("random", 1.0), ("mixed", 0.9), ("repeat", 0.5), ("repeat", 0.9), ("repeat", 1.0)):
    A, rhs = make(kind, beta, n=64)
    exact = solve(A.double(), rhs.double())
    scale = exact.abs().max().item()
    errs = []
    for name, f in CANDS.items():
        out = f(A.to("mps"), rhs.to("mps")).cpu().double()
        errs.append((out - exact).abs().max().item() / scale)
    print(f"{kind+' b='+str(beta):14s} " + " ".join(f"{e:10.1e}" for e in errs))

A, rhs = make("mixed", 0.9)
A, rhs = A.to("mps").requires_grad_(), rhs.to("mps").requires_grad_()
print("\nMPS fwd+bwd, 352 x 64x64, rhs 128 cols:")
for name, f in CANDS.items():
    for _ in range(5): f(A, rhs).sum().backward()
    runs = []
    for _ in range(5):
        torch.mps.synchronize(); t = time.perf_counter()
        for _ in range(10): f(A, rhs).sum().backward()
        torch.mps.synchronize(); runs.append((time.perf_counter() - t) / 10 * 1e3)
    ms = min(runs)
    with torch.no_grad():
        torch.mps.synchronize(); t = time.perf_counter()
        for _ in range(10): f(A, rhs)
        torch.mps.synchronize(); fwd = (time.perf_counter() - t) / 10 * 1e3
    print(f"  {name:10s} fwd {fwd:7.2f} ms   fwd+bwd {ms:7.2f} ms")
