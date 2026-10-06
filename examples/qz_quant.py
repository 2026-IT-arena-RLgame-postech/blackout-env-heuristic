"""QZ: faster P20 forwards for search / evaluation (docs/policy_v2_design_ko.md "QZ").

    PYTHONPATH=. python examples/qz_quant.py bench --device cuda --sizes 2048,4096,8192
    PYTHONPATH=. python examples/qz_quant.py profile --device cuda --size 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

P20 = "checkpoints/distill/frozen/p20_v30.pt"


def load_policy(ckpt: str, precision: str, device: str, **kw):
    from blackout_env.sim.batch import NetPolicy

    return NetPolicy.from_checkpoint(str(REPO / ckpt), device=device, precision=precision, **kw)


def device_forward(pol, g, t, a):
    """(logits, offsets | None) float32 on device tensors (as B1's ``pipeline._net_forward``)."""
    torch = pol.torch
    with torch.inference_mode():
        if pol.fp16:
            with torch.autocast(device_type=pol.device.type, dtype=torch.float16):
                out = pol._own(g, t, a)
        else:
            out = pol._own(g, t, a)
    q, off = out if pol.angle else (out, None)
    return q.float(), (None if off is None else off.float())


def fake_inputs(n: int, device, seed: int = 0, data: str | None = None):
    """Packed int16 maps [n, H, W], team [n, 4], agents [n, 10, 12] on ``device``: real recorded views
    (``data``, an npz from ``collect``) tiled to ``n``, else random-ish valid views."""
    import torch

    from blackout_env.sim.batch import N_AGENT_COLS
    from blackout_env.sim.mapdata import H, W

    if data:
        z = np.load(data)
        idx = np.random.default_rng(seed).integers(0, len(z["team"]), n)
        g, t, a = z["graphic"][idx], z["team"][idx], z["agents"][idx]
    else:
        r = np.random.default_rng(seed)
        g = (r.integers(0, 8, (n, H, W)) | (r.integers(0, 16, (n, H, W)) << 3)).astype(np.int16)
        t = r.random((n, 4), np.float32)
        a = r.random((n, 10, N_AGENT_COLS), np.float32) * 24
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in (g, t, a))


def cmd_bench(args):
    import torch

    torch.backends.cudnn.benchmark = True
    rows = []
    for mode in args.modes.split(","):
        for n in (int(x) for x in args.sizes.split(",")):
            pol = load_policy(args.ckpt, mode, args.device, max_batch=n)
            g, t, a = fake_inputs(n, args.device, data=args.data)
            f = lambda *x: device_forward(pol, *x)  # noqa: E731
            for _ in range(args.warmup):
                f(g, t, a)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                f(g, t, a)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ms = 1e3 * float(np.median(ts))
            row = {"mode": mode, "n": n, "ms": round(ms, 2), "views_per_s": round(n / ms * 1e3)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del pol
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "gpu": torch.cuda.get_device_name()}) + "\n")


def cmd_profile(args):
    import torch
    from torch.profiler import ProfilerActivity, profile

    pol = load_policy(args.ckpt, args.mode, args.device, max_batch=args.size)
    g, t, a = fake_inputs(args.size, args.device, data=args.data)
    for _ in range(3):
        device_forward(pol, g, t, a)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            device_forward(pol, g, t, a)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=70))


def _opponent(name: str, device: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name.startswith("net:"):
        return NetPolicy.from_checkpoint(str(REPO / name[4:]), device=device)
    return HeuristicPolicy(name)


def cmd_collect(args):
    """P20-driven sim states (P20's own views): P20 (fp32) vs each opponent, seeds x both sides; every view in
    decisions ``--dense`` (the race / staged-reserve join window) and every ``--stride``-th decision elsewhere
    (a per-game phase).  -> ``<out>/<opponent>_<seed0>.npz`` (packed int16 maps, team, agents, decision, seed, side)."""
    import torch

    from blackout_env.sim.batch import N_AGENT_COLS, BatchSim, NetPolicy
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ours = NetPolicy.from_checkpoint(str(REPO / args.ckpt), device=args.device)
    lo, hi = (int(x) for x in args.dense.split(","))
    for opp_name in args.opponents.split(";"):
        path = out / f"{opp_name.replace(':', '_').replace('/', '_')}_{args.seed0}.npz"
        if path.exists():
            continue
        opp = _opponent(opp_name, args.device)
        tasks = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)]
        rec = {k: [] for k in ("graphic", "team", "agents", "t", "seed", "side")}
        t0 = time.time()
        for c in range(0, len(tasks), args.chunk):
            ch = tasks[c:c + args.chunk]
            n = len(ch)
            seeds = np.array([s for s, _ in ch])
            sides = np.array([sd for _, sd in ch])
            phase = (seeds * 7 + sides) % args.stride
            bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
            ours.reset(n)
            opp.reset(n)
            bufs = (np.zeros((n, 2, H, W), np.int16), np.zeros((n, 2, 4), np.float32),
                    np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
            moves = np.zeros((n, 10, 2), np.float32)
            for t in range(args.cap_s * 25):
                live = np.nonzero(~bs.over)[0]
                if not len(live):
                    break
                bs.render_packed(live, *bufs)
                sub = tuple(x[: len(live)] for x in bufs)
                for team in (0, 1):
                    mine = sides[live] == team
                    for pol, sel in ((ours, mine), (opp, ~mine)):
                        if not sel.any():
                            continue
                        s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                        moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), s_in)[team]
                keep = ((lo <= t) & (t < hi)) | (t % args.stride == phase[live])
                for j in np.nonzero(keep)[0]:
                    b = live[j]
                    sd = sides[b]
                    rec["graphic"].append(sub[0][j, sd].copy())
                    rec["team"].append(sub[1][j, sd].copy())
                    rec["agents"].append(sub[2][j, sd].copy())
                    rec["t"].append(t)
                    rec["seed"].append(seeds[b])
                    rec["side"].append(sd)
                bs.step(moves, live)
            print(f"[{opp_name}] {c + n}/{len(tasks)} games, {len(rec['t'])} views, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()})


def load_views(paths) -> dict:
    zs = [np.load(p) for p in paths]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def outputs(pol, v: dict, batch: int = 2048):
    """float32 numpy (logits [n, 5, K], offsets [n, 5, K'] | None) of ``pol`` on views ``v``."""
    torch = pol.torch
    qs, os_ = [], []
    for s in range(0, len(v["t"]), batch):
        g, t, a = (torch.from_numpy(np.ascontiguousarray(v[k][s:s + batch])).to(pol.device)
                   for k in ("graphic", "team", "agents"))
        q, off = device_forward(pol, g, t, a)
        qs.append(q.cpu().numpy())
        os_.append(None if off is None else off.cpu().numpy())
    return np.concatenate(qs), (None if os_[0] is None else np.concatenate(os_))


def fidelity(ref, cand, t, join=(85, 100)) -> dict:
    """Candidate vs reference outputs per subset of rows: arg max agreement, KL(ref || cand) (mean, p99), the decoded
    move's heading error (deg; rows where both move) and identical moves."""
    from blackout_env.sim.batch import decode_moves

    (q0, o0), (q1, o1) = ref, cand
    n = len(q0)
    tb = np.zeros(n, bool)
    m0, m1 = decode_moves(q0, o0, tb), decode_moves(q1, o1, tb)
    lp0 = q0 - np.logaddexp.reduce(q0, -1, keepdims=True)
    lp1 = q1 - np.logaddexp.reduce(q1, -1, keepdims=True)
    kl = (np.exp(lp0) * (lp0 - lp1)).sum(-1)  # [n, 5]
    agree = q0.argmax(-1) == q1.argmax(-1)
    both = (np.abs(m0).sum(-1) > 0) & (np.abs(m1).sum(-1) > 0)
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(m0[..., 1], m0[..., 0]) - np.arctan2(m1[..., 1], m1[..., 0]))))))
    same = np.all(np.abs(m0 - m1) < 1e-5, -1)
    subsets = {"all": np.ones(n, bool), "join85_99": (t >= join[0]) & (t < join[1]), "open_lt85": t < join[0],
               "late_ge100": t >= join[1]}
    res = {}
    for name, sel in subsets.items():
        if not sel.any():
            continue
        b = both[sel]
        res[name] = {"views": int(sel.sum()), "argmax_agree": float(agree[sel].mean()),
                     "kl_mean": float(kl[sel].mean()), "kl_p99": float(np.quantile(kl[sel], 0.99)),
                     "angle_err_mean_deg": float(ang[sel][b].mean()) if b.any() else float("nan"),
                     "angle_err_p99_deg": float(np.quantile(ang[sel][b], 0.99)) if b.any() else float("nan"),
                     "move_identical": float(same[sel].mean()),
                     "views_any_flip": float((~agree[sel]).any(-1).mean())}
    return res


def cmd_fidelity(args):
    import torch

    torch.backends.cudnn.benchmark = True
    v = load_views(sorted(Path(args.data).glob("*.npz")))
    ref_path = Path(args.data) / "ref_fp32.npz"
    if ref_path.exists():
        z = np.load(ref_path)
        ref = (z["q"], z["off"])
    else:
        ref = outputs(load_policy(args.ref, "fp32", args.device), v)
        np.savez(ref_path, q=ref[0], off=ref[1])
    for spec in args.cands.split(","):
        ckpt, prec = spec.split("@") if "@" in spec else (args.ref, spec)
        pol = load_policy(ckpt, prec, args.device)
        res = fidelity(ref, outputs(pol, v), v["t"])
        row = {"cand": spec, **res}
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        del pol
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--ckpt", default=P20)
    b.add_argument("--device", default="cuda")
    b.add_argument("--modes", default="fp32,fp16,fp16c")
    b.add_argument("--sizes", default="2048,4096,8192")
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--data", default=None)
    b.add_argument("--out", default=None)
    b.set_defaults(fn=cmd_bench)
    q = sub.add_parser("profile")
    q.add_argument("--ckpt", default=P20)
    q.add_argument("--device", default="cuda")
    q.add_argument("--mode", default="fp32")
    q.add_argument("--size", type=int, default=2048)
    q.add_argument("--rows", type=int, default=40)
    q.add_argument("--data", default=None)
    q.set_defaults(fn=cmd_profile)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", default=P20)
    c.add_argument("--device", default="cuda")
    c.add_argument("--opponents", default="net:" + P20 + ";strategic_v30:reserve_stage=0.5,trade_hold=advance,"
                   "reserve_join=reserve,solo_sentry=true;strategic_v28:reserve_stage=0.5,trade_hold=advance")
    c.add_argument("--seeds", type=int, default=32)
    c.add_argument("--seed0", type=int, default=3_000_000)
    c.add_argument("--chunk", type=int, default=64)
    c.add_argument("--cap-s", type=int, default=150)
    c.add_argument("--stride", type=int, default=25)
    c.add_argument("--dense", default="60,110", help="decisions where every view is kept")
    c.add_argument("--out", default=str(REPO / "datasets/qz/heldout"))
    c.set_defaults(fn=cmd_collect)
    f = sub.add_parser("fidelity")
    f.add_argument("--data", default=str(REPO / "datasets/qz/heldout"))
    f.add_argument("--ref", default=P20)
    f.add_argument("--cands", default="fp32,fp16,fp16c", help="',' list of precision or ckpt@precision")
    f.add_argument("--device", default="cuda")
    f.add_argument("--out", default=None)
    f.set_defaults(fn=cmd_fidelity)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
