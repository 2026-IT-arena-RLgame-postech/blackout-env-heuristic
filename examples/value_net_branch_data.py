"""S1.6 branch data for a rollout net's continuations, in the batched simulator (GPU).
(docs/policy_v2_design_ko.md §24 "새 롤아웃 망으로 탐색"; the CPU/heuristic original is
examples/value_branch_data.py, §17.)

The leaf of the S3 search ("5 s rollout + V") is strong only when V was fitted for the policy that
actually continues the game (§19 "리프 재검증").  This generator produces the §17 fitted-value-
iteration data with a rollout net ``--net`` as the continuation policy of *both* teams:

* Root matches, lockstep chunks of ``--chunk`` games (seeds ``--seed0``+ x both sides): our team
  (``side``) is the net, the other team the net (``self``), P11, r5 or strategic_v23 (``--mix``;
  one opponent per chunk).  To the end or ``--cap-s``.
* Root ticks along each chunk (``value_branch_data.root_schedule``: dense in the opening).  From each
  live game's root, our team plays each of the search branches (``--branches``, default the S3 set
  base / alt0-4 / samp1.0 x2 -- what the search will ask V to rank) for ``--window`` decisions, the
  other team the plain net (the search's self-play model); then both teams the plain net for K = 125
  decisions (5 s).  One reseed per root, shared by its branches (common random numbers, like the search).
* Recorded per branch (``kind`` 1): the window-end state ``s`` (both teams' views, the value net's
  uint8 training encoding), the lead at ``s``, +2 s and +5 s, and the state ``s'`` 5 s later (both
  views) or the exact final lead when the game ended.  On-trajectory rows (``kind`` 0, self-play
  chunks only -- elsewhere the trajectory's continuation policy is not the net): every second of the
  root match, its state and the leads 2 s / 5 s later and the state 5 s later.

Output = value_branch_data.py's raw format (``<out>/raw/<seed>_<A|B>.npz``), so
``value_branch_train.py train --data <out>`` fits V on it unchanged.

    PYTHONPATH=. python examples/value_net_branch_data.py --net checkpoints/rollout_net/arch/f3_x_d2.pt \\
        --seeds 512 --seed0 6000000 --out datasets/value_f3xd2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

DPS = 25
K = 125   # continuation (5 s)
K2 = 50   # +2 s lead inside the continuation
ROOT_SPACING_S = ((10, 2.0), (30, 3.0), (60, 4.0), (150, 5.0), (420, 10.0))  # = value_branch_data
P11 = "checkpoints/distill/frozen/p11_v22.pt"
R5 = "checkpoints/rollout_net/r5.pt"
DEFAULT_MIX = "self:0.5,p11:0.2,r5:0.15,strategic_v23:0.15"


def root_schedule(rng: np.random.Generator, max_ticks: int) -> list[int]:
    ticks, t = [], float(rng.uniform(0, ROOT_SPACING_S[0][1]))
    while t * DPS < max_ticks:
        ticks.append(int(t * DPS))
        spacing = next(s for hi, s in ROOT_SPACING_S if t < hi)
        t += spacing * float(rng.uniform(0.5, 1.5))
    return ticks


def encode_rendered(graphic: np.ndarray) -> np.ndarray:
    """Batch-render graphic float32 [n, 13, H, W] -> the value net's uint8 [n, H, W, 13] (x15)."""
    enc = np.rint(graphic * 15.0).astype(np.uint8)
    return np.ascontiguousarray(enc.transpose(0, 2, 3, 1))


class Views:
    """Rendered + encoded states of one chunk, both teams' views, ``own`` = the game's learner side."""

    def __init__(self):
        self.cols = {k: [] for k in ("g_own", "ts_own", "ag_own", "g_mir", "ts_mir", "ag_mir")}
        self.n = 0

    def add(self, bsim, idx: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """Render games ``idx`` of ``bsim``; ``sides[j]`` = the learner side of game idx[j].  Returns the
        new state indices (chunk-global)."""
        from blackout_env.sim.batch import N_AGENT_COLS, N_CH
        from blackout_env.sim.mapdata import H, W

        m = len(idx)
        if not m:
            return np.zeros(0, np.int64)
        g = np.zeros((m, 2, N_CH, H, W), np.float32)
        t = np.zeros((m, 2, 4), np.float32)
        a = np.zeros((m, 2, 10, N_AGENT_COLS), np.float32)
        bsim.render(np.asarray(idx, np.int64), g, t, a)
        j = np.arange(m)
        sides = np.asarray(sides, np.int64)
        for pre, sel in (("own", sides), ("mir", 1 - sides)):
            self.cols[f"g_{pre}"].append(encode_rendered(g[j, sel]))
            self.cols[f"ts_{pre}"].append(t[j, sel].copy())
            self.cols[f"ag_{pre}"].append(a[j, sel].copy())
        out = np.arange(self.n, self.n + m)
        self.n += m
        return out

    def arrays(self) -> dict:
        return {k: np.concatenate(v) for k, v in self.cols.items()}


def make_opponent(name: str, device: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if name == "self":
        return None
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device=device)
    if name == "r5":
        return NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    if name.startswith("rnet:"):
        return NetPolicy(RolloutNet.from_checkpoint(name[5:]), device=device)
    return HeuristicPolicy(name)


def branch_rows(mod, bsim, live, sides, tick, branches, window, reseed0, views):
    """Every branch of every live game's root -> row dict (lists) + state indices in ``views``."""
    from blackout_env.sim.batch import BatchRollout

    nb = len(branches)
    roots = [bsim.state(int(b)) for b in live]
    gside = np.repeat(sides[live], nb)
    sims = [s for s in roots for _ in range(nb)]
    br = branches * len(roots)
    reseeds = np.repeat([(reseed0 + 7919 * int(b) + 104729 * tick) % (2 ** 31) for b in live], nb)
    mod.configure(len(sims), team=gside, kind=[b.kind for b in br], unit=[b.unit for b in br],
                  param=[b.param for b in br], seed=[b.seed for b in br], left=np.full(len(sims), window))
    r1 = BatchRollout(mod, mod, repeat=1).run(sims, window, reseeds=reseeds)
    alive = np.nonzero(~r1.over)[0]  # ended inside the window: terminal, nothing to learn
    if not len(alive):
        return None
    s_idx = views.add(r1.bsim, alive, gside[alive])
    lead_a = r1.bsim.scores[alive, 0] - r1.bsim.scores[alive, 1]
    mid = [r1.bsim.state(int(j)) for j in alive]
    mod.configure(len(mid))  # plain net for both teams
    r2 = BatchRollout(mod, mod, repeat=1).run(mid, K, trace_steps=[K2])
    fin_a = r2.bsim.scores[:, 0] - r2.bsim.scores[:, 1]
    ended = r2.over.copy()
    nxt = np.full(len(alive), -1, np.int64)
    live2 = np.nonzero(~ended)[0]
    nxt[live2] = views.add(r2.bsim, live2, gside[alive][live2])
    sgn = np.where(gside[alive] == 0, 1, -1)
    return {"kind": np.ones(len(alive), np.int8), "idx": s_idx, "next_idx": nxt, "lead": sgn * lead_a,
            "lead2": sgn * r2.lead_trace[0], "lead5": sgn * fin_a, "ended": ended,
            "t_s": np.full(len(alive), (tick + window) / DPS), "root": np.full(len(alive), tick),
            "branch": (alive % nb).astype(np.int64), "game": np.asarray(live)[alive // nb]}


def play_chunk(seeds, sides, opp_name, net, opp, branches, args, log=print) -> dict:
    """One lockstep chunk of root matches + their branch rows."""
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.search import ModNetPolicy

    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    mod = ModNetPolicy(net)  # branches / continuations (and the root match's net moves)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    if opp is not None:
        opp.reset(n)
    cap = int(args.cap_s * DPS)
    rng = np.random.default_rng(zlib.crc32(f"{seeds[0]}:{opp_name}:{args.seed0}".encode()))
    roots = set(root_schedule(rng, cap))
    views = Views()
    parts = []
    lead_by_tick = np.zeros((cap + 1, n), np.int64)
    traj = {}  # tick -> (games, state idx)
    bufs = (np.zeros((n, 2, N_CH, H, W), np.float32), np.zeros((n, 2, 4), np.float32),
            np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    moves = np.zeros((n, 10, 2), np.float32)
    sgn = np.where(sides == 0, 1, -1)
    end_tick = np.full(n, cap, np.int64)
    t_br, t0 = 0.0, time.perf_counter()
    reseed0 = int(seeds[0]) * 31 + 17
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        lead_by_tick[t] = sgn * (bs.scores[:, 0] - bs.scores[:, 1])
        if not len(live):
            break
        if opp is None and t % DPS == 0:
            traj[t] = (live.copy(), views.add(bs, live, sides[live]))
        if t in roots:
            c0 = time.perf_counter()
            rows = branch_rows(mod, bs, live, sides, t, branches, args.window, reseed0, views)
            if rows is not None:
                parts.append(rows)
            t_br += time.perf_counter() - c0
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        mod.configure(n)
        mo = mod.moves(bs, live, (0, 1), sub)
        moves[live, :5], moves[live, 5:] = mo[0], mo[1]
        if opp is not None:
            for team in (0, 1):
                sel = sides[live] != team
                if sel.any():
                    s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                    moves[live[sel], 5 * team:5 * team + 5] = opp.moves(bs, live[sel], (team,), s_in)[team]
        bs.step(moves, live)
        ended_now = live[bs.over[live]]
        end_tick[ended_now] = t + 1
        if args.verbose and t % 500 == 0:
            log(f"  t={t // DPS}s live {len(live)} branch {t_br:.0f}s total {time.perf_counter() - t0:.0f}s rows "
                f"{sum(len(p['kind']) for p in parts)}")
    final = sgn * (bs.scores[:, 0] - bs.scores[:, 1])
    lead_by_tick[end_tick, np.arange(n)] = final  # (the tick after the end; later ticks unused)
    # on-trajectory rows
    tr = {k: [] for k in ("kind", "idx", "next_idx", "lead", "lead2", "lead5", "ended", "t_s", "root", "branch", "game")}
    for tick, (games, idx) in traj.items():
        nxt = traj.get(tick + K)
        nxt_of = dict(zip(nxt[0].tolist(), nxt[1].tolist())) if nxt else {}
        for g, i in zip(games.tolist(), idx.tolist()):
            e = end_tick[g]
            ended = bool(bs.over[g]) and tick + K >= e
            if not ended and g not in nxt_of:  # capped before +5 s
                continue
            tr["kind"].append(0)
            tr["idx"].append(i)
            tr["next_idx"].append(-1 if ended else nxt_of[g])
            tr["lead"].append(lead_by_tick[tick, g])
            tr["lead2"].append(lead_by_tick[min(tick + K2, e), g])
            tr["lead5"].append(lead_by_tick[min(tick + K, e), g])
            tr["ended"].append(ended)
            tr["t_s"].append(tick / DPS)
            tr["root"].append(-1)
            tr["branch"].append(-1)
            tr["game"].append(g)
    if tr["kind"]:
        parts.append({k: np.asarray(v) for k, v in tr.items()})
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None
    return {"rows": rows, "states": views.arrays() if views.n else None, "final": final, "end_tick": end_tick,
            "over": bs.over.copy(), "n_roots": len([r for r in roots if r < int(end_tick.max())]),
            "seconds": time.perf_counter() - t0, "branch_s": t_br}


def save_matches(out: Path, seeds, sides, opp_name, res, args) -> int:
    """Split a chunk into per-match files (value_branch.load_raw's format: state indices per file)."""
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows, st = res["rows"], res["states"]
    n_rows = 0
    for g, (seed, side) in enumerate(zip(seeds, sides)):
        tid = f"{seed}_{'AB'[side]}"
        sel = np.nonzero(rows["game"] == g)[0] if rows is not None else np.zeros(0, np.int64)
        used = np.unique(np.concatenate([rows["idx"][sel], rows["next_idx"][sel][rows["next_idx"][sel] >= 0]])) \
            if len(sel) else np.zeros(0, np.int64)
        remap = np.full(int(st["g_own"].shape[0]) if st is not None else 0, -1, np.int64)
        remap[used] = np.arange(len(used))
        r = {k: v[sel] for k, v in rows.items() if k != "game"} if len(sel) else \
            {k: np.zeros(0, np.int64) for k in ("kind", "idx", "next_idx", "lead", "lead2", "lead5", "ended", "t_s", "root", "branch")}
        if len(sel):
            r["idx"] = remap[r["idx"]]
            r["next_idx"] = np.where(r["next_idx"] >= 0, remap[np.maximum(r["next_idx"], 0)], -1)
            r["ended"] = r["ended"].astype(bool)
        states = {k: v[used] for k, v in st.items()} if st is not None and len(used) else \
            {"g_own": np.zeros((0, 24, 24, 13), np.uint8)}
        task = {"id": tid, "seed": int(seed), "learner_team": int(side), "opponent": opp_name, "net": args.net,
                "window": args.window, "branches": args.branches}
        meta = {"task": task, "final": int(res["final"][g]), "end_tick": int(res["end_tick"][g]), "over": bool(res["over"][g]),
                "n_roots": res["n_roots"], "seconds": res["seconds"] / len(seeds)}
        np.savez_compressed(raw / f"{tid}.tmp.npz", meta=json.dumps(meta), **{f"r_{k}": v for k, v in r.items()},
                            **{f"s_{k}": v for k, v in states.items()})
        os.replace(raw / f"{tid}.tmp.npz", raw / f"{tid}.npz")
        n_rows += len(sel)
    return n_rows


def parse_mix(spec: str) -> list[tuple[str, float]]:
    out = []
    for tok in spec.split(","):
        name, _, w = tok.partition(":")
        out.append((name, float(w or 1)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True, help="RolloutNet checkpoint: the continuation policy (both teams)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=512)
    p.add_argument("--seed0", type=int, default=6_000_000)
    p.add_argument("--mix", default=DEFAULT_MIX, help="opponent:weight list (self | p11 | r5 | rnet:<ckpt> | descriptor)")
    p.add_argument("--branches", default="alt*,samp:1.0x2")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--chunk", type=int, default=128, help="games per lockstep chunk (one opponent per chunk)")
    p.add_argument("--cap-s", type=float, default=300.0)
    p.add_argument("--device", default="")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    import numba
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import default_device, parse_branches

    torch.set_num_threads(args.threads)
    numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
    device = args.device or default_device()
    net = NetPolicy(RolloutNet.from_checkpoint(args.net), device=device)
    branches = parse_branches(args.branches)
    mix = parse_mix(args.mix)
    w = np.array([x[1] for x in mix])
    # deal seeds to opponents by weight (whole seeds: both sides of a seed share the opponent)
    counts = np.floor(w / w.sum() * args.seeds).astype(int)
    counts[0] += args.seeds - counts.sum()
    raw = args.out / "raw"
    started = time.time()
    s0, total = args.seed0, 0
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    for (name, _), cnt in zip(mix, counts):
        tasks = [(s, side) for s in range(s0, s0 + cnt) for side in (0, 1)]
        s0 += cnt
        todo = [x for x in tasks if not (raw / f"{x[0]}_{'AB'[x[1]]}.npz").exists()]
        opp = make_opponent(name, device) if todo else None
        print(f"opponent {name}: {len(tasks)} matches, {len(todo)} to play", flush=True)
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            seeds, sides = [x[0] for x in chunk], [x[1] for x in chunk]
            res = play_chunk(seeds, sides, name, net, opp, branches, args)
            nr = save_matches(args.out, seeds, sides, name, res, args) if res["rows"] is not None else 0
            total += nr
            kinds = np.bincount(res["rows"]["kind"], minlength=2) if res["rows"] is not None else [0, 0]
            print(f"[{name}] {c + len(chunk)}/{len(todo)}: {nr} rows (traj {kinds[0]}, branch {kinds[1]}), "
                  f"{res['n_roots']} root ticks, match len {res['end_tick'].mean() / DPS:.0f}s, chunk {res['seconds']:.0f}s "
                  f"(branches {res['branch_s']:.0f}s); total {total} rows, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
