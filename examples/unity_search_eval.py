"""S3 search as a Unity player: does the simulator gain of the 1-ply r5 search carry over to Unity?
(docs/policy_v2_design_ko.md §21; the search itself is blackout_env/sim/search.py, §20.)

Every worker owns one Unity player (``BlackOutEnv``).  Our team is r5 (``plain``) or r5 with the
S3 search (``search``); the opponent is any opponents.py descriptor (``p11`` / ``v23`` shorthands).

State reconstruction -- a **shadow simulator**: the match starts from ``SimState.from_seed(seed)``
(Unity's RNG included) and is stepped with the exact float32 moves Unity received (captured at
``set_actions``, examples/sim_record_traces.py).  Before every decision its rendered broadcast is
compared with Unity's (raw 44-float state + packed map, bit for bit); on a mismatch it is rebuilt
from the last 25 broadcasts (``SimState.from_history``, carrying the old RNG and spawn timer) and
the desync is counted.  So the search root is the exact hidden state (RNG, spawn timer, item
lists) whenever the shadow is in sync.  As in the simulator study the search does not use the
true RNG: every root is reseeded (common random numbers across its branches).

Our moves always come from Unity's own observation (``canonical_obs``), the same r5 forward on CPU
at batch 1 for both tags; the searching tag runs ``Searcher.search`` on the root every ``--every``
decisions (GPU batch of B branches) and executes the chosen branch for ``--window`` decisions.
A match ends at the game end, at an absorption that leaves no battery in play (the Unity evals'
``stop_when_exhausted``), or at ``--cap-s`` (150 s like the simulator Gate 1).

    PYTHONPATH=. python examples/unity_search_eval.py play --opponent p11 --seeds 100 --tags plain,search \\
        --workers 16 --out reports/unity_search/s8e25m3
    PYTHONPATH=. python examples/unity_search_eval.py report --out reports/unity_search/s8e25m3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"
OPPONENTS = {"p11": f"ckpt:{P11}", "v23": "strategic_v23", "v19": "strategic_v19", "r5": f"ckpt:{R5}"}
DPS = 25
HISTORY = 25

_W = None  # per-worker state


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(2.0)
    try:
        _W["env"].close()
    finally:
        os._exit(0)


def _init_worker(build: str, time_scale: float, gpus, cfg: dict) -> None:
    global _W
    gpu = gpus.get() if gpus is not None else None
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from multiprocessing import util

    import numba
    import torch

    torch.set_num_threads(1)
    numba.set_num_threads(1)
    from blackout_env import BlackOutEnv
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches
    from examples.sim_record_traces import _Capture

    env = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    util.Finalize(env, env.close, exitpriority=10)
    r5 = RolloutNet.from_checkpoint(str(REPO / R5))
    acting = ModNetPolicy(NetPolicy(r5, device="cpu"))
    dev = cfg["device"]
    searcher = Searcher(ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=dev)),
                        ValueNetLeaf(str(REPO / VALUE), dev), parse_branches(cfg["branches"]),
                        window=cfg["window"], margin=cfg["margin"])
    _W = {"env": env, "cap": _Capture(env), "acting": acting, "searcher": searcher, "cfg": cfg, "gpu": gpu}
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    obs, _ = env.reset(seed=7)  # warm-up: the first match after a player starts is not seed-reproducible
    for _ in range(5):
        obs, *_ = env.step({a: np.zeros(2, np.float32) for a in obs})
    # numba / CUDA warm-up off the clock
    from blackout_env.sim import SimState

    searcher.search([SimState.from_seed(1)], [0], [1])


def _inputs(view, side, bufs):
    from blackout_env.env.team_frame import canonical_obs

    c = canonical_obs(view)
    bufs[0][0, side] = np.asarray(c["graphic"], np.float32).transpose(2, 0, 1)
    bufs[1][0, side] = np.asarray(c["team_state"], np.float32)
    bufs[2][0, side] = np.asarray(c["agent_states"], np.float32)
    return bufs


def play_match(task: dict) -> dict:
    from blackout_env.env.constants import agent_name
    from blackout_env.sim import SimState, core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH
    from blackout_env.sim.mapdata import HUNTER, H, W
    from blackout_env.train.dead_segments import batteries_in_play
    from blackout_env.train.distill.opponents import make_opponent

    env, cap, our, searcher, cfg = _W["env"], _W["cap"], _W["acting"], _W["searcher"], _W["cfg"]
    seed, side, search = task["seed"], task["side"], task["tag"] != "plain"
    opp_desc = OPPONENTS.get(task["opponent"], task["opponent"])
    opponent = make_opponent(opp_desc, seed)
    if hasattr(opponent, "reset"):
        opponent.reset()
    our.configure(1)
    ours = [agent_name(k) for k in range(5 * side, 5 * side + 5)]
    theirs = [agent_name(k) for k in range(5 * (1 - side), 5 * (1 - side) + 5)]
    bufs = (np.zeros((1, 2, N_CH, H, W), np.float32), np.zeros((1, 2, 4), np.float32),
            np.zeros((1, 2, 10, N_AGENT_COLS), np.float32))
    idx = np.zeros(1, np.int64)
    nb = len(searcher.branches)
    chosen = np.zeros(nb, np.int64)
    gain_sum, lat, t_unity, t_policy = 0.0, [], 0.0, 0.0
    desync, first_desync, empty = 0, -1, 0
    first = [-1, -1]
    cur_branch = 0
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    shadow = SimState.from_seed(seed)
    hist_s, hist_m, hist_mv = [], [], []
    final_info, last_abs, t, end = {}, None, 0, "game"
    cap_ticks = int(cfg["cap_s"] * DPS)
    while env.agents:
        if not obs:
            c0 = time.perf_counter()
            obs, _, _, _, infos = env.step({})
            t_unity += time.perf_counter() - c0
            final_info = next(iter(infos.values()), final_info) if infos else final_info
            empty += 1
            if empty > 200:
                raise RuntimeError("empty observations for 200+ steps")
            continue
        rs, rm = cap.raw_state.copy(), cap.raw_map.copy()
        if not (np.array_equal(shadow.raw_state(), rs) and np.array_equal(shadow.raw_map(), rm)):
            desync += 1
            if first_desync < 0:
                first_desync = t
            kw = {"rng_state": shadow.rng.copy(), "spawn_timer": float(shadow.gf[core.GF_SPAWN])}
            if len(hist_s) >= 2:
                k = min(HISTORY, len(hist_s))
                shadow = SimState.from_history(np.stack(hist_s[-k:] + [rs]), np.stack(hist_m[-k:] + [rm]),
                                               np.stack(hist_mv[-k:]), **kw)
            else:
                shadow = SimState.from_raw(rs, rm, **kw)
        for team in (0, 1):
            if first[team] < 0 and (shadow.ui[5 * team:5 * team + 5, core.U_CLS] == HUNTER).any():
                first[team] = t
        if search and t % cfg["every"] == 0:
            c0 = time.perf_counter()
            reseed = (seed * 7919 + t * 104729 + 12345 + side * 31337) % (2 ** 31)
            out = searcher.search([shadow.copy()], [side], [reseed])
            k = int(out.choice[0])
            our.set_game(0, side, searcher.branches[k], cfg["window"])
            chosen[k] += 1
            gain_sum += float(out.values[0, k] - out.values[0, 0])
            cur_branch = k
            lat.append(time.perf_counter() - c0)
        c0 = time.perf_counter()
        actions = {}
        view = next((obs[a] for a in ours if a in obs), None)
        if view is not None:
            mv = our.moves(None, idx, (side,), _inputs(view, side, bufs))[side][0]
            actions.update({a: mv[i] for i, a in enumerate(ours) if a in obs})
        opp_obs = {a: obs[a] for a in theirs if a in obs}
        if opp_obs:
            actions.update(opponent.act(opp_obs))
        t_policy += time.perf_counter() - c0
        cap.moves[:] = 0.0
        cap.has_move[:] = False
        c0 = time.perf_counter()
        obs, _, terms, _, infos = env.step(actions)
        t_unity += time.perf_counter() - c0
        final_info = next(iter(infos.values()), final_info) if infos else final_info
        sent = cap.moves.copy()
        hist_s.append(rs)
        hist_m.append(rm)
        hist_mv.append(sent)
        if len(hist_s) > HISTORY + 1:
            del hist_s[0], hist_m[0], hist_mv[0]
        shadow.step(sent)
        t += 1
        if any(terms.values()) or not obs:
            continue
        v = next(iter(obs.values()))
        absorption = float(v["team_state"][3])
        fired = last_abs is not None and absorption > last_abs + 1e-6
        last_abs = absorption
        if fired and batteries_in_play(v["graphic"][None], v["agent_states"][None])[0] == 0:
            end = "exhausted"
            break
        if t >= cap_ticks:
            end = "cap"
            break
    if end == "game" and "score_0" in final_info:
        s0, s1 = round(float(final_info["score_0"]) * 100), round(float(final_info["score_1"]) * 100)
    else:
        s0, s1 = int(round(float(cap.raw_state[40]) * 100)), int(round(float(cap.raw_state[41]) * 100))
    sc = (s0, s1)
    lat_ms = np.asarray(lat) * 1e3
    return {**task, "ours": sc[side], "opp": sc[1 - side], "margin": sc[side] - sc[1 - side], "end": end,
            "decisions": t, "first_ours": first[side], "first_opp": first[1 - side],
            "searches": len(lat), "chosen": chosen.tolist(), "gain_sum": gain_sum, "branch_last": cur_branch,
            "search_ms_mean": float(lat_ms.mean()) if len(lat) else None,
            "search_ms_p50": float(np.median(lat_ms)) if len(lat) else None,
            "search_ms_p95": float(np.percentile(lat_ms, 95)) if len(lat) else None,
            "search_s": float(lat_ms.sum() / 1e3), "unity_s": t_unity, "policy_s": t_policy,
            "desyncs": desync, "first_desync": first_desync, "empty_steps": empty,
            "wall_s": time.perf_counter() - t0, "pid": os.getpid(), "gpu": _W["gpu"],
            "meta": {k: cfg[k] for k in ("branches", "every", "window", "margin", "cap_s")}}


def load_rows(out: Path) -> list[dict]:
    p = out / "matches.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cmd_play(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"branches": args.branches, "every": args.every, "window": args.window, "margin": args.margin,
           "cap_s": args.cap_s, "device": args.device}
    done = {(r["tag"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    tasks = [{"tag": tag, "opponent": opp, "seed": s, "side": side}
             for s in range(args.seed0, args.seed0 + args.seeds) for opp in args.opponent.split(",")
             for tag in args.tags.split(",") for side in (0, 1)]
    tasks = [x for x in tasks if (x["tag"], x["opponent"], x["seed"], x["side"]) not in done]
    print(f"{len(tasks)} matches to play ({len(done)} done), {args.workers} workers, cfg {cfg}", flush=True)
    ctx = mp.get_context("spawn")
    gpus = None
    if args.gpus:
        gpus = ctx.Manager().Queue()
        ids = [int(g) for g in args.gpus.split(",")]
        for w in range(args.workers):
            gpus.put(ids[w % len(ids)])
    ex = ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(args.build, args.time_scale, gpus, cfg))
    t0 = time.time()
    try:
        pending = {ex.submit(play_match, x) for x in tasks}
        n = 0
        with (out / "matches.jsonl").open("a") as fh:
            while pending:
                fin, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for f in fin:
                    r = f.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    n += 1
                    if n % 10 == 0 or not pending:
                        el = time.time() - t0
                        print(f"[{n}/{len(tasks)}] {el:.0f}s  {n / el * 60:.1f} matches/min  last: {r['tag']} vs {r['opponent']} "
                              f"seed {r['seed']} side {r['side']} margin {r['margin']:+d} wall {r['wall_s']:.0f}s "
                              f"search {r['search_s']:.0f}s desync {r['desyncs']}", flush=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race(rows):
    big = 10 ** 9
    w = t = l = 0
    for r in rows:
        a = r["first_ours"] if r["first_ours"] >= 0 else big
        b = r["first_opp"] if r["first_opp"] >= 0 else big
        w, t, l = w + (a < b), t + (a == b), l + (a > b)
    return w, t, l


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["opponent"])][(r["seed"], r["side"])] = r
    lines, summary = [], {}
    for (tag, opp), d in sorted(by.items()):
        ms = [r["margin"] for r in d.values()]
        line = (f"{tag:8s} vs {opp:6s} n={len(d):4d} margin {np.mean(ms):+6.1f} ± {se(ms):4.1f}  race w/t/l {race(d.values())}  "
                f"win {np.mean([m > 0 for m in ms]):.2f}  end {dict(sorted(defaultdict(int, {e: sum(r['end'] == e for r in d.values()) for e in ('game', 'exhausted', 'cap')}).items()))}  "
                f"len {np.mean([r['decisions'] for r in d.values()]) / DPS:.0f}s  wall {np.mean([r['wall_s'] for r in d.values()]):.0f}s/match  "
                f"desync matches {sum(r['desyncs'] > 0 for r in d.values())} (ticks {sum(r['desyncs'] for r in d.values())})")
        lines.append(line)
        summary[f"{tag}/{opp}"] = {"n": len(d), "margin": float(np.mean(ms)), "se": se(ms), "race": race(d.values())}
        if tag != "plain" and ("plain", opp) in by:
            base = by[("plain", opp)]
            keys = sorted(k for k in d if k in base)
            per_seed = defaultdict(list)
            for k in keys:
                per_seed[k[0]].append(d[k]["margin"] - base[k]["margin"])
            ps = [np.mean(v) for v in per_seed.values()]
            rs, rb = race([d[k] for k in keys]), race([base[k] for k in keys])
            lat = [d[k]["search_ms_mean"] for k in keys if d[k]["search_ms_mean"] is not None]
            p95 = [d[k]["search_ms_p95"] for k in keys if d[k]["search_ms_p95"] is not None]
            ch = np.sum([d[k]["chosen"] for k in keys], 0)
            # gain split by race outcome of the pair
            def rc(r):
                a = r["first_ours"] if r["first_ours"] >= 0 else 10 ** 9
                b = r["first_opp"] if r["first_opp"] >= 0 else 10 ** 9
                return int(np.sign(b - a))
            split = defaultdict(list)
            for k in keys:
                split[(rc(base[k]), rc(d[k]))].append(d[k]["margin"] - base[k]["margin"])
            lines.append(f"   paired {tag} - plain vs {opp}: {np.mean(ps):+6.1f} ± {se(ps):4.1f}  ({len(ps)} seeds, {len(keys)} matches)  "
                         f"race {rs} vs plain {rb}  search latency mean {np.mean(lat):.0f} ms (p95 {np.mean(p95):.0f})  "
                         f"base share {ch[0] / max(1, ch.sum()):.2f}  searches/match {np.mean([d[k]['searches'] for k in keys]):.0f}")
            lines.append("   by race (plain->search): " + ", ".join(f"{a:+d}->{b:+d}: n={len(v)} Δ{np.mean(v):+.1f}" for (a, b), v in sorted(split.items())))
            summary[f"{tag}-plain/{opp}"] = {"diff": float(np.mean(ps)), "se": se(ps), "seeds": len(ps), "matches": len(keys),
                                              "race_search": rs, "race_plain": rb, "search_ms_mean": float(np.mean(lat)),
                                              "search_ms_p95": float(np.mean(p95)), "base_share": float(ch[0] / max(1, ch.sum()))}
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--build", default="build/linux_server/BlackOut.x86_64")
    q.add_argument("--opponent", default="p11", help="comma list: p11 | v23 | v19 | r5 | an opponents.py descriptor")
    q.add_argument("--tags", default="plain,search")
    q.add_argument("--seeds", type=int, default=20)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--every", type=int, default=25)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--margin", type=float, default=3.0)
    q.add_argument("--cap-s", type=float, default=150.0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--gpus", default="0,1,2,3", help="CUDA devices dealt to workers round-robin ('' = none)")
    q.add_argument("--device", default="cuda")
    q.add_argument("--time-scale", type=float, default=20.0)
    q.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
