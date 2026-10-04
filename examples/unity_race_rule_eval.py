"""H9 rule check in Unity: ``strategic_v25`` vs V23 / P11 with the V23 baseline on the same seeds
(docs/policy_v2_design_ko.md §28; the simulator version is examples/race_rule_eval.py).

Every worker owns one Unity player (``BlackOutEnv``, headless Linux server build).  Both teams are
opponents.py descriptors acting on Unity's own observations (``make_opponent(desc).act``; the
``p11`` / ``v23`` / ``v25`` / ``r5`` shorthands).  A shadow simulator stepped with the exact float32
moves Unity received (as in examples/unity_search_eval.py) gives the first Hunter decision of each
team and counts desyncs (shadow broadcast != Unity broadcast, bit for bit; rebuilt from history on
a mismatch).  A match ends at the game end, at an absorption that leaves no battery in play, or at
``--cap-s`` (150 s like the simulator).

``report``: per (ours, opponent) the margin, race w/t/l, our first-Hunter histogram; the paired
difference to ``--base`` (default strategic_v23) on the same (seed, side), SE over seeds (both
sides averaged first); with ``--sim`` (race_rule_eval.py matches.jsonl) the per-match agreement
of Unity with the simulator (margin, first Hunter) and the simulator's paired numbers on the
same seeds.

    PYTHONPATH=. python examples/unity_race_rule_eval.py play --ours strategic_v23,strategic_v25 \\
        --opponent strategic_v23,p11 --seeds 150 --workers 8 --out reports/race_rule_eval_unity
    PYTHONPATH=. python examples/unity_race_rule_eval.py report --out reports/race_rule_eval_unity \\
        --sim reports/race_rule_eval_unity/sim/matches.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

P11 = "checkpoints/distill/frozen/p11_v22.pt"
R5 = "checkpoints/rollout_net/r5.pt"
DPS = 25
HISTORY = 25

_W = None  # per-worker state


def _desc(name: str, ckpt_root: str) -> str:
    short = {"p11": f"ckpt:{Path(ckpt_root) / P11}", "r5": f"ckpt:{Path(ckpt_root) / R5}",
             "v23": "strategic_v23", "v25": "strategic_v25"}
    return short.get(name, name)


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(2.0)
    try:
        _W["env"].close()
    finally:
        os._exit(0)


def _init_worker(build: str, time_scale: float) -> None:
    global _W
    import warnings
    from multiprocessing import util

    import numba
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    from blackout_env import BlackOutEnv
    from examples.sim_record_traces import _Capture

    env = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    util.Finalize(env, env.close, exitpriority=10)
    _W = {"env": env, "cap": _Capture(env)}
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    obs, _ = env.reset(seed=7)  # warm-up: the first match after a player starts is not seed-reproducible
    for _ in range(5):
        obs, *_ = env.step({a: np.zeros(2, np.float32) for a in obs})


def play_match(task: dict) -> dict:
    from blackout_env.env.constants import agent_name
    from blackout_env.sim import SimState, core
    from blackout_env.sim.mapdata import HUNTER
    from blackout_env.train.dead_segments import batteries_in_play
    from blackout_env.train.distill.opponents import make_opponent

    env, cap = _W["env"], _W["cap"]
    seed, side = task["seed"], task["side"]
    pols = {}
    for team, name in ((side, task["ours"]), (1 - side, task["opponent"])):
        pol = make_opponent(_desc(name, task["ckpt_root"]), seed)
        if hasattr(pol, "reset"):
            pol.reset()
        pols[team] = (pol, [agent_name(k) for k in range(5 * team, 5 * team + 5)])
    desync, first_desync, empty = 0, -1, 0
    first = [-1, -1]
    first_unit = [-1, -1]
    t_unity = t_policy = 0.0
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    shadow = SimState.from_seed(seed)
    hist_s, hist_m, hist_mv = [], [], []
    final_info, last_abs, t, end = {}, None, 0, "game"
    cap_ticks = int(task["cap_s"] * DPS)
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
            if first[team] < 0:
                hit = np.nonzero(shadow.ui[5 * team:5 * team + 5, core.U_CLS] == HUNTER)[0]
                if len(hit):
                    first[team], first_unit[team] = t, int(hit[0])
        c0 = time.perf_counter()
        actions = {}
        for team in (0, 1):
            pol, names = pols[team]
            sub = {a: obs[a] for a in names if a in obs}
            if sub:
                actions.update(pol.act(sub))
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
    out = {k: task[k] for k in ("ours", "opponent", "seed", "side")}
    return {**out, "score_ours": sc[side], "score_opp": sc[1 - side], "margin": sc[side] - sc[1 - side], "end": end,
            "decisions": t, "first_ours": first[side], "first_opp": first[1 - side],
            "first_unit_ours": first_unit[side], "first_unit_opp": first_unit[1 - side],
            "desyncs": desync, "first_desync": first_desync, "empty_steps": empty,
            "unity_s": t_unity, "policy_s": t_policy, "wall_s": time.perf_counter() - t0, "pid": os.getpid()}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def split_list(text: str) -> list[str]:
    """``;``-separated when any ``;`` is present (descriptors with several params contain commas), else ``,``."""
    return [x for x in text.split(";" if ";" in text else ",") if x]


def cmd_play(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["ours"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out / "matches.jsonl")}
    tasks = [{"ours": ours, "opponent": opp, "seed": s, "side": side, "cap_s": args.cap_s, "ckpt_root": args.ckpt_root}
             for s in range(args.seed0, args.seed0 + args.seeds) for opp in split_list(args.opponent)
             for ours in split_list(args.ours) for side in (0, 1)]
    tasks = [x for x in tasks if (x["ours"], x["opponent"], x["seed"], x["side"]) not in done]
    print(f"{len(tasks)} matches to play ({len(done)} done), {args.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(args.build, args.time_scale))
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
                    if n % 20 == 0 or not pending:
                        el = time.time() - t0
                        print(f"[{n}/{len(tasks)}] {el:.0f}s {n / el * 60:.1f} matches/min  last: {r['ours']} vs "
                              f"{r['opponent']} seed {r['seed']} side {r['side']} margin {r['margin']:+d} first "
                              f"{r['first_ours']}/{r['first_opp']} desync {r['desyncs']}", flush=True)
        print(f"done {n} matches in {time.time() - t0:.0f}s", flush=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race_sign(r) -> int:
    a = r["first_ours"] if r["first_ours"] >= 0 else 10 ** 9
    b = r["first_opp"] if r["first_opp"] >= 0 else 10 ** 9
    return int(np.sign(b - a))


def race(rows):
    s = [race_sign(r) for r in rows]
    return s.count(1), s.count(0), s.count(-1)


def paired(d, base):
    keys = sorted(k for k in d if k in base)
    per = defaultdict(list)
    for k in keys:
        per[k[0]].append(d[k]["margin"] - base[k]["margin"])
    ps = [np.mean(v) for v in per.values()]
    return float(np.mean(ps)) if ps else float("nan"), se(ps), len(ps)


def _index(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["ours"], _short(r["opponent"]))][(r["seed"], r["side"])] = r
    return by


def _short(opp: str) -> str:
    return "p11" if opp.endswith("p11_v22.pt") else opp


def cmd_report(args):
    by = _index(load_rows(Path(args.out) / "matches.jsonl"))
    sim = _index(load_rows(Path(args.sim))) if args.sim else {}
    lines, summary = [], {}
    for (ours, opp), d in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows = list(d.values())
        ms = [r["margin"] for r in rows]
        hist = Counter(r["first_ours"] for r in rows)
        top = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:4])
        ends = Counter(r["end"] for r in rows)
        line = (f"{ours:16s} vs {opp:14s} n={len(d):3d} margin {np.mean(ms):+6.1f}±{se(ms):4.1f} "
                f"win {np.mean([m > 0 for m in ms]):.2f} race w/t/l {race(rows)} first-ours {{{top}}} "
                f"end {dict(ends)} desync matches {sum(r['desyncs'] > 0 for r in rows)}")
        s = {"n": len(d), "margin": float(np.mean(ms)), "se": se(ms), "race": race(rows),
             "first_ours_hist": dict(hist), "desync_matches": sum(r["desyncs"] > 0 for r in rows)}
        base = by.get((args.base, opp), {})
        if base and ours != args.base:
            m, e, ns = paired(d, base)
            keys = [k for k in d if k in base]
            tr = Counter((race_sign(base[k]), race_sign(d[k])) for k in keys)
            name = {1: "w", 0: "t", -1: "l"}
            line += (f"\n   paired {ours} - {args.base} vs {opp}: {m:+6.1f} ± {e:4.1f} ({ns} seeds)  race "
                     + " ".join(f"{name[a]}>{name[b]}:{v}" for (a, b), v in sorted(tr.items()) if a != b))
            s |= {"paired": m, "paired_se": e, "seeds": ns}
        sd = sim.get((ours, opp), {})
        if sd:
            keys = sorted(k for k in d if k in sd)
            same_m = sum(d[k]["margin"] == sd[k]["margin"] for k in keys)
            same_f = sum((d[k]["first_ours"], d[k]["first_opp"]) == (sd[k]["first_ours"], sd[k]["first_opp"]) for k in keys)
            dm = [d[k]["margin"] - sd[k]["margin"] for k in keys]
            line += (f"\n   vs sim (same {len(keys)} matches): margin equal {same_m}, first-Hunter equal {same_f}, "
                     f"mean Δ(unity-sim) {np.mean(dm):+.2f}±{se(dm):.2f}; sim margin {np.mean([sd[k]['margin'] for k in keys]):+.1f}")
            sbase = sim.get((args.base, opp), {})
            if sbase and ours != args.base:
                sub = {k: sd[k] for k in keys}
                m, e, ns = paired(sub, {k: v for k, v in sbase.items() if k in d})
                line += f", sim paired − {args.base} {m:+.1f} ± {e:.1f} ({ns} seeds)"
                s |= {"sim_paired": m, "sim_paired_se": e}
            s |= {"sim_margin_equal": same_m, "sim_first_equal": same_f, "sim_n": len(keys)}
        lines.append(line)
        summary[f"{ours}/{opp}"] = s
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--build", default="build/linux_server/BlackOut.x86_64")
    q.add_argument("--ckpt-root", default=str(REPO), help="directory holding checkpoints/ (p11 / r5 shorthands)")
    q.add_argument("--ours", default="strategic_v23,strategic_v25", help="comma (or ;) list of descriptors")
    q.add_argument("--opponent", default="strategic_v23,p11", help="comma (or ;) list: p11 | r5 | v23 | v25 | descriptors")
    q.add_argument("--seeds", type=int, default=150)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--cap-s", type=float, default=150.0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--time-scale", type=float, default=20.0)
    q.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--base", default="strategic_v23")
    q.add_argument("--sim", default=None, help="race_rule_eval.py matches.jsonl to compare with")
    q.add_argument("--out", required=True)
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
