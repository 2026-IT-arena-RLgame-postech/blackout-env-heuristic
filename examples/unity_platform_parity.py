"""Platform parity of Unity builds: the Linux Dedicated Server build vs the Mac build.

Records exact per-decision traces (examples/sim_record_traces.py format: raw 44-float broadcast,
packed semantic map, the float32 moves Unity received) and compares them bit for bit.

``record``   closed-loop matches: ``--pair A,B`` (opponents.py descriptors, team A vs team B) on
             seeds ``--seed0 ..``, one ``.npz`` per seed.  Every worker plays a throwaway warm-up
             match first (the first match after a player starts is not reproducible from its seed,
             docs §13), so every recorded match is an ordinary one.
``replay``   open-loop: the moves of recorded traces sent unchanged to another build; only the
             Unity side can make the observations differ.
``compare``  two trace directories (same file names): per match the first decision whose
             observation (state / map) or move differs, whether the first difference is in the
             observation (Unity) or in the move on an identical observation (the Python policy's
             numerics), final scores, and summary counts.  Writes ``<out>/compare.json``.

    PYTHONPATH=. python examples/unity_platform_parity.py record --build build/linux_server/BlackOut.x86_64 \\
        --pair strategic_v23,strategic_v19 --seeds 50 --out reports/unity_parity/linux/v23_v19
    PYTHONPATH=. python examples/unity_platform_parity.py replay --build build/linux_server/BlackOut.x86_64 \\
        --traces reports/unity_parity/mac/v23_v19 --out reports/unity_parity/linux_replay/v23_v19
    PYTHONPATH=. python examples/unity_platform_parity.py compare reports/unity_parity/mac/v23_v19 \\
        reports/unity_parity/linux/v23_v19

docs/policy_v2_design_ko.md §21.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_ENV = None


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(2.0)
    try:
        _ENV[0].close()
    finally:
        os._exit(0)


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    from multiprocessing import util

    import torch

    from blackout_env import BlackOutEnv
    from examples.sim_record_traces import _Capture

    torch.set_num_threads(1)
    env = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    util.Finalize(env, env.close, exitpriority=10)
    _ENV = (env, _Capture(env))
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    obs, _ = env.reset(seed=7)  # warm-up: the player's first match is not seed-reproducible
    for _ in range(5):
        obs, *_ = env.step({a: np.zeros(2, np.float32) for a in obs})


def _record(task: dict) -> dict:
    from examples.sim_record_traces import play

    env, cap = _ENV
    t0 = time.time()
    arrays, meta = play(env, cap, task)
    np.savez_compressed(task["path"], **arrays)
    meta["wall_s"] = round(time.time() - t0, 2)
    return meta


def _replay(task: dict) -> dict:
    """Send a trace's recorded moves to this build (open loop) and record what it observes."""
    env, cap = _ENV
    from blackout_env.env.constants import agent_name

    src = np.load(task["src"])
    meta_src = json.loads(str(src["meta"]))
    moves, has = src["moves"], src["has_move"]
    t0 = time.time()
    obs, _ = env.reset(seed=meta_src["seed"])
    states, maps, sent, sent_has = [], [], [], []
    info = {}
    for t in range(len(moves)):
        states.append(cap.raw_state.copy())
        maps.append(cap.raw_map.copy())
        if not env.agents or t == len(moves) - 1:
            sent.append(np.zeros((10, 2), np.float32))
            sent_has.append(np.zeros(10, bool))
            break
        cap.moves[:] = 0.0
        cap.has_move[:] = False
        actions = {agent_name(k): moves[t, k] for k in range(10) if has[t, k] and agent_name(k) in obs}
        obs, _, terms, _, infos = env.step(actions)
        sent.append(cap.moves.copy())
        sent_has.append(cap.has_move.copy())
        if infos:
            info = next(iter(infos.values()))
        if any(terms.values()):
            states.append(cap.raw_state.copy())
            maps.append(cap.raw_map.copy())
            sent.append(np.zeros((10, 2), np.float32))
            sent_has.append(np.zeros(10, bool))
            break
    meta = {**meta_src, "replay_of": task["src"], "steps": len(states), "winner": info.get("winner"),
            "score_0": info.get("score_0"), "score_1": info.get("score_1"), "wall_s": round(time.time() - t0, 2)}
    np.savez_compressed(task["path"], raw_state=np.stack(states), raw_map=np.stack(maps), moves=np.stack(sent),
                        has_move=np.stack(sent_has), meta=json.dumps(meta))
    return meta


def _pool(args, fn, tasks):
    out = []
    ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             initargs=(args.build, args.time_scale))
    try:
        for fut in as_completed([ex.submit(fn, t) for t in tasks]):
            meta = fut.result()
            out.append(meta)
            print(f"{Path(meta.get('path', meta.get('replay_of', ''))).name} seed {meta['seed']}: "
                  f"{meta.get('score_0')} {meta.get('score_1')} steps={meta['steps']} {meta['wall_s']}s", flush=True)
    finally:
        ex.shutdown(cancel_futures=True)
    return out


def cmd_record(args):
    a, b = args.pair.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{"a": a, "b": b, "seed": s, "max_steps": args.max_steps, "path": str(out / f"seed_{s}.npz")}
             for s in range(args.seed0, args.seed0 + args.seeds) if not (out / f"seed_{s}.npz").exists()]
    t0 = time.time()
    metas = _pool(args, _record, tasks)
    print(f"recorded {len(metas)} matches in {time.time() - t0:.0f}s", flush=True)


def cmd_replay(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{"src": str(p), "path": str(out / p.name)} for p in sorted(Path(args.traces).glob("*.npz"))
             if not (out / p.name).exists()]
    _pool(args, _replay, tasks)


def _first_diff(a, b, n):
    for t in range(n):
        if not np.array_equal(a[t], b[t]):
            return t
    return None


def compare_pair(pa: Path, pb: Path) -> dict:
    A, B = np.load(pa), np.load(pb)
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    n = min(len(A["raw_state"]), len(B["raw_state"]))
    d_state = _first_diff(A["raw_state"], B["raw_state"], n)
    d_map = _first_diff(A["raw_map"], B["raw_map"], n)
    d_move = _first_diff(A["moves"], B["moves"], n - 1)
    obs_first = min(x for x in (d_state, d_map, n) if x is not None)
    # a move differs on an identical observation (moves[t] is sent after observation t) -> policy numerics
    if d_move is not None and d_move < obs_first:
        first = {"tick": d_move, "kind": "move"}
        unit = int(np.nonzero(np.any(A["moves"][d_move] != B["moves"][d_move], axis=1))[0][0])
        first["max_abs_move_diff"] = float(np.abs(A["moves"][d_move] - B["moves"][d_move]).max())
        first["unit"] = unit
    elif obs_first < n:
        first = {"tick": obs_first, "kind": "state" if obs_first == d_state else "map"}
    else:
        first = None
    blank = [int((A["raw_map"] == 0).all(axis=(1, 2)).sum()), int((B["raw_map"] == 0).all(axis=(1, 2)).sum())]
    const = [int((A["raw_map"] == A["raw_map"][:, :1, :1]).all(axis=(1, 2)).sum()),
             int((B["raw_map"] == B["raw_map"][:, :1, :1]).all(axis=(1, 2)).sum())]
    sa = (ma.get("score_0"), ma.get("score_1"))
    sb = (mb.get("score_0"), mb.get("score_1"))
    margin = lambda s: None if s[0] is None else round(100 * (s[0] - s[1]), 1)  # noqa: E731
    return {"file": pa.name, "seed": ma["seed"], "steps": [len(A["raw_state"]), len(B["raw_state"])],
            "identical": first is None and len(A["raw_state"]) == len(B["raw_state"]), "first_diff": first,
            "blank_map_frames": blank, "constant_map_frames": const,
            "map_bits_equal_frac": float(np.mean([np.array_equal(A["raw_map"][t], B["raw_map"][t]) for t in range(n)])),
            "scores": [sa, sb], "margin": [margin(sa), margin(sb)]}


def cmd_compare(args):
    da, db = Path(args.a), Path(args.b)
    rows = [compare_pair(p, db / p.name) for p in sorted(da.glob("*.npz")) if (db / p.name).exists()]
    for r in rows:
        print(f"{r['file']}: identical={r['identical']} first={r['first_diff']} steps={r['steps']} "
              f"margin {r['margin']} blank maps {r['blank_map_frames']}", flush=True)
    m = [(r["margin"][1] - r["margin"][0]) for r in rows if None not in r["margin"]]
    summ = {"a": str(da), "b": str(db), "matches": len(rows), "identical": sum(r["identical"] for r in rows),
            "first_diff_kind": {k: sum(1 for r in rows if r["first_diff"] and r["first_diff"]["kind"] == k)
                                for k in ("state", "map", "move")},
            "first_diff_tick": sorted(r["first_diff"]["tick"] for r in rows if r["first_diff"]),
            "margin_diff_mean": float(np.mean(m)) if m else None,
            "margin_diff_abs_mean": float(np.mean(np.abs(m))) if m else None,
            "margin_same_frac": float(np.mean(np.asarray(m) == 0)) if m else None,
            "margin_diff_se": float(np.std(m, ddof=1) / np.sqrt(len(m))) if len(m) > 1 else None,
            "blank_map_frames": [sum(r["blank_map_frames"][i] for r in rows) for i in (0, 1)],
            "rows": rows}
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))
    out = Path(args.out or db)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(summ, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("record", "replay"):
        q = sub.add_parser(name)
        q.add_argument("--build", required=True)
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--time-scale", type=float, default=20.0)
        q.add_argument("--out", required=True)
        if name == "record":
            q.add_argument("--pair", required=True, help="teamA,teamB descriptors")
            q.add_argument("--seeds", type=int, default=10)
            q.add_argument("--seed0", type=int, default=3_000_000)
            q.add_argument("--max-steps", type=int, default=0)
        else:
            q.add_argument("--traces", required=True)
    q = sub.add_parser("compare")
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("--out", default="")
    args = p.parse_args()
    {"record": cmd_record, "replay": cmd_replay, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
