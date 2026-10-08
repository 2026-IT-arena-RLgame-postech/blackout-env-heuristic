"""Record exact replay traces of Unity matches for validating the Python simulator (S2, S0).

Each match is driven headless through BlackOutEnv with one policy per team and saved as one
compressed ``.npz`` holding, for every decision step ``t`` (25 per second):

  raw_state  float32 [T, 44]      MapObsAgent's raw vector broadcast at step t, before any
                                  preprocessing (positions normalised to [-1, 1], held item code,
                                  class id, scores / target, time left, absorption left)
  raw_map    uint16  [T, 24, 24]  the packed semantic map (bit layout in my_obs_preprocessor.py),
                                  rows top-down exactly as the env receives it
  moves      float32 [T, 10, 2]   the continuous action Unity received for every unit after the
                                  env's clipping, captured at ``UnityEnvironment.set_actions`` (so
                                  bit-exact); a unit with no decision this step gets 0
  has_move   bool    [T, 10]      whether the unit had a decision step (its move was sent)
  meta       json                 seed, policy descriptors, final scores, winner, steps

``moves[t]`` is the move applied after observation ``t`` (twice: decision tick + repeat tick), so
``raw_state[t + 1]`` is its result.  The last row has no move (terminal observation).

Policies are ``train/distill/opponents.make_opponent`` descriptors: any heuristic id
(``strategic_v21_anyangle``), ``mixture``, ``random``, ``ckpt:<path>`` for a student.

    PYTHONPATH=. .venv/bin/python examples/sim_record_traces.py --matches 40 --workers 4 \\
        --out reports/sim/traces
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import util as mp_util
from pathlib import Path

import numpy as np

STUDENT = "ckpt:checkpoints/distill/frozen/p5a_v22.pt"
DEFAULT_POOL = ["strategic_v21_anyangle", STUDENT, "strategic_v21", "strategic_v17", "mixture", "random"]

_ENV = None


class _Capture:
    """Instance-level hooks on one BlackOutEnv: the raw map broadcast and the sent actions."""

    def __init__(self, env) -> None:
        self.env = env
        self.raw_state: np.ndarray | None = None
        self.raw_map: np.ndarray | None = None
        self.moves = np.zeros((10, 2), dtype=np.float32)
        self.has_move = np.zeros(10, dtype=bool)
        pre = env._preprocessor
        orig_states, orig_graphics = pre.preprocess_agent_states, pre.preprocess_team_graphics

        def states(raw):
            self.raw_state = np.array(raw, dtype=np.float32, copy=True)
            return orig_states(raw)

        def graphics(raw):  # (H, W, 1) normalised packed value
            self.raw_map = np.rint(np.asarray(raw)[:, :, 0] * pre.PACK_DIVISOR).astype(np.uint16)
            return orig_graphics(raw)

        pre.preprocess_agent_states, pre.preprocess_team_graphics = states, graphics
        unity = env._unity_env
        orig_set = unity.set_actions

        def set_actions(behavior, action_tuple):
            if behavior.startswith("BlackOutUnit"):
                steps, _ = unity.get_steps(behavior)
                cont = np.asarray(action_tuple.continuous, dtype=np.float32)
                for i, agent_id in enumerate(steps.agent_id):
                    name = env._agent_name_cache.get((behavior, agent_id))
                    if name is not None:
                        k = int(name.split("_")[1])
                        self.moves[k] = cont[i]
                        self.has_move[k] = True
            return orig_set(behavior, action_tuple)

        unity.set_actions = set_actions


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    import torch

    from blackout_env import BlackOutEnv

    torch.set_num_threads(1)
    env = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    mp_util.Finalize(env, env.close, exitpriority=10)
    _ENV = (env, _Capture(env))


def play(env, cap: _Capture, task: dict) -> dict:
    from blackout_env.env.constants import team_a_agents, team_b_agents
    from blackout_env.train.distill.opponents import make_opponent

    pol_a = make_opponent(task["a"], task["seed"])
    pol_b = make_opponent(task["b"], task["seed"] + 1_000_003)
    names_a, names_b = set(team_a_agents()), set(team_b_agents())
    obs, _ = env.reset(seed=task["seed"])
    for p in (pol_a, pol_b):
        if hasattr(p, "reset"):
            p.reset()
    states, maps, moves, has = [], [], [], []
    info = {}
    while True:
        states.append(cap.raw_state.copy())
        maps.append(cap.raw_map.copy())
        if not env.agents or (task["max_steps"] and len(states) > task["max_steps"]):
            moves.append(np.zeros((10, 2), np.float32))
            has.append(np.zeros(10, bool))
            break
        cap.moves[:] = 0.0
        cap.has_move[:] = False
        actions = {}
        obs_a = {n: obs[n] for n in obs if n in names_a}
        obs_b = {n: obs[n] for n in obs if n in names_b}
        if obs_a:
            actions.update(pol_a.act(obs_a))
        if obs_b:
            actions.update(pol_b.act(obs_b))
        obs, _, terms, _, infos = env.step(actions)
        moves.append(cap.moves.copy())
        has.append(cap.has_move.copy())
        if infos:
            info = next(iter(infos.values()))
        if any(terms.values()):
            # the terminal broadcast: kept, then stop (the env has already reset Unity's match)
            states.append(cap.raw_state.copy())
            maps.append(cap.raw_map.copy())
            moves.append(np.zeros((10, 2), np.float32))
            has.append(np.zeros(10, bool))
            break
    meta = {**task, "steps": len(states), "winner": info.get("winner"),
            "score_0": info.get("score_0"), "score_1": info.get("score_1"),
            "time_left": info.get("time_left")}
    return dict(raw_state=np.stack(states), raw_map=np.stack(maps), moves=np.stack(moves),
                has_move=np.stack(has), meta=json.dumps(meta)), meta


def _run(task: dict) -> dict:
    env, cap = _ENV
    started = time.time()
    arrays, meta = play(env, cap, task)
    np.savez_compressed(task["path"], **arrays)
    meta["wall_s"] = round(time.time() - started, 1)
    return meta


def make_tasks(n: int, pool: list[str], seed: int, out: Path, max_steps: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        a, b = pool[i % len(pool)], pool[int(rng.integers(len(pool)))]
        if i % 2:
            a, b = b, a
        tasks.append({"a": a, "b": b, "seed": int(rng.integers(1, 2**30)), "max_steps": max_steps,
                      "path": str(out / f"trace_{i:03d}.npz")})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("reports/sim/traces"))
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--pool", nargs="+", default=DEFAULT_POOL)
    parser.add_argument("--max-steps", type=int, default=0, help="stop a match after this many decisions (0 = full)")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args.matches, args.pool, args.seed, args.out, args.max_steps)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             initargs=(args.build, args.time_scale)) as ex, open(args.out / "index.jsonl", "a") as index:
        for fut in as_completed([ex.submit(_run, t) for t in tasks]):
            meta = fut.result()
            index.write(json.dumps(meta) + "\n")
            index.flush()
            print(f"{Path(meta['path']).name} {meta['a']} vs {meta['b']}: {meta['score_0']} {meta['score_1']} "
                  f"steps={meta['steps']} {meta['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
