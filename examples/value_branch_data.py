"""S1.6: simulator branch data for the value net (docs/policy_v2_design_ko.md "S1.6 분기 데이터 가치망").

Generation only (training: examples/value_branch_train.py, targets: blackout_env/value/branch.py).

Root matches: strategic_v23 (the learner) vs strategic_v23 / v22 / v21_anyangle / v17 / v19, seeds
960000+ (disjoint from S1.5's 950000-950699), both sides, played to the end (420 s at most).  Root
ticks along each match at a per-bucket spacing (ROOT_SPACING_S, with jitter).  From each root the
learner plays each of value_branch_eval.BRANCHES for W = 2 s (the opponent its own policy; the
learner's history-dependent branch policies shadowed through the match exactly as in S1.5), then
both teams continue for K = 5 s (learner: fresh strategic_v23, opponent: its instance; the sim is
reseeded once per root -- the same reseed for every branch).  Recorded per branch: the window-end
state ``s`` (both teams' views, training encoding), the lead at +2 s and +5 s of the continuation,
and the state ``s'`` after the 5 s (both views) -- or the exact final lead when the match ended.

On-trajectory rows: every TRAJ_EVERY_S of the root match itself (V23 vs the opponent), the state,
the leads 2 s / 5 s later and the state 5 s later (or the final lead).

    PYTHONPATH=$PWD .venv/bin/python examples/value_branch_data.py --workers 6 --seeds 450 \
        --out datasets/value_s2v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

import value_branch_eval as vb  # noqa: E402

DPS = vb.DPS
W = 50  # branch window, decisions (2 s)
K = 125  # continuation, decisions (5 s)
K2 = 50  # +2 s lead inside the continuation
OPPONENTS = vb.OPPONENTS  # no P10 student: heuristics only
# root spacing (seconds) by elapsed time: dense early (fast-changing, the opening race), sparse late
ROOT_SPACING_S = ((10, 2.0), (30, 3.0), (60, 4.0), (150, 5.0), (420, 10.0))
TRAJ_EVERY_S = 1
MAX_TICKS = 420 * DPS


def root_schedule(rng: np.random.Generator) -> list[int]:
    ticks, t = [], float(rng.uniform(0, ROOT_SPACING_S[0][1]))
    while t * DPS < MAX_TICKS:
        ticks.append(int(t * DPS))
        spacing = next(s for hi, s in ROOT_SPACING_S if t < hi)
        t += spacing * float(rng.uniform(0.5, 1.5))
    return ticks


class Recorder:
    """Encoded states (both teams' views, learner first) appended to flat arrays."""

    def __init__(self, team: int):
        self.team = team
        self.cols = {k: [] for k in ("g_own", "ts_own", "ag_own", "g_mir", "ts_mir", "ag_mir")}

    def add(self, obs) -> int:
        from blackout_env.value.evaluate import encode_view, team_view

        for side, pre in ((self.team, "own"), (1 - self.team, "mir")):
            g, ts, ag = encode_view(team_view(obs, side))
            self.cols[f"g_{pre}"].append(g)
            self.cols[f"ts_{pre}"].append(ts)
            self.cols[f"ag_{pre}"].append(ag)
        return len(self.cols["g_own"]) - 1

    def arrays(self) -> dict:
        if not self.cols["g_own"]:
            return {"g_own": np.zeros((0, 24, 24, 13), np.uint8)}
        return {k: np.stack(v) for k, v in self.cols.items()}


def continue_k(sim, opp, team: int, reseed: int | None):
    """Both teams continue for K decisions (learner fresh V23).  Returns (lead +2 s, lead +5 s,
    over, obs at +5 s or None)."""
    if reseed is not None:
        sim.reseed(reseed)
    learner = vb._make("strategic_v23")
    lead2 = None
    for step in range(K):
        if sim.over:
            break
        if step == K2:
            lead2 = vb.lead_of(sim, team)
        sim.step(vb.joint_moves(sim.obs(), learner, opp, team))
    lead5 = vb.lead_of(sim, team)
    if lead2 is None:  # ended before +2 s
        lead2 = lead5
    return lead2, lead5, sim.over, (None if sim.over else sim.obs())


def _worker(task: dict) -> dict:
    vb._init_threads()
    from blackout_env.sim import SimState

    started = time.time()
    team, opp_desc = task["learner_team"], task["opponent"]
    rng = np.random.default_rng(zlib.crc32(task["id"].encode()))
    roots = set(root_schedule(rng))
    sim = SimState.from_seed(task["seed"])
    learner, opp = vb._make("strategic_v23"), vb._make(opp_desc, task["seed"])
    shadow = {h: vb._make(h) for h in vb.HISTORY}
    rec = Recorder(team)
    # per training row: kind (0 traj, 1 branch), state idx, next idx (-1 = ended), leads, ...
    rows = {k: [] for k in ("kind", "idx", "next_idx", "lead", "lead2", "lead5", "ended", "t_s", "root", "branch")}
    traj_ticks, traj_idx, lead_by_tick = [], [], []
    n_roots = 0
    tick = 0
    while not sim.over and tick < MAX_TICKS:
        lead_by_tick.append(vb.lead_of(sim, team))
        obs = sim.obs()
        if tick % (TRAJ_EVERY_S * DPS) == 0:
            traj_ticks.append(tick)
            traj_idx.append(rec.add(obs))
        if tick in roots:
            n_roots += 1
            seed = zlib.crc32(f"{task['id']}:{tick}".encode()) % 1_000_000
            for b, name in enumerate(vb.BRANCHES):
                pol = vb.branch_policy(name, shadow, seed * 31 + b, team)
                o = vb.clone_opponent(opp_desc, opp)
                s = sim.copy()
                for _ in range(W):
                    if s.over:
                        break
                    s.step(vb.joint_moves(s.obs(), pol, o, team))
                if s.over:  # ended inside the window: the end state is terminal, nothing to learn
                    continue
                i = rec.add(s.obs())
                lead = vb.lead_of(s, team)
                lead2, lead5, ended, obs5 = continue_k(s, o, team, 2_000_003 + seed)
                rows["kind"].append(1)
                rows["idx"].append(i)
                rows["next_idx"].append(-1 if ended else rec.add(obs5))
                rows["lead"].append(lead)
                rows["lead2"].append(lead2)
                rows["lead5"].append(lead5)
                rows["ended"].append(ended)
                rows["t_s"].append((tick + W) / DPS)
                rows["root"].append(tick)
                rows["branch"].append(b)
        own = {n: obs[n] for n in vb.team_names(team)}
        for h in shadow.values():
            h.act(own)
        sim.step(vb.joint_moves(obs, learner, opp, team))
        tick += 1
    end_tick, final = tick, vb.lead_of(sim, team)
    lead_by_tick.append(final)
    by_tick = dict(zip(traj_ticks, traj_idx))
    for t, i in zip(traj_ticks, traj_idx):
        ended = t + K >= end_tick and sim.over
        if t + K > end_tick and not sim.over:  # hit MAX_TICKS without an end (should not happen)
            continue
        rows["kind"].append(0)
        rows["idx"].append(i)
        rows["next_idx"].append(-1 if ended else by_tick.get(t + K, -1))
        rows["lead"].append(lead_by_tick[t])
        rows["lead2"].append(lead_by_tick[min(t + K2, end_tick)])
        rows["lead5"].append(lead_by_tick[min(t + K, end_tick)])
        rows["ended"].append(ended)
        rows["t_s"].append(t / DPS)
        rows["root"].append(-1)
        rows["branch"].append(-1)
    out = {k: np.asarray(v) for k, v in rows.items()}
    # a traj row whose +5 s tick was not recorded (only possible at MAX_TICKS) would have next -1
    # without ``ended``: drop it
    keep = out["ended"] | (out["next_idx"] >= 0)
    out = {k: v[keep] for k, v in out.items()}
    return {"task": task, "rows": out, "states": rec.arrays(), "final": final, "end_tick": end_tick,
            "over": bool(sim.over), "n_roots": n_roots, "seconds": time.time() - started}


def tasks_for(n_seeds: int, seed0: int) -> list[dict]:
    out = []
    for j in range(n_seeds):
        for side in (0, 1):
            out.append({"id": f"{seed0 + j}_{'AB'[side]}", "seed": seed0 + j, "learner_team": side,
                        "opponent": OPPONENTS[j % len(OPPONENTS)]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "datasets/value_s2v")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seeds", type=int, default=450)
    p.add_argument("--seed0", type=int, default=960000)
    p.add_argument("--limit", type=int, default=None, help="first N tasks only (timing)")
    args = p.parse_args()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks = tasks_for(args.seeds, args.seed0)[: args.limit]
    todo = [t for t in tasks if not (raw / f"{t['id']}.npz").exists()]
    print(f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    started = time.time()
    n_rows = n_branch = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_worker, t) for t in todo]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            tid = res["task"]["id"]
            meta = {"task": res["task"], "final": res["final"], "end_tick": res["end_tick"], "over": res["over"],
                    "n_roots": res["n_roots"], "seconds": res["seconds"]}
            np.savez(raw / f"{tid}.tmp.npz", meta=json.dumps(meta), **{f"r_{k}": v for k, v in res["rows"].items()},
                     **{f"s_{k}": v for k, v in res["states"].items()})
            os.replace(raw / f"{tid}.tmp.npz", raw / f"{tid}.npz")
            n_rows += len(res["rows"]["kind"])
            n_branch += int((res["rows"]["kind"] == 1).sum())
            if i % 20 == 0 or i == len(todo) - 1:
                el = time.time() - started
                print(f"[{i + 1}/{len(todo)}] {tid} {res['end_tick'] / DPS:.0f}s match, {res['n_roots']} roots, "
                      f"{res['seconds']:.0f}s; total rows {n_rows} (branch {n_branch}) elapsed {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
