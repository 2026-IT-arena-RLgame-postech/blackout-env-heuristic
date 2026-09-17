"""
Why is a checkpoint losing? Plays it against a heuristic and reports where its game breaks down.

Three questions the aggregate eval numbers (win rate, margin, blocked rate) could not answer
after Run 6, each of which took a separate ad-hoc script at the time -- see
docs/run6_diagnosis_20260916.md:

  time       when in a match do units start failing to move, and what do the Q-values look
             like at that moment (Run 6: 1.4% blocked in the first 50 ticks, 33% by tick 200,
             with the greedy margin halving over the same window)
  blocked    what is in front of a unit when its move fails -- a wall, another unit, or open
             floor (Run 6: 100% wall, 0% unit; units pass through each other)
  objectives approach -> pickup -> carry -> deliver, for the model and for its opponent in the
             same matches (Run 6: half the heuristic's pickups, but delivering fine once it
             had something -- the break was "does not go get a battery")

Usage:
    python -m examples.diagnose_checkpoint --checkpoint checkpoints/offline/<run>/final.pt
    python -m examples.diagnose_checkpoint --checkpoint ... --report time --seeds 1 2 3 --gui
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.heuristics import RecommendedStrategicHeuristic
from blackout_env.model.action_mask import DIR_CELL
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy, direction_vector_to_idx
from blackout_env.train.objective_monitor import ObjectiveCounts, ObjectiveMonitor, aggregate_objectives
from blackout_env.train.stall_monitor import StallMonitor, aggregate_stalls
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

WALL = 1
BLOCKED_MOVEMENT = 2e-4  # same threshold MovementMonitor uses
BLOCKED_ACTION_NORM = 0.35
TRANSITION_MOVEMENT = 0.15  # a respawn teleport, not a failed step


def _cell(position: np.ndarray, height: int, width: int) -> tuple[int, int]:
    return (min(height - 1, max(0, int(round((1.0 - float(position[1])) * 0.5 * height - 0.5)))),
            min(width - 1, max(0, int(round((float(position[0]) + 1.0) * 0.5 * width - 0.5)))))


class MatchRecord:
    """Per-tick candidate-side telemetry for one match."""

    def __init__(self, side: str) -> None:
        self.side = side
        self.ticks: list[dict[str, float]] = []
        self.blocked_causes: Counter = Counter()
        self.moving_ahead: Counter = Counter()
        self.objectives = ObjectiveMonitor()
        self.opponent_objectives = ObjectiveMonitor()
        self.stalls = StallMonitor()
        self.final_margin = 0.0


def play(env, policy: MyPolicy, opponent, seed: int, swap: bool) -> MatchRecord:
    team_a, team_b = list(team_a_agents()), list(team_b_agents())
    candidate_names = set(team_b if swap else team_a)
    opponent_names = set(team_a if swap else team_b)
    candidate_rows = sorted(unit_index(a) for a in candidate_names)
    opponent_rows = sorted(unit_index(a) for a in opponent_names)
    record = MatchRecord("TeamB" if swap else "TeamA")

    obs, _ = env.reset(seed=seed)
    empty_steps = 0
    final_info: dict = {}
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            if infos:
                final_info = next(iter(infos.values()))
            empty_steps += 1
            if empty_steps > 200:
                raise RuntimeError("Unity returned empty observations for over 200 steps")
            continue
        empty_steps = 0

        candidate_obs = {a: obs[a] for a in env.agents if a in candidate_names and a in obs}
        opponent_obs = {a: obs[a] for a in env.agents if a in opponent_names and a in obs}
        q_rows = policy.action_values(candidate_obs) if candidate_obs else {}
        actions = {agent: DIRECTION_VECTORS[int(np.argmax(row))] for agent, row in q_rows.items()}
        actions.update(opponent.act(opponent_obs) if opponent_obs else {})

        if candidate_obs:
            shared = candidate_obs[next(iter(candidate_obs))]
            record.objectives.observe(shared["graphic"], shared["agent_states"], candidate_rows)
        if opponent_obs:
            shared_opp = opponent_obs[next(iter(opponent_obs))]
            record.opponent_objectives.observe(shared_opp["graphic"], shared_opp["agent_states"], opponent_rows)

        before = next(iter(obs.values()))["agent_states"].copy()
        graphic = next(iter(obs.values()))["graphic"]
        next_obs, _, _, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))

        if next_obs and q_rows:
            after = next(iter(next_obs.values()))["agent_states"]
            shared = candidate_obs[next(iter(candidate_obs))]
            record.stalls.observe(shared["graphic"], before, after, actions_for_stalls := {
                agent: actions[agent] for agent in q_rows
            }, {agent: unit_index(agent) for agent in actions_for_stalls})
            height, width = graphic.shape[:2]
            walls = graphic[..., WALL] > 0.5
            cells = {row: _cell(before[row, :2], height, width) for row in range(before.shape[0])}
            blocked_flags, margins, tops = [], [], []
            for agent, row_q in q_rows.items():
                row = unit_index(agent)
                movement = float(np.linalg.norm(after[row, :2] - before[row, :2]))
                transition = movement > TRANSITION_MOVEMENT or not np.array_equal(before[row, 3:], after[row, 3:])
                blocked = (
                    float(np.linalg.norm(actions[agent])) >= BLOCKED_ACTION_NORM
                    and movement <= BLOCKED_MOVEMENT
                    and not transition
                )
                blocked_flags.append(float(blocked))
                ordered = np.sort(row_q)
                margins.append(ordered[-1] - ordered[-2])
                tops.append(ordered[-1])

                drow, dcol = DIR_CELL[int(direction_vector_to_idx(actions[agent][None, :])[0])]
                target = (cells[row][0] + drow, cells[row][1] + dcol)
                inside = 0 <= target[0] < height and 0 <= target[1] < width
                occupants = [o for o in cells if o != row and cells[o] == target]
                if not inside or walls[target]:
                    cause = "wall"
                elif occupants:
                    cause = "ally" if all((o < 5) == (row < 5) for o in occupants) else "enemy"
                else:
                    cause = "open"
                (record.blocked_causes if blocked else record.moving_ahead)[cause] += 1

            score_0 = float(final_info.get("score_0", 0.0)) * 100 if final_info else 0.0
            score_1 = float(final_info.get("score_1", 0.0)) * 100 if final_info else 0.0
            record.final_margin = (score_1 - score_0) if swap else (score_0 - score_1)
            record.ticks.append({
                "tick": len(record.ticks),
                "blocked": float(np.mean(blocked_flags)),
                "q_margin": float(np.mean(margins)),
                "q_max": float(np.mean(tops)),
                "score_diff": record.final_margin,
            })
        obs = next_obs
    return record


def report_time(records: list[MatchRecord]) -> None:
    print("\n=== when do things break down (all matches pooled by absolute tick) ===")
    rows = np.array([[t["tick"], t["blocked"], t["q_margin"], t["q_max"], t["score_diff"]]
                     for r in records for t in r.ticks])
    print(f"{'ticks':>12} {'n':>7} {'blocked':>9} {'q_margin':>9} {'q_max':>8} {'score_diff':>11}")
    edges = [0, 50, 100, 150, 200, 250, 300, 400, 500, 10_000]
    for low, high in zip(edges[:-1], edges[1:]):
        window = rows[(rows[:, 0] >= low) & (rows[:, 0] < high)]
        if len(window) < 20:
            continue
        print(f"{low:5d}-{high:<6d} {len(window):7d} {window[:, 1].mean():9.3f} {window[:, 2].mean():9.3f} "
              f"{window[:, 3].mean():8.3f} {window[:, 4].mean():11.1f}")

    runs: list[int] = []
    for record in records:
        length = 0
        for tick in record.ticks:
            if tick["blocked"] >= 0.4:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    if runs:
        runs_arr = np.array(runs)
        print(f"\nstretches with >=40% of the team blocked: n={len(runs_arr)} mean={runs_arr.mean():.0f} "
              f"median={np.median(runs_arr):.0f} max={runs_arr.max()} ticks")


def report_blocked(records: list[MatchRecord]) -> None:
    print("\n=== what is in front of a unit when its move fails ===")
    blocked, moving = Counter(), Counter()
    for record in records:
        blocked.update(record.blocked_causes)
        moving.update(record.moving_ahead)
    for label, counts in (("blocked unit-ticks", blocked), ("moving unit-ticks", moving)):
        total = max(1, sum(counts.values()))
        breakdown = "  ".join(f"{name}={count} ({count / total:.1%})" for name, count in counts.most_common())
        print(f"{label:>20}: {total:6d}   {breakdown}")


def report_objectives(records: list[MatchRecord]) -> None:
    print("\n=== scoring pipeline (model vs the heuristic it is playing) ===")
    model = aggregate_objectives([r.objectives.counts for r in records], "")
    opponent = aggregate_objectives([r.opponent_objectives.counts for r in records], "")
    print(f"{'metric':32} {'model':>12} {'opponent':>12}")
    for key in model:
        print(f"{key:32} {model[key]:12.4f} {opponent[key]:12.4f}")

    print("\n=== by side (the canonical team frame should make these match) ===")
    for side in ("TeamA", "TeamB"):
        subset = [r for r in records if r.side == side]
        if not subset:
            continue
        stats = aggregate_objectives([r.objectives.counts for r in subset], "")
        blocked = np.mean([t["blocked"] for r in subset for t in r.ticks])
        print(f"{side}: margin={np.mean([r.final_margin for r in subset]):7.1f}  blocked={blocked:.3f}  "
              f"approach_battery={stats['approach_battery']:+.4f}  pickups/1k={stats['pickups_per_1000_ticks']:.2f}")


def report_stalls(records: list[MatchRecord]) -> None:
    print("\n=== why a unit is standing still (see train/stall_monitor.py) ===")
    stats = aggregate_stalls([r.stalls.counts for r in records])
    labels = {
        "idle_on_storage_per_1000": "empty-handed, parked on a storage tile",
        "cargo_wait_full_per_1000": "carrying, at a storage that cannot take it",
        "cargo_wait_full_spawn_frac": "  ...of those, at the spawn storage",
        "cargo_wait_avoidable_frac": "  ...of those, another storage had room",
        "blocked_per_1000": "failed moves into a wall",
        "corner_block_frac": "  ...of those, a sidestep would have cleared it",
    }
    for key, label in labels.items():
        print(f"  {label:48s} {stats[key]:8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--report", nargs="+", default=["all"], choices=["all", "time", "blocked", "objectives", "stalls"])
    parser.add_argument("--gui", action="store_true", help="show the Unity window (slower, real-time by default)")
    parser.add_argument("--time-scale", type=float, default=None, help="default: 1.0 with --gui, else 20.0")
    parser.add_argument("--mask-walls", action="store_true", help="evaluate with wall-ward actions masked out")
    args = parser.parse_args()

    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=args.device, tb_log_dir=None))
    trainer.load(args.checkpoint)
    trainer.net.eval()
    policy = MyPolicy(trainer.net, device=args.device, mask_walls=args.mask_walls)
    opponent = RecommendedStrategicHeuristic()

    time_scale = args.time_scale if args.time_scale is not None else (1.0 if args.gui else 20.0)
    env = BlackOutEnv(str(args.build), time_scale=time_scale, no_graphics=not args.gui, unity_shaping=False)
    records: list[MatchRecord] = []
    try:
        for seed in args.seeds:
            for swap in (False, True):
                if hasattr(opponent, "reset"):
                    opponent.reset()
                record = play(env, policy, opponent, seed, swap)
                records.append(record)
                print(f"seed={seed} {record.side}: ticks={len(record.ticks)} margin={record.final_margin:+.0f} "
                      f"blocked={np.mean([t['blocked'] for t in record.ticks]):.3f}", flush=True)
    finally:
        env.close()

    wanted = set(args.report)
    if wanted & {"all", "time"}:
        report_time(records)
    if wanted & {"all", "blocked"}:
        report_blocked(records)
    if wanted & {"all", "objectives"}:
        report_objectives(records)
    if wanted & {"all", "stalls"}:
        report_stalls(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
