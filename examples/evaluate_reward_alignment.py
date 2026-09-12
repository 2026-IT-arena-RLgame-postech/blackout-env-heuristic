"""
Empirical reward-model / outcome alignment check, run at high time_scale using existing
heuristic policies (no training needed). Answers, with real trajectories from the shipped
production reward_config.json (all fixed event rewards 0; only Psi_k/Phi_i shaping + terminal
+-1 remain -- see reward_proposal.md Sections 12/14/15):

  1) Macro (game outcome): does cumulative pre-terminal shaped return actually track who wins?
  2) Meso (absorption events): when a team's CONFIRMED score changes at an absorption tick,
     does the shaped reward at that same tick move the same direction?
  3) Micro (battery pickup/deposit, and combat's *absence* of direct reward): does an
     individual custody change (pickup from field, deposit into storage, or losing a carried
     battery to death/steal) move the shaped reward the way it should, and is it true that
     kills/deaths themselves (independent of item custody) carry zero direct reward?

No new Unity instrumentation is used -- everything here is decoded from the same
obs/rewards/infos a training loop already receives:
  - obs["agent_states"]: per-unit (pos_x, pos_y, team_sign, ..., battery_amount, ...) -- see
    blackout_env/env/my_obs_preprocessor.py's preprocess_agent_states docstring. Column 4 is
    always the battery-holding slot (index 1 of the item one-hot), regardless of n_items.
  - infos: score_0, score_1, absorption_time_left, time_left, winner (terminal step only).
  - rewards: per-agent scalar reward for that tick.

Usage
-----
    cd /Users/mac/project/26rl/blackout-env
    ./.venv/bin/python examples/evaluate_reward_alignment.py --build build/mac/BlackOut.app
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents, N_TEAM_A
from blackout_env.heuristics import (
    StrategicHeuristicV1, StrategicHeuristicV4, StrategicHeuristicV6, StrategicHeuristicV8,
)
from blackout_env.model.base import BaseModel

# Config variants for isolating which shaping term drives a given tick's reward (same idea as
# examples/evaluate_reward_model.py's VARIANTS). "psi_only" zeroes individual nav-shaping so
# team_a_reward is driven purely by the team potential Psi_k -- used to check whether a
# meso-level misalignment comes from Psi_k itself or from nav-shaping's per-unit discontinuity.
BASE_REWARD_CONFIG = {
    "itemRewards": [{"itemName": "Battery", "value": 0.0}, {"itemName": "SpeedBuff", "value": 0.0}],
    "killReward": 0.0, "deathPenalty": 0.0, "teamScoreReward": 0.0, "teamScorePenalty": 0.0,
    "potentialEta": 0.25, "potentialGamma": 0.99995, "potentialScale": 40.0,
    "hazardCoefficient": 0.05, "navPotentialEta": 0.08, "navPotentialScale": 12.0,
}
REWARD_VARIANTS = {
    "full": BASE_REWARD_CONFIG,
    "psi_only": {**BASE_REWARD_CONFIG, "navPotentialEta": 0.0},
}


def write_reward_config(build: Path, variant: str) -> str:
    path = build / "Contents/Resources/Data/StreamingAssets/reward_config.json"
    original = path.read_text()
    path.write_text(json.dumps(REWARD_VARIANTS[variant], indent=4))
    return original

BATTERY_COL = 4          # agent_states[:, 4] -- see module docstring
POS_COLS = slice(0, 2)   # agent_states[:, 0:2]
TELEPORT_DIST = 0.35     # normalized-position jump beyond which we call it a death/respawn,
                          # not a voluntary stop-and-deposit (map coords are in [-1, 1])


@dataclass
class StepRecord:
    team_a_reward: float
    team_b_reward: float
    per_agent_reward: dict[str, float]
    agent_states: np.ndarray   # (10, agent_state_size), team-A perspective (objective pos/item)
    score_a: float
    score_b: float


@dataclass
class EpisodeLog:
    steps: list[StepRecord] = field(default_factory=list)
    winner: int | None = None
    seed: int | None = None


def run_logged_match(env: BlackOutEnv, model_a: BaseModel, model_b: BaseModel, seed: int) -> EpisodeLog:
    team_a_names = set(team_a_agents())
    team_b_names = set(team_b_agents())
    controller = {n: model_a for n in team_a_names}
    controller.update({n: model_b for n in team_b_names})

    obs, _ = env.reset(seed=seed)
    model_a.reset() if hasattr(model_a, "reset") else None
    model_b.reset() if hasattr(model_b, "reset") else None
    log = EpisodeLog(seed=seed)
    final_info: dict = {}
    all_agent_names = team_a_names | team_b_names
    last_states: np.ndarray | None = None  # forward-filled: same shared array every agent sees

    while env.agents:
        a_obs = {a: obs[a] for a in env.agents if controller[a] is model_a and a in obs}
        b_obs = {a: obs[a] for a in env.agents if controller[a] is model_b and a in obs}
        actions: dict[str, np.ndarray] = {}
        if a_obs:
            actions.update(model_a.act(a_obs))
        if b_obs:
            actions.update(model_b.act(b_obs))

        obs, rewards, terminations, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))

        r_a = sum(rewards[a] for a in team_a_names)
        r_b = sum(rewards[a] for a in team_b_names)
        # Only agents whose decision is due THIS tick appear in obs (DecisionPeriod-gated), so
        # a fixed agent name can be absent on any given step -- grab agent_states from whichever
        # of the 10 happens to be present (they all share the identical MapObsAgent-sourced
        # table) and forward-fill on the rare step where none of them decided this tick.
        for name in all_agent_names:
            if name in obs and obs[name].get("agent_states") is not None:
                last_states = np.array(obs[name]["agent_states"], copy=True)
                break
        log.steps.append(StepRecord(
            team_a_reward=r_a, team_b_reward=r_b,
            per_agent_reward=dict(rewards),
            agent_states=last_states,
            score_a=float(final_info.get("score_0", 0.0)),
            score_b=float(final_info.get("score_1", 0.0)),
        ))

    physical_winner = final_info.get("winner")
    log.winner = None if physical_winner in (-1, None) else int(physical_winner)
    return log


# ---------------------------------------------------------------------------
# Macro: cumulative shaped return vs actual outcome
# ---------------------------------------------------------------------------

def analyze_macro(logs: list[EpisodeLog]) -> dict:
    per_ep_corr = []
    final_adv = []       # sign(cumulative shaped advantage) at episode end
    winner_sign = []     # +1 if A won, -1 if B won (draws excluded)
    terminal_matches = 0
    terminal_total = 0

    for log in logs:
        if log.winner is None or len(log.steps) < 2:
            continue
        a_cum = np.cumsum([s.team_a_reward for s in log.steps])
        b_cum = np.cumsum([s.team_b_reward for s in log.steps])
        adv = a_cum - b_cum
        score_diff = np.array([s.score_a - s.score_b for s in log.steps], dtype=float)

        # Exclude the terminal step's +-1 broadcast so this measures the *shaping* signal,
        # not the terminal reward re-deriving the outcome it's defined from.
        if len(adv) > 1 and np.std(adv[:-1]) > 1e-9 and np.std(score_diff[:-1]) > 1e-9:
            per_ep_corr.append(float(np.corrcoef(adv[:-1], score_diff[:-1])[0, 1]))

        final_adv.append(np.sign(adv[-2] if len(adv) > 1 else adv[-1]))
        winner_sign.append(1.0 if log.winner == 0 else -1.0)

        terminal_total += 1
        if np.sign(log.steps[-1].team_a_reward) == winner_sign[-1]:
            terminal_matches += 1

    final_adv = np.array(final_adv)
    winner_sign = np.array(winner_sign)
    concord = float(np.mean(final_adv == winner_sign)) if len(final_adv) else float("nan")

    return {
        "n_decisive_episodes": int(len(winner_sign)),
        "mean_corr_adv_vs_scorediff": float(np.mean(per_ep_corr)) if per_ep_corr else float("nan"),
        "pre_terminal_advantage_sign_matches_winner": concord,
        "terminal_reward_sign_matches_winner": (terminal_matches / terminal_total) if terminal_total else float("nan"),
    }


# ---------------------------------------------------------------------------
# Meso: absorption (confirmed score change) events
# ---------------------------------------------------------------------------

def analyze_meso_offsets(logs: list[EpisodeLog], offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)) -> dict:
    """Same event set as analyze_meso, but pairs each score-change tick's delta with the team
    reward at tick+offset instead of tick+0, to check whether a fixed decision/observation
    timing lag (not a reward-model defect) explains a same-tick mismatch."""
    out = {}
    for off in offsets:
        deltas, rewards_ = [], []
        for log in logs:
            steps = log.steps
            prev_diff = None
            for t, s in enumerate(steps):
                diff = s.score_a - s.score_b
                if prev_diff is not None and abs(diff - prev_diff) > 1e-6:
                    rt = t + off
                    if 0 <= rt < len(steps):
                        deltas.append(diff - prev_diff)
                        rewards_.append(steps[rt].team_a_reward)
                prev_diff = diff
        d, r = np.array(deltas), np.array(rewards_)
        if len(d) == 0:
            out[off] = {"n": 0}
            continue
        out[off] = {
            "n": int(len(d)),
            "sign_agreement_rate": float(np.mean(np.sign(d) == np.sign(r))),
            "corr": float(np.corrcoef(d, r)[0, 1]) if np.std(r) > 1e-9 else float("nan"),
        }
    return out


def analyze_meso_clean(logs: list[EpisodeLog], window: int = 3, offset: int = 1) -> dict:
    """Same event set as analyze_meso, but restricted to "clean" events -- score-change ticks
    with no OTHER score-change tick within +-window steps in the same episode. Tests whether the
    ~50% sign agreement that remains even after the offset=+1 timing correction (see
    analyze_meso_offsets) is caused by concurrent, unrelated custody-change events from the other
    9 units landing nearby and adding noise to any one isolated event, as reward_proposal.md
    §16.5 speculated -- if so, isolated events should show much better agreement than the pooled
    set."""
    event_steps: list[tuple[list, int, float]] = []  # (steps, index, delta_score)
    for log in logs:
        steps = log.steps
        prev_diff = None
        idxs = []
        for t, s in enumerate(steps):
            diff = s.score_a - s.score_b
            if prev_diff is not None and abs(diff - prev_diff) > 1e-6:
                idxs.append((t, diff - prev_diff))
            prev_diff = diff
        for i, (t, d) in enumerate(idxs):
            isolated = (i == 0 or t - idxs[i - 1][0] > window) and \
                       (i == len(idxs) - 1 or idxs[i + 1][0] - t > window)
            if isolated:
                event_steps.append((steps, t, d))

    deltas, rewards_ = [], []
    for steps, t, d in event_steps:
        rt = t + offset
        if 0 <= rt < len(steps):
            deltas.append(d)
            rewards_.append(steps[rt].team_a_reward)
    d, r = np.array(deltas), np.array(rewards_)
    if len(d) == 0:
        return {"n_clean_events": 0}
    return {
        "n_clean_events": int(len(d)),
        "sign_agreement_rate": float(np.mean(np.sign(d) == np.sign(r))),
        "corr": float(np.corrcoef(d, r)[0, 1]) if np.std(r) > 1e-9 else float("nan"),
    }


def analyze_meso_magnitude(logs: list[EpisodeLog], offset: int = 1) -> dict:
    """Checks not just the SIGN but the SCALE of the meso reward signal: does a bigger score
    swing (a bigger battery, or several at once) actually produce a proportionally bigger
    reward, or does the reward saturate/flatten regardless of event size? Uses the offset=+1
    timing correction established by analyze_meso_offsets."""
    deltas, rewards_ = [], []
    for log in logs:
        steps = log.steps
        prev_diff = None
        for t, s in enumerate(steps):
            diff = s.score_a - s.score_b
            if prev_diff is not None and abs(diff - prev_diff) > 1e-6:
                rt = t + offset
                if 0 <= rt < len(steps):
                    deltas.append(diff - prev_diff)
                    rewards_.append(steps[rt].team_a_reward)
            prev_diff = diff
    d, r = np.array(deltas), np.abs(np.array(rewards_))
    ad = np.abs(d)
    if len(d) == 0:
        return {"n": 0}

    # Bin by |delta_score| and report mean |reward| per bin -- if magnitude tracked event size,
    # mean |reward| should increase with |delta_score| across bins, not stay flat. Quantile-based
    # so this adapts to whatever the actual delta_score distribution turns out to be, instead of
    # guessing fixed thresholds that might all land in (or miss) the same bucket.
    bins: dict[str, dict] = {}
    edges = sorted(set(np.quantile(ad, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ad >= lo) & (ad <= hi) if hi == edges[-1] else (ad >= lo) & (ad < hi)
        if mask.any():
            bins[f"[{lo:.3g}, {hi:.3g}{']' if hi == edges[-1] else ')'}"] = {
                "n": int(mask.sum()), "mean_abs_reward": float(np.mean(r[mask])),
            }
    top_vals = [float(v) for v, _ in Counter(np.round(ad, 4)).most_common(5)]

    slope, intercept = (np.polyfit(ad, r, 1) if np.std(ad) > 1e-9 else (float("nan"), float("nan")))
    corr = float(np.corrcoef(ad, r)[0, 1]) if np.std(ad) > 1e-9 and np.std(r) > 1e-9 else float("nan")
    return {
        "n": int(len(d)),
        "corr_abs_delta_vs_abs_reward": corr,
        "linear_fit_abs_reward_per_abs_delta": float(slope),
        "by_magnitude_bin": bins,
        "most_common_abs_delta_values": top_vals,
    }


def analyze_meso(logs: list[EpisodeLog]) -> dict:
    delta_scores = []
    step_rewards = []

    for log in logs:
        prev_diff = None
        for s in log.steps:
            diff = s.score_a - s.score_b
            if prev_diff is not None and abs(diff - prev_diff) > 1e-6:
                delta_scores.append(diff - prev_diff)
                step_rewards.append(s.team_a_reward)
            prev_diff = diff

    delta_scores = np.array(delta_scores)
    step_rewards = np.array(step_rewards)
    n = len(delta_scores)
    if n == 0:
        return {"n_absorption_events": 0}

    def signed_stats(mask: np.ndarray) -> dict:
        d, r = delta_scores[mask], step_rewards[mask]
        if len(d) == 0:
            return {"n": 0}
        out = {
            "n": int(len(d)),
            "sign_agreement_rate": float(np.mean(np.sign(d) == np.sign(r))),
            "mean_reward_at_event": float(np.mean(r)),
        }
        if np.std(r) > 1e-9:
            out["corr"] = float(np.corrcoef(d, r)[0, 1])
        return out

    sign_match = float(np.mean(np.sign(delta_scores) == np.sign(step_rewards)))
    corr = float(np.corrcoef(delta_scores, step_rewards)[0, 1]) if np.std(step_rewards) > 1e-9 else float("nan")
    return {
        "n_absorption_events": n,
        "sign_agreement_rate": sign_match,
        "corr_delta_score_vs_reward_at_tick": corr,
        "mean_abs_reward_at_event": float(np.mean(np.abs(step_rewards))),
        # Pooling score-up (deposit-like) and score-down (steal/loss-like) events can hide a
        # one-sided problem -- split them out.
        "score_increased_events": signed_stats(delta_scores > 0),
        "score_decreased_events": signed_stats(delta_scores < 0),
    }


# ---------------------------------------------------------------------------
# Micro: per-unit battery pickup / deposit / death-drop custody transitions
# ---------------------------------------------------------------------------

def analyze_micro(logs: list[EpisodeLog]) -> dict:
    pickup_rewards = []    # (reward, item_amount) at the tick a unit's battery slot goes 0 -> >0
    deposit_rewards = []   # ... goes >0 -> 0 AND unit doesn't teleport (voluntary drop)
    death_drop_rewards = []  # ... goes >0 -> 0 AND unit teleports (death/respawn while carrying)

    for log in logs:
        steps = log.steps
        if any(s.agent_states is None for s in steps):
            continue
        for t in range(1, len(steps) - 1):
            prev_states = steps[t - 1].agent_states
            cur_states = steps[t].agent_states
            team_a_reward = steps[t].team_a_reward
            team_b_reward = steps[t].team_b_reward

            for unit_idx in range(cur_states.shape[0]):
                prev_amt = prev_states[unit_idx, BATTERY_COL]
                cur_amt = cur_states[unit_idx, BATTERY_COL]
                is_team_a = unit_idx < N_TEAM_A
                own_reward = team_a_reward if is_team_a else team_b_reward

                if prev_amt <= 1e-6 < cur_amt:
                    pickup_rewards.append((own_reward, cur_amt))
                elif prev_amt > 1e-6 >= cur_amt:
                    # Look a few ticks ahead: did this unit teleport (death/respawn) or stay put
                    # (voluntary deposit / stolen in place)?
                    look_ahead = min(t + 3, len(steps) - 1)
                    later_pos = steps[look_ahead].agent_states[unit_idx, POS_COLS]
                    cur_pos = cur_states[unit_idx, POS_COLS]
                    teleported = float(np.linalg.norm(later_pos - cur_pos)) > TELEPORT_DIST
                    (death_drop_rewards if teleported else deposit_rewards).append((own_reward, prev_amt))

    def summarize(vals: list[tuple[float, float]], expect_positive: bool) -> dict:
        if not vals:
            return {"n": 0}
        arr = np.array([v[0] for v in vals])
        amounts = np.array([v[1] for v in vals])
        good_sign = arr >= 0 if expect_positive else arr <= 0
        out = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "frac_expected_sign": float(np.mean(good_sign)),
        }
        # Does a bigger battery (larger held amount) produce a proportionally bigger |reward|?
        if np.std(amounts) > 1e-9 and np.std(np.abs(arr)) > 1e-9:
            out["corr_amount_vs_abs_reward"] = float(np.corrcoef(amounts, np.abs(arr))[0, 1])
        return out

    return {
        "pickup_from_field": summarize(pickup_rewards, expect_positive=True),
        # NOTE: "expect_positive" is a rough default, not a hard requirement here. Once
        # IndividualNavPotentialCalculator's fetch potential accounts for the full remaining
        # delivery distance (2026-09-12 fix, see reward_proposal.md §16.5-1), a deposit
        # legitimately CAN show a net-negative reward at that tick if the unit's next fetch
        # target is far away -- finishing one delivery honestly means more work remains for the
        # next one, which isn't a reward-model defect. Don't read a low frac_expected_sign here
        # as proof of a bug without checking whether it's just this effect.
        "voluntary_deposit_or_inplace_loss": summarize(deposit_rewards, expect_positive=True),
        "death_drop_while_carrying": summarize(death_drop_rewards, expect_positive=False),
    }


POLICY_MAP = {
    "V1": StrategicHeuristicV1, "V4": StrategicHeuristicV4,
    "V6": StrategicHeuristicV6, "V8": StrategicHeuristicV8,
}


def run_shard(build_str: str, time_scale: float, no_graphics: bool,
              jobs: list[tuple[str, str, int, bool]]) -> list[tuple[str, str, int, bool, EpisodeLog]]:
    """Runs one BlackOutEnv instance (its own Unity process/port) through a list of
    (name_a, name_b, seed, swap) jobs sequentially. Used as the unit of work handed to each
    worker process so N shards -- and therefore N Unity instances -- run concurrently."""
    env = BlackOutEnv(build_str, time_scale=time_scale, no_graphics=no_graphics)
    results = []
    try:
        for name_a, name_b, seed, swap in jobs:
            cls_a, cls_b = POLICY_MAP[name_a], POLICY_MAP[name_b]
            model_a, model_b = (cls_a(), cls_b()) if not swap else (cls_b(), cls_a())
            log = run_logged_match(env, model_a, model_b, seed=seed)
            results.append((name_a, name_b, seed, swap, log))
    finally:
        env.close()
    return results


def shard(items: list, n: int) -> list[list]:
    return [items[i::n] for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--time-scale", type=float, default=200.0)
    parser.add_argument("--pairs", type=str, nargs="+", default=["V4:V1", "V6:V1", "V8:V4"])
    parser.add_argument("--no-graphics", action="store_true", help="Run headless instead of the GUI player")
    parser.add_argument("--variants", type=str, nargs="+", default=["full", "psi_only"],
                         choices=list(REWARD_VARIANTS))
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of parallel Unity instances (each gets its own port).")
    args = parser.parse_args()

    build = args.build.resolve()
    config_path = build / "Contents/Resources/Data/StreamingAssets/reward_config.json"
    original_config = config_path.read_text()

    all_jobs: list[tuple[str, str, int, bool]] = []
    for pair in args.pairs:
        name_a, name_b = pair.split(":")
        for seed in args.seeds:
            for swap in (False, True):
                all_jobs.append((name_a, name_b, seed, swap))

    logs_by_variant: dict[str, list[EpisodeLog]] = {}
    try:
        for variant in args.variants:
            write_reward_config(build, variant)
            all_logs: list[EpisodeLog] = []

            shards = shard(all_jobs, max(1, args.workers))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(run_shard, str(build), args.time_scale, args.no_graphics, j)
                    for j in shards if j
                ]
                for fut in as_completed(futures):
                    for name_a, name_b, seed, swap, log in fut.result():
                        all_logs.append(log)
                        label = "?" if log.winner is None else ("A" if log.winner == 0 else "B")
                        a_name = name_a if not swap else name_b
                        b_name = name_b if not swap else name_a
                        print(f"[{variant}] [{a_name} vs {b_name}] seed={seed} winner={label} "
                              f"steps={len(log.steps)} score={log.steps[-1].score_a:.0f}-{log.steps[-1].score_b:.0f}")

            logs_by_variant[variant] = all_logs
    finally:
        config_path.write_text(original_config)

    for variant, all_logs in logs_by_variant.items():
        print(f"\n=== variant={variant}: {len(all_logs)} episodes across pairs {args.pairs} ===")

        print("\n--- 거시 (Macro): cumulative shaped return vs actual win/loss ---")
        for k, v in analyze_macro(all_logs).items():
            print(f"  {k}: {v}")

        print("\n--- 중간 (Meso): score-change tick (deposit/steal/absorb) vs reward at that tick ---")
        for k, v in analyze_meso(all_logs).items():
            print(f"  {k}: {v}")

        print("\n--- 중간 (Meso) offset check: reward at tick+offset vs score-change at tick 0 ---")
        for off, v in analyze_meso_offsets(all_logs).items():
            print(f"  offset={off:+d}: {v}")

        print("\n--- 중간 (Meso) isolated-event check (offset=+1, no other score change within +-3 steps) ---")
        print(f"  {analyze_meso_clean(all_logs)}")

        print("\n--- 중간 (Meso) magnitude check: does a bigger score swing get a bigger |reward|? ---")
        mag = analyze_meso_magnitude(all_logs)
        for k, v in mag.items():
            print(f"  {k}: {v}")

        print("\n--- 미시 (Micro): battery pickup / deposit / death-drop custody transitions ---")
        micro = analyze_micro(all_logs)
        for event, stats in micro.items():
            print(f"  {event}: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
