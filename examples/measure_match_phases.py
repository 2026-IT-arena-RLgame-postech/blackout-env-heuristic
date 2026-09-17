"""
How a match is lost over time. Plays a policy against V4 headless and reports, per time
window (--bin-seconds), for the candidate and for V4:

score        displayed score at the end of the window (includes stored, unabsorbed batteries)
stolen       (and stolen_outer: from a storage outside the 4x4 base) battery points taken out of that side's storages by an enemy unit standing next to it
             (storage count drop, not at an absorption, matched to an enemy cargo gain within 1.5 tiles)
withdrawn    storage drops matched to an own-unit cargo gain (a unit emptying its own storage)
lost_cargo   carried battery points that vanished away from any own storage (death while carrying)
delivered    (and delivered_outer) carried battery points that went into an own storage

Usage:
    python examples/measure_match_phases.py --checkpoint checkpoints/offline/<run>/step_N.pt
    python examples/measure_match_phases.py --heuristic strategic_v17
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import StrategicHeuristicV4, make_heuristic
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.reward_v2 import (
    BATTERY, BATTERY_SCALE, CARGO_COL, SPAWN_ENEMY, SPAWN_OWN, STORAGE_ENEMY, STORAGE_OWN, base_mask, unit_positions,
)

ABSORPTION_IDX = 3
EVENTS = ("stolen", "stolen_outer", "withdrawn", "lost_cargo", "delivered", "delivered_outer")


def _near(cells: np.ndarray, r: float, c: float, radius: float = 1.5) -> bool:
    return len(cells) > 0 and float(np.min(np.hypot(cells[:, 0] - r, cells[:, 1] - c))) <= radius


def play(env, policy_a, policy_b, seed: int, bin_decisions: int) -> tuple[dict, list]:
    """Returns per-window stats keyed by physical team (0 = A, 1 = B) and window index."""
    names_a, names_b = set(team_a_agents()), set(team_b_agents())
    obs, _ = env.reset(seed=seed)
    for p in (policy_a, policy_b):
        if hasattr(p, "reset"):
            p.reset()
    stats = defaultdict(lambda: defaultdict(float))  # (team, window) -> event -> points
    window, prev, decision = 0, None, 0
    scores = []
    while env.agents:
        if not obs:
            obs, _, _, _, _ = env.step({})
            continue
        view = obs.get("unit_0") or next(iter(obs.values()))
        if "unit_0" in obs:
            g, s, t = view["graphic"], view["agent_states"], view["team_state"]
            battery = np.rint(g[..., BATTERY] * BATTERY_SCALE)
            stores = [(np.argwhere(g[..., ch] > 0.5).astype(float), (battery * (g[..., ch] > 0.5)).sum()) for ch in (STORAGE_OWN, STORAGE_ENEMY)]
            cargo = np.rint(np.maximum(s[:, CARGO_COL], 0) * BATTERY_SCALE)
            row, col = unit_positions(s[None])
            team = np.where(s[:, 2] > 0, 0, 1)  # team A view: + = team A
            bases = []
            for ch in (SPAWN_OWN, SPAWN_ENEMY):
                sp = np.argwhere(g[..., ch] > 0.5)
                bases.append(base_mask(tuple(int(v) for v in sp[0])) if len(sp) else np.zeros(g.shape[:2], bool))

            def outer(side, r, c):  # nearest storage tile of `side` to (r, c) lies outside its base
                cells = stores[side][0]
                if not len(cells):
                    return False
                y, x = cells[int(np.argmin(np.hypot(cells[:, 0] - r, cells[:, 1] - c)))].astype(int)
                return not bases[side][y, x]
            absorbed = prev is not None and t[ABSORPTION_IDX] > prev["abs"] + 1e-6
            decision += 1
            if decision // bin_decisions > window:
                scores.append((window, float(t[0]) * 100, float(t[1]) * 100))
                window = decision // bin_decisions
            if prev is not None:
                if not absorbed:
                    for side in (0, 1):
                        drop = prev["stores"][side][1] - stores[side][1]
                        cells = stores[side][0]
                        gains = [(u, cargo[u] - prev["cargo"][u]) for u in range(10) if cargo[u] > prev["cargo"][u]]
                        for u, gain in gains:
                            if drop <= 0 or not _near(cells, row[0, u], col[0, u]):
                                continue
                            take = min(gain, drop)
                            stats[(side, window)]["stolen" if team[u] != side else "withdrawn"] += take
                            if team[u] != side and outer(side, row[0, u], col[0, u]):
                                stats[(side, window)]["stolen_outer"] += take
                            drop -= take
                    for u in range(10):
                        lost = prev["cargo"][u] - cargo[u]
                        if lost <= 0:
                            continue
                        own_cells = stores[team[u]][0]
                        moved = abs(row[0, u] - prev["row"][u]) + abs(col[0, u] - prev["col"][u])
                        key = "delivered" if _near(own_cells, prev["row"][u], prev["col"][u]) and moved < 2.0 else "lost_cargo"
                        stats[(int(team[u]), window)][key] += lost
                        if key == "delivered" and outer(int(team[u]), prev["row"][u], prev["col"][u]):
                            stats[(int(team[u]), window)]["delivered_outer"] += lost
            prev = {"abs": float(t[ABSORPTION_IDX]), "stores": stores, "cargo": cargo, "row": row[0].copy(), "col": col[0].copy(), "t": t.copy()}
        actions = {}
        oa = {n: obs[n] for n in obs if n in names_a}
        ob = {n: obs[n] for n in obs if n in names_b}
        if oa:
            actions.update(policy_a.act(oa))
        if ob:
            actions.update(policy_b.act(ob))
        obs, _, _, _, _ = env.step(actions)
    if prev is not None:
        scores.append((window, float(prev["t"][0]) * 100, float(prev["t"][1]) * 100))
    return stats, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--heuristic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33, 34, 35])
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--bin-seconds", type=float, default=5.0, help="window length (25 decisions = 1 s)")
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    args = parser.parse_args()

    if args.checkpoint:
        from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

        trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device="cpu", tb_log_dir=None))
        trainer.load(args.checkpoint)
        trainer.net.eval()
        cand, name = MyPolicy(trainer.net, device="cpu"), f"{args.checkpoint.parent.name[-11:]}/{args.checkpoint.stem}"
    else:
        cand, name = make_heuristic(args.heuristic), args.heuristic
    opp = StrategicHeuristicV4()
    env = BlackOutEnv(args.build, time_scale=20, no_graphics=True, additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    per = defaultdict(lambda: defaultdict(list))  # role -> key -> [per-match list over windows]
    n = 0
    try:
        for seed in args.seeds:
            for swap in (False, True):
                a, b = (opp, cand) if swap else (cand, opp)
                stats, scores = play(env, a, b, seed, int(args.bin_seconds * 25))
                cand_team = 1 if swap else 0
                n += 1
                for w in range(args.windows):
                    for role, side in (("cand", cand_team), ("v4", 1 - cand_team)):
                        for ev in EVENTS:
                            per[role][ev].append((w, stats[(side, w)][ev]))
                    reached = [x for x in scores if x[0] <= w]
                    last = reached[-1] if reached else (w, 0.0, 0.0)
                    ended = scores and scores[-1][0] < w
                    per["cand"]["score"].append((w, (last[2] if swap else last[1]), ended))
                    per["v4"]["score"].append((w, (last[1] if swap else last[2]), ended))
    finally:
        env.close()

    print(f"{name} vs V4: {n} matches, per {args.bin_seconds:g} s window (points summed over matches; score = mean displayed at window end, held after the match ends)")
    header = "".join(f"{f'{w * args.bin_seconds:g}-{(w + 1) * args.bin_seconds:g}s':>9s}" for w in range(args.windows))
    print(f"  {'':22s}{header}")
    for role in ("cand", "v4"):
        label = "model" if role == "cand" and args.checkpoint else ("cand" if role == "cand" else "V4")
        vals = defaultdict(float)
        for w, s, _ in per[role]["score"]:
            vals[w] += s / n
        print(f"  {label + ' score':22s}" + "".join(f"{vals[w]:9.1f}" for w in range(args.windows)))
        for ev in EVENTS:
            sums = defaultdict(float)
            for w, v in per[role][ev]:
                sums[w] += v
            print(f"  {label + ' ' + ev:22s}" + "".join(f"{sums[w]:9.0f}" for w in range(args.windows)))


if __name__ == "__main__":
    main()
