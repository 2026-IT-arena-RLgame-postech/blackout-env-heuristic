"""
Where do units reverse direction, and which storage do they deliver to? Plays a policy against a
heuristic headless and reports, for each side:

reversal   share of unit-decisions whose commanded direction points back against the previous
           one (dot < -0.5, i.e. 135 or 180 degrees), binned by the unit's tile distance to the
           nearest wall (Chebyshev, 1 = next to a wall), split Hunter / other. Death respawns,
           class changes and the first decision of a unit are skipped.
storage    deliveries (cargo -> none within 1.5 tiles of an own storage tile) by destination:
           the storage cluster nearest the own spawn vs any other, plus how many own storage
           clusters were active.

--hysteresis M keeps a checkpoint's previous direction unless another direction's Q is more
than M higher (an inference-only check of whether argmax flips between near-tied actions cause
the oscillation).

Usage:
    python examples/measure_movement_and_storage.py --checkpoint checkpoints/offline/<run>/final.pt
    python examples/measure_movement_and_storage.py --heuristic strategic_v17
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.heuristics import StrategicHeuristicV4, make_heuristic
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy, direction_vector_to_idx
from blackout_env.train.reward_v2 import CARGO_COL, CLASS_COLS, HUNTER, STORAGE_OWN, WALL, unit_positions

H = W = 24
DELIVERY_RADIUS = 1.5
WALL_BINS = (1, 2, 3, 4)  # 4 = "4 or more"


class HysteresisPolicy:
    def __init__(self, policy: MyPolicy, margin: float) -> None:
        self.policy, self.margin, self.prev = policy, margin, {}

    def reset(self) -> None:
        self.prev = {}

    def act(self, obs):
        if not obs:
            return {}
        out = {}
        for agent, q in self.policy.action_values(obs).items():
            best = int(np.argmax(q))
            prev = self.prev.get(agent)
            if prev is not None and q[prev] >= q[best] - self.margin:
                best = prev
            self.prev[agent] = best
            out[agent] = DIRECTION_VECTORS[best]
        return out


def _clusters(mask: np.ndarray) -> list[np.ndarray]:
    seen, out = np.zeros_like(mask), []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        out.append(np.array(cells, dtype=np.float64))
    return out


class SideRecorder:
    """Wraps one team's policy and records its own units' commanded directions and deliveries."""

    def __init__(self, inner, spawn_channel: int = 4) -> None:
        self.inner, self.spawn_channel = inner, spawn_channel
        self.reversals = defaultdict(lambda: [0, 0])  # (hunter, wall_bin) -> [reversed, total]
        self.deliveries = {"spawn": 0, "other": 0}
        self.clusters_seen: list[int] = []
        self.reset()

    def reset(self) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self.prev: dict[int, tuple] = {}
        self._clusters_this_match = 0

    def end_match(self) -> None:
        self.clusters_seen.append(self._clusters_this_match)

    def act(self, obs):
        actions = self.inner.act(obs) if obs else {}
        if not obs:
            return actions
        view = next(iter(obs.values()))
        graphic, states = view["graphic"], view["agent_states"]
        wall = graphic[..., WALL] > 0.5
        wall_cells = np.argwhere(wall)
        storages = _clusters(graphic[..., STORAGE_OWN] > 0.5)
        self._clusters_this_match = max(self._clusters_this_match, len(storages))
        spawn = np.argwhere(graphic[..., self.spawn_channel] > 0.5)
        spawn_idx = None
        if storages and len(spawn):
            spawn_idx = int(np.argmin([np.min(np.abs(cl - spawn[0]).max(1)) for cl in storages]))
        row, col = unit_positions(states[None])
        for agent, vec in actions.items():
            u = unit_index(agent)
            r, c = float(row[0, u]), float(col[0, u])
            cls = int(states[u, CLASS_COLS].argmax())
            cargo = states[u, CARGO_COL] > 1e-3
            d = int(direction_vector_to_idx(np.asarray(vec)[None])[0])
            cell = np.array([round(r), round(c)], dtype=np.float64)
            wall_bin = min(WALL_BINS[-1], int(np.min(np.abs(wall_cells - cell).max(1)))) if len(wall_cells) else WALL_BINS[-1]
            prev = self.prev.get(u)
            if prev is not None:
                pr, pc, pcls, pcargo, pd = prev
                teleported = abs(r - pr) + abs(c - pc) > 2.0
                if not teleported and cls == pcls:
                    stat = self.reversals[(cls == HUNTER, max(1, wall_bin))]
                    stat[0] += int(DIRECTION_VECTORS[d] @ DIRECTION_VECTORS[pd] < -0.5)
                    stat[1] += 1
                if pcargo and not cargo and not teleported and storages:
                    dists = [np.min(np.hypot(cl[:, 0] - pr, cl[:, 1] - pc)) for cl in storages]
                    k = int(np.argmin(dists))
                    if dists[k] <= DELIVERY_RADIUS:
                        self.deliveries["spawn" if k == spawn_idx else "other"] += 1
            self.prev[u] = (r, c, cls, cargo, d)
        return actions


def play(env, cand: SideRecorder, opp: SideRecorder, seed: int, swap: bool) -> tuple[float, float]:
    team_a, team_b = (opp, cand) if swap else (cand, opp)
    names_a, names_b = set(team_a_agents()), set(team_b_agents())
    obs, _ = env.reset(seed=seed)
    cand.reset(), opp.reset()
    last = (0.0, 0.0)
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
        else:
            view = next(iter(obs.values()))
            if unit_index(next(iter(obs))) < 5:
                last = (float(view["team_state"][0]), float(view["team_state"][1]))
            actions = {}
            oa = {n: obs[n] for n in obs if n in names_a}
            ob = {n: obs[n] for n in obs if n in names_b}
            if oa:
                actions.update(team_a.act(oa))
            if ob:
                actions.update(team_b.act(ob))
            obs, _, _, _, infos = env.step(actions)
        if infos and not env.agents:
            info = next(iter(infos.values()))
            last = (float(info.get("score_0", last[0])), float(info.get("score_1", last[1])))
    cand.end_match(), opp.end_match()
    a, b = last[0] * 100, last[1] * 100
    return (b - a, b + a) if swap else (a - b, a + b)


def report(name: str, rec: SideRecorder) -> None:
    print(f"  {name}")
    for hunter in (True, False):
        cells = []
        for wb in WALL_BINS:
            rev, tot = rec.reversals[(hunter, wb)]
            label = f"{wb}{'+' if wb == WALL_BINS[-1] else ''}"
            cells.append(f"wall {label}: {rev / max(1, tot):6.1%} (n={tot})")
        print(f"    {'hunter' if hunter else 'other ':6s} reversal  " + "  ".join(cells))
    dv = rec.deliveries
    total = dv["spawn"] + dv["other"]
    print(f"    deliveries {total}: spawn storage {dv['spawn'] / max(1, total):.0%}, other {dv['other'] / max(1, total):.0%}"
          f" | own storage clusters per match {np.mean(rec.clusters_seen):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--heuristic")
    parser.add_argument("--hysteresis", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.checkpoint:
        from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

        trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=args.device, tb_log_dir=None))
        trainer.load(args.checkpoint)
        trainer.net.eval()
        policy = MyPolicy(trainer.net, device=args.device)
        inner = HysteresisPolicy(policy, args.hysteresis) if args.hysteresis > 0 else policy
        name = f"{args.checkpoint.parent.name}/{args.checkpoint.stem}" + (f" hysteresis {args.hysteresis}" if args.hysteresis else "")
    else:
        inner, name = make_heuristic(args.heuristic), args.heuristic
    cand, opp = SideRecorder(inner), SideRecorder(StrategicHeuristicV4())
    env = BlackOutEnv(args.build, time_scale=20, no_graphics=True, additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    margins = []
    try:
        for seed in args.seeds:
            for swap in (False, True):
                margin, _ = play(env, cand, opp, seed, swap)
                margins.append(margin)
    finally:
        env.close()
    print(f"{name} vs V4: {len(margins)} matches, margin {np.mean(margins):+.1f} ({', '.join(f'{m:+.0f}' for m in margins)})")
    report(name, cand)
    report("V4 (opponent)", opp)


if __name__ == "__main__":
    main()
