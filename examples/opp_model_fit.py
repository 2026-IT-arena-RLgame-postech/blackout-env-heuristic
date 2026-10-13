"""How well does each candidate rollout net explain an opponent's moves?  (opponent identification, step 0)

The R6 search models the other team with one rollout net (r24cd256 samples + arg max).  The idea
under test: keep several candidate nets and weight them by how well each predicted the opponent's
moves so far (a posterior), so the search's opponent model follows the opponent actually faced.

This measures the premise in the simulator: P24c plays ``--opp``; at every decision each
candidate net sees the opponent team's canonical view and is scored on the opponent's actual move
(its bin, STAND for a zero move): log-likelihood, top-1 agreement, and the posterior after
5 / 30 / 150 s (which net would the search pick, how sure).  ``--from-positions`` scores the
move recovered from the opponent's displacement over the decision (what the official obs give)
instead of the exact move.

    PYTHONPATH=. .venv/bin/python examples/opp_model_fit.py --opp v321 --seeds 8 --out reports/opp_model_fit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CANDIDATES = {
    "r24c": "checkpoints/rollout_net/r6/r24cd256.pt",
    "r22": "checkpoints/rollout_net/r5/r22d256.pt",
    "r21": "checkpoints/rollout_net/r4/r21d256.pt",
    "r20": "checkpoints/rollout_net/r3/r20d256.pt",
    "f3": "checkpoints/rollout_net/arch/f3_x_d2.pt",
    "r5": "checkpoints/rollout_net/r5.pt",
}
P24C = "checkpoints/distill/frozen/p24c_v321.pt"
CHECKPOINTS_AT = (125, 750, 3750)   # decisions: 5 / 30 / 150 s


def opponent(name: str, seed: int):
    """blind_search_eval's opponent names (students, heuristics, perturbed / V27 settings, OOD-26) as act() policies."""
    from blackout_env.heuristics import make_heuristic
    from blackout_env.train.distill.opponents import make_opponent
    from examples.blind_search_eval import BASES, EXTRA_HEUR, PERTURBED, V27

    if name in BASES:
        from blackout_env.train.distill.inference import load_checkpoint_net, make_student

        return make_student(load_checkpoint_net(str(REPO / BASES[name])), "iqn")
    if name in PERTURBED or name in V27:
        pid, params = PERTURBED.get(name) or V27[name]
        return make_heuristic(pid, **params)
    if name in ("v19", "v23", "v25"):
        return make_opponent(f"strategic_{name}", seed)
    return make_opponent(EXTRA_HEUR.get(name, name), seed)


def observed_bin(move: np.ndarray, k: int, stand: bool) -> int:
    """Canonical move -> bin index (K = STAND for a zero move; -1 when the net has no STAND)."""
    if float(np.hypot(*move)) < 1e-3:
        return k if stand else -1
    return int(round(math.atan2(move[1], move[0]) / (2 * math.pi / k))) % k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=6100000)
    ap.add_argument("--decisions", type=int, default=3750)
    ap.add_argument("--from-positions", action="store_true")
    ap.add_argument("--out", default="reports/opp_model_fit")
    args = ap.parse_args()

    import torch

    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, NetPolicy
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.state import SimState
    from blackout_env.train.distill.inference import load_checkpoint_net, make_student

    torch.set_num_threads(1)
    nets = {n: NetPolicy(RolloutNet.from_checkpoint(str(REPO / p)), device="cpu")
            for n, p in CANDIDATES.items() if (REPO / p).is_file()}
    names = list(nets)
    ours = make_student(load_checkpoint_net(str(REPO / P24C)), "iqn")
    g = np.zeros((1, 2, N_CH, H, W), np.float32)
    t = np.zeros((1, 2, 4), np.float32)
    a = np.zeros((1, 2, 10, N_AGENT_COLS), np.float32)
    rows = []
    for m in range(2 * args.seeds):
        seed, side = args.seed0 + m // 2, m % 2           # side = our (P24c) team
        o = 1 - side
        sim = SimState.from_seed(seed)
        opp = opponent(args.opp, seed)
        if hasattr(opp, "reset"):
            opp.reset()
        ll = {n: 0.0 for n in names}
        top1 = {n: 0 for n in names}
        n_obs, post_at = 0, {}
        for d in range(args.decisions):
            if sim.over:
                break
            obs = sim.obs()
            mv = np.zeros((10, 2), np.float32)
            for pol, team in ((ours, side), (opp, o)):
                for name, act in pol.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5 * team, 5 * team + 5)}).items():
                    mv[int(name[5:])] = act
            bs = BatchSim([sim])
            bs.render(np.zeros(1, np.int64), g, t, a)
            before = sim.uf[5 * o:5 * o + 5].copy()
            sim.step(mv)
            seen = (sim.uf[5 * o:5 * o + 5] - before) if args.from_positions else mv[5 * o:5 * o + 5]
            if o == 1:
                seen = seen[:, ::-1]                       # world -> canonical (reflection across y = x)
            for name in names:
                q, _ = nets[name].forward(np.ascontiguousarray(g[:, o]), np.ascontiguousarray(t[:, o]),
                                          np.ascontiguousarray(a[:, o]))
                q = q[0].astype(np.float64)                 # [5, K(+1)]
                width = q.shape[-1]
                k, stand = width - width % 2, width % 2 == 1
                lp = q - q.max(-1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
                for u in range(5):
                    if args.from_positions and float(np.hypot(*seen[u])) > 0.5:   # a death / respawn teleport
                        continue
                    b = observed_bin(seen[u], k, stand)
                    if b < 0:
                        continue
                    ll[name] += float(lp[u, b])
                    top1[name] += int(int(q[u].argmax()) == b)
                    if name == names[0]:
                        n_obs += 1
            if d + 1 in CHECKPOINTS_AT:
                v = np.array([ll[n] for n in names])
                p = np.exp(v - v.max())
                post_at[str(d + 1)] = {n: float(x) for n, x in zip(names, p / p.sum())}
        row = {"opp": args.opp, "seed": seed, "side": side, "decisions": d + 1, "units_scored": n_obs,
               "ll_per_move": {n: ll[n] / max(n_obs, 1) for n in names},
               "top1": {n: top1[n] / max(n_obs, 1) for n in names}, "posterior_at": post_at,
               "from_positions": args.from_positions}
        rows.append(row)
        best = max(names, key=lambda n: ll[n])
        print(f"{args.opp} seed {seed} side {side}: best {best} " +
              " ".join(f"{n} {ll[n] / max(n_obs, 1):.3f}/{top1[n] / max(n_obs, 1):.2f}" for n in names), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "pos" if args.from_positions else "exact"
    with (out / f"fit_{args.opp}_{tag}.jsonl").open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
