"""Fast batched evaluation: every (opponent, seed, side) game of an eval in ONE ``BatchRollout``.

``grpo.eval_matches`` plays one opponent and one side per rollout (8 sequential rollouts of 16 games for the in-run
eval): 10k+ latency-bound steps each.  Here all ``len(opps) * len(seeds) * 2`` games advance together, the learner
(greedy ``LearnerPolicy``, optional fp16 CUDA-graph forward) and every opponent seated through ``Seats`` /
``JointSeats`` exactly like the trainers' rollouts.

Equivalence with ``eval_matches`` (fp32, no graphs): per-game margins are identical.  Heuristic opponents build
their per-(game, team) instance with ``make_opponent(desc, seed + 2 b + team)``; ``eval_matches`` has game index
``b`` = seed index ``j`` in each side's run, so here a heuristic gets one policy object per (spec, side) whose seed
is shifted by ``-2 * b0`` so that ``seed + 2 b + team`` equals the old ``2 j + team``.  Net opponents are stateless:
one object per spec, shared by its games.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from blackout_env.rl.grpo import JointSeats, LearnerPolicy, Seats, _heuristic_aliases
from blackout_env.sim.batch import BatchRollout, HeuristicPolicy, NetPolicy
from blackout_env.sim.state import SimState
from blackout_env.train.distill.inference import _OwnQ, load_net


def _heur(desc: str, shift: int, workers):
    seed = -2 * shift   # worker / instance seed = seed + 2 b + team, b = shift + j  ->  2 j + team
    if workers is not None:
        from blackout_env.rl.hpool import ParallelHeuristic

        return ParallelHeuristic(desc, seed, workers)
    return HeuristicPolicy(desc, seed)


def eval_batched(net_or_path, opp_specs, seeds, device: str, horizon: int = 11000, graphs: bool = True,
                 fp16: bool = True, heur_workers: int = 0, joint: bool = False, workers=None, fine=None):
    """Greedy learner vs each of ``opp_specs`` on ``seeds``, both sides, all games in one rollout.

    Returns ``{opp_spec: {"margin_side0": [...], "margin_side1": [...], "win", "margin", "se"}}`` (margins from the
    learner's side, in seed order; win counts draws as 1/2; se = std(ddof=1)/sqrt(n) of the 2*len(seeds) margins).
    ``fp16`` / ``graphs`` only act on CUDA (graphs additionally need fp16).  ``heur_workers`` > 0: heuristic
    opponents run in that many worker processes (or pass an existing ``HeuristicWorkers`` as ``workers``).
    Fine direction head (PPO ``fine_bins``): a checkpoint path whose file has ``fine_head_state`` is played with it (greedy
    argmax coarse bin, then argmax fine sub-bin), a net object takes ``fine`` (a ``rl.finehead.FineHead``); otherwise unchanged."""
    opp_specs, seeds = list(opp_specs), list(seeds)
    dev = torch.device(device)
    if isinstance(net_or_path, str):
        ck = torch.load(net_or_path, map_location="cpu", weights_only=True)
        net = load_net(ck["policy_state"])
        if fine is None and "fine_head_state" in ck:
            from blackout_env.rl.finehead import FineHead

            fine = FineHead.from_state(ck["fine_head_state"])
    else:
        net = net_or_path
    net = net.to(dev).eval()
    own = _OwnQ(net)
    tap = None
    if fine is not None:
        from blackout_env.rl.finehead import FeatTap

        fine, tap = fine.to(dev).eval(), FeatTap.of(net)
    cuda = dev.type == "cuda"
    use_fp16 = bool(fp16) and cuda
    use_graphs = bool(graphs) and use_fp16
    O, S = len(opp_specs), len(seeds)
    n = O * S * 2
    gf = None
    if use_graphs:
        from blackout_env.sim.graphfwd import GraphForward

        gf = GraphForward(own, dev, angle=True, max_batch=max(2048, n), **({"tap": tap, "fine": fine} if fine is not None else {}))   # fine decoded inside the graph
    learner = LearnerPolicy(own, dev, 0, fp16=use_fp16, graphs=gf, **({"fine": fine, "tap": tap} if fine is not None and gf is None else {}))

    own_workers = None
    if workers is None and heur_workers > 0 and any(not s.startswith("net:") for s in opp_specs):
        from blackout_env.rl.hpool import HeuristicWorkers

        workers = own_workers = HeuristicWorkers(heur_workers)
    # game b = (side * O + o) * S + j
    opps, opp_of, side = [], np.zeros(n, np.int64), np.zeros(n, np.int64)
    aliases = _heuristic_aliases()
    for o, spec in enumerate(opp_specs):
        if spec.startswith("net:"):
            opps.append(NetPolicy.from_checkpoint(spec[4:], device=device, fp16=use_fp16, graphs=use_graphs))
            k = [len(opps) - 1] * 2
        else:
            desc = aliases.get(spec, spec)
            k = []
            for sd in (0, 1):
                opps.append(_heur(desc, (sd * O + o) * S, workers))
                k.append(len(opps) - 1)
        for sd in (0, 1):
            sl = slice((sd * O + o) * S, (sd * O + o + 1) * S)
            opp_of[sl], side[sl] = k[sd], sd
    roots = [SimState.from_seed(s) for _ in range(O * 2) for s in seeds]
    try:
        if joint or workers is not None:
            pj = JointSeats(learner, opps, opp_of, side, split=not joint)
            res = BatchRollout(pj, pj, packed=True).run(roots, horizon)
        else:
            res = BatchRollout(Seats(0, learner, opps, opp_of, side), Seats(1, learner, opps, opp_of, side),
                               packed=True).run(roots, horizon)
    finally:
        for p in opps:
            if hasattr(p, "w") and hasattr(p, "key"):
                p.reset(0)   # free the worker-side instances
        if own_workers is not None:
            own_workers.close()
        del gf
        if cuda:
            torch.cuda.empty_cache()
    out = {}
    for o, spec in enumerate(opp_specs):
        ms = []
        for sd in (0, 1):
            sl = slice((sd * O + o) * S, (sd * O + o + 1) * S)
            ms.append(res.scores[sl, sd] - res.scores[sl, 1 - sd])
        m = np.concatenate(ms).astype(np.float64)
        out[spec] = {"margin_side0": [float(x) for x in ms[0]], "margin_side1": [float(x) for x in ms[1]],
                     "win": float(((m > 0) + 0.5 * (m == 0)).mean()), "margin": float(m.mean()),
                     "se": float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else 0.0}
    return out
