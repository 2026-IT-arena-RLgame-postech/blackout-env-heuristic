"""R1: search that knows nothing about its opponent, against a held-out opponent pool (docs/policy_v2_design_ko.md §34).
Simulator only (Unity: examples/unity_search_eval.py --base).

Our player is P14b (``checkpoints/distill/frozen/p14_v25.pt``) acting; a search tag searches from 5 s (decision 125,
§32) to the end with the rollout net (f3_x_d2 unless the tag says otherwise), leaf V s2v_f3xd2 iter2, B8
``alt*,samp:1.0x2``, every 25, window 25, margin 3, overlay (the base branch and a unit branch's other units = P14b).
The opponent model inside the search never uses the real opponent's identity -- it is always a rollout net:

* ``det``        the deterministic rollout net (S3, §20/§32 ``p14b_from5``);
* ``o4f``        S4 ``o4fmean`` (§26): the other team = rollout-net samples (tau 1) x3 + its arg max over the whole
                 150-decision rollout, the branch value = their mean;
* ``o4f_mix3`` / ``o4f_soft5`` / ``o4f_min``  the same variants, a mild pessimistic aggregate
                 (0.3 min + 0.7 mean / soft-min T = 5 points / max-min);
* ``mix_r5``     two opponent-model nets: f3_x_d2 {arg max, one sample} + r5 {two samples}, mean;
* ``*_v25r`` / ``*_v25f``  the same with the P14b-distilled rollout nets (§31: v25d256 / v25f3x) as the rollout and
                 opponent model (leaf V still s2v_f3xd2 iter2, fit to the f3_x_d2 continuation);
  ``mix_v25r`` = f3_x_d2 {arg max, sample} + v25r {arg max... samples}, see ``VARIANTS``.

The held-out pool (``OPPONENTS``; none is used to fit anything in the search): V19, V23, V25, P5a, P11, P14b, the plain
d256_f3_x32_d2 / f3_x_d2 nets (f3_x_d2 = the search's own model: the S3 exact-model case), strategic_v27 (§33,
reserve_stage 1.5 = ``v27`` and 1.0 = ``v27s1``) and four fixed perturbed
V23 / V25 settings drawn from the data-collection mixture's V17-family parameter cloud (``PERTURBED``).

``play``: one opponent, the given tags, seeds x both sides in lockstep chunks -> ``<out>/matches_<opponent>.jsonl``
(a rerun skips rows already there).  ``report``: per opponent and tag the paired difference vs ``plain`` (same seed,
side; SE over seeds, sides averaged), the pooled mean over the opponents (equal weight; SE = sqrt(sum SE^2) / n) and
the worst opponent, and the search cost.

    PYTHONPATH=. python examples/blind_search_eval.py play --opponent v23 --tags plain,det,o4f --seeds 128
    PYTHONPATH=. python examples/blind_search_eval.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

F3 = "checkpoints/rollout_net/arch/f3_x_d2.pt"
D256 = "checkpoints/rollout_net/arch/d256_f3_x32_d2.pt"
R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/s2v_f3xd2/iter2.pt"
P14B = "checkpoints/distill/frozen/p14_v25.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"
P5A = "checkpoints/distill/frozen/p5a_v22.pt"
P16 = "checkpoints/distill/frozen/p16_v27.pt"  # the V27 student (docs §35); --base p16 = the next loop step
P17 = "checkpoints/distill/frozen/p17_v28.pt"  # the V28 student (docs §38)
P20 = "checkpoints/distill/frozen/p20_v30.pt"  # P18bp2 + 5k V30 distil (v_0005, docs §47)
P20C = "checkpoints/distill/frozen/p20_v30_fp16c.pt"  # QZ: P20 + precision "fp16c" (sim/fastnet.fast_checkpoint)
P21E = "checkpoints/distill/frozen/p21e_v30.pt"  # P21e: staged-reserve upweight + short V30 fine-tune (docs §50)
P22 = "checkpoints/distill/frozen/p22_v31.pt"  # P22 = P22b: P21e + 2k V31 fine-tune (docs §55)
P24C = "checkpoints/distill/frozen/p24c_v321.pt"  # P24c v9: P22 + V32.1 teacher (P24 agent; gpu1 ~/p24nets/p24c9.pt)
BASES = {"p14b": P14B, "p16": P16, "p17": P17, "p20": P20, "p20c": P20C, "p21e": P21E, "p22": P22, "p24c": P24C}
V25R = "checkpoints/rollout_net/v25/v25d256.pt"  # the P14b-distilled rollout nets (docs §31; --v25r overrides)
V25F = "checkpoints/rollout_net/v25/v25f3x.pt"
V27R = "checkpoints/rollout_net/v27/v27d256.pt"  # distilled from P16 (docs §37)
V28R = "checkpoints/rollout_net/v28/v28d256.pt"  # distilled from P17 (Q9, docs §39)
R20 = "checkpoints/rollout_net/r3/r20d256.pt"  # distilled from P20 (R3, docs §49)
R21 = "checkpoints/rollout_net/r4/r21d256.pt"  # distilled from P21e, init r20d256 (R4, docs §54)
R22 = "checkpoints/rollout_net/r5/r22d256.pt"  # distilled from P22, init r21d256 (R5, docs §59)
R24C = "checkpoints/rollout_net/r6/r24cd256.pt"  # distilled from P24c, init r22d256, V32/V32.1/h4 in the pools (R6, docs §61)
TRACE_S = (5, 10, 30, 60, 150, 300)
SEARCH_FROM = 125  # decisions (5 s, docs §32)

# fixed draws from the mixture's V17-family cloud (heuristics/mixture.py _top_tier_parameters + replan / threat),
# np.random.default_rng(34000 + k), k = 1, 5, 4, 2
PERTURBED = {
    "v23pA": ("strategic_v23", {"replan_interval": 8, "threat_radius": 0.153, "hunter_quota": 3, "hunter_mode": "mixed",
                                "exposure_cost": 0.627, "steal_weight": 1.985, "camp_engage_radius": 5.033,
                                "special_weights": (0.442, 0.8, 0.097, 0.089)}),
    "v23pB": ("strategic_v23", {"replan_interval": 12, "threat_radius": 0.165, "hunter_quota": 2, "hunter_mode": "hunt",
                                "exposure_cost": 0.964, "steal_weight": 2.294, "camp_engage_radius": 4.423,
                                "special_weights": (0.341, 0.947, 0.104, 0.104)}),
    "v25pA": ("strategic_v25", {"replan_interval": 11, "threat_radius": 0.18, "hunter_quota": 3, "hunter_mode": "hunt",
                                "exposure_cost": 0.724, "steal_weight": 1.684, "camp_engage_radius": 4.846,
                                "special_weights": (0.349, 0.865, 0.092, 0.144)}),
    "v25pB": ("strategic_v25", {"replan_interval": 9, "threat_radius": 0.178, "hunter_quota": 2, "hunter_mode": "mixed",
                                "exposure_cost": 0.895, "steal_weight": 1.531, "camp_engage_radius": 4.95,
                                "special_weights": (0.449, 0.685, 0.084, 0.103)}),
}
# strategic_v27 (docs §33: V25 + a staged reserve walker for the second race; the strongest opponent)
V27 = {"v27": ("strategic_v27", {"reserve_stage": 1.5}), "v27s1": ("strategic_v27", {"reserve_stage": 1.0}),
       "v27s05": ("strategic_v27", {"reserve_stage": 0.5}),
       "v28": ("strategic_v28", {"reserve_stage": 0.5, "trade_hold": "advance"}),  # docs §36
       "v30": ("strategic_v30", {"reserve_stage": 0.5, "trade_hold": "advance", "reserve_join": "reserve",
                                 "solo_sentry": True}),  # docs §44
       "v31": ("strategic_v30", {"reserve_stage": 0.5, "trade_hold": "advance", "reserve_join": "reserve",
                                 "solo_sentry": True, "threat_radius": 0.13, "camp_engage_radius": 6.5})}  # V31, docs §46
OPPONENTS = ("p14b", "p16", "p17", "p20", "p21e", "p22", "p24c", "v32b", "v321", "v28", "v30", "v31", "v25", "v27", "v27s1", "v27s05", "v23", "v19", "p5a", "p11", "d256", "f3", "v23pA", "v23pB", "v25pA", "v25pB")

# V32 B (docs §58, branch feat/v32): V31 + four Hunters, exit guard, sanctuary sweep, cargo transform
V32B = ("strategic_v32:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,"
        "camp_engage_radius=6.5,hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,"
        "sweep=true,cargo_transform=true")


def _extra_heuristics() -> dict:
    """V32 B + the OOD-26 suite's heuristic entries by short name (docs §56; ``v30_h4``, ``v30_h4hunt``, ``v30_c0``, ...)."""
    from blackout_env.train.distill.ood_suite import FOLLOWUP, OOD26

    out = {n: d for n, d, _, _ in OOD26 + FOLLOWUP if not d.startswith(("ckpt:", "temp:")) and n != "v31"}
    out["v32b"] = V32B
    out["v321"] = V32B + ",adapt_k=3"  # V32.1 (docs §58.7), the champion heuristic
    return out


EXTRA_HEUR = _extra_heuristics()

O4 = {"opp_branches": "samp:1.0x3", "opp_left": 150}
# tag -> search options (None = plain P14b); "net": rollout net override ("v25r"), "deep": Searcher S4 options
VARIANTS = {
    "plain": None,
    "det": {},
    "o4f": {"deep": {**O4, "opp_agg": "mean"}},
    "o4f_mix3": {"deep": {**O4, "opp_agg": "mix:0.3"}},
    "o4f_soft5": {"deep": {**O4, "opp_agg": "softmin:5"}},
    "o4f_min": {"deep": {**O4, "opp_agg": "min"}},
    # base, samp s0 = f3_x_d2; samp s1, s2 = r5 (opponent draws other than ours: seed + 32)
    "mix_r5": {"deep": {**O4, "opp_agg": "mean", "opp_net_idx": [0, 0, 1, 1], "opp_nets": [R5]}},
    "det_v25r": {"net": "v25r"},
    "det_v25f": {"net": V25F},
    "o4f_v27r": {"net": V27R, "deep": {**O4, "opp_agg": "mean"}},
    "o4f_v28r": {"net": V28R, "deep": {**O4, "opp_agg": "mean"}},
    "det_v28r": {"net": V28R},
    # leaf V refit on v28d256 continuations (Q9, docs §39) -- "value": a per-tag leaf V override
    "o4f_v28r_V1": {"net": V28R, "value": "checkpoints/value/s2v_v28d256/iter1.pt", "deep": {**O4, "opp_agg": "mean"}},
    "o4f_v28r_V2": {"net": V28R, "value": "checkpoints/value/s2v_v28d256/iter2.pt", "deep": {**O4, "opp_agg": "mean"}},
    # rollout / opponent model distilled from P20 (R3, docs §49); _V = leaf V refit on r20d256 continuations
    "o4f_r20": {"net": R20, "deep": {**O4, "opp_agg": "mean"}},
    # rollout / opponent model distilled from P21e (R4, docs §54)
    "o4f_r21": {"net": R21, "deep": {**O4, "opp_agg": "mean"}},
    # rollout / opponent model distilled from P22 (R5, docs §59)
    "o4f_r22": {"net": R22, "deep": {**O4, "opp_agg": "mean"}},
    # rollout / opponent model distilled from P24c (R6, docs §61)
    "o4f_r24c": {"net": R24C, "deep": {**O4, "opp_agg": "mean"}},
    # opponent identification (docs §63): o4f_r24c whose 4 opponent variants are dealt to the candidate nets
    # {r24c, r22, r21, r20, f3_x_d2} by their posterior on the opponent's moves so far (OpponentID)
    "o4f_id": {"net": R24C, "deep": {**O4, "opp_agg": "mean", "opp_nets": [R22, R21, R20, F3]},
               "opp_id": {"temper": 0.1, "half_life": 750}},
    "o4f_r20_V": {"net": R20, "value": "checkpoints/value/s2v_r20d256/iter1.pt", "deep": {**O4, "opp_agg": "mean"}},
    # rollout v27d256 (our continuation), opponent model = v28d256 {base, s0, s1, s2}: separates the two roles
    "o4f_v27r_opp28": {"net": V27R, "deep": {**O4, "opp_agg": "mean", "opp_net_idx": [1, 1, 1, 1], "opp_nets": [V28R]}},
    "o4f_v25f": {"net": V25F, "deep": {**O4, "opp_agg": "mean"}},
    "o4f_v25r": {"net": "v25r", "deep": {**O4, "opp_agg": "mean"}},
    # rollout f3_x_d2, opponent model: f3 {base, s0} + v25r {s1, s2}
    "mix_v25r": {"deep": {**O4, "opp_agg": "mean", "opp_net_idx": [0, 0, 1, 1], "opp_nets": ["v25r"]}},
    # rollout v25r, opponent model: v25r {base, s0} + f3 {s1, s2}
    "mix_v25r_f3": {"net": "v25r", "deep": {**O4, "opp_agg": "mean", "opp_net_idx": [0, 0, 1, 1], "opp_nets": [F3]}},
}


from blackout_env.sim.batch import HeuristicPolicy  # noqa: E402


class FixedHeuristicPolicy(HeuristicPolicy):
    """``HeuristicPolicy`` with constructor parameters a descriptor cannot carry (tuples): one instance per (game, team)."""

    def __init__(self, policy_id: str, params: dict):
        super().__init__(policy_id)
        self.params = dict(params)

    def _instance(self, b: int, team: int):
        from blackout_env.heuristics import make_heuristic

        key = (b, team)
        if key not in self.instances:
            self.instances[key] = make_heuristic(self.descriptor, **self.params)
        return self.instances[key]


def rollout_policy(path: str, device: str):
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    return NetPolicy(RolloutNet.from_checkpoint(str(REPO / path)), device=device)


def make_opponent(name: str, device: str):
    from blackout_env.sim.batch import NetPolicy

    if name in BASES:
        return NetPolicy.from_checkpoint(str(REPO / BASES[name]), device=device)
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device=device)
    if name == "p5a":
        return NetPolicy.from_checkpoint(str(REPO / P5A), device=device)
    if name == "f3":
        return rollout_policy(F3, device)
    if name == "d256":
        return rollout_policy(D256, device)
    if name in PERTURBED or name in V27:
        return FixedHeuristicPolicy(*(PERTURBED.get(name) or V27[name]))
    if name in ("v19", "v23", "v25"):
        return HeuristicPolicy(f"strategic_{name}")
    if name in EXTRA_HEUR:
        return HeuristicPolicy(EXTRA_HEUR[name])
    raise ValueError(name)


def load_rows(out: Path) -> list[dict]:
    return [json.loads(x) for path in sorted(out.glob("matches_*.jsonl")) for x in path.read_text().splitlines() if x.strip()]


def build_searcher(tag: str, base, device: str, args):
    from blackout_env.sim.search import (ModNetPolicy, Searcher, ValueNetLeaf, deep_opp_nets, parse_branches,
                                         searcher_kwargs)

    v = VARIANTS[tag]
    res = lambda p: args.v25r if p == "v25r" else p  # noqa: E731
    net_path = res(v.get("net", F3))
    deep = dict(v.get("deep") or {})
    opp = [rollout_policy(res(p), device) for p in deep_opp_nets(deep)]
    net = rollout_policy(net_path, device)
    value = v.get("value", args.value)
    pipe = int(getattr(args, "pipeline", 0) or 0)  # B1: PipelinedRollout (1 = bit-identical chunks)
    s = Searcher(ModNetPolicy(net, opp_nets=opp), ValueNetLeaf(str(REPO / value), device), parse_branches(args.branches),
                 window=args.window, margin=args.margin, base_policy=base, overlay=True,
                 pipeline=(pipe == 1) or pipe, pipeline_graphs=bool(getattr(args, "pipeline_graphs", False)),
                 **searcher_kwargs(deep))
    meta = {"net": net_path, "value": value, "opp_nets": [res(p) for p in deep_opp_nets(deep)],
            **{k: x for k, x in deep.items() if k != "opp_nets"}}
    return s, net, meta


def cmd_play(args):
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.search import MatchConfig, ModNetPolicy, play_matches

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = NetPolicy.from_checkpoint(str(REPO / BASES[args.base]), device=args.device)
    plain_net = rollout_policy(F3, args.device)
    done = {(r["tag"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    opp = make_opponent(args.opponent, args.device)
    for tag in args.tags.split(","):
        todo = [t for t in tasks if (tag, args.opponent, t[0], t[1]) not in done]
        if not todo:
            continue
        if VARIANTS[tag] is None:
            searcher, net, smeta = None, plain_net, {}
        else:
            searcher, net, smeta = build_searcher(tag, base, args.device, args)
        cfg = MatchConfig(every=args.every, window=args.window, cap_s=args.cap_s, trace_s=TRACE_S,
                          search_from=args.search_from, search_until=None)
        meta = {"base": args.base, "search_from": args.search_from, "branches": args.branches, "every": args.every,
                "window": args.window, "margin": args.margin, "value": args.value, "cap_s": args.cap_s, **smeta}
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            t0 = time.time()
            opp_id = None
            if VARIANTS[tag] and VARIANTS[tag].get("opp_id"):
                from blackout_env.sim.search import OpponentID

                opp_id = OpponentID([searcher.policy.net] + searcher.policy.opp_nets, len(searcher.opp_branches),
                                    **VARIANTS[tag]["opp_id"])
            res = play_matches([s for s, _ in chunk], [sd for _, sd in chunk], ModNetPolicy(net), opp, searcher, cfg,
                               log=print if args.verbose else None, base=base, overlay=True, opp_id=opp_id)
            wall = time.time() - t0
            with (out / f"matches_{args.opponent}.jsonl").open("a") as fh:
                for j, (s, sd) in enumerate(chunk):
                    fh.write(json.dumps({
                        "tag": tag, "opponent": args.opponent, "seed": s, "side": sd,
                        "margin": int(res["margin"][j]), "ours": int(res["ours"][j]), "opp": int(res["opp"][j]),
                        "over": bool(res["over"][j]), "ticks": int(res["ticks"][j]),
                        "lead_at": {str(k): int(v) for k, v in zip(TRACE_S, res["lead_at"][j])},
                        "first_ours": int(res["first_ours"][j]), "first_opp": int(res["first_opp"][j]),
                        "searches": int(res["searches"][j]), "chosen": res["chosen"][j].tolist(),
                        "gain_sum": float(res["gain_sum"][j]), "chunk_wall_s": wall, "chunk_n": len(chunk),
                        "chunk_searches": int(res["searches"].sum()), "search_s": res["search_s"],
                        "search_game_decisions": int(res["search_game_decisions"]), "meta": meta,
                        **({"id_assigned": res["id_assigned"][j].tolist(),
                            "id_posterior": [round(float(x), 4) for x in res["id_posterior"][j]]}
                           if "id_assigned" in res else {})}) + "\n")
            m = res["margin"]
            print(f"[{tag} vs {args.opponent}] {c + len(chunk)}/{len(todo)} margin {m.mean():+.1f} wall {wall:.0f}s "
                  f"(search {res['search_s']:.0f}s, {res['searches'].sum()} roots)", flush=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def paired(a: dict, b: dict, key=lambda x: x["margin"]):
    """Mean paired difference a - b over common (seed, side), SE over seeds (sides averaged first)."""
    by = defaultdict(list)
    for k in sorted(k for k in a if k in b):
        by[k[0]].append(key(a[k]) - key(b[k]))
    ps = [np.mean(v) for v in by.values()]
    return (float(np.mean(ps)) if ps else float("nan")), se(ps), len(ps)


def cost(rows: list[dict]) -> dict:
    """Search cost of one (tag, opponent): s/match, lockstep roots/s, game-decisions/root (chunk-level fields)."""
    chunks = {}
    for r in rows:
        chunks[(r["chunk_wall_s"], r["search_s"])] = r
    ss = sum(c["search_s"] for c in chunks.values())
    nroot = sum(c.get("chunk_searches", 0) for c in chunks.values())
    gd = sum(c.get("search_game_decisions", 0) for c in chunks.values())
    n = sum(c["chunk_n"] for c in chunks.values())
    return {"search_s_per_match": ss / max(1, n), "roots_per_s": nroot / ss if ss else float("nan"),
            "gd_per_root": gd / max(1, nroot), "roots_per_match": nroot / max(1, n),
            "wall_s_per_match": sum(c["chunk_wall_s"] for c in chunks.values()) / max(1, n)}


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["opponent"])][(r["seed"], r["side"])] = r
    opps = [o for o in OPPONENTS if (args.ref, o) in by] + sorted({o for _, o in by} - set(OPPONENTS))
    tags = [t for t in VARIANTS if t != args.ref and any(tg == t for tg, _ in by)]
    lines = [f"R1 blind search: paired difference vs {args.ref!r} (same opponent, seed, side); +-SE over seeds, sides averaged",
             f"opponents: {', '.join(opps)}", ""]
    w = 13
    lines.append(f"{'tag':12s} " + " ".join(f"{o:>{w}s}" for o in opps) + f" {'pooled':>{w}s} {'worst':>{w + 7}s}  s/match  roots/s")
    summary = {"ref": args.ref, "opponents": opps, "tags": {}}
    refm = " ".join(f"{np.mean([x['margin'] for x in by[(args.ref, o)].values()]):>+{w}.1f}" for o in opps)
    lines.append(f"{args.ref + ' margin':12s} {refm}")
    for tag in tags:
        cells, ds = [], {}
        for o in opps:
            if (tag, o) not in by:
                cells.append(f"{'':>{w}s}")
                continue
            d, s_, n = paired(by[(tag, o)], by[(args.ref, o)])
            ds[o] = (d, s_, n)
            cells.append(f"{f'{d:+.1f}±{s_:.1f}':>{w}s}")
        full = len(ds) == len(opps)
        pooled = (np.mean([v[0] for v in ds.values()]), np.sqrt(sum(v[1] ** 2 for v in ds.values())) / max(1, len(ds)))
        worst = min(ds.items(), key=lambda kv: kv[1][0]) if ds else ("-", (np.nan, np.nan, 0))
        allrows = [x for o in opps for x in by.get((tag, o), {}).values()]
        c = cost(allrows)
        lines.append(f"{tag:12s} " + " ".join(cells) + f" {f'{pooled[0]:+.1f}±{pooled[1]:.1f}' + ('' if full else '*'):>{w}s} "
                     f"{f'{worst[1][0]:+.1f}±{worst[1][1]:.1f} ({worst[0]})':>{w + 7}s}  {c['search_s_per_match']:7.1f}  {c['roots_per_s']:7.1f}")
        summary["tags"][tag] = {"by_opponent": {o: {"diff": v[0], "se": v[1], "seeds": v[2]} for o, v in ds.items()},
                                "pooled": pooled[0], "pooled_se": pooled[1], "complete": full,
                                "worst": worst[0], "worst_diff": worst[1][0], "worst_se": worst[1][1], "cost": c}
    lines += ["", "(* = pooled over the opponents present only)", "", "seeds per (tag, opponent):"]
    for tag in [args.ref] + tags:
        lines.append(f"  {tag:12s} " + " ".join(f"{o}:{len(by.get((tag, o), {})) // 2}" for o in opps if (tag, o) in by))
    lines += ["", "cost per tag (all opponents): search s/match, roots/match, lockstep roots/s, game-decisions/root, wall s/match"]
    for tag in tags:
        c = summary["tags"][tag]["cost"]
        lines.append(f"  {tag:12s} {c['search_s_per_match']:6.1f} s  {c['roots_per_match']:5.1f} roots  {c['roots_per_s']:6.1f} roots/s  "
                     f"{c['gd_per_root']:6.0f} gd/root  wall {c['wall_s_per_match']:5.1f} s")
    # pairwise between search variants (same opponent), pooled
    if args.pairs:
        lines += ["", "pairwise (a - b), pooled over common opponents:"]
        for spec in args.pairs:
            a, b = spec.split("-", 1) if ":" not in spec else spec.split(":")
            ds = {o: paired(by[(a, o)], by[(b, o)]) for o in opps if (a, o) in by and (b, o) in by}
            if not ds:
                continue
            pm = np.mean([v[0] for v in ds.values()])
            ps = np.sqrt(sum(v[1] ** 2 for v in ds.values())) / len(ds)
            lines.append(f"  {a} - {b}: pooled {pm:+.1f}±{ps:.1f}  " + " ".join(f"{o} {v[0]:+.1f}±{v[1]:.1f}" for o, v in ds.items()))
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / f"report_vs_{args.ref}.txt").write_text(txt + "\n")
    (Path(args.out) / f"summary_vs_{args.ref}.json").write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--opponent", required=True, help="|".join(OPPONENTS))
    q.add_argument("--tags", default="plain,det,o4f")
    q.add_argument("--base", default="p14b", choices=sorted(BASES), help="our acting policy (use its own --out)")
    q.add_argument("--value", default=VALUE)
    q.add_argument("--v25r", default=V25R, help="the P14b-distilled rollout net for the *_v25r tags")
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--every", type=int, default=25)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--margin", type=float, default=3.0)
    q.add_argument("--search-from", type=int, default=SEARCH_FROM)
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=2_000_000)
    q.add_argument("--chunk", type=int, default=128)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--pipeline", type=int, default=0,
                   help="B1: 1 = pipelined search rollouts (GPU/CPU overlap, bit-identical); n > 1 = chunk_games n")
    q.add_argument("--pipeline-graphs", action="store_true", help="B1: the pipeline's full-chunk forwards as CUDA graphs")
    q.add_argument("--out", default=str(REPO / "reports/blind_search/sim"))
    q = sub.add_parser("report")
    q.add_argument("--out", default=str(REPO / "reports/blind_search/sim"))
    q.add_argument("--ref", default="plain")
    q.add_argument("--pairs", nargs="*", default=[], help="a:b -- tag a minus tag b per opponent, pooled")
    args = p.parse_args()
    if getattr(args, "device", None) == "":
        from blackout_env.sim.search import default_device

        args.device = default_device()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
