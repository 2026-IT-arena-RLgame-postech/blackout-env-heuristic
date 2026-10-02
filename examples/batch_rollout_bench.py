"""Batched simulator rollouts (S3 prep): throughput, exactness, action repeat, rollout-net fidelity.
(docs/policy_v2_design_ko.md "S3 준비 — 배치 롤아웃").

    PYTHONPATH=. .venv/bin/python examples/batch_rollout_bench.py throughput --out reports/batch_rollout
    PYTHONPATH=. .venv/bin/python examples/batch_rollout_bench.py exact
    PYTHONPATH=. .venv/bin/python examples/batch_rollout_bench.py repeat --games 200
    PYTHONPATH=. .venv/bin/python examples/batch_rollout_bench.py fidelity --small checkpoints/rollout_net/r1.pt

Roots for throughput: mid-game states (10-60 s) of P11-vs-P11 sim games from seeds 970000+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MAIN = Path("/Users/mac/project/26rl/blackout-env")  # shared checkpoints (read-only)
P11 = str(MAIN / "checkpoints/distill/frozen/p11_v22.pt")
DPS = 25


def _threads(n: int) -> None:
    import numba
    import torch

    torch.set_num_threads(n)
    numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))


def p11_policy(device="cpu", fp16=False, **kw):
    from blackout_env.sim.batch import NetPolicy

    return NetPolicy.from_checkpoint(P11, device=device, fp16=fp16, **kw)


def mid_game_roots(n_seeds: int, seed0: int = 970000, every: int = 50, until: int = 1500, policy=None):
    """States every ``every`` decisions of P11-vs-P11 games (from_seed, reseeded), until ``until``."""
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout

    policy = policy or p11_policy("cpu")
    roll = BatchRollout(policy, policy)
    states = [SimState.from_seed(seed0 + i) for i in range(n_seeds)]
    for s, k in zip(states, range(n_seeds)):
        s.reseed(seed0 + k)
    out = []
    for t in range(0, until, every):
        res = roll.run(states, every)
        states = [s for s in res.states() if not s.over]
        if not states:
            break
        out += [(t + every, s) for s in states]
    return out


def cmd_throughput(args):
    import pickle

    from blackout_env.sim.batch import BatchRollout, HeuristicPolicy

    _threads(args.threads)
    cache = Path(args.out) / "roots.pkl"
    if cache.exists():
        roots = pickle.load(open(cache, "rb"))
    else:
        roots = [s for t, s in mid_game_roots(64, every=50, until=1500) if 250 <= t <= 1500]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(roots, open(cache, "wb"))
    print("roots", len(roots), flush=True)
    rows = []
    configs = [c.split(":") for c in args.configs]
    for dev, prec in configs:
        pol = p11_policy(dev, prec == "fp16")
        roll = BatchRollout(pol, pol)
        for n in args.sizes:
            batch = [roots[i % len(roots)] for i in range(n)]
            roll.run(batch[: min(n, 16)], 3)  # warm-up (compile, MPS kernels)
            reps = max(1, args.min_decisions // (n * args.horizon))
            t0 = time.perf_counter()
            tot = {"render": 0.0, "policy": 0.0, "step": 0.0, "game_decisions": 0}
            for _ in range(reps):
                roll.run(batch, args.horizon, reseeds=range(n))
                for k in tot:
                    tot[k] += roll.timing[k]
            wall = time.perf_counter() - t0
            gd = tot["game_decisions"]
            row = {"device": dev, "precision": prec, "N": n, "game_decisions_per_s": gd / wall,
                   "us_per_game_decision": wall / gd * 1e6,
                   **{f"{k}_us": tot[k] / gd * 1e6 for k in ("render", "policy", "step")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.heuristic:
        pol = HeuristicPolicy("strategic_v23")
        roll = BatchRollout(pol, pol)
        batch = roots[:16]
        roll.run(batch[:2], 3)
        t0 = time.perf_counter()
        roll.run(batch, args.horizon)
        wall = time.perf_counter() - t0
        gd = roll.timing["game_decisions"]
        row = {"device": "cpu", "precision": "heuristic_v23", "N": 16, "game_decisions_per_s": gd / wall,
               "us_per_game_decision": wall / gd * 1e6}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(args.out) / f"throughput{args.tag}.json", "w"), indent=1)


def race_outcome(first: np.ndarray) -> np.ndarray:
    """[N, 2] first-Hunter decision per team (-1 never) -> +1 A first, -1 B first, 0 tie / none."""
    a = np.where(first[:, 0] < 0, 1 << 30, first[:, 0])
    b = np.where(first[:, 1] < 0, 1 << 30, first[:, 1])
    return np.sign(b - a)


def paired(ref: dict, arm: dict) -> dict:
    d = arm["lead"] - ref["lead"]
    n = len(d)
    out = {"same_winner": float(np.mean(np.sign(arm["lead"]) == np.sign(ref["lead"]))),
           "same_lead": float(np.mean(d == 0)), "mean_abs_dlead": float(np.abs(d).mean()),
           "dlead_mean": float(d.mean()), "dlead_se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0}
    if "race" in ref:
        out["same_race"] = float(np.mean(arm["race"] == ref["race"]))
    return out


def play_arm(policy_a, policy_b, roots, horizon, reseeds, repeat=(1, 1)) -> dict:
    from blackout_env.sim.batch import BatchRollout

    roll = BatchRollout(policy_a, policy_b, repeat=repeat[0], repeat_b=repeat[1])
    t0 = time.perf_counter()
    res = roll.run(roots, horizon, reseeds=reseeds)
    wall = time.perf_counter() - t0
    return {"lead": res.lead(0), "race": race_outcome(res.first_hunter), "over": res.over,
            "steps": res.steps, "game_decisions_per_s": roll.timing["game_decisions"] / wall,
            "policy_calls_per_decision": roll.timing["calls"] / max(1, roll.timing["game_decisions"])}


def summarize_arm(arm: dict) -> dict:
    return {"lead_A_mean": float(arm["lead"].mean()), "lead_A_se": float(arm["lead"].std(ddof=1) / np.sqrt(len(arm["lead"]))),
            "A_win": float(np.mean(arm["lead"] > 0)), "B_win": float(np.mean(arm["lead"] < 0)),
            "race_A/tie/B": [int((arm["race"] == v).sum()) for v in (1, 0, -1)],
            "over": float(arm["over"].mean()), "mean_steps": float(arm["steps"].mean()),
            "game_decisions_per_s": arm["game_decisions_per_s"],
            "policy_calls_per_decision": arm["policy_calls_per_decision"]}


def cmd_repeat(args):
    import pickle

    from blackout_env.sim import SimState

    _threads(args.threads)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    starts = [SimState.from_seed(s) for s in seeds]
    pol = p11_policy(args.device, args.precision == "fp16")
    out = {"config": vars(args), "full": {}, "rollout5s": {}}
    arms = [tuple(int(x) for x in a.split(",")) for a in args.arms]
    full = {}
    for rep in arms:
        full[rep] = play_arm(pol, pol, starts, args.cap_s * DPS, seeds, rep)
        print("full", rep, summarize_arm(full[rep]), flush=True)
    if args.noise:
        dev, prec = args.noise.split(":")
        noise = p11_policy(dev, prec == "fp16")
        full["noise"] = play_arm(noise, noise, starts, args.cap_s * DPS, seeds, (1, 1))
        print("full noise", summarize_arm(full["noise"]), flush=True)
    ref = full[(1, 1)]
    for k, arm in full.items():
        out["full"][str(k)] = {**summarize_arm(arm), "vs_k1": paired(ref, arm)}
    roots = pickle.load(open(Path(args.roots), "rb"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(roots), size=min(args.games, len(roots)), replace=False)
    sub = [roots[i] for i in pick]
    rs = [int(1000 + i) for i in pick]
    short = {}
    for rep in arms:
        short[rep] = play_arm(pol, pol, sub, args.rollout_s * DPS, rs, rep)
    if args.noise:
        short["noise"] = play_arm(noise, noise, sub, args.rollout_s * DPS, rs, (1, 1))
    for k, arm in short.items():
        out["rollout5s"][str(k)] = {**summarize_arm(arm), "vs_k1": paired(short[(1, 1)], arm)}
    print(json.dumps(out, indent=1, default=str))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out) / f"repeat{args.tag}.json", "w"), indent=1, default=str)


def make_policy(spec: str, device: str = "mps", fp16: bool = False):
    """p11 | v23 (strategic_v23 heuristic) | small:<ckpt> | hybrid:<ckpt> (P11 first 10 s, then small)
    | heur:<descriptor>."""
    from blackout_env.sim.batch import HeuristicPolicy, HybridPolicy, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if spec == "p11":
        return p11_policy(device, fp16)
    if spec == "v23":
        return HeuristicPolicy("strategic_v23")
    if spec.startswith("heur:"):
        return HeuristicPolicy(spec[5:])
    kind, _, path = spec.partition(":")
    small = NetPolicy(RolloutNet.from_checkpoint(path), device=device, fp16=fp16)
    if kind == "small":
        return small
    if kind == "hybrid":
        return HybridPolicy(p11_policy(device, fp16), small)
    raise ValueError(spec)


def cmd_fidelity(args):
    """Candidate vs opponent full sim matches from seeds, both sides: race (first Hunter) and margins,
    paired by (opponent, seed, side) against the reference candidate (P11)."""
    from blackout_env.sim import SimState

    _threads(args.threads)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    cands = {c: make_policy(c, args.device, args.precision == "fp16") for c in args.candidates}
    opps = {o: (cands[o] if o in cands else make_policy(o, args.device, args.precision == "fp16")) for o in args.opponents}
    res = {}
    for o, opp in opps.items():
        for c, cand in cands.items():
            leads, races, steps, overs = [], [], [], []
            for side in (0, 1):
                starts = [SimState.from_seed(s) for s in seeds]
                pa, pb = (cand, opp) if side == 0 else (opp, cand)
                t0 = time.perf_counter()
                arm = play_arm(pa, pb, starts, args.cap_s * DPS, seeds, (1, 1))
                sign = 1 if side == 0 else -1
                leads.append(sign * arm["lead"])
                races.append(sign * arm["race"])
                steps.append(arm["steps"])
                overs.append(arm["over"])
                print(o, c, side, f"{time.perf_counter() - t0:.0f}s", flush=True)
            lead, race = np.concatenate(leads), np.concatenate(races)
            res[(c, o)] = {"lead": lead, "race": race}
            print(json.dumps({"candidate": c, "opponent": o, "margin": float(lead.mean()),
                              "margin_se": float(lead.std(ddof=1) / np.sqrt(len(lead))),
                              "win": float(np.mean(lead > 0)), "loss": float(np.mean(lead < 0)),
                              "race_won/tie/lost": [int((race == v).sum()) for v in (1, 0, -1)],
                              "over": float(np.concatenate(overs).mean())}), flush=True)
    ref = args.candidates[0]
    out = []
    for (c, o), r in res.items():
        row = {"candidate": c, "opponent": o, "n": len(r["lead"]), "margin": float(r["lead"].mean()),
               "margin_se": float(r["lead"].std(ddof=1) / np.sqrt(len(r["lead"]))),
               "win": float(np.mean(r["lead"] > 0)), "loss": float(np.mean(r["lead"] < 0)),
               "race_won/tie/lost": [int((r["race"] == v).sum()) for v in (1, 0, -1)]}
        if c != ref:
            base = res[(ref, o)]
            d = r["lead"] - base["lead"]
            row.update({f"vs_{ref}_dmargin": float(d.mean()), f"vs_{ref}_dmargin_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                        f"vs_{ref}_same_race": float(np.mean(r["race"] == base["race"]))})
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out) / f"fidelity{args.tag}.json", "w"), indent=1)


def cmd_short(args):
    """5 s continuations from mid-game roots (roots.pkl, 10-60 s), both teams one policy spec:
    agreement of the lead change over the rollout with the reference (first spec)."""
    import pickle

    _threads(args.threads)
    roots = pickle.load(open(Path(args.roots), "rb"))
    pick = np.random.default_rng(0).choice(len(roots), size=min(args.roots_n, len(roots)), replace=False)
    sub = [roots[i] for i in pick]
    lead0 = np.array([r.scores[0] - r.scores[1] for r in sub])
    rs = [int(1000 + i) for i in pick]
    arms = {}
    for spec in args.specs:
        name, _, rep = spec.partition("@")
        prec = "fp16" if name.endswith("~fp16") else args.precision
        pol = make_policy(name.replace("~fp16", ""), args.device, prec == "fp16")
        k = int(rep or 1)
        arm = play_arm(pol, pol, sub, args.rollout_s * DPS, rs, (k, k))
        arm["dlead"] = arm["lead"] - lead0
        arms[spec] = arm
        print(spec, f"{arm['game_decisions_per_s']:.0f} dec/s", flush=True)
    ref = arms[args.specs[0]]
    out = []
    for spec, arm in arms.items():
        d = arm["dlead"] - ref["dlead"]
        row = {"spec": spec, "n": len(d), "dlead_mean": float(arm["dlead"].mean()),
               "same_dlead": float(np.mean(d == 0)), "mean_abs_diff": float(np.abs(d).mean()),
               "bias": float(d.mean()), "bias_se": float(d.std(ddof=1) / np.sqrt(len(d))),
               "corr": float(np.corrcoef(arm["dlead"], ref["dlead"])[0, 1]) if d.any() else 1.0,
               "game_decisions_per_s": arm["game_decisions_per_s"]}
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out) / f"short{args.tag}.json", "w"), indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("throughput")
    p.add_argument("--configs", nargs="+", default=["cpu:fp32", "mps:fp32", "mps:fp16"])
    p.add_argument("--sizes", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--min-decisions", type=int, default=4000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="reports/batch_rollout")
    p = sub.add_parser("repeat")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=980000)
    p.add_argument("--arms", nargs="+", default=["1,1", "2,2", "4,4", "2,1", "4,1"],
                   help="repeat_a,repeat_b; 1,1 is the reference")
    p.add_argument("--noise", default="cpu:fp32", help="k=1 on another device:precision = numeric-noise floor")
    p.add_argument("--device", default="mps")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--cap-s", type=int, default=150)
    p.add_argument("--rollout-s", type=int, default=5)
    p.add_argument("--roots", default="reports/batch_rollout/roots.pkl")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="reports/batch_rollout")
    p = sub.add_parser("fidelity")
    p.add_argument("--candidates", nargs="+", default=["p11"], help="first = the paired reference")
    p.add_argument("--opponents", nargs="+", default=["v23", "p11"])
    p.add_argument("--games", type=int, default=24, help="seeds; x2 sides")
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--cap-s", type=int, default=150)
    p.add_argument("--device", default="mps")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="reports/batch_rollout")
    p = sub.add_parser("short")
    p.add_argument("--specs", nargs="+", default=["p11", "p11~fp16", "p11@2", "v23"],
                   help="policy spec (make_policy) [~fp16] [@repeat]; first = reference")
    p.add_argument("--roots", default="reports/batch_rollout/roots.pkl")
    p.add_argument("--roots-n", type=int, default=300)
    p.add_argument("--rollout-s", type=int, default=5)
    p.add_argument("--device", default="mps")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="reports/batch_rollout")
    args = parser.parse_args()
    {"throughput": cmd_throughput, "repeat": cmd_repeat, "fidelity": cmd_fidelity, "short": cmd_short}[args.cmd](args)


if __name__ == "__main__":
    main()
