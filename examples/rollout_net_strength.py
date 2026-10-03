"""Is the small rollout net (r5, RolloutNet) as strong as P11?  Simulator matches, paired by seed.

Every (candidate, opponent) pairing plays the same seeds from both sides, full matches (a match
also ends at an absorption that leaves no battery in play, as the Unity evaluations' default
``stop_when_exhausted``), so a candidate's margin minus the reference candidate's on the same
(opponent, seed, side) is a paired difference.  Worker processes play chunks of seeds; every game
is one line of ``<out>/games.jsonl`` (a rerun skips what is there), then ``summary`` prints the table.

    PYTHONPATH=. .venv/bin/python examples/rollout_net_strength.py sim --seeds 1000 --workers 12 \\
        --out reports/rollout_net_strength/sim
    PYTHONPATH=. .venv/bin/python examples/rollout_net_strength.py summary --out reports/rollout_net_strength/sim
    PYTHONPATH=. .venv/bin/python examples/rollout_net_strength.py latency

Specs: ``p11``, ``r5``, ``p5a`` (checkpoints under MAIN), ``net:<path>`` (any stateless distill
checkpoint), ``small:<path>`` (a RolloutNet), ``v23`` / ``v22`` / ``v19`` / ``heur:<descriptor>``.
docs/policy_v2_design_ko.md §19 "r5 대 P11 강도 검증".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MAIN = Path(os.environ.get("BLACKOUT_MAIN", "/Users/mac/project/26rl/blackout-env"))  # shared checkpoints (read-only)
CKPT = {"p11": MAIN / "checkpoints/distill/frozen/p11_v22.pt", "p5a": MAIN / "checkpoints/distill/frozen/p5a_v22.pt",
        "r5": MAIN / "checkpoints/rollout_net/r5.pt"}
HEUR = {"v23": "strategic_v23", "v22": "strategic_v22", "v19": "strategic_v19", "v24": "strategic_v24"}
DPS = 25  # decisions per second
TRACE_S = (10, 20, 30, 45, 60, 90, 120, 150, 240, 300)
HORIZON = 420 * DPS + 50

_POLICIES: dict = {}


def make_policy(spec: str, device: str):
    if spec in _POLICIES:
        return _POLICIES[spec]
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if spec in HEUR or spec.startswith("heur:"):
        pol = HeuristicPolicy(HEUR.get(spec, spec[5:]))
    elif spec == "r5" or spec.startswith("small:"):
        pol = NetPolicy(RolloutNet.from_checkpoint(str(CKPT["r5"] if spec == "r5" else spec[6:])), device=device)
    elif spec in CKPT or spec.startswith("net:"):
        pol = NetPolicy.from_checkpoint(str(CKPT[spec] if spec in CKPT else spec[4:]), device=device)
    else:
        raise ValueError(spec)
    _POLICIES[spec] = pol
    return pol


def _init_worker():
    import numba
    import torch

    torch.set_num_threads(1)
    numba.set_num_threads(1)


def play_chunk(cand: str, opp: str, side: int, seeds: list[int], device: str) -> list[dict]:
    """``cand`` on team ``side`` against ``opp``: one batched rollout of every seed."""
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout

    pc, po = make_policy(cand, device), make_policy(opp, device)
    pa, pb = (pc, po) if side == 0 else (po, pc)
    t0 = time.perf_counter()
    res = BatchRollout(pa, pb).run([SimState.from_seed(s) for s in seeds], HORIZON, reseeds=seeds,
                                   stop_when_exhausted=True, trace_steps=[s * DPS for s in TRACE_S])
    wall = time.perf_counter() - t0
    sign = 1 if side == 0 else -1
    rows = []
    for b, seed in enumerate(seeds):
        fc, fo = int(res.first_hunter[b, side]), int(res.first_hunter[b, 1 - side])
        fc_, fo_ = (fc if fc >= 0 else 1 << 30), (fo if fo >= 0 else 1 << 30)
        rows.append({"candidate": cand, "opponent": opp, "seed": int(seed), "side": side,
                     "lead": int(sign * res.lead(0)[b]), "race": int(np.sign(fo_ - fc_)),
                     "first_hunter_s": [fc / DPS if fc >= 0 else None, fo / DPS if fo >= 0 else None],
                     "steps": int(res.steps[b]), "over": bool(res.over[b]), "exhausted": bool(res.exhausted[b]),
                     "trace": {str(s): int(sign * res.lead_trace[i, b]) for i, s in enumerate(TRACE_S)},
                     "wall_s": round(wall / len(seeds), 3)})
    return rows


def cmd_sim(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "games.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            done.add((d["candidate"], d["opponent"], d["side"], d["seed"]))
    pairs = [(c, o) for c in args.candidates for o in args.opponents if c != o]
    pairs += [tuple(p.split(",")) for p in args.h2h]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = []
    for c, o in pairs:
        for side in (0, 1):
            todo = [s for s in seeds if (c, o, side, s) not in done]
            for i in range(0, len(todo), args.chunk):
                tasks.append((c, o, side, todo[i:i + args.chunk]))
    # heuristic opponents are the slow ones: start them first
    tasks.sort(key=lambda t: (t[1] not in HEUR, t[0]))
    print(f"{len(tasks)} chunks, {sum(len(t[3]) for t in tasks)} games, {args.workers} workers", flush=True)
    import multiprocessing as mp

    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex, \
            open(log, "a") as f:
        futs = {ex.submit(play_chunk, *t, args.device): t for t in tasks}
        for k, fut in enumerate(as_completed(futs)):
            rows = fut.result()
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            c, o, side, ss = futs[fut]
            print(f"[{k + 1}/{len(tasks)}] {c} vs {o} side {side} {len(ss)} games, "
                  f"{rows[0]['wall_s'] * len(ss):.0f}s, elapsed {time.time() - t0:.0f}s", flush=True)
    cmd_summary(args)


def _se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(rows: list[dict], ref: str, key=lambda d: d["lead"]) -> list[dict]:
    """Per (candidate, opponent): margin, W/L, race won/tie/lost; and against ``ref`` on the same
    (opponent, seed, side): the paired margin difference (se over seeds, both sides averaged)."""
    by = defaultdict(dict)
    for d in rows:
        by[(d["candidate"], d["opponent"])][(d["seed"], d["side"])] = d
    table = []
    for (c, o), games in sorted(by.items()):
        lead = np.array([key(d) for d in games.values()], float)
        race = np.array([d["race"] for d in games.values()])
        row = {"candidate": c, "opponent": o, "n": len(lead), "margin": float(lead.mean()), "margin_se": _se(lead),
               "win": float(np.mean(lead > 0)), "draw": float(np.mean(lead == 0)), "loss": float(np.mean(lead < 0)),
               "race": [int((race == v).sum()) for v in (1, 0, -1)]}
        base = by.get((ref, o))
        if c != ref and base:
            per_seed = defaultdict(list)
            for (seed, side), d in games.items():
                if (seed, side) in base:
                    per_seed[seed].append(key(d) - key(base[(seed, side)]))
            dd = np.array([np.mean(v) for v in per_seed.values()])
            row.update({"vs_ref": ref, "paired_seeds": len(dd), "dmargin": float(dd.mean()), "dmargin_se": _se(dd),
                        "same_race": float(np.mean([games[k]["race"] == base[k]["race"] for k in games if k in base]))})
        table.append(row)
    return table


def cmd_summary(args):
    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    lines = []
    table = summarize(rows, args.ref)
    lines.append(f"{'candidate':10s} {'opponent':10s} {'n':>5s} {'margin':>7s} {'±se':>5s} {'W/D/L %':>14s} "
                 f"{'race W/T/L':>14s} {'Δ vs ' + args.ref:>10s} {'±se':>5s} {'same race':>9s}")
    for r in table:
        d = f"{r['dmargin']:+10.1f} {r['dmargin_se']:5.1f} {r['same_race']:9.3f}" if "dmargin" in r else ""
        lines.append(f"{r['candidate']:10s} {r['opponent']:10s} {r['n']:5d} {r['margin']:+7.1f} {r['margin_se']:5.1f} "
                     f"{100 * r['win']:4.0f}/{100 * r['draw']:3.0f}/{100 * r['loss']:3.0f}  "
                     f"{'/'.join(map(str, r['race'])):>14s} {d}")
    # the lead over time (paired, against ref): where the difference appears
    lines.append("")
    lines.append("paired Δ lead vs " + args.ref + " by match time (s): " + " ".join(f"{s:>9d}" for s in TRACE_S) + "      final")
    for r in table:
        if "dmargin" not in r:
            continue
        cells = []
        for s in TRACE_S:
            t = summarize([d for d in rows if (d["candidate"], d["opponent"]) in ((r["candidate"], r["opponent"]), (args.ref, r["opponent"]))],
                          args.ref, key=lambda d, s=s: d["trace"][str(s)])
            t = next(x for x in t if x["candidate"] == r["candidate"])
            cells.append(f"{t['dmargin']:+5.1f}±{t['dmargin_se']:3.1f}")
        lines.append(f"{r['candidate']:10s} vs {r['opponent']:10s}{'':22s} " + " ".join(cells) + f" {r['dmargin']:+5.1f}±{r['dmargin_se']:3.1f}")
    # match length and how often the race decides
    lines.append("")
    for (c, o), grp in sorted(defaultdict(list, {k: [d for d in rows if (d['candidate'], d['opponent']) == k]
                                                for k in {(d['candidate'], d['opponent']) for d in rows}}).items()):
        steps = np.array([d["steps"] for d in grp])
        ex = np.mean([d["exhausted"] for d in grp])
        ov = np.mean([d["over"] for d in grp])
        lead = np.array([d["lead"] for d in grp])
        race = np.array([d["race"] for d in grp])
        cond = " ".join(f"{name} {lead[race == v].mean():+.1f}(n{(race == v).sum()})" for name, v in (("raceW", 1), ("raceT", 0), ("raceL", -1)) if (race == v).any())
        lines.append(f"{c:10s} vs {o:10s} length {steps.mean() / DPS:5.1f}s (p90 {np.percentile(steps, 90) / DPS:5.1f}) "
                     f"target/timeout {ov:.2f} exhausted {ex:.2f} | margin by race: {cond}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    json.dump(table, open(out / "summary.json", "w"), indent=1)


def unity_rows(path: Path) -> list[dict]:
    """examples/distill_heldout_eval.py's matches.jsonl -> this script's game rows (candidate /
    opponent short names, lead, race from first Hunter times, lead at 30/60/120 s or the final lead
    when the match ended earlier) plus the teacher-agreement counts."""
    def short(x):
        x = x.removeprefix("ckpt:")
        for k, v in CKPT.items():
            if Path(x).name == Path(v).name:
                return k
        return {v: k for k, v in HEUR.items()}.get(x, x)

    rows = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        lead = d["learner_score"] - d["opponent_score"]
        fr = d.get("first_reached", {})
        fc, fo = fr.get("first_hunter1_s", 1e9), fr.get("opp_first_hunter1_s", 1e9)
        trace = {}
        for s in (30, 60, 120):
            m = d.get("margins", {}).get(f"margin_{s}s")
            trace[str(s)] = m if m is not None else (lead if d["elapsed_s"] < s else float("nan"))
        rows.append({"candidate": short(d["candidate"]), "opponent": short(d["opponent"]), "seed": d["seed"],
                     "side": int(d["swap"]), "lead": lead, "race": int(np.sign(fo - fc)) if abs(fo - fc) > 0.05 else 0,
                     "steps": int(d["elapsed_s"] * DPS), "over": d["elapsed_s"] >= 419, "exhausted": False,
                     "trace": trace, "counts": d.get("counts", {})})
    return rows


def cmd_unity_summary(args):
    rows = unity_rows(Path(args.out) / "matches.jsonl")
    table = summarize(rows, args.ref)
    lines = [f"{'candidate':10s} {'opponent':10s} {'n':>5s} {'margin':>7s} {'±se':>5s} {'W/D/L %':>14s} "
             f"{'race W/T/L':>14s} {'Δ vs ' + args.ref:>10s} {'±se':>5s} {'same race':>9s}"]
    for r in table:
        d = f"{r['dmargin']:+10.1f} {r['dmargin_se']:5.1f} {r['same_race']:9.3f}" if "dmargin" in r else ""
        lines.append(f"{r['candidate']:10s} {r['opponent']:10s} {r['n']:5d} {r['margin']:+7.1f} {r['margin_se']:5.1f} "
                     f"{100 * r['win']:4.0f}/{100 * r['draw']:3.0f}/{100 * r['loss']:3.0f}  "
                     f"{'/'.join(map(str, r['race'])):>14s} {d}")
    lines += ["", f"paired Δ lead vs {args.ref} at 30 / 60 / 120 s / final (ended matches: their final lead)"]
    for r in table:
        if "dmargin" not in r:
            continue
        sub = [d for d in rows if (d["candidate"], d["opponent"]) in ((r["candidate"], r["opponent"]), (args.ref, r["opponent"]))]
        cells = []
        for s in ("30", "60", "120"):
            ok = [d for d in sub if not math.isnan(d["trace"][s])]
            t = next(x for x in summarize(ok, args.ref, key=lambda d, s=s: d["trace"][s]) if x["candidate"] == r["candidate"])
            cells.append(f"{t['dmargin']:+5.1f}±{t['dmargin_se']:3.1f}")
        lines.append(f"{r['candidate']:10s} vs {r['opponent']:10s} " + " ".join(cells) + f" {r['dmargin']:+5.1f}±{r['dmargin_se']:3.1f}")
    # agreement with the V23 act-teacher on the student's own states, by match time
    lines += ["", "agreement with act:strategic_v23 (label hit) / heading error (deg) on own states, by time bucket"]
    for (c, o) in sorted({(d["candidate"], d["opponent"]) for d in rows}):
        tot = defaultdict(float)
        for d in rows:
            if (d["candidate"], d["opponent"]) == (c, o):
                for k, v in d["counts"].items():
                    tot[k] += v
        cells = []
        for b in ("all", "t_0_30s", "t_30_60s", "t_60_120s", "t_120s_plus"):
            hit, n = tot.get(f"agree_hit/{b}", 0), tot.get(f"agree_n/{b}", 0)
            ae, an = tot.get(f"angle_err_sum/{b}", 0), tot.get(f"angle_n/{b}", 0)
            cells.append(f"{b}: {hit / max(n, 1):.3f}/{ae / max(an, 1):4.1f}")
        ae, an = tot.get("angle_err_sum/opening", 0), tot.get("angle_n/opening", 0)
        w2 = tot.get("angle_within2_hit/opening", 0)
        cells.append(f"opening<10s: {ae / max(an, 1):4.1f} deg, <=2deg {w2 / max(an, 1):.2f}")
        lines.append(f"{c:10s} vs {o:10s} " + "  ".join(cells))
    lines.append("")
    for (c, o) in sorted({(d["candidate"], d["opponent"]) for d in rows}):
        grp = [d for d in rows if (d["candidate"], d["opponent"]) == (c, o)]
        lead, race = np.array([d["lead"] for d in grp]), np.array([d["race"] for d in grp])
        steps = np.array([d["steps"] for d in grp])
        cond = " ".join(f"{name} {lead[race == v].mean():+.1f}(n{(race == v).sum()})"
                        for name, v in (("raceW", 1), ("raceT", 0), ("raceL", -1)) if (race == v).any())
        lines.append(f"{c:10s} vs {o:10s} length {steps.mean() / DPS:5.1f}s timeout {np.mean([d['over'] for d in grp]):.2f} | margin by race: {cond}")
    text = "\n".join(lines)
    print(text)
    (Path(args.out) / "summary_paired.txt").write_text(text + "\n")
    json.dump(table, open(Path(args.out) / "summary_paired.json", "w"), indent=1)


def cmd_latency(args):
    """Per-decision latency of one team's move: MyPolicy.act on an observation dict (the Unity path:
    canonicalise, tensors, forward, decode) and the bare forward, CPU (1 thread and default) and MPS."""
    import torch

    from blackout_env.sim import SimState
    from blackout_env.train.distill.inference import load_checkpoint_net, make_student

    sim = SimState.from_seed(990000)
    for _ in range(400):
        sim.step(np.zeros((10, 2), np.float32))
    obs = sim.obs()
    team = {k: obs[k] for k in (f"unit_{i}" for i in range(5))}
    from blackout_env.env.team_frame import canonical_obs

    c = canonical_obs(obs["unit_0"])
    g0 = torch.tensor(c["graphic"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    t0_ = torch.tensor(c["team_state"], dtype=torch.float32).unsqueeze(0)
    a0 = torch.tensor(c["agent_states"], dtype=torch.float32).unsqueeze(0)
    out = {}
    for name in args.nets:
        path = str(CKPT.get(name, name))
        net = load_checkpoint_net(path)
        params = sum(p.numel() for p in net.parameters())
        res = {"params": params}
        for threads in (1, 6):
            torch.set_num_threads(threads)
            stu = make_student(load_checkpoint_net(path))
            for _ in range(30):
                stu.act(team)
            ts = []
            for _ in range(args.reps):
                s = time.perf_counter()
                stu.act(team)
                ts.append(time.perf_counter() - s)
            res[f"act_cpu_t{threads}_ms"] = 1e3 * float(np.median(ts))
            own = stu._net.own
            with torch.inference_mode():
                for _ in range(30):
                    own(g0, t0_, a0)
                ts = []
                for _ in range(args.reps):
                    s = time.perf_counter()
                    own(g0, t0_, a0)
                    ts.append(time.perf_counter() - s)
            res[f"forward_cpu_t{threads}_ms"] = 1e3 * float(np.median(ts))
        if torch.backends.mps.is_available():
            from blackout_env.sim.batch import NetPolicy

            pol = NetPolicy(load_checkpoint_net(path), device="mps")
            g, t, a = (x.numpy() for x in (g0, t0_, a0))
            for bs in (1, 64):
                gb, tb, ab = (np.repeat(x, bs, 0) for x in (g, t, a))
                for _ in range(30):
                    pol.forward(gb, tb, ab)
                ts = []
                for _ in range(args.reps):
                    s = time.perf_counter()
                    pol.forward(gb, tb, ab)  # includes the copy back (synchronises)
                    ts.append(time.perf_counter() - s)
                res[f"forward_mps_b{bs}_ms"] = 1e3 * float(np.median(ts))
        out[name] = res
        print(name, json.dumps(res), flush=True)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        json.dump(out, open(Path(args.out) / "latency.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--candidates", nargs="+", default=["p11", "r5"], help="first = the paired reference")
    s.add_argument("--opponents", nargs="+", default=["v23", "v22", "v19", "p5a"])
    s.add_argument("--h2h", nargs="*", default=["r5,p11"], help="extra candidate,opponent pairings")
    s.add_argument("--seeds", type=int, default=1000)
    s.add_argument("--seed0", type=int, default=1_100_000)
    s.add_argument("--chunk", type=int, default=50)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--device", default="mps")
    s.add_argument("--ref", default="p11")
    s.add_argument("--out", default="reports/rollout_net_strength/sim")
    s = sub.add_parser("summary")
    s.add_argument("--ref", default="p11")
    s.add_argument("--out", default="reports/rollout_net_strength/sim")
    s = sub.add_parser("unity-summary", help="pair a distill_heldout_eval.py --out (r5 / p11 checkpoints)")
    s.add_argument("--ref", default="p11")
    s.add_argument("--out", default="reports/rollout_net_strength/unity")
    s = sub.add_parser("latency")
    s.add_argument("--nets", nargs="+", default=["p11", "r5"])
    s.add_argument("--reps", type=int, default=300)
    s.add_argument("--out", default="")
    args = p.parse_args()
    {"sim": cmd_sim, "summary": cmd_summary, "unity-summary": cmd_unity_summary, "latency": cmd_latency}[args.cmd](args)


if __name__ == "__main__":
    main()
