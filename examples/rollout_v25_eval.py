"""Q2 (docs/policy_v2_design_ko.md §31 "Q2: P14b 교사 롤아웃 망"): checks and row metrics of rollout nets
re-distilled from P14b (the V25-level student) inside the batched simulator.

    # P14b batched in the simulator (NetPolicy, CUDA, many games per forward) == its per-match student
    # (make_student(...).act on sim.obs(), CPU batch 1): per-decision moves on the same states, then
    # free-running batched vs per-match games (first Hunter decision, final score)
    PYTHONPATH=. .venv/bin/python examples/rollout_v25_eval.py teacher-check --games 8 --out reports/rollout_v25/teacher_check.json
    # agreement with the teacher (P14b labels: arg max bin, heading) on held-out simulator states, by match time
    PYTHONPATH=. .venv/bin/python examples/rollout_v25_eval.py agree --sets p14b_states=D1/train.npz own_new=D2/train.npz \\
        --nets new=small:N.pt f3x=small:F.pt p11=net:P11.pt --json reports/rollout_v25/agree.json

Held-out sets are examples/rollout_net_collect.py ``--teacher net:P14B`` outputs (p11_q / p11_off = the
teacher's logits / offsets on every recorded view, both teams).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))
DPS = 25
BUCKETS = ((0, 10), (10, 30), (30, 60), (60, 120), (120, 1e9))


def cmd_teacher_check(args):
    import torch

    from blackout_env.sim import SimState, obs_from_raw
    from blackout_env.sim.batch import N_AGENT_COLS, BatchRollout, BatchSim, NetPolicy
    from blackout_env.train.distill.inference import load_checkpoint_net, make_student

    torch.set_num_threads(4)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    pol = NetPolicy.from_checkpoint(args.teacher, device=args.device)
    stu = make_student(load_checkpoint_net(args.teacher))  # the Unity / per-match path: CPU, batch 1

    def ref_moves(obs):
        mv = np.zeros((10, 2), np.float32)
        for tm in (0, 1):
            acts = stu.act({f"unit_{u}": obs[f"unit_{u}"] for u in range(5 * tm, 5 * tm + 5)})
            for k, v in acts.items():
                mv[int(k.split("_")[1])] = np.clip(np.asarray(v, np.float32), -1, 1)
        return mv

    # 1. same states: step every game with the per-match moves, compare the batched moves on each state
    sims = [SimState.from_seed(s) for s in seeds]
    for s, k in zip(sims, seeds):
        s.reseed(k)
    bs = BatchSim(sims)
    n = bs.n
    bufs = (np.zeros((n, 2, 13, 24, 24), np.float32), np.zeros((n, 2, 4), np.float32), np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    diffs, same_bin, units = [], 0, 0
    t0 = time.time()
    for t in range(args.steps):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        bs.render(live, *bufs)
        out = pol.moves(bs, live, (0, 1), bufs)
        bat = np.concatenate([out[0], out[1]], 1)  # [m, 10, 2]
        mv = np.zeros((n, 10, 2), np.float32)
        for j, b in enumerate(live):
            ref = ref_moves(obs_from_raw(*bs.raw(int(b))))
            mv[b] = ref
            a_ref = np.arctan2(ref[:, 1], ref[:, 0])
            a_bat = np.arctan2(bat[j, :, 1], bat[j, :, 0])
            d = np.abs(np.remainder(a_bat - a_ref + math.pi, 2 * math.pi) - math.pi)
            diffs.append(d)
            same_bin += int((np.round(a_ref / (2 * math.pi / 32)) % 32 == np.round(a_bat / (2 * math.pi / 32)) % 32).sum())
            units += 10
        bs.step(mv, live)
    d = np.concatenate(diffs) * 180 / math.pi
    same = {"decisions": int(len(diffs)), "units": units, "max_deg": float(d.max()), "mean_deg": float(d.mean()),
            "frac_gt_0p01deg": float(np.mean(d > 0.01)), "frac_gt_1deg": float(np.mean(d > 1.0)),
            "bin_agree": same_bin / max(units, 1), "s": round(time.time() - t0, 1)}
    print("same states", json.dumps(same), flush=True)

    # 2. free-running: batched P14b vs P14b against per-match P14b vs P14b, same seeds
    t0 = time.time()
    res = BatchRollout(pol, pol).run([SimState.from_seed(s) for s in seeds], args.steps, reseeds=seeds)
    games = []
    for b, k in enumerate(seeds):
        sim = SimState.from_seed(k)
        sim.reseed(k)
        first = [-1, -1]
        for t in range(args.steps):
            if sim.over:
                break
            sim.step(ref_moves(sim.obs()))
            first = _first_hunter(sim, first, t)
        games.append({"seed": k, "batched_first_hunter": res.first_hunter[b].tolist(), "ref_first_hunter": first,
                      "batched_score": res.scores[b].tolist(), "ref_score": [int(x) for x in sim.scores]})
    free = {"games": games, "same_first_hunter": float(np.mean([g["batched_first_hunter"] == g["ref_first_hunter"] for g in games])),
            "same_score": float(np.mean([g["batched_score"] == g["ref_score"] for g in games])), "s": round(time.time() - t0, 1)}
    print("free running", json.dumps({k: v for k, v in free.items() if k != "games"}), flush=True)
    for g in games:
        print(g, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"teacher": args.teacher, "device": args.device, "same_states": same, "free_running": free},
                  open(args.out, "w"), indent=1)


def _first_hunter(sim, first, t):
    from blackout_env.sim import core
    from blackout_env.sim.mapdata import HUNTER

    for tm in (0, 1):
        if first[tm] < 0 and (sim.ui[5 * tm:5 * tm + 5, core.U_CLS] == HUNTER).any():
            first[tm] = t + 1
    return first


def _forward_fn(spec, device):
    """spec small:PATH / net:PATH -> f(g, t, a) device tensors -> (logits, offsets) of the own rows."""
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if spec.startswith("small:"):
        pol = NetPolicy(RolloutNet.from_checkpoint(spec[6:]), device=device)
    else:
        pol = NetPolicy.from_checkpoint(spec[4:], device=device)

    def f(g, t, a):
        with torch.inference_mode():
            return pol._own(g, t, a)
    return f


def bucket_metrics(f, data, device, batch=4096):
    """Per time bucket: arc-label hit (net arg max in the teacher's arg max +- 1), exact arg max agreement,
    heading error vs the teacher's executed heading (mean, <= 2 deg share), on every own unit."""
    import torch

    from rollout_net_train import ang_err_deg, headings, unpack_labels

    n = len(data["tick"])
    tick = np.asarray(data["tick"])
    acc = {}
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        g = torch.from_numpy(np.asarray(data["map"][idx]).view(np.int16)).to(device)
        t = torch.from_numpy(np.asarray(data["team_state"][idx])).to(device)
        a = torch.from_numpy(np.asarray(data["agent_states"][idx])).to(device)
        q, off = f(g, t, a)
        q, off = q.float(), off.float()
        tq = torch.from_numpy(np.asarray(data["p11_q"][idx]).astype(np.float32)).to(device)
        labels = torch.from_numpy(unpack_labels(np.asarray(data["labels"][idx]))).to(device)
        ang = torch.from_numpy(np.asarray(data["angles"][idx])).to(device)
        hit = labels.gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1).float()
        exact = (q.argmax(-1) == tq.argmax(-1)).float()
        err = ang_err_deg(headings(q, off), ang)
        sec = tick[idx] / DPS
        for name, (lo, hi) in [("all", (0, 1e9))] + [(f"{lo}-{hi if hi < 1e8 else ''}s", (lo, hi)) for lo, hi in BUCKETS]:
            m = torch.from_numpy((sec >= lo) & (sec < hi)).to(device)
            if not bool(m.any()):
                continue
            r = acc.setdefault(name, np.zeros(5))
            r += [float(hit[m].sum()), float(exact[m].sum()), float(err[m].sum()), float((err[m] <= 2).sum()), float(m.sum()) * 5]
    return {k: {"hit": v[0] / v[4], "argmax_agree": v[1] / v[4], "err_deg": v[2] / v[4], "within2": v[3] / v[4],
                "units": int(v[4])} for k, v in acc.items()}


def cmd_agree(args):
    import torch

    from rollout_net_train import load_split

    device = torch.device(args.device)
    sets = {kv.split("=", 1)[0]: load_split(kv.split("=", 1)[1], True) for kv in args.sets}
    out = {}
    for kv in args.nets:
        name, spec = kv.split("=", 1)
        f = _forward_fn(spec, args.device)
        out[name] = {s: bucket_metrics(f, d, device) for s, d in sets.items()}
        for s, r in out[name].items():
            print(f"{name:10s} {s:14s} " + "  ".join(f"{b}: {x['hit']:.3f}/{x['argmax_agree']:.3f}/{x['err_deg']:.1f}°"
                                                     for b, x in r.items()), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)


def cmd_table(args):
    """reports/rollout_v25/table.md: per candidate x opponent the paired Δ vs f3x and vs P14b (± seed SE),
    first-Hunter histogram, rollout cost x r5 (BatchRollout loop, N = 2048, mean of the rounds), and the
    agreement with P14b on held-out states by match time."""
    from collections import defaultdict

    from rollout_net_strength import DPS as SDPS, summarize

    root = Path(args.root)
    rows = [json.loads(x) for x in (root / "sim/games.jsonl").read_text().splitlines()]
    by = {ref: {(r["candidate"], r["opponent"]): r for r in summarize(rows, ref)} for ref in ("f3x", "p14b")}
    cands = [c for c in args.candidates if any(k[0] == c for k in by["f3x"])]
    opps = [o for o in args.opponents if any(k[1] == o for k in by["f3x"])]
    thr = defaultdict(list)
    for f in sorted((root / "throughput").glob("throughput_*.json")):
        name = f.stem.removeprefix("throughput_").rsplit("_r", 1)[0]
        for r in json.load(open(f)):
            if r["N"] == 2048:
                thr[(name, r["precision"])].append(r["game_decisions_per_s"])
    cost = {k: np.mean(thr[("r5", k[1])]) / np.mean(v) for k, v in thr.items()}
    lines = ["| 후보 | 상대 | n | 점수차 | Δ vs f3x | Δ vs P14b | 경주 승/동/패 | 첫 Hunter 98-/99/100/101+/없음 |", "|" + "---|" * 8]
    for c in cands:
        for o in opps:
            r = by["f3x"].get((c, o))
            if r is None:
                continue
            grp = [d for d in rows if (d["candidate"], d["opponent"]) == (c, o)]
            dec = [round(d["first_hunter_s"][0] * SDPS) if d["first_hunter_s"][0] is not None else None for d in grp]
            hist = "/".join(str(x) for x in (sum(v is not None and v <= 98 for v in dec), dec.count(99), dec.count(100),
                                             sum(v is not None and v > 100 for v in dec), dec.count(None)))
            d1 = f"{r['dmargin']:+.1f}±{r['dmargin_se']:.1f}" if "dmargin" in r else "–"
            r2 = by["p14b"].get((c, o), {})
            d2 = f"{r2['dmargin']:+.1f}±{r2['dmargin_se']:.1f}" if "dmargin" in r2 else "–"
            lines.append(f"| {c} | {o} | {r['n']} | {r['margin']:+.1f}±{r['margin_se']:.1f} | {d1} | {d2} | "
                         f"{'/'.join(map(str, r['race']))} | {hist} |")
    lines += ["", "| 망 | 처리량 fp32 / fp16 (게임 결정/초, N=2048) | 비용 ×r5 fp32 / fp16 |", "|---|---|---|"]
    for n in ("r5", "f3x", "v25f3x", "d256x32", "v25d256"):
        if (n, "fp32") in thr:
            lines.append(f"| {n} | {np.mean(thr[(n, 'fp32')]):.0f} / {np.mean(thr[(n, 'fp16')]):.0f} | "
                         f"{cost[(n, 'fp32')]:.2f} / {cost[(n, 'fp16')]:.2f} |")
    ag = json.load(open(root / "agree.json"))
    buckets = ["all", "0-10s", "10-30s", "30-60s", "60-120s", "120-s"]
    for s in next(iter(ag.values())):
        lines += ["", f"P14b 일치 — {s} (arc hit / arg max 일치 / 각도 오차°)", "| 망 | " + " | ".join(buckets) + " |", "|" + "---|" * 7]
        for n, d in ag.items():
            lines.append(f"| {n} | " + " | ".join(f"{d[s][b]['hit']:.3f} / {d[s][b]['argmax_agree']:.3f} / {d[s][b]['err_deg']:.1f}"
                                                 if b in d[s] else "–" for b in buckets) + " |")
    txt = "\n".join(lines)
    print(txt)
    (root / "table.md").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("teacher-check")
    q.add_argument("--teacher", default=str(Path.home() / "p14frozen/p14b.pt"))
    q.add_argument("--games", type=int, default=8)
    q.add_argument("--steps", type=int, default=1500)
    q.add_argument("--seed0", type=int, default=6_100_000)
    q.add_argument("--device", default="cuda")
    q.add_argument("--out", default="")
    q = sub.add_parser("agree")
    q.add_argument("--sets", nargs="+", required=True, help="NAME=PATH.npz (collect --teacher net: output)")
    q.add_argument("--nets", nargs="+", required=True, help="NAME=small:PATH | NAME=net:PATH")
    q.add_argument("--device", default="cuda")
    q.add_argument("--json", default="")
    q = sub.add_parser("table")
    q.add_argument("--root", default="reports/rollout_v25")
    q.add_argument("--candidates", nargs="+", default=["v25f3x", "v25d256", "f3x", "d256x32", "p14b"])
    q.add_argument("--opponents", nargs="+", default=["p11", "v23", "v25", "p14b"])
    args = p.parse_args()
    {"teacher-check": cmd_teacher_check, "agree": cmd_agree, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
