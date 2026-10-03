"""S3 Gate 1: does the 1-ply search over r5 branches (blackout_env/sim/search.py) beat plain r5?
(docs/policy_v2_design_ko.md §20 "S3 자기 대전 탐색 교사".)  Simulator only.

``play``: seeds x both sides, in lockstep chunks of ``--chunk`` matches; for each (opponent, seed,
side) the plain rollout net (tag ``plain``) and the searching one (tag ``--tag``) play the same
match start (``SimState.from_seed``, Unity RNG).  The search models the opponent as r5 whatever
it really is (self-play assumption).  Rows append to ``<out>/matches.jsonl`` (a rerun skips rows
already there).  ``--record`` also stores the search team's rows (canonical inputs + executed
moves, examples/rollout_net_train.py's packed format) for expert iteration.

``report``: per (tag, opponent) the search margin minus the plain margin, paired by (seed, side),
standard error over seeds (both sides of a seed averaged first), wins, race (first Hunter)
win/tie/loss for both, branch choice shares and wall-clock per match.

    PYTHONPATH=. .venv/bin/python examples/s3_search_eval.py play --opponent r5 --seeds 64 \\
        --branches 'alt*,samp:1.0x2' --every 25 --window 25 --tag s8e25 --out reports/s3_search
    PYTHONPATH=. .venv/bin/python examples/s3_search_eval.py report --out reports/s3_search
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

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"


def make_opponent_policy(name: str, our, device: str, same_net: bool = True):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if name == "r5":  # the same net: one merged forward (unless ours is another net, e.g. r6 vs r5)
        return our if same_net else NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device=device)
    if name.startswith("rnet:"):  # another rollout net checkpoint
        return NetPolicy(RolloutNet.from_checkpoint(name[5:]), device=device)
    return HeuristicPolicy(name)  # opponents.py descriptor, e.g. strategic_v23


def load_rows(out: Path) -> list[dict]:
    path = out / "matches.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_play(args):
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import MatchConfig, ModNetPolicy, Searcher, ValueNetLeaf, parse_branches, play_matches

    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = NetPolicy(RolloutNet.from_checkpoint(args.net), device=args.device)
    is_r5 = Path(args.net).resolve() == (REPO / R5).resolve()
    net_plain = net if is_r5 else NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=args.device)
    branches = parse_branches(args.branches)
    leaf = ValueNetLeaf(str(REPO / args.value), args.device)
    searcher = Searcher(ModNetPolicy(net), leaf, branches, window=args.window, margin=args.margin)
    cfg = MatchConfig(every=args.every, window=args.window, cap_s=args.cap_s)
    done = {(r["tag"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    meta = {"branches": [b.name for b in branches], "every": args.every, "window": args.window,
            "margin": args.margin, "net": args.net, "value": args.value, "cap_s": args.cap_s}
    print(json.dumps(meta), flush=True)
    rec_parts = []
    for tag, srch in (("plain", None), (args.tag, None if args.no_search else searcher)):
        if tag == "plain" and args.no_plain:
            continue
        todo = [t for t in tasks if (tag, args.opponent, t[0], t[1]) not in done]
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            our = ModNetPolicy(net_plain if tag == "plain" else net)  # "plain" rows are always plain r5
            opp = make_opponent_policy(args.opponent, our, args.device, same_net=tag == "plain" or is_r5)
            t0 = time.time()
            cfg.record = bool(args.record) and srch is not None
            res = play_matches([s for s, _ in chunk], [sd for _, sd in chunk], our, opp, srch, cfg,
                               log=print if args.verbose else None)
            wall = time.time() - t0
            with (out / "matches.jsonl").open("a") as fh:
                for j, (s, sd) in enumerate(chunk):
                    row = {"tag": tag, "net": args.net if tag != "plain" else R5, "opponent": args.opponent, "seed": s, "side": sd,
                           "margin": int(res["margin"][j]), "ours": int(res["ours"][j]), "opp": int(res["opp"][j]),
                           "over": bool(res["over"][j]), "ticks": int(res["ticks"][j]),
                           "first_ours": int(res["first_ours"][j]), "first_opp": int(res["first_opp"][j]),
                           "searches": int(res["searches"][j]), "chosen": res["chosen"][j].tolist(),
                           "gain_sum": float(res["gain_sum"][j]), "chunk_wall_s": wall, "chunk_n": len(chunk),
                           "search_s": res["search_s"], "meta": meta if srch is not None else None}
                    fh.write(json.dumps(row) + "\n")
            if cfg.record:
                rows = res["rows"]
                rows["seed"] = np.asarray([chunk[g][0] for g in rows["game"]], np.int64)
                rows["side"] = np.asarray([chunk[g][1] for g in rows["game"]], np.int8)
                part = out / "records" / f"{args.tag}_{args.opponent}_{chunk[0][0]}_{chunk[0][1]}_{c}.npz"
                part.parent.mkdir(exist_ok=True)
                np.savez(part, **rows)
                rec_parts.append(str(part))
            m = res["margin"]
            print(f"[{tag} vs {args.opponent}] {c + len(chunk)}/{len(todo)} margin {m.mean():+.1f} "
                  f"wall {wall:.0f}s ({wall / len(chunk):.1f}s/match, search {res['search_s']:.0f}s)", flush=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race_counts(rows):
    from blackout_env.sim.search import race

    r = race(np.array([x["first_ours"] for x in rows]), np.array([x["first_opp"] for x in rows]))
    return int((r > 0).sum()), int((r == 0).sum()), int((r < 0).sum())


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["opponent"])][(r["seed"], r["side"])] = r
    lines = []
    for (tag, opp), d in sorted(by.items()):
        if tag == "plain":
            continue
        base = by.get(("plain", opp), {})
        keys = sorted(k for k in d if k in base)
        if not keys:
            continue
        diff = defaultdict(list)
        for k in keys:
            diff[k[0]].append(d[k]["margin"] - base[k]["margin"])
        per_seed = [np.mean(v) for v in diff.values()]
        ms = [d[k]["margin"] for k in keys]
        mb = [base[k]["margin"] for k in keys]
        rs, rb = race_counts([d[k] for k in keys]), race_counts([base[k] for k in keys])
        meta = next((d[k]["meta"] for k in keys if d[k].get("meta")), None) or {"branches": ["base"], "every": 0, "window": 0, "margin": 0}
        chosen = np.sum([d[k]["chosen"] for k in keys], 0)
        share = {n: round(float(c / max(1, chosen.sum())), 3) for n, c in zip(meta["branches"], chosen)}
        wall = np.mean([d[k]["chunk_wall_s"] / d[k]["chunk_n"] for k in keys])
        wall_b = np.mean([base[k]["chunk_wall_s"] / base[k]["chunk_n"] for k in keys])
        win_s = np.mean([np.sign(x) for x in ms])
        win_b = np.mean([np.sign(x) for x in mb])
        lines.append(f"{tag:14s} vs {opp:16s} n={len(keys):4d} seeds={len(per_seed):4d}  search {np.mean(ms):+6.1f}  plain {np.mean(mb):+6.1f}  "
                     f"diff {np.mean(per_seed):+6.1f} ± {se(per_seed):4.1f}  win(sign) {win_s:+.2f} vs {win_b:+.2f}  "
                     f"race w/t/l {rs} vs {rb}  capped {np.mean([not d[k]['over'] for k in keys]):.2f}/{np.mean([not base[k]['over'] for k in keys]):.2f}  "
                     f"len {np.mean([d[k]['ticks'] for k in keys]) / 25:.0f}s  wall {wall:.1f}s/match (plain {wall_b:.2f})  "
                     f"e{meta['every']} w{meta['window']} m{meta['margin']} B{len(meta['branches'])}")
        lines.append(f"{'':14s}    base share {share.get('base', 0):.2f}; " + ", ".join(f"{k} {v:.2f}" for k, v in share.items() if k != "base"))
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--opponent", default="r5", help="r5 | p11 | rnet:<ckpt> | an opponents.py descriptor")
    q.add_argument("--net", default=R5)
    q.add_argument("--value", default=VALUE)
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--every", type=int, default=25)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--margin", type=float, default=0.0)
    q.add_argument("--seeds", type=int, default=64)
    q.add_argument("--seed0", type=int, default=2_000_000)
    q.add_argument("--chunk", type=int, default=128)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--tag", required=True)
    q.add_argument("--no-plain", action="store_true")
    q.add_argument("--no-search", action="store_true", help="play --net plain under --tag (e.g. r6 vs the plain r5 rows)")
    q.add_argument("--record", action="store_true")
    q.add_argument("--device", default="mps")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/s3_search"))
    q = sub.add_parser("report")
    q.add_argument("--out", default=str(REPO / "reports/s3_search"))
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
