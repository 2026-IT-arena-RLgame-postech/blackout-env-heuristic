"""Q5: is searching during the opening race still worth it when the acting policy already races
optimally (P14b, the V25 student, arrives at decision 99)?  (docs/policy_v2_design_ko.md §32.)
Simulator only.

``play``: seeds x both sides in lockstep chunks; every tag plays the same match starts
(``SimState.from_seed``) against ``--opponent``.  Tags (``TAGS``):

* ``p14b``          plain P14b (no search);
* ``p14b_r10``      P14b acting + the S3 search (rollout net f3_x_d2, V s2v_f3xd2 iter2, B8
                    ``alt*,samp:1.0x2``, every 25, window 25, margin 3) only while t < 10 s; the base
                    branch = P14b, a unit branch = P14b for the other units (overlay);
* ``p14b_r10_team`` the same, a non-base branch = the whole team f3_x_d2 (no overlay);
* ``p14b_s5_10``    search only 5 s <= t < 10 s (after the race);
* ``p14b_full`` / ``p14b_from5`` / ``p14b_from10``  search the whole match / from 5 s / from 10 s;
* ``p14b_r10_rand`` control: like ``p14b_r10`` but a uniformly random non-base branch with
                    probability ``--rand-rate`` per search point (no rollouts) -- symmetry breaking alone;
* ``v25`` / ``v25_r10``  the V25 heuristic acting (plain / race10);
* ``f3`` / ``f3_r10``  plain f3_x_d2 / search-new race10 (§24), f3_x_d2 acting (reference).

Rows append to ``<out>/matches_<opponent>.jsonl`` (a rerun skips rows already there): margin, our lead at
``TRACE_S`` seconds, first Hunter decisions of both teams, search choices.

``report``: per opponent and tag: race W/T/L, our first-Hunter decision distribution (99 rate),
margin and the paired difference vs ``--ref`` (same seed, side; SE over seeds, both sides averaged).

    PYTHONPATH=. .venv/bin/python examples/race_search_p14b_eval.py play --opponent p14b --tags p14b,p14b_r10
    PYTHONPATH=. .venv/bin/python examples/race_search_p14b_eval.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

F3 = "checkpoints/rollout_net/arch/f3_x_d2.pt"
VALUE = "checkpoints/value/s2v_f3xd2/iter2.pt"
P14B = "checkpoints/distill/frozen/p14_v25.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"
TRACE_S = (4.2, 5, 10, 15, 30, 60)

# tag -> (acting base: "p14b" | "v25" | None = the rollout net, search window seconds or None, overlay)
TAGS = {
    "p14b": ("p14b", None, True),
    "p14b_r10": ("p14b", (0, 10), True),
    "p14b_r10_team": ("p14b", (0, 10), False),
    "p14b_s5_10": ("p14b", (5, 10), True),
    "p14b_full": ("p14b", (0, 150), True),
    "p14b_from5": ("p14b", (5, 150), True),
    "p14b_from10": ("p14b", (10, 150), True),
    "p14b_r10_rand": ("p14b", (0, 10), True),
    "v25": ("v25", None, True),
    "v25_r10": ("v25", (0, 10), True),
    "f3": (None, None, True),
    "f3_r10": (None, (0, 10), True),
}


class RandomBranchSearcher:
    """Control for a search: per root a uniformly random non-base branch with probability ``rate``."""

    def __init__(self, branches, rate: float, seed: int = 0):
        self.branches, self.rate = branches, float(rate)
        self.rng = np.random.default_rng(seed)

    def search(self, roots, teams, reseeds=None):
        from blackout_env.sim.search import SearchOut

        r = len(roots)
        go = self.rng.random(r) < self.rate
        choice = np.where(go, 1 + self.rng.integers(0, len(self.branches) - 1, r), 0)
        return SearchOut(choice, np.zeros((r, len(self.branches))), {"games": r})


def make_policy(name: str, device: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name == "p14b":
        return NetPolicy.from_checkpoint(str(REPO / P14B), device=device)
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device=device)
    if name in ("v25", "v23"):
        return HeuristicPolicy(f"strategic_{name}")
    return HeuristicPolicy(name)


def load_rows(out: Path) -> list[dict]:
    """Every ``matches_<opponent>.jsonl`` (one file per opponent: parallel processes never share a file)."""
    return [json.loads(x) for path in sorted(out.glob("matches_*.jsonl")) for x in path.read_text().splitlines() if x.strip()]


def cmd_play(args):
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import MatchConfig, ModNetPolicy, Searcher, ValueNetLeaf, parse_branches, play_matches

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = NetPolicy(RolloutNet.from_checkpoint(str(REPO / args.net)), device=args.device)
    leaf = ValueNetLeaf(str(REPO / args.value), args.device)
    branches = parse_branches(args.branches)
    bases = {}
    done = {(r["tag"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    opp = make_policy(args.opponent, args.device)
    for tag in args.tags.split(","):
        base_name, window, overlay = TAGS[tag]
        if base_name and base_name not in bases:
            bases[base_name] = make_policy(base_name, args.device)
        base = bases.get(base_name)
        searcher = None
        if tag.endswith("_rand"):
            searcher = RandomBranchSearcher(branches, args.rand_rate, seed=args.seed0)
        elif window is not None:
            searcher = Searcher(ModNetPolicy(net), leaf, branches, window=args.window, margin=args.margin,
                                leaf_decisions=args.leaf, base_policy=base, overlay=overlay)
        cfg = MatchConfig(every=args.every, window=args.window, cap_s=args.cap_s, trace_s=TRACE_S,
                          search_from=int(window[0] * 25) if window else 0,
                          search_until=int(window[1] * 25) if window else None)
        meta = {"base": base_name, "window_s": window, "overlay": overlay, "branches": [b.name for b in branches],
                "every": args.every, "window": args.window, "margin": args.margin, "net": args.net,
                "value": args.value, "leaf": args.leaf, "cap_s": args.cap_s}
        todo = [t for t in tasks if (tag, args.opponent, t[0], t[1]) not in done]
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            t0 = time.time()
            res = play_matches([s for s, _ in chunk], [sd for _, sd in chunk], ModNetPolicy(net), opp, searcher, cfg,
                               log=print if args.verbose else None, base=base, overlay=overlay)
            wall = time.time() - t0
            with (out / f"matches_{args.opponent}.jsonl").open("a") as fh:
                for j, (s, sd) in enumerate(chunk):
                    fh.write(json.dumps({
                        "tag": tag, "opponent": args.opponent, "seed": s, "side": sd,
                        "margin": int(res["margin"][j]), "ours": int(res["ours"][j]), "opp": int(res["opp"][j]),
                        "over": bool(res["over"][j]), "ticks": int(res["ticks"][j]),
                        "lead_at": {str(k): int(v) for k, v in zip(TRACE_S, res["lead_at"][j])},
                        "hunters_at": {str(k): res["hunters_at"][j, i].tolist() for i, k in enumerate(TRACE_S)},
                        "first_ours": int(res["first_ours"][j]), "first_opp": int(res["first_opp"][j]),
                        "searches": int(res["searches"][j]), "chosen": res["chosen"][j].tolist(),
                        "gain_sum": float(res["gain_sum"][j]), "chunk_wall_s": wall, "chunk_n": len(chunk),
                        "search_s": res["search_s"], "meta": meta}) + "\n")
            m = res["margin"]
            print(f"[{tag} vs {args.opponent}] {c + len(chunk)}/{len(todo)} margin {m.mean():+.1f} "
                  f"first99 {np.mean(res['first_ours'] == 99):.2f} wall {wall:.0f}s (search {res['search_s']:.0f}s)",
                  flush=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race_wtl(rows):
    from blackout_env.sim.search import race

    r = race(np.array([x["first_ours"] for x in rows]), np.array([x["first_opp"] for x in rows]))
    return int((r > 0).sum()), int((r == 0).sum()), int((r < 0).sum())


def first_dist(rows) -> str:
    c = Counter()
    for x in rows:
        f = x["first_ours"]
        c["none" if f < 0 else str(f) if f <= 100 else "101+"] += 1
    keys = sorted((k for k in c if k not in ("none", "101+")), key=int) + [k for k in ("101+", "none") if k in c]
    return " ".join(f"{k}:{c[k]}" for k in keys)


def paired(a: dict, b: dict, key) -> tuple[float, float, int]:
    keys = sorted(k for k in a if k in b)
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(key(a[k]) - key(b[k]))
    ps = [np.mean(v) for v in by.values()]
    return (float(np.mean(ps)) if ps else float("nan")), se(ps), len(keys)


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["opponent"])][(r["seed"], r["side"])] = r
    opps = [o for o in ("p14b", "v25", "v23", "p11") if any(op == o for _, op in by)]
    opps += sorted({op for _, op in by} - set(opps))
    tags = [t for t in TAGS if any(tg == t for tg, _ in by)]
    lines = [f"paired differences vs {args.ref!r} (same opponent, seed, side); +-SE over seeds (both sides averaged)", ""]
    hdr = (f"{'opponent':8s} {'tag':14s} {'n':>4s}  {'race W/T/L':>12s}  {'ours@99':>7s}  {'opp@99':>6s}  {'margin':>7s}  "
           f"{'d margin':>13s}  {'d lead@10s':>12s}  {'d lead@30s':>12s}  {'nonbase':>7s}  {'s/match':>7s}  first Hunter (ours)")
    lines.append(hdr)
    for opp in opps:
        ref = by.get((args.ref, opp), {})
        for tag in tags:
            d = by.get((tag, opp))
            if not d:
                continue
            rs = list(d.values())
            w, t_, l_ = race_wtl(rs)
            f99 = np.mean([x["first_ours"] == 99 for x in rs])
            o99 = np.mean([x["first_opp"] == 99 for x in rs])
            ch = np.sum([x["chosen"] for x in rs], 0)
            nonbase = 1 - ch[0] / max(1, ch.sum()) if np.ndim(ch) and len(ch) > 1 else 0.0
            spm = np.mean([x["search_s"] / x["chunk_n"] for x in rs])
            if ref and tag != args.ref:
                dm, sm, _ = paired(d, ref, lambda x: x["margin"])
                d10, s10, _ = paired(d, ref, lambda x: x["lead_at"]["10"])
                d30, s30, _ = paired(d, ref, lambda x: x["lead_at"]["30"])
                dtxt = f"{dm:+6.1f}±{sm:4.1f}  {d10:+5.1f}±{s10:4.1f}  {d30:+5.1f}±{s30:4.1f}"
            else:
                dtxt = f"{'—':>13s}  {'—':>12s}  {'—':>12s}"
            lines.append(f"{opp:8s} {tag:14s} {len(rs):4d}  {f'{w}/{t_}/{l_}':>12s}  {f99:7.2f}  {o99:6.2f}  "
                         f"{np.mean([x['margin'] for x in rs]):+7.1f}  {dtxt}  {nonbase:7.2f}  {spm:7.1f}  {first_dist(rs)}")
        lines.append("")
    hrows = [(k, d) for k, d in sorted(by.items()) if any("hunters_at" in x for x in d.values())]
    if hrows:
        lines.append("Hunters (ours / opponent's, mean over matches) at 4.2 s, 5 s, 10 s; lead at 5 s, 10 s")
        for (tag, opp), d in hrows:
            rs = [x for x in d.values() if "hunters_at" in x]
            h = {t: np.mean([x["hunters_at"][t] for x in rs], 0) for t in ("4.2", "5", "10")}
            lines.append(f"  {opp:8s} {tag:14s} n={len(rs):4d}  " + "  ".join(f"{t}s {v[0]:.2f}/{v[1]:.2f}" for t, v in h.items())
                         + f"  lead {np.mean([x['lead_at']['5'] for x in rs]):+5.1f} {np.mean([x['lead_at']['10'] for x in rs]):+5.1f}")
        lines.append("")
    # race transitions vs ref
    lines.append(f"race outcome transitions vs {args.ref!r} (ref -> tag: count)")
    from blackout_env.sim.search import race

    for opp in opps:
        ref = by.get((args.ref, opp), {})
        for tag in tags:
            d = by.get((tag, opp))
            if not d or tag == args.ref or not ref:
                continue
            keys = sorted(k for k in d if k in ref)
            ra = race(np.array([ref[k]["first_ours"] for k in keys]), np.array([ref[k]["first_opp"] for k in keys]))
            rb = race(np.array([d[k]["first_ours"] for k in keys]), np.array([d[k]["first_opp"] for k in keys]))
            nm = {1: "W", 0: "T", -1: "L"}
            c = Counter(f"{nm[int(x)]}->{nm[int(y)]}" for x, y in zip(ra, rb) if x != y)
            same_first = np.mean([d[k]["first_ours"] == ref[k]["first_ours"] for k in keys])
            lines.append(f"  {opp:8s} {tag:14s} changed {sum(c.values()):3d}/{len(keys)}  {dict(c)}  same first-Hunter decision {same_first:.3f}")
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / f"report_vs_{args.ref}.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--opponent", default="p14b", help="p14b | p11 | v25 | v23 | an opponents.py descriptor")
    q.add_argument("--tags", default="p14b,p14b_r10")
    q.add_argument("--net", default=F3)
    q.add_argument("--value", default=VALUE)
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--every", type=int, default=25)
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--margin", type=float, default=3.0)
    q.add_argument("--leaf", type=int, default=125)
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=128)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--rand-rate", type=float, default=0.65, help="_rand tags: non-base rate (p14b_r10: ~0.65)")
    q.add_argument("--device", default="")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/race_search_p14b"))
    q = sub.add_parser("report")
    q.add_argument("--out", default=str(REPO / "reports/race_search_p14b"))
    q.add_argument("--ref", default="p14b")
    args = p.parse_args()
    if getattr(args, "device", None) == "":
        from blackout_env.sim.search import default_device

        args.device = default_device()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
