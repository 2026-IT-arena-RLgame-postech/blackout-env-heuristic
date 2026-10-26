"""Where does a policy win?  Per-match economy / survival stats of both teams (simulator only).

``play``: like ``examples/v27_eval.py play`` (same policy descriptors, paired (seed, side)), but every match also
logs, for each team, from the sim's event stream:

* battery economy: neutral pickups (count, amount), steals from enemy storage (amount), deposits (amount),
  absorbed (= locked score), cargo lost by dying (amount) / by transforming (amount), specials picked up;
* deaths of each class (Collector / Hunter / Carrier) and who killed them (Collector / Hunter / Carrier / none),
  deaths while holding a battery;
* kills made by our Hunters / Collectors / Carriers;
* unit-seconds per class, score at 60 / 120 / 180 / 240 / 300 s, the decision of the first battery deposit.

``report``: means per (ours, opponent), side by side, with paired diffs ``ours - base`` on the same opponent.

    PYTHONPATH=. .venv/bin/python examples/econ_stats.py play --ours 'V32.1;net:checkpoints/rl/abl_beta01/final.pt' \\
        --opponent V32.1 --seeds 64 --workers 12 --out reports/heur_econ/e4
    PYTHONPATH=. .venv/bin/python examples/econ_stats.py report --base V32.1 --out reports/heur_econ/e4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

CLS = ("col", "hun", "car")
SNAP_S = (60, 120, 180, 240, 300)


def _team(u: int) -> int:
    return 0 if u < 5 else 1


def play_chunk(job):
    import torch

    torch.set_num_threads(1)
    from v27_eval import _policy

    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState

    ours_name, opp_name, seeds, sides, cap_s = job
    F32 = np.float32
    t0 = time.time()
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    ours, opp = _policy(ours_name), _policy(opp_name)
    ours.reset(n)
    opp.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    need = getattr(opp, "needs_inputs", True) or getattr(ours, "needs_inputs", True)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    st = [[defaultdict(float), defaultdict(float)] for _ in range(n)]
    for t in range(cap_s * 25):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if need:
            bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if not sel.any():
                    continue
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), s_in)[team]
        bs.step(moves, live)
        for b in live:
            s = st[b]
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                kind = int(e[1])
                if kind == core.EV_PICKUP or kind == core.EV_STEAL:
                    d = s[_team(int(e[2]))]
                    if int(e[3]) == core.BATTERY:
                        key = "pick" if kind == core.EV_PICKUP else "steal"
                        d[key + "_n"] += 1
                        d[key + "_amt"] += int(e[4])
                    else:
                        d["special_n"] += 1
                elif kind == core.EV_DEPOSIT:
                    d = s[_team(int(e[2]))]
                    if int(e[3]) == core.BATTERY:
                        d["dep_n"] += 1
                        d["dep_amt"] += int(e[4])
                        if "first_dep_t" not in d:
                            d["first_dep_t"] = t + 1
                elif kind == core.EV_ABSORB:
                    if int(e[3]) == core.BATTERY and int(e[2]) in (0, 1):
                        s[int(e[2])]["abs_amt"] += int(e[4])
                elif kind == core.EV_LOST:
                    if int(e[3]) == core.BATTERY:
                        s[_team(int(e[2]))]["lost_die_amt" if int(e[5]) == 1 else "lost_tx_amt"] += int(e[4])
                elif kind == core.EV_KILL:
                    killer, victim, vcls, kcls = int(e[2]), int(e[3]), int(e[4]), int(e[5])
                    d = s[_team(victim)]
                    kname = CLS[kcls] if killer >= 0 else "none"
                    d[f"die_{CLS[vcls]}"] += 1
                    d[f"die_{CLS[vcls]}_by_{kname}"] += 1
                    if killer >= 0:
                        s[_team(killer)][f"kill_by_{CLS[kcls]}"] += 1
        cls = bs.ui[live, :, core.U_CLS]
        held = bs.ui[live, :, core.U_HELD] >= 0
        for j, b in enumerate(live):
            for team in (0, 1):
                c = cls[j, 5 * team:5 * team + 5]
                d = st[b][team]
                for k, nm in enumerate(CLS):
                    d[f"{nm}_s"] += float((c == k).sum()) / 25.0
                d["held_s"] += float(held[j, 5 * team:5 * team + 5].sum()) / 25.0
        if (t + 1) % 25 == 0 and (t + 1) // 25 in SNAP_S:
            sc = bs.scores
            for b in live:
                for team in (0, 1):
                    st[b][team][f"score{(t + 1) // 25}"] = float(sc[b, team])
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        o, p = int(sc[j, sd]), int(sc[j, 1 - sd])
        for team in (0, 1):  # matches that ended early keep their final score in later snapshots
            for k in SNAP_S:
                st[j][team].setdefault(f"score{k}", float(sc[j, team]))
        rows.append({"ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd), "margin": o - p,
                     "score_ours": o, "score_opp": p, "ticks": int(bs.ticks[j]),
                     "us": dict(st[j][sd]), "them": dict(st[j][1 - sd]), "chunk_wall_s": time.time() - t0})
    return rows


def load_rows(out: Path) -> list[dict]:
    path = out / "matches.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_play(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["ours"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    jobs = []
    for opp in args.opponent.split(";"):
        for ours in args.ours.split(";"):
            todo = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)
                    if (ours, opp, s, sd) not in done]
            for c in range(0, len(todo), args.chunk):
                ch = todo[c:c + args.chunk]
                jobs.append((ours, opp, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s))
    print(f"{len(jobs)} chunks", flush=True)
    os.environ["BLACKOUT_NET_DEVICE"] = args.device
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(play_chunk, jobs):
            with (out / "matches.jsonl").open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            m = np.mean([r["margin"] for r in rows])
            print(f"[{rows[0]['ours']} vs {rows[0]['opponent']}] {len(rows)} margin {m:+.1f} "
                  f"wall {rows[0]['chunk_wall_s']:.0f}s", flush=True)


KEYS = ["margin", "win", "score_ours", "score_opp",
        "us.pick_amt", "us.steal_amt", "us.dep_amt", "us.abs_amt", "us.lost_die_amt", "us.lost_tx_amt",
        "them.pick_amt", "them.steal_amt", "them.dep_amt", "them.abs_amt", "them.lost_die_amt",
        "us.die_col", "us.die_col_by_hun", "us.die_col_by_car", "us.die_car", "us.die_car_by_hun", "us.die_car_by_col",
        "us.die_hun", "us.die_hun_by_hun",
        "them.die_col", "them.die_col_by_hun", "them.die_car", "them.die_hun",
        "us.kill_by_hun", "us.kill_by_col", "them.kill_by_hun",
        "us.col_s", "us.hun_s", "us.car_s", "us.held_s", "them.hun_s", "them.car_s", "us.first_dep_t",
        "us.special_n", "them.special_n"] + \
       [f"{w}.score{k}" for k in SNAP_S for w in ("us", "them")]


def value(r, key):
    if key == "win":
        return (np.sign(r["margin"]) + 1) / 2
    if "." in key:
        w, k = key.split(".", 1)
        return float(r[w].get(k, np.nan if k == "first_dep_t" else 0.0))
    return float(r[key])


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    opps = sorted({k[1] for k in by})
    lines = []
    for opp in opps:
        names = [o for o, p in by if p == opp]
        names.sort(key=lambda x: (x != args.base, x))
        base = by.get((args.base, opp), {})
        short = [nm.split("/")[-2] if nm.startswith("net:") else nm for nm in names]
        lines.append(f"\n=== vs {opp}  (n={[len(by[(nm, opp)]) for nm in names]})")
        lines.append(f"{'stat':22s}" + "".join(f"{s[:16]:>18s}" for s in short)
                     + "".join(f"{'Δ' + s[:12]:>18s}" for s in short if s != args.base))
        for key in KEYS:
            vals, diffs = [], []
            for nm in names:
                d = by[(nm, opp)]
                vals.append(np.nanmean([value(r, key) for r in d.values()]))
                if nm != args.base and base:
                    ks = sorted(k for k in d if k in base)
                    per = defaultdict(list)
                    for k in ks:
                        per[k[0]].append(value(d[k], key) - value(base[k], key))
                    x = np.array([np.nanmean(v) for v in per.values()])
                    x = x[np.isfinite(x)]
                    diffs.append(f"{x.mean():+8.2f}±{x.std(ddof=1) / np.sqrt(len(x)):5.2f}" if len(x) > 1 else "")
            lines.append(f"{key:22s}" + "".join(f"{v:18.2f}" for v in vals) + "".join(f"{s:>18s}" for s in diffs))
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--ours", required=True)
    q.add_argument("--opponent", required=True)
    q.add_argument("--seeds", type=int, default=64)
    q.add_argument("--seed0", type=int, default=5_100_000)
    q.add_argument("--chunk", type=int, default=16)
    q.add_argument("--cap-s", type=int, default=425)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--device", default="cpu")
    q.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--base", required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
