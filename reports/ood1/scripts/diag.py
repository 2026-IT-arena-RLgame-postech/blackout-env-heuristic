"""OOD1 (docs §56): copy of reports/p20_diag/scripts/diag.py with the OOD-26 short names as aliases (ood_suite.py).
P20 diagnosis (docs §47; copy of reports/p18b_diag/scripts/diag.py + V30 alias, more score marks).
P18b diagnosis (docs §43): why P18 (V29 distilled from P17) loses to P17 vs V28 and plays longer games.
Full sim matches as examples/v27_eval.py play (cap 425 s, --stop-exhausted), plus per match: Hunters at decision 99
(ours, theirs), our/their Hunter count at decisions 150/250/500/1000/2000, our Hunter transforms after the opening,
kills by class, score at 20/40/80 s.  Rows -> <out>/diag.jsonl;  `summary` prints per (ours, opponent).
    python diag.py play --ours 'p18;p17;V29' --opponent 'V28;V23' --seeds 32 --seed0 1100000 --workers 9 --out DIR
    python diag.py summary --out DIR
"""
import argparse, json, multiprocessing as mp, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "examples"))
ALIAS = {"V29": "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve",
         "V28": "strategic_v28:reserve_stage=0.5,trade_hold=advance", "V27": "strategic_v27:reserve_stage=1.5",
         "V30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
         "V25": "strategic_v25", "V23": "strategic_v23"}
from blackout_env.train.distill.ood_suite import FOLLOWUP, OOD26, REFERENCE, V31  # noqa: E402
ALIAS.update({n: d for n, d, _, _ in OOD26 + REFERENCE + FOLLOWUP})
ALIAS["V31"] = V31
MARKS = (150, 250, 500, 1000, 2000)
SCORE_AT = (250, 375, 500, 750, 1000, 1500, 2000, 3000)


def play_chunk(job):
    import torch
    torch.set_num_threads(1)
    from v27_eval import _policy
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, batteries_in_play
    from blackout_env.sim.mapdata import HUNTER, H, W
    from blackout_env.sim.state import SimState
    ours_name, opp_name, seeds, sides, cap_s = job
    n = len(seeds); sides = np.asarray(sides)
    ours, opp = _policy(ALIAS.get(ours_name, ours_name)), _policy(ALIAS.get(opp_name, opp_name))
    ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), np.float32), np.zeros((n, 2, 4), np.float32), np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    mv = np.zeros((n, 10, 2), np.float32)
    stopped = np.zeros(n, bool)
    h99 = np.zeros((n, 2), int); hmark = np.full((n, len(MARKS), 2), -1); smark = np.full((n, len(SCORE_AT), 2), -1)
    trans = np.zeros((n, 2), int); kills = [[Counter(), Counter()] for _ in range(n)]
    for t in range(cap_s * 25):
        live = np.nonzero(~bs.over & ~stopped)[0]
        if not len(live): break
        bs.render(live, *bufs)
        sub = tuple(x[:len(live)] for x in bufs)
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if sel.any():
                    s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                    mv[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), s_in)[team]
        ab = bs.gf[live, core.GF_ABS].copy()
        bs.step(mv, live)
        fired = live[(bs.gf[live, core.GF_ABS] < ab) & ~bs.over[live]]
        if len(fired): stopped[fired[batteries_in_play(bs, fired) == 0]] = True
        d = t + 1
        for b in live:
            sd = int(sides[b])
            for e in bs.ev[b, :int(bs.g[b, core.G_NEV])]:
                if e[1] == core.EV_TRANSFORM and e[3] == HUNTER and d > 101:
                    trans[b, 0 if (e[2] < 5) == (sd == 0) else 1] += 1
                elif e[1] == core.EV_KILL and e[2] >= 0:
                    k_ours = (e[2] < 5) == (sd == 0)
                    kills[b][0 if k_ours else 1][int(e[4])] += 1
        hc = np.stack([(bs.ui[live, 5 * tm:5 * tm + 5, core.U_CLS] == HUNTER).sum(1) for tm in (0, 1)], 1)
        o = sides[live]; hc_o = np.where(o[:, None] == 0, hc, hc[:, ::-1])
        if d == 99: h99[live] = hc_o
        if d in MARKS: hmark[live, MARKS.index(d)] = hc_o
        if d in SCORE_AT:
            sc = bs.scores[live]; smark[live, SCORE_AT.index(d)] = np.where(o[:, None] == 0, sc, sc[:, ::-1])
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        rows.append({"ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd),
                     "margin": int(sc[j, sd] - sc[j, 1 - sd]), "ticks": int(bs.ticks[j]), "h99": h99[j].tolist(),
                     "hmark": hmark[j].tolist(), "smark": smark[j].tolist(), "trans": trans[j].tolist(),
                     "kills_by_us": {str(k): v for k, v in kills[j][0].items()}, "kills_of_us": {str(k): v for k, v in kills[j][1].items()}})
    return rows


def cmd_play(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for opp in a.opponent.split(";"):
        for ours in a.ours.split(";"):
            todo = [(s, sd) for s in range(a.seed0, a.seed0 + a.seeds) for sd in (0, 1)]
            for c in range(0, len(todo), a.chunk):
                ch = todo[c:c + a.chunk]
                jobs.append((ours, opp, [s for s, _ in ch], [sd for _, sd in ch], a.cap_s))
    os.environ["BLACKOUT_NET_DEVICE"] = a.device
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(play_chunk, jobs):
            with (out / "diag.jsonl").open("a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"[{rows[0]['ours']} vs {rows[0]['opponent']}] {len(rows)}", flush=True)


def cmd_summary(a):
    rows = [json.loads(x) for x in (Path(a.out) / "diag.jsonl").read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows: by[(r["opponent"], r["ours"])].append(r)
    CL = {"0": "C", "1": "H", "2": "Ca"}
    for (opp, ours), R in sorted(by.items()):
        h = Counter(tuple(r["h99"]) for r in R)
        hm = np.array([r["hmark"] for r in R], float); hm[hm < 0] = np.nan
        sm = np.array([r["smark"] for r in R], float); sm[sm < 0] = np.nan
        kb = Counter(); ko = Counter()
        for r in R: kb.update(r["kills_by_us"]); ko.update(r["kills_of_us"])
        n = len(R)
        print(f"{ours:4s} vs {opp:4s} n={n} margin {np.mean([r['margin'] for r in R]):+6.1f} ticks {np.mean([r['ticks'] for r in R]):5.0f}"
              f" | H@99 {dict(sorted(h.items(), key=lambda kv: -kv[1])[:4])}")
        print(f"     H ours/theirs @" + " ".join(f"{m}:{np.nanmean(hm[:, i, 0]):.1f}/{np.nanmean(hm[:, i, 1]):.1f}" for i, m in enumerate(MARKS))
              + " | score-diff @" + " ".join(f"{m}:{np.nanmean(sm[:, i, 0] - sm[:, i, 1]):+.1f}" for i, m in enumerate(SCORE_AT))
              + f" | new H ours/theirs {np.mean([r['trans'][0] for r in R]):.2f}/{np.mean([r['trans'][1] for r in R]):.2f}"
              + " | kills by us " + ",".join(f"{CL[k]}{v / n:.2f}" for k, v in sorted(kb.items()))
              + " of us " + ",".join(f"{CL[k]}{v / n:.2f}" for k, v in sorted(ko.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("play")
    for k, v in (("--ours", str), ("--opponent", str), ("--out", str), ("--device", str)): p.add_argument(k, type=v, default="cpu" if k == "--device" else None)
    for k, v in (("--seeds", 32), ("--seed0", 1100000), ("--workers", 9), ("--chunk", 16), ("--cap-s", 425)): p.add_argument(k, type=int, default=v)
    s = sp.add_parser("summary"); s.add_argument("--out", required=True)
    a = ap.parse_args(); (cmd_play if a.cmd == "play" else cmd_summary)(a)
