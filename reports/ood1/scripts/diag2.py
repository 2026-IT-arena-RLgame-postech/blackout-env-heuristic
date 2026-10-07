"""OOD1 (docs §56): why do our policies weaken against the V30 variants?  Full sim matches as examples/v27_eval.py
play (cap 425 s, --stop-exhausted) with per-phase bookkeeping, phases = 0-10 / 10-20 / 20-30 / 30-60 / 60-120 / 120+ s
(decisions at 25/s):
  gain / loss of each team's score in the phase (score up = deposits/absorptions; down = items stolen from storage),
  deaths of each team by victim class (Collector/Hunter/Carrier), steals (EV_STEAL) and carried items lost on death,
  Hunter-seconds, plus first-Hunter / second race (as v27_eval) and Hunters at decision 99.
Rows -> <out>/diag2.jsonl.  ``summary`` prints, per (opponent, ours): all games and the lost games, the phase where
the deficit accumulates and the event deltas vs the same policy against plain V30 (``--ref``).
    python diag2.py play --ours 'p21e;p20;V30' --opponent 'v30_h4hunt;V30(ref)' --seeds 32 --seed0 2600000 --out DIR
    python diag2.py summary --out DIR
"""
import argparse, json, multiprocessing as mp, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "examples"))
from blackout_env.train.distill.ood_suite import FOLLOWUP, OOD26, REFERENCE, V30, V31  # noqa: E402
ALIAS = {n: d for n, d, _, _ in OOD26 + REFERENCE + FOLLOWUP}
ALIAS.update({"V30": V30, "V31": V31})
EDGES = (0, 250, 500, 750, 1500, 3000, 10 ** 9)
PH = ("0-10", "10-20", "20-30", "30-60", "60-120", "120+")
CL = ("C", "H", "Ca")


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
    P = len(PH)
    gain = np.zeros((n, P, 2)); loss = np.zeros((n, P, 2)); deaths = np.zeros((n, P, 2, 3), int)
    steals = np.zeros((n, P, 2), int); lost = np.zeros((n, P, 2)); hsec = np.zeros((n, P, 2))
    h99 = np.zeros((n, 2), int); first = np.full((n, 2), -1); second = np.full((n, 2), -1)
    prev = bs.scores.astype(float).copy()
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
        p = int(np.searchsorted(EDGES, d, side="right") - 1)
        sc = bs.scores.astype(float)
        for b in live:
            sd = int(sides[b]); rel = lambda team: 0 if team == sd else 1  # noqa: E731
            for tm in (0, 1):
                dd = sc[b, tm] - prev[b, tm]
                if dd > 0: gain[b, p, rel(tm)] += dd
                elif dd < 0: loss[b, p, rel(tm)] -= dd
            for e in bs.ev[b, :int(bs.g[b, core.G_NEV])]:
                k = int(e[1])
                if k == core.EV_KILL and e[3] >= 0:
                    deaths[b, p, rel(0 if e[3] < 5 else 1), int(e[4])] += 1
                elif k == core.EV_STEAL:
                    steals[b, p, rel(0 if e[2] < 5 else 1)] += 1
                elif k == core.EV_LOST:
                    lost[b, p, rel(0 if e[2] < 5 else 1)] += float(e[4])
                elif k == core.EV_TRANSFORM and e[3] == HUNTER:
                    tm = rel(0 if e[2] < 5 else 1)
                    if first[b, tm] >= 0 and second[b, tm] < 0 and d > first[b, tm] + 8:
                        second[b, tm] = d
            for tm in (0, 1):
                h = int((bs.ui[b, 5 * tm:5 * tm + 5, core.U_CLS] == HUNTER).sum())
                hsec[b, p, rel(tm)] += h / 25.0
                if h and first[b, rel(tm)] < 0: first[b, rel(tm)] = d
                if d == 99: h99[b, rel(tm)] = h
        prev = sc
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        rows.append({"ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd),
                     "margin": int(sc[j, sd] - sc[j, 1 - sd]), "ticks": int(bs.ticks[j]), "h99": h99[j].tolist(),
                     "first": first[j].tolist(), "second": second[j].tolist(), "gain": gain[j].tolist(), "loss": loss[j].tolist(),
                     "deaths": deaths[j].tolist(), "steals": steals[j].tolist(), "lost": lost[j].tolist(), "hsec": hsec[j].tolist()})
    return rows


def cmd_play(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "diag2.jsonl"
    done = {(r["ours"], r["opponent"], r["seed"], r["side"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    jobs = []
    for opp in a.opponent.split(";"):
        for ours in a.ours.split(";"):
            todo = [(s, sd) for s in range(a.seed0, a.seed0 + a.seeds) for sd in (0, 1) if (ours, opp, s, sd) not in done]
            for c in range(0, len(todo), a.chunk):
                ch = todo[c:c + a.chunk]
                jobs.append((ours, opp, [s for s, _ in ch], [sd for _, sd in ch], a.cap_s))
    print(f"{len(jobs)} chunks", flush=True)
    os.environ["BLACKOUT_NET_DEVICE"] = "cpu"
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(play_chunk, jobs):
            with path.open("a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"[{rows[0]['ours']} vs {rows[0]['opponent']}] {len(rows)}", flush=True)


def race(r, i):
    x = [v if v >= 0 else 10 ** 9 for v in r[i]]
    return "W" if x[0] < x[1] else ("T" if x[0] == x[1] else "L")


def agg(R):
    """Per-phase mean score-diff increment (ours - theirs), events per match."""
    A = lambda k: np.array([r[k] for r in R], float)  # noqa: E731
    g, l = A("gain"), A("loss")
    inc = (g[..., 0] - l[..., 0]) - (g[..., 1] - l[..., 1])  # [n, P]
    de = A("deaths")  # [n, P, 2, 3]
    return {"n": len(R), "margin": float(np.mean([r["margin"] for r in R])), "sec": float(np.mean([r["ticks"] for r in R])) / 50, "inc": inc.mean(0),
            "cum": np.cumsum(inc.mean(0)), "gain": g.sum(1).mean(0), "loss": l.sum(1).mean(0),
            "deaths": de.sum(1).mean(0), "deaths_early": de[:, :4].sum(1).mean(0),
            "steals": A("steals").sum(1).mean(0), "lost": A("lost").sum(1).mean(0), "hsec": A("hsec").sum(1).mean(0),
            "race2": "/".join(str(sum(race(r, "second") == c for r in R)) for c in "WTL"),
            "race1": "/".join(str(sum(race(r, "first") == c for r in R)) for c in "WTL"),
            "h99": np.mean([r["h99"][0] for r in R]), "h99o": np.mean([r["h99"][1] for r in R])}


def fmt(s, tag):
    de, d0 = s["deaths"], s["deaths_early"]
    return (f"  {tag:10s} n {s['n']:3d} margin {s['margin']:+6.1f} len {s['sec']:3.0f}s | cum diff @10/20/30/60/120/end "
            + "/".join(f"{x:+.0f}" for x in s["cum"])
            + f" | gain {s['gain'][0]:.0f}/{s['gain'][1]:.0f} stolen-from {s['loss'][0]:.0f}/{s['loss'][1]:.0f}"
            + f" | deaths ours C{de[0, 0]:.1f} H{de[0, 1]:.1f} Ca{de[0, 2]:.1f} (<60s C{d0[0, 0]:.1f} Ca{d0[0, 2]:.1f})"
            + f" theirs C{de[1, 0]:.1f} H{de[1, 1]:.1f} Ca{de[1, 2]:.1f} | steals {s['steals'][0]:.1f}/{s['steals'][1]:.1f}"
            + f" lost {s['lost'][0]:.0f}/{s['lost'][1]:.0f} | H-s {s['hsec'][0]:.0f}/{s['hsec'][1]:.0f}"
            + f" | race1 {s['race1']} race2 {s['race2']} H@99 {s['h99']:.1f}/{s['h99o']:.1f}")


def cmd_summary(a):
    rows = [json.loads(x) for x in (Path(a.out) / "diag2.jsonl").read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows: by[(r["opponent"], r["ours"])].append(r)
    opps = list(dict.fromkeys(r["opponent"] for r in rows))
    for o in opps:
        print(f"== vs {o}")
        for (oo, ours), R in by.items():
            if oo != o: continue
            print(f" {ours}")
            print(fmt(agg(R), "all"))
            L = [r for r in R if r["margin"] < 0]
            if L: print(fmt(agg(L), "lost"))
            C = [r for r in R if r["margin"] < -20]
            if C: print(fmt(agg(C), "cat<-20"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("play")
    for k in ("--ours", "--opponent", "--out"): p.add_argument(k, required=True)
    for k, v in (("--seeds", 32), ("--seed0", 2600000), ("--workers", 4), ("--chunk", 16), ("--cap-s", 425)): p.add_argument(k, type=int, default=v)
    s = sp.add_parser("summary"); s.add_argument("--out", required=True)
    a = ap.parse_args(); (cmd_play if a.cmd == "play" else cmd_summary)(a)
