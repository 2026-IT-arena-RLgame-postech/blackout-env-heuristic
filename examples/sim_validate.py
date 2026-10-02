"""Validate the Python simulator (blackout_env/sim) against recorded Unity traces (S2).

For every trace (examples/sim_record_traces.py) the recorded moves of both teams are replayed
open-loop through the simulator, special-item spawns forced from the trace, and the simulator's
rendered observations are compared with Unity's:

* full match from the first observation -- initialised from the reset seed (Unity's own RNG,
  exact) and, separately, from the first observation alone (``SimState.from_raw``);
* mid-match starts every ``--every`` seconds from one Unity observation (``from_raw``: the hidden
  state is reconstructed, positions inverted from the float32 normalisation) for 1 / 5 / 20 s,
  and the same starts with positions made exact from the 25 preceding broadcasts
  (``from_history``; keys ``*_hist25``).

Reported: per-tick position error (world units, both sides inverted through the same
observation map), observation events (blackout_env/sim/trace.obs_events) and their agreement,
the score timeline, and the first divergence of every non-identical replay with a cause probe
(alternative agent orders, the unity dt).

    PYTHONPATH=. .venv/bin/python -m examples.sim_validate --traces 'reports/sim/traces/*.npz' \\
        --out reports/sim/validation
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

from blackout_env.sim import DEFAULT_ORDER
from blackout_env.sim.trace import Trace, obs_events, positions, replay

HORIZONS = {"1s": 25, "5s": 125, "20s": 500}
HISTORY = 25  # broadcasts before a mid-match start used to make positions exact (SimState.from_history)


def _row_equal(a, b, ma, mb) -> bool:
    return np.array_equal(a, b) and np.array_equal(ma, mb)


def first_divergence(s, m, u, um, tol=1e-3):
    """(step, kind) of the first observation that differs, and the first beyond ``tol``."""
    exact = None
    for k in range(len(s)):
        if not _row_equal(s[k], u[k], m[k], um[k]):
            exact = k
            break
    big = None
    pe = np.abs(positions(s) - positions(u)).max(axis=(1, 2))
    for k in range(len(s)):
        discrete = not (np.array_equal(s[k][2:40:4], u[k][2:40:4]) and np.array_equal(s[k][3:40:4], u[k][3:40:4])
                        and np.array_equal(s[k][40:42], u[k][40:42]) and np.array_equal(m[k] >> 3, um[k] >> 3))
        if pe[k] > tol or discrete:
            big = (k, "discrete" if discrete else "position", float(pe[k]))
            break
    return exact, big


def describe(s, m, u, um, k) -> dict:
    d = np.nonzero(s[k] != u[k])[0]
    md = np.argwhere(m[k] != um[k])
    fields = []
    for i in d.tolist():
        if i < 40:
            fields.append(f"unit{i // 4}." + ("x", "y", "held", "class")[i % 4])
        else:
            fields.append(("score_a", "score_b", "time", "absorb")[i - 40])
    return {"fields": fields, "sim": s[k][d].tolist(), "unity": u[k][d].tolist(),
            "map_cells": [(int(23 - r), int(c), int(m[k][r, c]), int(um[k][r, c])) for r, c in md[:6]]}


def event_agreement(ev_u, ev_s) -> dict:
    cu, cs = collections.Counter(ev_u), collections.Counter(ev_s)
    out = {}
    kinds = {e[1] for e in ev_u} | {e[1] for e in ev_s}
    for kind in sorted(kinds) + ["all"]:
        fu = collections.Counter({e: n for e, n in cu.items() if kind == "all" or e[1] == kind})
        fs = collections.Counter({e: n for e, n in cs.items() if kind == "all" or e[1] == kind})
        common = sum((fu & fs).values())
        out[kind] = {"unity": sum(fu.values()), "sim": sum(fs.values()), "matched": common}
    return out


def _merge(acc, new):
    for kind, v in new.items():
        a = acc.setdefault(kind, {"unity": 0, "sim": 0, "matched": 0})
        for key in a:
            a[key] += v[key]


def probe_cause(tr, t0, exact_seed, k_div) -> list[str]:
    """Which change makes the first divergence go away (or move later)."""
    hints = []
    base = k_div
    rng = np.random.default_rng(0)
    for trial in range(40):  # random adjacent swaps of the agent order
        order = DEFAULT_ORDER.copy()
        i = int(rng.integers(9))
        order[i], order[i + 1] = order[i + 1], order[i]
        s, m, _ = replay(tr, t0=t0, steps=base + 5, order=order, exact_seed=exact_seed)
        n = len(s)
        u, um = tr.raw_state[t0:t0 + n], tr.raw_map[t0:t0 + n]
        e, _ = first_divergence(s, m, u, um)
        if e is None or e > base:
            hints.append(f"agent order swap {order.tolist()} moves divergence {base}->{e}")
            break
    return hints


def closed_loop(tr) -> dict | None:
    """Both recorded policies re-run *inside* the simulator on its own rendered observations
    (heuristics are deterministic given their seed; skipped for students, whose IQN tau draws
    are not seeded): the moves they choose must equal the recorded ones step for step."""
    if any(str(tr.meta.get(k, "")).startswith("ckpt:") for k in ("a", "b")):
        return None
    from blackout_env.sim.trace import init_sim
    from blackout_env.train.distill.opponents import make_opponent

    sim = init_sim(tr, 0, exact_seed=False)
    pa = make_opponent(tr.meta["a"], tr.meta["seed"])
    pb = make_opponent(tr.meta["b"], tr.meta["seed"] + 1_000_003)
    for p in (pa, pb):
        if hasattr(p, "reset"):
            p.reset()
    # the env's observation dicts list agents in Unity's decision-step order -- the same
    # OnEnable order as AgentAct (DEFAULT_ORDER); a policy drawing random numbers per agent in
    # dict order (RandomWalker) needs it to reproduce the recording
    names_a = [f"unit_{u}" for u in DEFAULT_ORDER if u < 5]
    names_b = [f"unit_{u}" for u in DEFAULT_ORDER if u >= 5]
    same, first = 0, None
    for t in range(len(tr.raw_state)):
        obs = sim.obs()
        acts = pa.act({a: obs[a] for a in names_a})
        acts.update(pb.act({a: obs[a] for a in names_b}))
        mv = np.zeros((10, 2), np.float32)
        for name, a in acts.items():
            mv[int(name.split("_")[1])] = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
        if not np.array_equal(mv, tr.moves[t]):
            first = t
            break
        same += 1
        sim.step(mv)
        if sim.over:
            break
    return {"identical_steps": same, "first_move_mismatch": first, "over": sim.over, "scores": list(sim.scores)}


def run(paths, every_s: int, out: Path, probe: bool, closed: bool = True) -> dict:
    report = {"traces": [], "horizons": {}, "events_full": {}, "events_mid": {}}
    pos_err = {"full": []}
    mid_exact = collections.Counter()
    mid_tick_err = {}
    score_equal = final_equal = winner_equal = 0
    exact_full = 0
    for path in paths:
        tr = Trace.load(path)
        u_all, um_all = tr.raw_state, tr.raw_map
        rec = {"trace": Path(path).name, "a": tr.meta.get("a"), "b": tr.meta.get("b"), "seed": tr.meta.get("seed"),
               "steps": tr.steps}
        for mode, exact_seed in (("seed", True), ("obs", False), ("rng", True)):
            try:
                s, m, sim = replay(tr, exact_seed=exact_seed, spawns="rng" if mode == "rng" else "trace")
            except ValueError:  # the first match after a player start: seed not reproducible
                rec[mode] = None
                continue
            n = len(s)
            u, um = u_all[:n], um_all[:n]
            e, big = first_divergence(s, m, u, um)
            pe = np.abs(positions(s) - positions(u)).max(axis=2)  # [n, 10]
            sc_s = np.rint(s[:, 40:42] * 100).astype(int)
            sc_u = np.rint(u[:, 40:42] * 100).astype(int)
            rec[mode] = {
                "replayed": n, "exact_steps": int(sum(_row_equal(s[k], u[k], m[k], um[k]) for k in range(n))),
                "first_exact_divergence": e, "first_divergence_1e-3": big,
                "pos_err_max": float(pe.max()), "pos_err_p99": float(np.percentile(pe, 99)),
                "score_timeline_equal": bool(np.array_equal(sc_s, sc_u)),
                "final_scores": {"sim": sc_s[-1].tolist(), "unity": sc_u[-1].tolist()},
                "ended_sim": bool(sim.over), "winner_sim": sim.winner,
            }
            if e is not None:
                rec[mode]["divergence"] = describe(s, m, u, um, e)
                if probe and mode == "seed":
                    rec[mode]["probe"] = probe_cause(tr, 0, True, e)
            ev = event_agreement(obs_events(u, um), obs_events(s, m))
            rec[mode]["events"] = ev
            if mode == "seed":
                _merge(report["events_full"], ev)
                pos_err["full"].append(pe.ravel())
                score_equal += rec[mode]["score_timeline_equal"]
                final_equal += sc_s[-1].tolist() == sc_u[-1].tolist()
                exact_full += e is None
                w = tr.meta.get("winner")
                winner_equal += sim.over and (w if w is not None else -1) == sim.winner
        if closed:
            rec["closed_loop"] = closed_loop(tr)
        # mid-match starts: from one observation, and with positions refined from a history
        for hist in (0, HISTORY):
            suffix = "" if hist == 0 else f"_hist{hist}"
            for t0 in range(every_s * 25, tr.steps - 1, every_s * 25):
                s, m, _ = replay(tr, t0=t0, steps=max(HORIZONS.values()), exact_seed=False, history=hist)
                n = len(s)
                u, um = u_all[t0:t0 + n], um_all[t0:t0 + n]
                pe = np.abs(positions(s) - positions(u)).max(axis=2)
                for h, steps in HORIZONS.items():
                    if n - 1 >= min(steps, tr.steps - t0):
                        k = min(steps, n - 1) + 1
                        pos_err.setdefault(h + suffix, []).append(pe[:k].ravel())
                        mid_tick_err.setdefault(h + suffix, []).append(float(pe[:k].max()))
                        mid_exact[h + suffix] += all(_row_equal(s[j], u[j], m[j], um[j]) for j in range(k))
                _merge(report.setdefault("events_mid" + suffix, {}),
                       event_agreement(obs_events(u[:126], um[:126]), obs_events(s[:126], m[:126])))
        report["traces"].append(rec)
        print(f"{rec['trace']}: seed-init exact {rec['seed']['exact_steps']}/{rec['seed']['replayed']}"
              f" div={rec['seed']['first_exact_divergence']} | obs-init exact {rec['obs']['exact_steps']}"
              f" maxerr={rec['obs']['pos_err_max']:.2g} | scores {rec['seed']['final_scores']}", flush=True)
    for h, arrs in pos_err.items():
        a = np.concatenate(arrs) if arrs else np.zeros(1)
        report["horizons"][h] = {"n": int(a.size), "median": float(np.median(a)), "p99": float(np.percentile(a, 99)),
                                 "max": float(a.max()), "frac_le_1e-3": float(np.mean(a <= 1e-3))}
        if h in mid_tick_err:
            report["horizons"][h]["starts"] = len(mid_tick_err[h])
            report["horizons"][h]["starts_max_err_le_1e-3"] = float(np.mean(np.asarray(mid_tick_err[h]) <= 1e-3))
            report["horizons"][h]["starts_bit_exact"] = int(mid_exact[h])
    rng = [r["rng"] for r in report["traces"] if r.get("rng")]
    report["summary_rng_spawns"] = {"seed_reproducible": len(rng),
                                    "bit_exact": sum(r["first_exact_divergence"] is None for r in rng)}
    cl = [r["closed_loop"] for r in report["traces"] if r.get("closed_loop")]
    report["summary_closed_loop"] = {"heuristic_matches": len(cl),
                                     "all_moves_identical": sum(c["first_move_mismatch"] is None for c in cl)}
    report["summary"] = {"traces": len(paths), "full_match_bit_exact": exact_full,
                         "score_timeline_equal": score_equal, "final_score_equal": final_equal,
                         "match_over_with_env_winner": winner_equal}
    for key in ("events_full", "events_mid", f"events_mid_hist{HISTORY}"):
        allv = report.get(key, {}).get("all", {"unity": 0, "sim": 0, "matched": 0})
        report["summary"][key + "_agreement"] = allv["matched"] / max(1, max(allv["unity"], allv["sim"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def fit_order(paths, start=None, cap: int = 11000) -> list[int]:
    """Local search over AgentAct orders (all pair swaps until none helps), maximising the summed
    first-divergence step of full-match replays.  Needed again if a new build changes the
    OnEnable order of the BlackOutAgent components."""
    import itertools

    trs = [Trace.load(p) for p in paths]

    def score(order):
        tot = 0
        for tr in trs:
            s, m, _ = replay(tr, order=np.asarray(order), steps=cap)
            e, _ = first_divergence(s, m, tr.raw_state[:len(s)], tr.raw_map[:len(s)])
            tot += len(s) if e is None else e
        return tot

    order = list(DEFAULT_ORDER if start is None else start)
    best = score(order)
    improved = True
    while improved:
        improved = False
        for i, j in itertools.combinations(range(10), 2):
            cand = list(order)
            cand[i], cand[j] = cand[j], cand[i]
            c = score(cand)
            if c > best:
                best, order, improved = c, cand, True
                print("better", order, best, flush=True)
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", default="reports/sim/traces/trace_*.npz")
    parser.add_argument("--every", type=int, default=10, help="seconds between mid-match starts")
    parser.add_argument("--out", type=Path, default=Path("reports/sim/validation"))
    parser.add_argument("--probe", action="store_true", help="probe agent-order swaps at divergences")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--fit-order", action="store_true",
                        help="search the ML-Agents AgentAct order (pair swaps) that replays the traces longest")
    args = parser.parse_args()
    paths = sorted(glob.glob(args.traces))
    if args.fit_order:
        print("best order", fit_order(paths))
        return
    rep = run(paths, args.every, args.out, args.probe, closed=not args.no_closed_loop)
    print(json.dumps(rep["summary"], indent=1), json.dumps(rep["summary_rng_spawns"]), json.dumps(rep["summary_closed_loop"]))
    print(json.dumps(rep["horizons"], indent=1))
    for key in ("events_full", "events_mid", f"events_mid_hist{HISTORY}"):
        print(key, json.dumps(rep.get(key)))


if __name__ == "__main__":
    main()
