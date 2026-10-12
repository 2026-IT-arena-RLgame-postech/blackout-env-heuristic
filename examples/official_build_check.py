"""Fork policies playing the *official* Unity build through OfficialTracker, verified every tick.

The official build (ist-tech-AI-games/blackout @ d2220a7) sends only the official observations.
Team A's unit_0 vector + the map feed one tracker; the tracker's reconstruction (fork format)
drives both teams' policies (25 Hz decisions, each action held for 2 ticks, as trained).  Every
tick it is cross-checked against everything the official build shows: each of the 10 agents' own
class, every held item type, positions, both scores (battery amounts: an amount the tracker had
wrong shows up as a score mismatch) and ground items -- the tracker's resync counters must stay 0.

Matches follow the official ``run_series`` pattern: UnityEnvironment(seed=0), ``reset()`` without
a seed after every match.

    PYTHONPATH=. .venv/bin/python examples/official_build_check.py --build <BlackOut.app> --matches 4
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from blackout_env.official.render import parse_vector
from blackout_env.official.seed_search import RngFinder
from blackout_env.official.tracker import MultiTracker
from blackout_env.sim import core
from blackout_env.train.distill.opponents import make_opponent

V32 = ("strategic_v32:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,"
       "camp_engage_radius=6.5,hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,"
       "sweep=true,cargo_transform=true,adapt_k=3")
V31 = ("strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,"
       "camp_engage_radius=6.5")


def collect(env, unit_names, map_name):
    """(vectors [10, 45] or None rows, agent ids per behaviour, terminal?, team A id map)."""
    vectors = [None] * 10
    ids, terminal = {}, False
    for name in unit_names:
        dec, term = env.get_steps(name)
        rows = []
        for steps, is_term in ((dec, False), (term, True)):
            for aid in steps.agent_id:
                v = steps[aid].obs[0]
                u = int(round(float(v[44])))
                vectors[u] = v
                if is_term:
                    terminal = True
                else:
                    rows.append((aid, u))
        ids[name] = rows
    dec, _ = env.get_steps(map_name)
    gmap = None
    if len(dec):
        vis = [o for o in dec[dec.agent_id[0]].obs if o.ndim == 3][0]   # (1, 96, 96)
        gmap = np.rint(vis[0] * 255).astype(np.uint8)
    return vectors, ids, terminal, gmap


def check(tr, vectors, bad):
    sim = tr.sim
    for u, v in enumerate(vectors):
        if v is None:
            continue
        p = parse_vector(v)
        if int(sim.ui[u, core.U_CLS]) != p["my_class"]:
            bad["cls"] += 1
        h = sim.ui[u, core.U_HELD]
        have = 0 if h < 0 else int(sim.it[h, core.I_TYPE]) + 1
        if have != int(p["held"][u]):
            bad["held"] += 1
        if tuple(sim.scores) != p["scores"]:
            bad["score"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--a", default=V32)
    ap.add_argument("--b", default=V31)
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--out", default="reports/official_obs/official_build_check.jsonl")
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()

    eng = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=args.build, seed=0, no_graphics=True, side_channels=[eng], worker_id=args.worker)
    eng.set_configuration_parameters(time_scale=args.time_scale)
    env.reset()
    unit_names = [n for n in env.behavior_specs if n.startswith("BlackOutUnit")]
    map_name = [n for n in env.behavior_specs if n.startswith("BlackOutMap")][0]
    tr = MultiTracker(RngFinder(full_scan=False))
    rows = []
    for m in range(args.matches):
        descs = [d.strip() for d in (args.a + "|" + args.b).split("|")]
        da, db = descs[m % (len(descs) - 1)], descs[-1] if m % 2 == 0 else descs[0]
        pa, pb = make_opponent(da, 100 * args.worker + 2 * m), make_opponent(db, 100 * args.worker + 2 * m + 1)
        first_bad, settled = None, None
        moves = np.zeros((10, 2), np.float32)
        bad = {"cls": 0, "held": 0, "score": 0}
        ticks, t_track, t0 = 0, 0.0, time.time()
        stats0 = None
        while True:
            vectors, ids, terminal, gmap = collect(env, unit_names, map_name)
            if terminal:
                break
            if vectors[0] is not None:
                t1 = time.perf_counter()
                tr.observe(vectors[0], gmap)
                t_track += time.perf_counter() - t1
                if stats0 is None:
                    stats0 = tr.seed
                before = sum(bad.values())
                check(tr, vectors, bad)
                if first_bad is None and sum(bad.values()) > before:
                    first_bad = ticks
                if settled is None and tr.stats["hypotheses"] == 1:
                    settled = ticks
                if ticks % 2 == 0:
                    obs = tr.obs()
                    for pol, units in ((pa, range(5)), (pb, range(5, 10))):
                        for n, a in pol.act({f"unit_{u}": obs[f"unit_{u}"] for u in units}).items():
                            moves[int(n[5:])] = a
                ticks += 1
            for name in unit_names:
                if ids[name]:
                    env.set_actions(name, ActionTuple(continuous=np.stack([moves[u] for _, u in ids[name]])))
            dec, _ = env.get_steps(map_name)
            if len(dec):
                env.set_actions(map_name, ActionTuple(continuous=np.zeros((len(dec), 0), np.float32)))
            env.step()
        sc = list(tr.sim.scores)
        row = {"worker": args.worker, "match": m, "a": da[:20], "b": db[:20], "rng": tr.seed, "settled_tick": settled,
               "end": "score" if max(sc) >= 100 else "timeout", "first_bad_tick": first_bad,
               "ticks": ticks, "scores": sc, "mismatch": bad,
               "resync": {k: v for k, v in tr.stats.items() if k not in ("ticks", "seeded")},
               "track_ms_per_tick": 1000 * t_track / max(ticks, 1), "wall_s": time.time() - t0}
        print(json.dumps(row), flush=True)
        rows.append(row)
        env.reset()   # official run_match: reset() without a seed
    env.close()
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
