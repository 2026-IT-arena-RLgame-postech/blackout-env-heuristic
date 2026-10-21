"""Behaviour fingerprints of heuristic descriptors: play a Python-sim match and hash every action.

Used to prove that refactors / new knobs (heuristics/params.py) keep the default behaviour bit-exact:
``match_hash(descriptor, seed, side, steps)`` plays ``descriptor`` against a fixed reference opponent in the
Python simulator and returns the SHA-1 of all the moves it made plus the final score.  The golden values live in
``tests/data/heuristic_hashes.json`` (written by ``examples/heur_hash.py capture`` on the code *before* the knobs
were added).
"""

from __future__ import annotations

import hashlib

import numpy as np

# Reference opponent: V28 (a fixed, stateful heuristic that is not in the lineage under test).
REFERENCE = "strategic_v28:reserve_stage=0.5,trade_hold=advance"
_V31 = ("reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,"
        "threat_radius=0.13,camp_engage_radius=6.5")
_V32 = (_V31 + ",hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,"
        "sweep=true,cargo_transform=true")
# the original (pre-knob) spellings of the champion lineage
BASELINES = {
    "V19": "strategic_v19",
    "V30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
    "V31": "strategic_v31:" + _V31,
    "V32": "strategic_v32:" + _V32,
    "V32.1": "strategic_v32:" + _V32 + ",adapt_k=3",
}


def match_hash(descriptor: str, seed: int, side: int = 0, steps: int = 1500, reference: str = REFERENCE) -> dict:
    """Play ``steps`` decisions of ``descriptor`` (team ``side``) vs ``reference`` on map seed ``seed``."""
    from blackout_env.sim.state import SimState, obs_from_raw
    from blackout_env.train.distill.opponents import make_opponent

    sim = SimState.from_seed(seed)
    pols = {side: make_opponent(descriptor, seed), 1 - side: make_opponent(reference, seed + 1)}
    h = hashlib.sha1()
    for _ in range(steps):
        obs = obs_from_raw(sim.raw_state(), sim.raw_map())
        mv = np.zeros((10, 2), np.float32)
        for team, pol in pols.items():
            acts = pol.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * team, 5 * team + 5)})
            for name, act in acts.items():
                mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
        h.update(mv.tobytes())
        sim.step(mv)
        if sim.over:
            break
    return {"sha1": h.hexdigest(), "score": list(sim.scores)}
