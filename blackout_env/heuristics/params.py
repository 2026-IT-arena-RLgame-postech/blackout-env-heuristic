"""Named, ranged behaviour knobs of the champion heuristic lineage (V19, V30 ... V32.1) and a variant sampler.

Every knob is an ordinary constructor keyword of the heuristic classes (``make_heuristic(policy_id, **params)``),
so a parameterised variant is just a descriptor string -- the same ``policy_id:k=v,k=v`` text
``train/distill/opponents.make_opponent`` already parses, which is also what ``HeuristicPolicy`` /
``ParallelHeuristic`` (GRPO opponent workers) and ``act:<descriptor>`` teachers take.  This module adds

* ``KNOBS`` -- name, type, search range, group and one-line meaning of every knob (the knob table in
  ``docs/heuristic_params.md`` is generated from it: ``python -m blackout_env.heuristics.params``);
* ``ALIASES`` -- short names for the champion descriptors (``V19 V30 V31 V32 V32.1``); ``make_opponent`` accepts
  ``V32.1`` or ``V32.1:hunter_quota=3,sweep_radius=7.5`` (or ``V32.1,hunter_quota=3``): the alias's parameters first,
  then the overrides;
* ``sample_params`` -- a seeded random neighbour of a base descriptor, for retuning searches and opponent pools.

Defaults reproduce the current behaviour bit for bit (``tests/test_heuristic_params.py``).  Ranges are *sane search
ranges around the champion's values*, checked by ``examples/heur_variants.py`` to play non-degenerate games; they are
not guarantees.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ------------------------------------------------------------------------------------------------ descriptors

_V31 = ("reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,"
        "threat_radius=0.13,camp_engage_radius=6.5")
_V32 = (_V31 + ",hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,"
        "sweep=true,cargo_transform=true")
ALIASES = {
    "V19": "strategic_v19",
    "V30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
    "V31": "strategic_v31:" + _V31,
    "V32": "strategic_v32:" + _V32,
    "V32.1": "strategic_v32:" + _V32 + ",adapt_k=3",
    # V33 (reports/heur_v33.md): V32.1 with commit_margin < 0 (walkers step into the sanctuary instead of fleeing a
    # fresh enemy Hunter), CEM-retuned knobs and much higher special-item weights (BuffSpeed / DebuffSpeed)
    "V33": "strategic_v33:reserve_stage=0.5523,trade_hold=advance,reserve_join=reserve,solo_sentry=false,threat_radius=0.1113,camp_engage_radius=6.5,hunter_quota=4,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,sweep=true,cargo_transform=true,adapt_k=4,commit_margin=-0.3,guard_radius=7.691,late_after=16.0,carrier_col_scale=0.9054,camp_tiles=2.457,breakout_units=1,join_reach=1.137,trade_radius=0.7359,steal_weight=1.94,exposure_cost=0.7841,danger_hunter_radius=4.137,contest_factor=0.5747,w_dspd=5.0,w_bspd=4.5",
}
# Balance patch (max 2 Hunters per team): V33 with its Hunter knobs in the new ranges (same behaviour as V33)
ALIASES["V33bp"] = ALIASES["V33"].replace("hunter_quota=4", "hunter_quota=2")
ALIASES["V34"] = ALIASES["V33bp"].replace("strategic_v33:", "strategic_v34:")
# V34a: + two staged reserves and cap_sync (reports/bp: +10 vs V34 + reserve_count=2, side gap 26 -> 24)
ALIASES["V34a"] = ALIASES["V34"] + ",reserve_count=2,cap_sync=true,cap_enter=0.2"


def parse_value(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_params(text: str) -> dict:
    out = {}
    for item in filter(None, text.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"heuristic parameter {item!r} is not key=value")
        out[key] = parse_value(value)
    return out


def format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    return str(value)


def format_params(params: dict) -> str:
    return ",".join(f"{k}={format_value(v)}" for k, v in params.items())


def _head(descriptor: str) -> str:
    return descriptor.replace(",", ":", 1).split(":", 1)[0]


def split_descriptor(descriptor: str) -> tuple[str, dict]:
    """``policy_id[:k=v,...]`` or ``ALIAS[:k=v,...]`` / ``ALIAS,k=v,...`` -> (policy_id, merged params)."""
    head = _head(descriptor)
    if head in ALIASES:
        rest = descriptor[len(head):].lstrip(":,")
        policy, _, base = ALIASES[head].partition(":")
        return policy, {**parse_params(base), **parse_params(rest)}
    policy, _, rest = descriptor.partition(":")
    return policy, parse_params(rest)


def resolve_descriptor(descriptor: str) -> str:
    """Expand an alias descriptor into a plain ``policy_id:k=v,...`` one (other descriptors are returned as is)."""
    if _head(descriptor) not in ALIASES:
        return descriptor
    policy, params = split_descriptor(descriptor)
    return policy + (":" + format_params(params) if params else "")


# ------------------------------------------------------------------------------------------------ knobs


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str          # "float" | "int" | "bool"
    lo: float
    hi: float
    group: str
    doc: str
    structural: bool = False   # switches a rule on/off: sampled only with ``structural=True``
    teach_only: bool = False   # only used by the V22+ goal-labeller lineage (V30..V32.1), inert in V19

    def clip(self, value):
        if self.kind == "bool":
            return bool(value)
        value = min(self.hi, max(self.lo, value))
        return int(round(value)) if self.kind == "int" else value


def _k(name, kind, lo, hi, group, doc, structural=False):
    return Knob(name, kind, lo, hi, group, doc, structural, name in TEACH_ONLY)


# measured inert for V19 (examples/heur_variants.py sensitivity): V19 does not use the teach-labeller code paths
TEACH_ONLY = {"threat_radius", "deposit_band", "idle_band", "exclusion_margin", "score_band", "carrier_col_scale",
              "carrier_hunter_scale", "camp_tiles", "danger_tiles", "breakout_units", "opening_seconds", "commit_margin",
              "commit_reach", "reserve_stage", "reserve_count", "join_margin", "join_reach", "trade_radius",
              "trade_reach", "trade_floor", "sentry_min_hunters", "adapt_k", "adapt_at", "late_after",
              "race_commit", "late_hunter_quota",
              "cargo_transform", "sweep", "sweep_camper", "solo_sentry"}


KNOBS: dict[str, Knob] = {k.name: k for k in [
    # -- Hunter allocation / roles
    _k("hunter_quota", "int", 1, 2, "hunters", "Hunters the team keeps (walkers sent to the sanctuary until reached)"),
    _k("exit_guards", "int", 0, 2, "hunters", "Hunters holding our base exit (strike enemy Hunters near it)"),
    _k("guard_min_hunters", "int", 1, 2, "hunters", "Hunters needed before exit guards are posted"),
    _k("guard_radius", "float", 4.0, 10.0, "hunters", "tiles around our exit in which a guard strikes enemy Hunters"),
    _k("sweep_radius", "float", 5.0, 12.0, "hunters", "tiles from the sanctuary / Hunter in which Hunters strike enemy Collectors"),
    _k("sentry_min_hunters", "int", 1, 3, "hunters", "Hunters needed before one is posted as sanctuary sentry"),
    _k("sentry_offset", "float", 1.0, 4.0, "hunters", "tiles from sanctuary centre toward the enemy spawn where the sentry stands"),
    _k("sentry_radius", "float", 4.0, 10.0, "hunters", "tiles around the sanctuary in which the sentry strikes enemy Collectors"),
    _k("camp_engage_radius", "float", 3.0, 9.0, "hunters", "camper strikes enemies within this many tiles of its post"),
    _k("late_after", "float", 4.0, 60.0, "hunters", "seconds into the match after which late_hunter_quota refills Hunters (inert while that is off)"),
    _k("adapt_k", "int", 0, 4, "hunters", "latch: enemy showed fewer than this many Hunters by adapt_at -> V30 roles (0 = off)"),
    _k("adapt_at", "float", 3.0, 12.0, "hunters", "seconds at which the adapt_k latch is read"),
    _k("hunter_speed", "float", 4.5, 7.5, "hunters", "assumed Hunter speed (tiles/s) in all chase/arrival estimates (own and enemy)"),
    # -- hunting value model
    _k("hunt_threshold", "float", 0.15, 0.7, "hunting", "a chaser only chases targets scoring above this"),
    _k("hunt_spawn_radius", "float", 4.0, 10.0, "hunting", "tiles from our spawn within which an enemy Hunter is a camper worth trading"),
    _k("hunt_spawn_value", "float", 6.0, 20.0, "hunting", "value of trading with a Hunter camping our spawn"),
    _k("hunt_camp_radius", "float", 2.0, 5.0, "hunting", "tiles from exposed storage within which an enemy Hunter is worth trading"),
    _k("hunt_camp_value", "float", 1.0, 8.0, "hunting", "value of trading with a Hunter next to exposed storage"),
    _k("chase_equal", "float", 1.0, 4.0, "hunting", "chase-time penalty vs an equally fast target (1 = chase eagerly)"),
    # -- engagement / evasion
    _k("threat_radius", "float", 0.08, 0.2, "evasion", "normalised radius within which Collectors/Carriers flee Hunters (x12 = tiles)"),
    _k("carrier_col_scale", "float", 0.0, 1.0, "evasion", "Carrier's flee radius from enemy Collectors/Carriers, x threat_radius; <1 only"),
    _k("carrier_hunter_scale", "float", 0.5, 1.0, "evasion", "Carrier's flee radius from enemy Hunters, x threat_radius; <1 only"),
    _k("camp_tiles", "float", 2.0, 5.0, "evasion", "tiles: enemy Hunter this close to our base is camping the exit"),
    _k("danger_tiles", "float", 2.0, 5.0, "evasion", "tiles around a camping Hunter avoided when a route around exists"),
    _k("breakout_units", "int", 1, 4, "evasion", "blocked units waiting in the base that leave together"),
    # -- race / opening
    _k("opening_seconds", "float", 6.0, 20.0, "race", "seconds in which a battery-holding walker still goes to the sanctuary"),
    _k("commit_margin", "float", -1.0, 0.25, "race", "seconds a committed walker must reach the sanctuary ahead of an enemy Hunter"),
    _k("commit_reach", "float", 0.5, 2.0, "race", "tiles from the sanctuary within which the last-step commit applies"),
    _k("reserve_stage", "float", 0.0, 2.0, "race", "tiles: the staged reserve walker waits this far from the sanctuary (0 = none)"),
    _k("reserve_count", "int", 1, 3, "race", "number of staged reserve walkers"),
    _k("join_margin", "float", 0.0, 0.3, "race", "tiles the enemy's lead may trail ours for the race to count as tied"),
    _k("join_reach", "float", 0.5, 2.0, "race", "tiles: enemy Collectors this close to its lead walker count as its reserve"),
    _k("trade_radius", "float", 0.5, 3.0, "race", "tiles: enemy Hunters this close to ours count as a pending trade"),
    _k("trade_reach", "float", 0.5, 3.0, "race", "tiles from the sanctuary within which our Collectors hold/advance during a trade"),
    _k("trade_floor", "float", 0.05, 0.6, "race", "tiles from the sanctuary where an advancing Collector stops"),
    # -- economy / carriers / deposits
    _k("steal_weight", "float", 0.8, 3.0, "economy", "value multiplier of batteries stolen from enemy storage"),
    _k("exposure_cost", "float", 0.3, 2.0, "economy", "cost of leaving a deposit stealable for an absorption interval"),
    _k("carrier_field_value", "float", 10.0, 40.0, "economy", "field battery value below which no new Carrier is made"),
    _k("danger_hunter_radius", "float", 2.5, 6.0, "economy", "tiles: enemy Hunters this near a pickup discount its value"),
    _k("danger_collector_radius", "float", 1.5, 4.0, "economy", "tiles: enemy Collectors this near a pickup discount it for a Carrier"),
    _k("danger_weight", "float", 0.3, 0.9, "economy", "max discount of a pickup next to an enemy"),
    _k("contest_factor", "float", 0.5, 1.0, "economy", "value factor of a field battery an enemy reaches first"),
    _k("score_band", "float", 1.05, 1.3, "economy", "tasks within this score ratio of the best are kept as options"),
    _k("exclusion_margin", "float", 0.3, 2.0, "economy", "seconds a teammate must arrive earlier to take a shared task"),
    _k("deposit_band", "float", 0.1, 0.6, "economy", "deposit cost slack (s) within which storages are tied"),
    _k("idle_band", "float", 0.1, 0.6, "economy", "idle-patrol score slack within which enemy storages are tied"),
    # -- special items (V33)
    _k("w_bspd", "float", 0.0, 6.0, "specials", "planner value per second of an owned BuffSpeed"),
    _k("w_dspd", "float", 0.0, 8.0, "specials", "planner value per second of an owned DebuffSpeed (enemy Workers -90 %)"),
    _k("w_bsize", "float", 0.0, 1.5, "specials", "planner value per second of an owned BuffSize"),
    _k("w_dsize", "float", 0.0, 1.5, "specials", "planner value per second of an owned DebuffSize"),
    _k("dom_threat_radius", "float", 0.1, 0.4, "evasion", "threat_radius used while the enemy dominates the Hunter war (V33; -1 = off)"),
    _k("dom_danger_hunter_radius", "float", 3.0, 9.0, "evasion", "danger_hunter_radius used while the enemy dominates (V33; -1 = off)"),
    _k("dom_margin", "int", 1, 2, "evasion", "enemy Hunters out minus ours that counts as dominated (V33)"),
    _k("h_evade_radius", "float", 1.0, 6.0, "hunters", "tiles: an outnumbered Hunter is repelled by enemy Hunters this close (V33; -1 = off)"),
    _k("carrier_delay", "float", 0.0, 30.0, "race", "s: no Carrier before this (0 = off; V34)"),
    _k("cap_sync", "bool", 0, 1, "structural", "hold extra walkers out of the sanctuary while the Hunter cap is full (V34)", True),
    _k("cap_enter", "float", 0.1, 0.6, "race", "tiles: Collectors this close to the sanctuary count as entering (V34 cap_sync)"),
    _k("storm_units", "int", 0, 3, "structural", "Collectors committed to the sanctuary while the enemy has more Hunters (V34)", True),
    _k("storm_after", "float", 4.0, 20.0, "race", "s: storm only after this (V34)"),
    _k("carrier_on_monopoly", "bool", 0, 1, "race", "carrier_delay ends early once we hold a Hunter monopoly (V34)"),
    _k("h_trade_up", "bool", 0, 1, "structural", "when we have more Hunters, the closest pair becomes a strike (V33)", True),
    # -- structural switches (sampled only with structural=True)
    _k("sweep", "bool", 0, 1, "structural", "Hunters first strike enemy Collectors near the sanctuary", True),
    _k("cargo_transform", "bool", 0, 1, "structural", "battery-holding Collectors stay Hunter candidates all match", True),
    _k("sweep_camper", "bool", 0, 1, "structural", "the camper also sweeps (false keeps it on its post)", True),
    _k("solo_sentry", "bool", 0, 1, "structural", "a lone Hunter keeps the sanctuary (V30 role order)", True),
    _k("race_commit", "bool", 0, 1, "structural", "last-step commit of a walker into the sanctuary", True),
    _k("late_hunter_quota", "int", -1, 2, "structural", "late Hunter refill target (-1 = off)", True),
]}


@lru_cache(maxsize=None)
def _ctor_defaults(policy_id: str) -> dict:
    """Keyword -> constructor default over the class's whole MRO (the most derived definition wins)."""
    from .mixture import POLICY_REGISTRY

    out: dict = {}
    for klass in reversed(POLICY_REGISTRY[policy_id].__mro__):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, p in inspect.signature(init).parameters.items():
            if name != "self" and p.kind == p.KEYWORD_ONLY and p.default is not inspect.Parameter.empty:
                out[name] = p.default
    return out


def base_values(base: str) -> tuple[str, dict]:
    """(policy_id, {knob: effective value}) of a base descriptor: its explicit parameters over constructor defaults,
    for the knobs this policy class accepts."""
    from .teach import TeachLabeller
    from .mixture import POLICY_REGISTRY

    policy, explicit = split_descriptor(base)
    defaults = _ctor_defaults(policy)
    teach = issubclass(POLICY_REGISTRY[policy], TeachLabeller)
    return policy, {k: explicit.get(k, defaults[k]) for k, kn in KNOBS.items()
                    if k in defaults and (teach or not kn.teach_only)}


def sample_overrides(base: str, rng: np.random.Generator, scale: float = 1.0, *, groups=None,
                     structural: bool = False, p: float | None = None, exclude=()) -> dict:
    """Random neighbours of the base values: each eligible knob is perturbed with probability ``p`` (default
    ``min(1, 0.35 * scale)``) by a Gaussian of std ``scale * (hi - lo) / 6`` around the base value, clipped to its
    range (ints rounded; bools flip with probability ``0.3 * scale``).  Only changed values are returned."""
    policy, values = base_values(base)
    p = min(1.0, 0.35 * scale) if p is None else p
    out = {}
    for name, value in values.items():
        knob = KNOBS[name]
        if (knob.structural and not structural) or (groups and knob.group not in groups) or name in exclude:
            continue
        if rng.random() >= p:
            continue
        if knob.kind == "bool":
            if rng.random() < 0.3 * scale:
                out[name] = not value
            continue
        if not math.isfinite(float(value)):
            continue
        new = knob.clip(float(value) + rng.normal() * scale * (knob.hi - knob.lo) / 6.0)
        if knob.kind == "float":
            new = float(f"{new:.4g}")
            new = min(knob.hi, max(knob.lo, new))
        if new != value:
            out[name] = new
    return out


def with_params(base: str, **overrides) -> str:
    """The plain descriptor for ``base`` (alias or descriptor) with ``overrides`` applied."""
    policy, params = split_descriptor(base)
    params = {**params, **overrides}
    return policy + (":" + format_params(params) if params else "")


def sample_params(base: str, rng: np.random.Generator, scale: float = 1.0, **kwargs) -> str:
    """A random variant descriptor around ``base`` (``"V32.1"`` or any descriptor); see ``sample_overrides``."""
    return with_params(base, **sample_overrides(base, rng, scale, **kwargs))


# Hunter-count knobs: since the balance patch only values up to MAX_HUNTERS matter; older descriptors keep theirs
HUNTER_COUNT_KNOBS = {"hunter_quota", "guard_min_hunters", "sentry_min_hunters", "late_hunter_quota"}


def validate(descriptor: str) -> None:
    """Raise ValueError if a parameter is outside its knob's range (unknown names are left to the constructor)."""
    for name, value in split_descriptor(descriptor)[1].items():
        knob = KNOBS.get(name)
        if knob is None or knob.kind == "bool" or isinstance(value, str):
            continue
        if name in HUNTER_COUNT_KNOBS and value > knob.hi:
            continue  # pre-patch values (up to 4) act as the cap (env.constants.MAX_HUNTERS)
        if not (knob.lo <= value <= knob.hi) and not (name == "adapt_k" and value == 0):
            raise ValueError(f"{name}={value} outside the sane range [{knob.lo}, {knob.hi}]")


def knob_table(base: str = "V32.1") -> str:
    """Markdown table of the knobs with the defaults of the champion bases."""
    cols = ["V19", "V30", "V31", "V32.1"]
    vals = {c: base_values(c)[1] for c in cols}
    rows = ["| group | knob | type | range | " + " | ".join(cols) + " | meaning |",
            "|---|---|---|---|" + "---|" * len(cols) + "---|"]
    for k in KNOBS.values():
        cells = [format_value(vals[c][k.name]) if k.name in vals[c] else "-" for c in cols]
        rng = "on/off" if k.kind == "bool" else f"{format_value(k.lo)} .. {format_value(k.hi)}"
        rows.append(f"| {k.group} | `{k.name}` | {k.kind} | {rng} | " + " | ".join(cells) + f" | {k.doc} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(knob_table())
