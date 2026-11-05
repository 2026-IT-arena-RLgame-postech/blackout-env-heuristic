"""OOD-BP: out-of-distribution opponents for the post-balance-patch game (50 Hz, max 2 Hunters per team).

OOD-26 (``ood_suite.py``) is built from pre-patch heuristics (V30 family) tuned to old rules.  Our post-patch RL
training pool is B6, B5, B4, C0, B3R (one heuristic lineage with different CEM knobs) plus our own past RL checkpoints,
so RL has never met rushes / blockades / exit campers / pure stealers / decoys / economy-only play / noisy execution.
OOD-BP is a fixed, named set of such opponents; ``examples/v27_eval.py play --hz 50 --opponent oodbp`` expands it
(``oodbp:<group>[+<group>]``, ``oodbp=<name>,<name>``); ``examples/oodbp_report.py`` summarises the rows.

Design facts that shaped it (reports/bp/ood, NOTES.md 2026-11-02):

* the post-patch match is decided in the first ~40 s: whoever wins the Hunter-sanctuary race (tick ~98-100) and the
  next Hunter exchange owns the sanctuary.  A policy that loses the race is crushed (margin about -80 vs B6) whatever
  it does afterwards, so the *competent* personalities keep B6's opening (B6 descriptor + knobs, or ``switch:`` from B6
  after the opening) and differ in what they do with their Hunters and Collectors afterwards;
* execution noise from tick 0 only ruins the race, so the noise entries start after the opening (``switch:25:B6|perturb:..``)
  except one ``sloppy`` entry with a mild noise from the start;
* the group ``crushed`` (random, idle, the pre-patch V17 style policies) is expected to lose by 80-100: read it by win
  rate (a regression there is a bug, not a strength gap).

Each entry: (short name, descriptor, group, personality / why it is OOD).  ``B6`` etc. are the aliases of
``blackout_env.heuristics.params``.
"""

from __future__ import annotations

from blackout_env.heuristics.params import resolve_descriptor


def _oodbp(extra: str) -> str:
    """B6 on the ``strategic_oodbp`` class (V34 + Hunter plan / hide) with extra knobs."""
    return resolve_descriptor("B6," + extra).replace("strategic_v34:", "strategic_oodbp:", 1)


B6 = "B6"
CAMP = "B6,hunter_roles=south/east,roles_when=always"

OODBP: list[tuple[str, str, str, str]] = [
    # (a) Hunter-war personalities: B6 opening, then another use of the two Hunters
    ("exit_camp", CAMP, "hunter", "both Hunters camp the enemy base exits (south + east), strike what leaves"),
    ("raid_south", "B6,hunter_roles=raid/south,roles_when=always", "hunter", "one Hunter raids Collectors, one holds the enemy south exit"),
    ("blockade", "B6,hunter_roles=,solo_sentry=true,sentry_min_hunters=1,sentry_offset=1.5,sentry_radius=10,guard_fill=true",
     "hunter", "Hunters sit on the Hunter sanctuary (post 1.5 tiles out, radius 10) and deny the race entrance"),
    ("hunter_duel", "B6,hunter_roles=,hunt_hh=30,guard_fill=false", "hunter", "Hunters hunt the enemy Hunters first (duelist), no raiding"),
    ("fwd_post", "B6,sentry_offset=9,hunter_roles=", "hunter", "sentry post 9 tiles out toward the enemy spawn (forward defence)"),
    ("turtle", "B6,hunter_roles=,exit_guards=2,guard_radius=10,exposure_cost=2.0,steal_weight=0.8,guard_fill=false",
     "hunter", "both Hunters guard our own exit/storages, deposits only to the safe storage, minimal offence"),
    # (b) economy personalities
    ("stealer", _oodbp("raid_squad=3,steal_weight=3.0,exposure_cost=2.0,hunter_roles=raid/raid,raid_value=4"), "econ",
     "three Collectors never take field batteries and live off the enemy's storages; loot goes to the safe storage"),
    ("pure_stealer", _oodbp("raid_squad=5,field_weight=0,steal_weight=3.0,exposure_cost=2.0"), "econ",
     "all five units only steal (no field pickups at all); stalemate-prone, long games"),
    ("eco_boom", _oodbp("hunter_plan=0:2/12:0,hide=home"), "econ", "wins/loses the opening, then wants no Hunters: survivors hide in base, pure economy"),
    ("late_burst", _oodbp("hunter_plan=0:2/20:0/50:2"), "econ", "no Hunter refills 20-50 s (economy), then refills to two again"),
    # (c) tempo / tricks
    ("storm", "B6,storm_units=3,storm_after=4", "tempo", "three Collectors committed to the sanctuary whenever the enemy has more Hunters out"),
    ("decoy", "B6,bait=6", "tempo", "Collectors run at the enemy Hunter that approaches our Carrier (bait)"),
    ("camp_after_open", f"switch:30:B6|{CAMP}", "tempo", "plays B6 for 30 s, then turns into exit_camp"),
    # (d) execution noise after the opening (clean race), plus one sloppy-from-the-start entry
    ("noisy_eps", "switch:25:B6|perturb:eps=0.15:B6", "noise", "B6 whose units act randomly 15 % of the unit-ticks after 25 s"),
    ("laggy", "switch:25:B6|perturb:lag=6:B6", "noise", "B6 acting on 6-tick-old observations after 25 s"),
    ("sloppy", "perturb:freeze=0.04:B6", "noise", "B6 whose units stall 4 % of the time from tick 0 (loses some races)"),
    # (e) legacy heuristics outside the RL pool
    ("v34c", "V34c", "legacy", "pre-B3 post-patch heuristic: both Hunters raid always, long stalemate games"),
    # (f) crushed: expected to lose heavily (read by win rate)
    ("random", "random", "crushed", "random walkers"),
    ("idle", "idle", "crushed", "units stand still"),
    ("early_rush", "strategic_early_rush", "crushed", "pre-patch V17 style: 4-Hunter opening sweep (capped to 2), no race logic"),
]

NAMES = {d: n for n, d, _, _ in OODBP}
GROUPS = sorted({g for _, _, g, _ in OODBP})

# OOD-HO (2026-11-05): HELD-OUT evaluation personalities.  OOD-BP entries were later put into the RL training pool
# (pool12: stealer, storm, late_burst, ...), so OOD-BP no longer measures out-of-distribution play.  These are new
# combinations / phase changes that no training pool contains -- never add them to a pool (``oodho`` in v27_eval).
# v2: matches end at a 100-point lead after ~40 s (median; 10 % by 24 s), so switches happen by 8-15 s (v1's 45/60 s
# switches never fired: those entries were plain B6).
_BL = "B6,hunter_roles=,solo_sentry=true,sentry_min_hunters=1,sentry_offset=1.5,sentry_radius=10,guard_fill=true"
_RS = "B6,hunter_roles=raid/south,roles_when=always"
OODHO: list[tuple[str, str, str, str]] = [
    ("ho_one_hunter", _oodbp("hunter_plan=0:2/10:1"), "ho_econ", "race with two, then keeps a single Hunter (refills to one)"),
    ("ho_mid_steal", "switch:12:B6|" + _oodbp("raid_squad=3,steal_weight=3.0,exposure_cost=2.0,hunter_roles=raid/raid,raid_value=4"),
     "ho_econ", "plays B6 for 12 s, then turns into a three-Collector storage thief"),
    ("ho_steal_camp", _oodbp("raid_squad=2,steal_weight=2.5,hunter_roles=south/east,roles_when=always"), "ho_econ",
     "two Collectors steal while both Hunters camp our exits"),
    ("ho_lead_turtle", "switch:15:B6|B6,hunter_roles=,exit_guards=2,guard_radius=10,exposure_cost=2.0,steal_weight=0.8,guard_fill=false",
     "ho_tempo", "plays B6 for 15 s, then turtles"),
    ("ho_phase3", f"switch:8:B6|switch:20:{_RS}|{_BL}", "ho_tempo", "B6 opening, raid_south 8-20 s, then blockade"),
    ("ho_storm_bait", "B6,storm_units=2,storm_after=8,bait=5", "ho_tempo", "two-unit storm plus Carrier bait"),
    ("ho_east_camp", "B6,hunter_roles=east/east,roles_when=always", "ho_hunter", "both Hunters camp our east exit only"),
    ("ho_fwd_raid", "B6,sentry_offset=6,hunter_roles=raid/raid,roles_when=always,raid_value=4", "ho_hunter",
     "forward post plus double raid on Collectors"),
    ("ho_mid_chaos", "switch:12:B6|perturb:eps=0.3:B6", "ho_noise", "B6 that degrades to 30 % random unit-ticks after 12 s"),
    ("ho_v34b", "V34b", "ho_legacy", "older post-patch heuristic (not in any pool)"),
]
HO_NAMES = {d: n for n, d, _, _ in OODHO}


def expand(entry: str) -> list[str]:
    """``oodbp`` -> every descriptor; ``oodbp:<group>[+<group>]`` -> those groups; ``oodbp=<name>,<name>`` -> those
    names; anything else -> itself."""
    if entry == "oodbp":
        return [d for _, d, _, _ in OODBP]
    if entry == "oodho":
        return [d for _, d, _, _ in OODHO]
    if entry.startswith("oodbp:"):
        groups = set(entry[6:].split("+"))
        unknown = groups - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown OOD-BP groups {sorted(unknown)} (have {GROUPS})")
        return [d for _, d, g, _ in OODBP if g in groups]
    if entry.startswith("oodbp="):
        by = {n: d for n, d, _, _ in OODBP}
        return [by[n] for n in entry[6:].split(",")]
    return [entry]


def short(descriptor: str) -> str:
    return NAMES.get(descriptor) or HO_NAMES.get(descriptor, descriptor if len(descriptor) < 40 else descriptor[:37] + "...")
