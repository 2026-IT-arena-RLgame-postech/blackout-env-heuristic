"""OOD-26: the out-of-distribution opponent suite for adoption gates (docs/policy_v2_design_ko.md §56).

The P14b...P21e gates (§38-§50) only played strong strategic heuristics and our own nets, all of
them in (or next to) the students' distillation pool.  OOD-26 is a fixed, named set of opponents
the students were *not* trained against -- or met only as a rare unperturbed extra -- so a gate
can check that a candidate does not collapse off-distribution.  ``examples/v27_eval.py play
--opponent ood26`` expands it (``ood26:<group>`` for one group); ``examples/ood_report.py``
turns the rows into the §56 metrics.

What the students *did* train against (the honest baseline, P21e's E pool, §50 + §43):

* ``mixture`` -- strategic_v1...v19 + v4_near, their numeric parameter cloud redrawn every
  absorption; V17-V19 also draw hunter_quota (default-1 / default) and hunter_mode mixed/hunt;
* ``checkpoints`` -- the offline QMIX runs of 2026-09-17/18 (not RUN16 = 2026-09-19);
* ``extra`` -- the five style policies at their defaults, perturb:eps=0.05:v19, lag=2:v17,
  sticky=3:v21, dither=0.1:v19, a perturbed v21-teach, distill d3b/d4, V28 x4, V29, V30 x2,
  V27(1.0), P17 x2, P16, self, V25 x2, V23 -- each unperturbed;
* labels from the V30 teacher (P20/P21e), V28 (P17), V25 (P14b) on the student's own states.

So the style defaults (group ``style``) are *near*-distribution controls (each ~2 % of matches),
not OOD; everything else is outside: V30-family parameters (never perturbed in training), heuristics
outside the pool (v20+ besides the extras), RUN16, random/idle, execution noise on the *strong*
policies (training only perturbed v17-v21), and a temperature-sampled P20.

Each entry: (short name, descriptor, group, why it is OOD).  Net descriptors use the
BLACKOUT_NETS names of examples/v27_eval.py (``p20``) and its ``temp:<T>:<net>`` sampler.
"""

from __future__ import annotations

V30 = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31 = V30 + ",threat_radius=0.13,camp_engage_radius=6.5"
RUN16 = "checkpoints/offline/20260919-010456_42178"  # never in any distill pool (pool = 09-17/18 runs)


def _v30(extra: str) -> str:
    return f"{V30},{extra}"


OOD26: list[tuple[str, str, str, str]] = [
    # (a) V30 parameter extremes: the V30 family is only ever met unperturbed
    ("v30_h4hunt", _v30("hunter_quota=4,hunter_mode=hunt"), "param", "4 Hunters chasing (default 3, mixed)"),
    ("v30_h2", _v30("hunter_quota=2"), "param", "2 Hunters, more economy"),
    ("v30_camp", _v30("hunter_mode=camp,camp_engage_radius=9.0"), "param", "all Hunters camp our spawn, wide engage"),
    ("v30_stage3", _v30("reserve_stage=3.0"), "param", "reserve staged far later (0.5 s trained)"),
    ("v30_nostage", _v30("reserve_stage=0.0"), "param", "no staged reserve at all"),
    ("v30_c0", _v30("carrier_quota=0"), "param", "no Carrier (default 1)"),
    ("v30_hunt", _v30("hunter_mode=hunt"), "param", "Hunters chase instead of mixed camp/hunt (h4hunt split)"),
    ("v30_reckless", _v30("exposure_cost=0.2,steal_weight=3.5"), "param", "reckless stealing, ignores exposure"),
    # (a') style policies: defaults are near-distribution controls (in the extra pool), one extreme
    ("early_rush", "strategic_early_rush", "style", "control: in extra pool at default"),
    ("fortress", "strategic_fortress", "style", "control: in extra pool at default"),
    ("raider", "strategic_raider", "style", "control: in extra pool at default"),
    ("duelist", "strategic_duelist", "style", "control: in extra pool at default"),
    ("cargo_hunter", "strategic_cargo_hunter", "style", "control: in extra pool at default"),
    ("rush_all", "strategic_early_rush:rush_hunters=5,rush_seconds=120.0,economy_hunters=2,economy_carriers=0",
     "param", "5-Hunter rush for 2 min, no Carrier"),
    # (b) old heuristics, unperturbed
    ("v4", "strategic_v4", "old", "mixture member, unperturbed (near-dist)"),
    ("v19", "strategic_v19", "old", "mixture member, unperturbed (near-dist)"),
    ("v21", "strategic_v21", "old", "only sticky-perturbed in pool"),
    ("v26", "strategic_v26", "old", "late-Hunter trade rule (§30), never in pool"),
    ("run16_final", f"ckpt:{RUN16}/final.pt", "old", "offline RL run kept out of every pool"),
    ("random", "random", "old", "random walkers"),
    ("idle", "idle", "old", "units stand still"),
    # (c) execution noise on the strong policies (training only perturbed v17-v21)
    ("v31_eps10", f"perturb:eps=0.1:{V31}", "noise", "V31 with 10 % random unit-ticks"),
    ("v30_lag4", f"perturb:lag=4:{V30}", "noise", "V30 acting on 4-tick-old observations"),
    ("v30_freeze10", f"perturb:freeze=0.1:{V30}", "noise", "V30 units stall 10 % of the time (2 s stretches)"),
    ("p20_t1", "temp:1.0:p20", "noise", "P20 sampling its logits at T=1 (77 % argmax agreement)"),
    # (d) mirror-adjacent: the stronger rule-teacher no student trained against
    ("v31", V31, "mirror", "V31 (§46), never in pool"),
]

# in-distribution reference rows (the teacher itself), reported next to the suite but not part of it.
# ``carrier_quota=3`` was in the first screen as "v30_c3" but is a no-op (V17 planner fields one Carrier at most:
# V30 vs it = 0 margin in 64/64 matches), so those rows are read as V30 (examples/ood_report.py).
C3_NOOP = _v30("carrier_quota=3")
REFERENCE: list[tuple[str, str, str, str]] = [("V30(ref)", V30, "ref", "teacher of P20/P21e, in pool")]
FOLLOWUP: list[tuple[str, str, str, str]] = [
    ("v30_h4", _v30("hunter_quota=4"), "param", "4 Hunters, mixed (h4hunt split)"),
]

NAMES = {d: n for n, d, _, _ in OOD26 + REFERENCE + FOLLOWUP}
GROUPS = sorted({g for _, _, g, _ in OOD26 + REFERENCE + FOLLOWUP})

# teacher of each candidate (for student - teacher gaps and the catastrophic-loss condition)
TEACHER_OF = {"p21e": V30, "p20": V30, "p19": V30, "p17": "strategic_v28:reserve_stage=0.5,trade_hold=advance",
              "p14b": "strategic_v25", "p16": "strategic_v27:reserve_stage=1.5"}


def expand(entry: str) -> list[str]:
    """``ood26`` -> every descriptor; ``ood26:<group>[+<group>]`` -> those groups; ``ood26=<name>,<name>`` -> those
    names; anything else -> itself."""
    if entry == "ood26":
        return [d for _, d, _, _ in OOD26]
    if entry.startswith("ood26:"):
        groups = set(entry[6:].split("+"))
        unknown = groups - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown OOD-26 groups {sorted(unknown)} (have {GROUPS})")
        return [d for _, d, g, _ in OOD26 + REFERENCE + FOLLOWUP if g in groups]
    if entry.startswith("ood26="):
        by = {n: d for n, d, _, _ in OOD26 + REFERENCE + FOLLOWUP}
        return [by[n] for n in entry[6:].split(",")]
    return [entry]


def short(descriptor: str) -> str:
    return NAMES.get(descriptor, descriptor if len(descriptor) < 40 else descriptor[:37] + "...")
