# Heuristic parameters (V19 / V30 / V31 / V32 / V32.1)

Goal: after the Hunter balance patch, (a) retune the best heuristic by search instead of code edits, and (b) generate
many diverse variants for the GRPO opponent pool.  Everything below is CPU-only and lives in
`blackout_env/heuristics/params.py` (registry + sampler), `examples/heur_variants.py` (tool) and
`tests/test_heuristic_params.py` (guard).

## Mechanism

Every knob is a constructor keyword of the heuristic classes, so a variant is just a descriptor string -- the
`policy_id:k=v,k=v` text that `make_opponent` already parses.  `HeuristicPolicy`, `ParallelHeuristic` (GRPO workers),
`act:<descriptor>` teachers and `--opponent-extra` therefore work unchanged.  Added on top:

* **Aliases** `V19 V30 V31 V32 V32.1` (`params.ALIASES`; `rl/grpo.py` uses the same table).  `make_opponent` accepts
  `V32.1`, `V32.1:hunter_quota=3,sweep_radius=7.5` and `V32.1,hunter_quota=3` (alias parameters first, then the overrides).
  Old descriptors (`strategic_v32:...`, `strategic_v19`) are unchanged.  In comma-separated CLI lists (`rl_train.py --set`)
  use `;` as list separator for descriptors with parameters, as before.
* **Registry** `params.KNOBS`: name, type, search range, group, meaning.  Defaults = the base descriptor's explicit value,
  else the constructor default (`params.base_values("V32.1")`).
* **Sampler** `sample_params(base, rng, scale)` -> descriptor.  Each eligible knob is perturbed with probability
  `min(1, 0.35*scale)` by a Gaussian with std `scale*(hi-lo)/6` around the *base* value, clipped to the range.
  Rule switches (`structural` knobs: `sweep`, `cargo_transform`, ...) are only flipped with `structural=True`.
  `sample_overrides(...)` returns just the changed values; `with_params(base, **kw)` builds a descriptor; `validate(d)`
  checks ranges.  Seeded `np.random.default_rng` => reproducible pools.

```python
from blackout_env.heuristics.params import sample_params, with_params
import numpy as np
rng = np.random.default_rng(0)
pool = [sample_params("V32.1", rng, scale=1.0) for _ in range(16)]     # opponent pool
d = with_params("V32.1", hunter_quota=3, threat_radius=0.15)            # hand-made variant
```

New constructor knobs (hard-coded literals before, defaults = the old literal, bit-exact): `hunt_threshold`,
`carrier_field_value`, `danger_hunter_radius`, `danger_collector_radius`, `danger_weight`, `contest_factor`
(economy kernel, Python reference and numba kernel both read them), `score_band`, `exclusion_margin`, `deposit_band`,
`idle_band`, `hunt_spawn_radius/value`, `hunt_camp_radius/value`, `chase_equal`, `hunter_speed` (assumed Hunter speed in
all arrival/chase estimates; it does not change the sim).  Game-engine constants (unit radii, base speeds of other
classes, absorption interval) are still module constants in `v17_planner.py` / `v22_teacher.py` -- change them there if
the patch changes the rules themselves.

## Knobs (defaults per base; `-` = not a parameter of that base)

Ranges are sane search ranges, narrowed after the demo below (e.g. `commit_margin` <= 0.25: 0.4 lost every game
vs the mirror).  "Conditional" knobs do nothing unless their rule is active: `reserve_stage/count`, `join_*`,
`trade_*` only matter when `hunter_quota <= 3` (with four Hunters there is no staged reserve walker); `late_after` needs
`late_hunter_quota >= 0`; `hunt_spawn_*`/`hunt_camp_*` only fire when an enemy Hunter camps our spawn/storage
(rare; measured never fired in 16 sensitivity games vs V31 / V19).

| group | knob | type | range | V19 | V30 | V31 | V32.1 | meaning |
|---|---|---|---|---|---|---|---|---|
| hunters | `hunter_quota` | int | 2 .. 4 | 4 | 3 | 3 | 4 | Hunters the team keeps (walkers sent to the sanctuary until reached) |
| hunters | `exit_guards` | int | 0 .. 2 | 1 | - | - | 1 | Hunters holding our base exit (strike enemy Hunters near it) |
| hunters | `guard_min_hunters` | int | 2 .. 4 | - | - | - | 2 | Hunters needed before exit guards are posted |
| hunters | `guard_radius` | float | 4.0 .. 10.0 | 7.0 | - | - | 7.0 | tiles around our exit in which a guard strikes enemy Hunters |
| hunters | `sweep_radius` | float | 5.0 .. 12.0 | 9.0 | - | - | 9.0 | tiles from the sanctuary / Hunter in which Hunters strike enemy Collectors |
| hunters | `sentry_min_hunters` | int | 1 .. 4 | - | 2 | 2 | 3 | Hunters needed before one is posted as sanctuary sentry |
| hunters | `sentry_offset` | float | 1.0 .. 4.0 | 2.5 | 2.5 | 2.5 | 2.5 | tiles from sanctuary centre toward the enemy spawn where the sentry stands |
| hunters | `sentry_radius` | float | 4.0 .. 10.0 | 7.0 | 7.0 | 7.0 | 7.0 | tiles around the sanctuary in which the sentry strikes enemy Collectors |
| hunters | `camp_engage_radius` | float | 3.0 .. 9.0 | 4.5 | 4.5 | 6.5 | 6.5 | camper strikes enemies within this many tiles of its post |
| hunters | `late_after` | float | 4.0 .. 60.0 | - | - | - | 10.0 | seconds into the match after which late_hunter_quota refills Hunters (inert while that is off) |
| hunters | `adapt_k` | int | 0 .. 4 | - | - | - | 3 | latch: enemy showed fewer than this many Hunters by adapt_at -> V30 roles (0 = off) |
| hunters | `adapt_at` | float | 3.0 .. 12.0 | - | - | - | 6.0 | seconds at which the adapt_k latch is read |
| hunters | `hunter_speed` | float | 4.5 .. 7.5 | 6.0 | 6.0 | 6.0 | 6.0 | assumed Hunter speed (tiles/s) in all chase/arrival estimates (own and enemy) |
| hunting | `hunt_threshold` | float | 0.15 .. 0.7 | 0.35 | 0.35 | 0.35 | 0.35 | a chaser only chases targets scoring above this |
| hunting | `hunt_spawn_radius` | float | 4.0 .. 10.0 | 7.0 | 7.0 | 7.0 | 7.0 | tiles from our spawn within which an enemy Hunter is a camper worth trading |
| hunting | `hunt_spawn_value` | float | 6.0 .. 20.0 | 12.0 | 12.0 | 12.0 | 12.0 | value of trading with a Hunter camping our spawn |
| hunting | `hunt_camp_radius` | float | 2.0 .. 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | tiles from exposed storage within which an enemy Hunter is worth trading |
| hunting | `hunt_camp_value` | float | 1.0 .. 8.0 | 4.0 | 4.0 | 4.0 | 4.0 | value of trading with a Hunter next to exposed storage |
| hunting | `chase_equal` | float | 1.0 .. 4.0 | 2.5 | 2.5 | 2.5 | 2.5 | chase-time penalty vs an equally fast target (1 = chase eagerly) |
| evasion | `threat_radius` | float | 0.08 .. 0.2 | - | 0.16 | 0.13 | 0.13 | normalised radius within which Collectors/Carriers flee Hunters (x12 = tiles) |
| evasion | `carrier_col_scale` | float | 0.0 .. 1.0 | - | - | 1.0 | 1.0 | Carrier's flee radius from enemy Collectors/Carriers, x threat_radius; <1 only |
| evasion | `carrier_hunter_scale` | float | 0.5 .. 1.0 | - | - | 1.0 | 1.0 | Carrier's flee radius from enemy Hunters, x threat_radius; <1 only |
| evasion | `camp_tiles` | float | 2.0 .. 5.0 | - | 3.0 | 3.0 | 3.0 | tiles: enemy Hunter this close to our base is camping the exit |
| evasion | `danger_tiles` | float | 2.0 .. 5.0 | - | 3.0 | 3.0 | 3.0 | tiles around a camping Hunter avoided when a route around exists |
| evasion | `breakout_units` | int | 1 .. 4 | - | 2 | 2 | 2 | blocked units waiting in the base that leave together |
| race | `opening_seconds` | float | 6.0 .. 20.0 | - | 10.0 | 10.0 | 10.0 | seconds in which a battery-holding walker still goes to the sanctuary |
| race | `commit_margin` | float | -0.1 .. 0.25 | - | 0.0 | 0.0 | 0.0 | seconds a committed walker must reach the sanctuary ahead of an enemy Hunter |
| race | `commit_reach` | float | 0.5 .. 2.0 | - | 1.0 | 1.0 | 1.0 | tiles from the sanctuary within which the last-step commit applies |
| race | `reserve_stage` | float | 0.0 .. 2.0 | - | 0.5 | 0.5 | 0.5 | tiles: the staged reserve walker waits this far from the sanctuary (0 = none) |
| race | `reserve_count` | int | 1 .. 2 | - | 1 | 1 | 1 | number of staged reserve walkers |
| race | `join_margin` | float | 0.0 .. 0.3 | - | 0.08 | 0.08 | 0.08 | tiles the enemy's lead may trail ours for the race to count as tied |
| race | `join_reach` | float | 0.5 .. 2.0 | - | 1.0 | 1.0 | 1.0 | tiles: enemy Collectors this close to its lead walker count as its reserve |
| race | `trade_radius` | float | 0.5 .. 3.0 | - | 1.0 | 1.0 | 1.0 | tiles: enemy Hunters this close to ours count as a pending trade |
| race | `trade_reach` | float | 0.5 .. 3.0 | - | 1.0 | 1.0 | 1.0 | tiles from the sanctuary within which our Collectors hold/advance during a trade |
| race | `trade_floor` | float | 0.05 .. 0.6 | - | 0.2 | 0.2 | 0.2 | tiles from the sanctuary where an advancing Collector stops |
| economy | `steal_weight` | float | 0.8 .. 3.0 | 1.8 | 1.8 | 1.8 | 1.8 | value multiplier of batteries stolen from enemy storage |
| economy | `exposure_cost` | float | 0.3 .. 2.0 | 0.9 | 0.9 | 0.9 | 0.9 | cost of leaving a deposit stealable for an absorption interval |
| economy | `carrier_field_value` | float | 10.0 .. 40.0 | 25.0 | 25.0 | 25.0 | 25.0 | field battery value below which no new Carrier is made |
| economy | `danger_hunter_radius` | float | 2.5 .. 6.0 | 4.0 | 4.0 | 4.0 | 4.0 | tiles: enemy Hunters this near a pickup discount its value |
| economy | `danger_collector_radius` | float | 1.5 .. 4.0 | 2.5 | 2.5 | 2.5 | 2.5 | tiles: enemy Collectors this near a pickup discount it for a Carrier |
| economy | `danger_weight` | float | 0.3 .. 0.9 | 0.6 | 0.6 | 0.6 | 0.6 | max discount of a pickup next to an enemy |
| economy | `contest_factor` | float | 0.5 .. 1.0 | 0.75 | 0.75 | 0.75 | 0.75 | value factor of a field battery an enemy reaches first |
| economy | `score_band` | float | 1.05 .. 1.3 | - | 1.12 | 1.12 | 1.12 | tasks within this score ratio of the best are kept as options |
| economy | `exclusion_margin` | float | 0.3 .. 2.0 | - | 1.0 | 1.0 | 1.0 | seconds a teammate must arrive earlier to take a shared task |
| economy | `deposit_band` | float | 0.1 .. 0.6 | - | 0.25 | 0.25 | 0.25 | deposit cost slack (s) within which storages are tied |
| economy | `idle_band` | float | 0.1 .. 0.6 | - | 0.3 | 0.3 | 0.3 | idle-patrol score slack within which enemy storages are tied |
| structural | `sweep` | bool | on/off | - | - | - | true | Hunters first strike enemy Collectors near the sanctuary |
| structural | `cargo_transform` | bool | on/off | - | - | - | true | battery-holding Collectors stay Hunter candidates all match |
| structural | `sweep_camper` | bool | on/off | - | - | - | true | the camper also sweeps (false keeps it on its post) |
| structural | `solo_sentry` | bool | on/off | - | true | true | false | a lone Hunter keeps the sanctuary (V30 role order) |
| structural | `race_commit` | bool | on/off | - | true | true | true | last-step commit of a walker into the sanctuary |
| structural | `late_hunter_quota` | int | -1 .. 4 | - | - | - | -1 | late Hunter refill target (-1 = off) |


## Which knobs matter after a Hunter nerf (starting points)

Sensitivity (`heur_variants.py sensitivity V32.1 --ref "V31;V19;V32.1"`, 2 maps x 2 sides, `reports/heur_variants/sens_v32_1.txt`):
largest effects on margin vs the V32.1 mirror: `commit_margin`, `exit_guards`, `sweep`/`sweep_camper`, `hunter_quota`,
`hunter_speed`, `adapt_at`, `score_band`, `exposure_cost`, `steal_weight`, `sentry_min_hunters`, `solo_sentry`,
`danger_tiles`, `threat_radius`.  Suggested first search group if Hunters get weaker: `hunter_quota`, `exit_guards`,
`guard_min_hunters`, `sweep_radius`, `sentry_*`, `adapt_k/adapt_at`, `threat_radius`, `carrier_*_scale`,
`opening_seconds`, `commit_*`; if Hunters get stronger/faster, also `hunter_speed`, `danger_hunter_radius`,
`danger_tiles`, `camp_tiles`.

## Tool

```
PYTHONPATH=. python examples/heur_variants.py sample V32.1 --n 8 --seed 0 [--scale 1.0] [--structural]
PYTHONPATH=. python examples/heur_variants.py run V32.1 --n 8 --seeds 16 --ref V31 --workers 4 --out reports/heur_variants/demo.json
PYTHONPATH=. python examples/heur_variants.py sensitivity V32.1 --seeds 2 --ref "V31;V19;V32.1" --workers 4
PYTHONPATH=. python examples/heur_hash.py check --golden tests/data/heuristic_hashes.json   # bit-exactness, 30 full matches
```

`run` plays each variant on both sides of every map vs `--ref` in the Python sim and reports win rate, mean margin,
*disagreement* (a shadow instance of the base is fed the variant's observations; share of unit-decisions whose heading
differs by > 22.5 degrees or stand/move differ) and `idle` (share of unit-decisions with no move; a frozen variant shows
up here).  `sensitivity` reports for each knob at its lo/hi value the share of games in which any action differed
(`fired`).

## Retuning after the patch

1. Record the new baseline: `heur_variants.py run V32.1 --n 0 --ref <strong opponents>` (or your own evaluation).
2. Start from the groups above; random search: sample with `scale` 0.5-1.5 around the best descriptor so far
   (`sample_params(best, rng, scale)`), evaluate N variants x paired seeds, keep the best, repeat (CEM-style: resample
   around the winner with shrinking scale).  Descriptors are plain strings, so results are directly usable as
   `--opponent-extra` / `act:` teacher descriptors.
3. If a knob hits its range limit, widen it in `params.KNOBS`; if the patch changes game rules (speeds, radii), also
   adjust the module constants named above.
4. Re-run `tests/test_heuristic_params.py`; the golden hashes only guard the *defaults* (they must be regenerated with
   `examples/heur_hash.py capture` if you intentionally change default behaviour).
