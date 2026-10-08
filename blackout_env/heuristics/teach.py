"""V4 as a relabelling teacher: consistent per-state action labels for states the learner visits.

DAgger-style relabelling asks a teacher "what would you do here?" in states the *learner* drove
into.  Played as a policy, V4 is a fine controller, but several of its outputs are not a
function of the observation, so the same state would get different answers:

* roles come from the unit's slot (team-local unit 0 = Carrier, 1 = Hunter): a Collector in
  slot 0/1 is always told to go and transform, whatever composition the learner chose;
* target assignment adds hysteresis from the previous target (``_economic_candidates``);
* movement follows an A* path cached for up to 10 ticks, so after the learner walked elsewhere
  the next waypoint can be stale;
* limit-cycle / stuck detection over the last 16-24 positions triggers 12-tick perpendicular
  escapes, and some fallbacks rotate with the tick counter;
* inside an arrival dead-zone it returns a zero vector, which an 8-way discrete action space
  cannot express (``direction_vector_to_idx`` would read it as "east").

A feed-forward learner cannot reproduce history-dependent labels, and a learner trained on them
sees one state labelled two ways.  Dropping the memory is not enough either: without its
hysteresis V4's global greedy matching re-sorts every unit whenever one unit moves, and its
storage choice compares straight-line distances while units walk paths.  Measured on Run 11
80k's trajectories (2026-09-18) that memoryless version switched a still-available target 28-67
times per unit-minute, up to 8.9 of them turning the label >= 90 degrees and turning back within
a second -- labels that would teach exactly the oscillation they are meant to cure.

``StrategicHeuristicV4Teach`` keeps V4's values and replaces the rest, so the label is a
function of the observation that also stays put while a unit follows it:

* role is read from the unit's current class (Collector -> collector, Hunter -> hunter,
  Carrier -> carrier), so by default no transformation is labelled: the learner owns the
  composition, the teacher shows how to execute the role each unit already has.  Opt-in
  ``target_hunters`` / ``target_carriers`` add transform labels that fill the team up to a
  composition from its *current* classes (_composition_goals), never from slots.  Run 15 (no
  composition labels) and Run 14 moved alike with composition equalised; Run 14's 25-point lead
  over it against V4 was its extra Hunters (2.0 vs 1.0), which it drifted into untaught;
* each unit starts from its own best battery/steal/special under V4's value, and gives one up
  only when a teammate's *own* best includes it too and that teammate arrives at least
  ``EXCLUSION_MARGIN`` seconds earlier.  Coordination is one level deep on purpose: global
  matching (V4's greedy without memory), claims by arrival time and soft "arriving later"
  penalties all let one unit's move re-sort everyone's targets, and flip-flopped labels 13-35
  times per unit-minute.  V4 itself only stays stable through hysteresis memory, which a
  feed-forward learner cannot see -- the same state then reads as either target.  Independent
  choices are self-reinforcing (walking toward a target raises its value at least as fast as any
  other's) but let every unit chase one battery: followed as a controller they lost to V4 by 45
  points vs 16 for full matching.  The margin leaves near-equal races alone (no flip at the
  boundary) and resolves clear ones: 25 points behind V4 with target-choice label cycles at
  0.05-2.0 per unit-minute, the level of V4's own memory-stabilised targets (0-2.2);
* every task within ``VALUE_BAND`` of the best (about one tile of travel at V4's 0.32 per tile)
  is kept and the label is the union of their directions -- hysteresis computed from the state
  alone: a near-tie no longer flips with each tile the unit crosses.  Measured on the same
  trajectories this matches V4's own memory-based hysteresis (0.0-1.2 direction flip-flops per
  unit-minute vs 0.0-0.9), and it removes ordering-based tie-breaks, which were what made the
  team-B mirror disagree;
* storage is chosen by V3's cost with *path* distance instead of straight-line distance, and
  the label heads for the whole storage cluster (slot choice inside it is not a direction);
  clusters within ``STORAGE_BAND`` tiles of the cheapest are kept the same way;
* a union whose directions roughly cancel (tied goals on opposite sides) is invalid: the teacher
  has no preference there, and teaching "both ways" is teaching the oscillation;
* the label is the *set* of compass moves that make the most shortest-path progress toward the
  goal from the unit's current tile (straight and diagonal moves tie when both lie on a
  shortest path; forcing one would be label noise);
* on a goal tile the label finishes the approach toward that tile's centre from the continuous
  position (pickups need contact, not just the tile), as V4's arrival logic does;
* V4's threat repulsion for cargo holders and Carriers, which depends only on current
  positions, is kept: when it changes the direction the label becomes that single direction;
* no goal, unreachable, or within V4's arrival radius (interacting) gives an invalid label:
  mask it out of the loss.

Labels are in the *world* frame of ``blackout_env.model.my_policy.DIRECTION_VECTORS`` (index i =
angle i*45 degrees, +x = column +1, +y = row -1), exactly like the continuous actions every
heuristic returns.  A trainer that stores team B rows in the mirrored canonical frame must map
label indices with the same ``mirror_direction_idx`` it applies to executed actions.

Played as a policy (``act``) it follows its own labels, which checks that the labels lead
somewhere; it has no escape logic and is not a stronger V4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..env.constants import unit_index
from . import _shared
from .spread_deposit import StrategicHeuristicV4
from .strategic import CARRIER, COLLECTOR, HUNTER, SITE_CARRIER, SITE_HUNTER, SPAWN_ENEMY, STORAGE_ALLY, WALL
from .v17_planner import StrategicHeuristicV17

N_DIRECTIONS = 8
# Compass index i -> (d_row, d_col); DIRECTION_VECTORS[i] = (cos(i*45), sin(i*45)) with +y = row - 1.
MOVES = tuple(
    (-int(round(math.sin(i * math.pi / 4))), int(round(math.cos(i * math.pi / 4))))
    for i in range(N_DIRECTIONS)
)
DIRECTIONS = np.array(
    [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(N_DIRECTIONS)], dtype=np.float32
)
_CLASS_ROLE = {COLLECTOR: "collector", HUNTER: "hunter", CARRIER: "carrier"}
VALUE_BAND = 0.3  # V4 value units; distance costs 0.32 per tile, so about one tile of travel
EXCLUSION_MARGIN = 1.0  # seconds a teammate must arrive earlier to take a shared best task
STORAGE_BAND = 1.0  # V3 storage cost units = tiles
_CANCELLING = 0.5  # |mean of the union's unit vectors| below this = tied goals on opposite sides
_PROGRESS_TOLERANCE = 1e-3
_ARRIVAL_RADIUS = 0.22  # StrategicHeuristic._navigate: closer than this counts as arrived

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a project dependency
    njit = None

# Speed (O1, docs/policy_v2_design_ko.md "O1"): a label mask has 256 possible values, so every
# quantity that depends on the mask alone is tabulated once, by the very expressions it replaces
# (bit-identical).  ``_BITS`` maps ``mask.tobytes()`` to the mask's bit pattern (bit i = move i).
_MASKS = np.array([[(bits >> i) & 1 for i in range(N_DIRECTIONS)] for bits in range(256)], dtype=bool)
_MASKS.flags.writeable = False
_BITS = {_MASKS[bits].tobytes(): bits for bits in range(256)}
_MEAN = [None] + [DIRECTIONS[_MASKS[bits]].mean(axis=0) for bits in range(1, 256)]  # DIRECTIONS[mask].mean(axis=0)
_MEAN_NORM = [math.nan] + [float(np.linalg.norm(_MEAN[bits])) for bits in range(1, 256)]
with np.errstate(invalid="ignore", divide="ignore"):  # cancelling masks are never valid labels
    _ACTION = [None] + [(_MEAN[bits] / float(np.linalg.norm(_MEAN[bits]))).astype(np.float32) for bits in range(1, 256)]
for _v in _MEAN[1:] + _ACTION[1:]:
    _v.flags.writeable = False
_SQRT2 = math.sqrt(2.0)
_MOVES_DR = np.array([m[0] for m in MOVES], dtype=np.int64)
_MOVES_DC = np.array([m[1] for m in MOVES], dtype=np.int64)
_EVASION_SLACK = 1e-4  # tiles/12: float64 pre-filter margin over the float32 distance (error < 1e-6)


def _progress_bits_py(distance, pos_r, pos_c, walkable, dr, dc, root2, tolerance):
    """``TeachLabeller._progress_moves`` as a bit pattern (the reference the kernel must match)."""
    h, w = distance.shape
    here = float(distance[pos_r, pos_c])
    if not math.isfinite(here) or here <= 0.0:
        return 0
    progress = [-math.inf] * 8
    for i in range(8):
        nr, nc = pos_r + dr[i], pos_c + dc[i]
        if not (0 <= nr < h and 0 <= nc < w) or not walkable[nr, nc]:
            continue
        if dr[i] != 0 and dc[i] != 0 and not (walkable[pos_r, nc] and walkable[nr, pos_c]):
            continue
        step = root2 if dr[i] != 0 and dc[i] != 0 else 1.0
        progress[i] = (here - float(distance[nr, nc])) / step
    best = -math.inf
    for i in range(8):
        if progress[i] > best:
            best = progress[i]
    bits = 0
    if math.isfinite(best) and best > 0.0:
        for i in range(8):
            if progress[i] >= best - tolerance:
                bits |= 1 << i
    return bits


_progress_bits = njit(cache=True)(_progress_bits_py) if njit is not None else _progress_bits_py

Goal = tuple[tuple[tuple[int, int], ...], str]  # (goal tiles, kind): progress toward the nearest tile


@dataclass(frozen=True)
class TeachLabel:
    """One unit's label. ``mask[i]`` marks compass move i as correct; all False when invalid.

    ``target`` is a representative goal tile; ``goals`` holds every tied goal the mask covers.
    """

    mask: np.ndarray
    valid: bool
    kind: str
    target: tuple[int, int] | None
    goals: tuple[Goal, ...] = ()
    evading: bool = False


def _invalid(goals: tuple[Goal, ...] = ()) -> TeachLabel:
    kind = goals[0][1] if goals else ""
    target = goals[0][0][0] if goals else None
    return TeachLabel(np.zeros(N_DIRECTIONS, dtype=bool), False, kind, target, goals)


def label_action(label: TeachLabel) -> np.ndarray:
    """The continuous action that follows ``label``: the mean of its directions, or standing
    still where the teacher has no label (interacting, nowhere to go)."""
    if not label.valid:
        return np.zeros(2, dtype=np.float32)
    bits = _BITS.get(label.mask.tobytes()) if label.mask.dtype == np.bool_ and label.mask.shape == (N_DIRECTIONS,) else None
    if bits:
        return _ACTION[bits].copy()  # the expression below, tabulated
    vector = DIRECTIONS[label.mask].mean(axis=0)  # never cancels: see _CANCELLING
    return (vector / float(np.linalg.norm(vector))).astype(np.float32)


class TeachLabeller:
    """Goal -> label machinery shared by the relabelling teachers; mix in before a heuristic.

    A teacher implements ``teach(obs) -> {name: TeachLabel}``: it picks goals as a function of the
    observation alone and turns each unit's goals into a label with ``_label``.  ``act`` follows
    the labels, which checks that they lead somewhere."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._teach_distance_cache: dict[tuple[bytes, tuple[tuple[int, int], ...]], np.ndarray] = {}
        self._teach_walkable: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._teach_distance_cache.clear()
        self._teach_walkable = None
        self._o1_memo = None
        self._enemy_rows_entry = None

    def teach(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, TeachLabel]:
        raise NotImplementedError

    def label_masks(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """``teach`` reduced to its ``[8]`` bool masks (all False = no label)."""
        return {name: label.mask for name, label in self.teach(obs).items()}

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {name: label_action(label) for name, label in self.teach(obs).items()}

    def _composition_targets(self, graphic: np.ndarray, states: np.ndarray, row_for: dict[str, int]) -> tuple[int, int]:
        """(Hunters, Carriers) the team should have; (0, 0) = no composition labels."""
        return 0, 0

    def _composition_goals(
        self,
        controlled: list[str],
        row_for: dict[str, int],
        states: np.ndarray,
        graphic: np.ndarray,
        walkable: np.ndarray,
    ) -> dict[str, tuple[Goal, ...]]:
        """Transform labels that fill the team up to ``_composition_targets``.

        V4 decides roles by slot (V17 by slot and memory), which conflicts with whatever composition the learner chose.  This
        reads the team's current classes instead: for each missing Carrier (the base sanctuary only
        works while none is alive) and each missing Hunter, the empty Collector with the shortest
        path to that sanctuary is sent there, closest first, ties by unit index.  Walking toward it
        only shortens its own path, so the choice holds; a pickup on the way takes it out of the
        running (cargo is deposited first, as in V4).  A surplus is never labelled -- a unit only
        changes class back by dying."""
        target_hunters, target_carriers = self._composition_targets(graphic, states, row_for)
        if not target_hunters and not target_carriers:
            return {}
        shape = graphic.shape[:2]
        classes = {name: self._class_id(states[row_for[name]]) for name in controlled}
        candidates = [
            name for name in controlled
            if classes[name] == COLLECTOR and not self._is_holding(states[row_for[name]])
        ]
        goals: dict[str, tuple[Goal, ...]] = {}
        for role, channel, target in ((CARRIER, SITE_CARRIER, target_carriers), (HUNTER, SITE_HUNTER, target_hunters)):
            missing = (target or 0) - sum(cls == role for cls in classes.values())
            if missing <= 0 or not candidates:
                continue
            # the enemy's own sanctuary sits inside its base, where we cannot walk
            tiles = self._site_tiles(graphic, channel, walkable)
            if not tiles:
                continue
            distance = self._distance_to(walkable, tiles)
            reach = {name: float(distance[self._to_pixel(states[row_for[name]][:2], shape)]) for name in candidates}
            ranked = sorted((name for name in candidates if math.isfinite(reach[name])),
                            key=lambda name: (reach[name], unit_index(name)))
            for name in ranked[:missing]:
                goals[name] = ((tiles, "transform"),)
                candidates.remove(name)
        return goals


    def _label(
        self,
        state: np.ndarray,
        states: np.ndarray,
        goals: tuple[Goal, ...],
        walkable: np.ndarray,
        shape: tuple[int, int],
    ) -> TeachLabel:
        if not goals:
            return _invalid()
        pos = self._to_pixel(state[:2], shape)
        for tiles, kind in goals:
            if pos in tiles:
                # On a goal tile but pickups/deposits/transforms need contact near its centre:
                # finish the approach from the continuous position, V4's arrival radius.
                row = (1.0 - float(state[1])) * 0.5 * shape[0] - 0.5
                col = (float(state[0]) + 1.0) * 0.5 * shape[1] - 0.5
                toward = np.array([pos[1] - col, row - pos[0]], dtype=np.float32)
                if float(np.linalg.norm(toward)) <= _ARRIVAL_RADIUS:
                    return _invalid(goals)  # interacting; any label here is noise
                single = np.zeros(N_DIRECTIONS, dtype=bool)
                single[int(np.argmax(DIRECTIONS @ toward))] = True
                return TeachLabel(single, True, kind, pos, goals)

        bits = 0
        for tiles, _ in goals:
            bits |= self._progress_bits(self._distance_to(walkable, tiles), pos, walkable, shape)
        if not bits or _MEAN_NORM[bits] < _CANCELLING:
            return _invalid(goals)  # nowhere to go, or tied goals pulling opposite ways
        kind, target = goals[0][1], goals[0][0][0]
        evade = self._evasion(state, states, _MEAN[bits])
        if evade is not None and not (bits >> evade) & 1:
            single = np.zeros(N_DIRECTIONS, dtype=bool)
            single[evade] = True
            return TeachLabel(single, True, kind, target, goals, evading=True)
        return TeachLabel(_MASKS[bits].copy(), True, kind, target, goals)

    @staticmethod
    def _progress_bits(distance: np.ndarray, pos: tuple[int, int], walkable: np.ndarray, shape: tuple[int, int]) -> int:
        """``_progress_moves`` as a bit pattern (bit i = compass move i), by the numba kernel."""
        if distance.dtype != np.float32 or walkable.dtype != np.bool_ or distance.shape != tuple(shape):
            return int(np.packbits(TeachLabeller._progress_moves(distance, pos, walkable, shape), bitorder="little")[0])
        return _progress_bits(distance, pos[0], pos[1], walkable, _MOVES_DR, _MOVES_DC, _SQRT2, _PROGRESS_TOLERANCE)

    @staticmethod
    def _progress_moves(
        distance: np.ndarray, pos: tuple[int, int], walkable: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """Compass moves with the most shortest-path progress per tile walked (none if stuck)."""
        mask = np.zeros(N_DIRECTIONS, dtype=bool)
        here = float(distance[pos])
        if not math.isfinite(here) or here <= 0.0:
            return mask
        progress = np.full(N_DIRECTIONS, -np.inf, dtype=np.float64)
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < shape[0] and 0 <= nc < shape[1]) or not walkable[nr, nc]:
                continue
            if dr and dc and not (walkable[pos[0], nc] and walkable[nr, pos[1]]):
                continue  # no corner cutting, same rule as the distance map and Unity's walls
            step = math.sqrt(2.0) if dr and dc else 1.0
            progress[i] = (here - float(distance[nr, nc])) / step
        best = float(progress.max())
        if math.isfinite(best) and best > 0.0:
            mask = progress >= best - _PROGRESS_TOLERANCE
        return mask

    evade_empty_collectors = False  # V4 (and V17) only steer cargo and Carriers away from Hunters

    def _evasion(self, state: np.ndarray, states: np.ndarray, delta: np.ndarray) -> int | None:
        """V4's threat repulsion (StrategicHeuristic._navigate), from current positions only."""
        cls = self._class_id(state)
        if not (self._is_holding(state) or cls == CARRIER or (self.evade_empty_collectors and cls == COLLECTOR)):
            return None
        # Enemies clearly outside the threat radius (float64 distance beyond it by a margin far above
        # the float32 rounding of the exact test) add nothing: only the others take the exact path.
        radius = self.threat_radius
        sx, sy = float(state[0]), float(state[1])
        near = [row for row, hunter, ex, ey in self._enemy_rows(states)
                if (hunter or cls == CARRIER) and math.hypot(sx - ex, sy - ey) < radius + _EVASION_SLACK]
        if not near:
            return None
        total = delta.astype(np.float32).copy()
        repelled = False
        for row in near:
            enemy = states[row]
            away = state[:2] - enemy[:2]
            dist = float(np.sqrt(away.dot(away)))
            if 1e-5 < dist < self.threat_radius:
                total += (away / dist) * ((self.threat_radius - dist) / self.threat_radius) * 9.0
                repelled = True
        if not repelled or float(np.linalg.norm(total)) < 1e-6:
            return None
        return int(np.argmax(DIRECTIONS @ total))

    def _enemy_rows(self, states: np.ndarray) -> list[tuple[int, bool, float, float]]:
        """(row, is Hunter, x, y) of every enemy row of ``states`` (``states[:, 2] < 0``), per array."""
        entry = getattr(self, "_enemy_rows_entry", None)
        tick = getattr(self, "_tick", None)
        if entry is not None and entry[0] is states and entry[2] == tick:
            return entry[1]
        rows = [(i, self._class_id(states[i]) == HUNTER, float(states[i, 0]), float(states[i, 1]))
                for i in np.flatnonzero(states[:, 2] < 0).tolist()]
        self._enemy_rows_entry = (states, rows, tick)
        return rows

    def _tick_memo(self) -> dict:
        """A dict that lives for the current decision tick (``_tick``): per-tick memos of pure
        functions of arrays held by reference (identity keys stay valid while the entry lives)."""
        tick = getattr(self, "_tick", None)
        memo = getattr(self, "_o1_memo", None)
        if memo is None or memo[0] != tick or len(memo[1]) > 4096:
            memo = self._o1_memo = (tick, {})
        return memo[1]

    def _site_tiles(self, graphic: np.ndarray, channel: int, walkable: np.ndarray) -> tuple[tuple[int, int], ...]:
        """``tuple((int(y), int(x)) for y, x in np.argwhere((graphic[..., channel] > 0.5) & walkable))`` (row-major,
        i.e. already sorted), memoised per tick and array pair."""
        memo = self._tick_memo()
        key = ("site_tiles", id(graphic), channel, id(walkable))
        hit = memo.get(key)
        if hit is not None and hit[0] is graphic and hit[1] is walkable:
            return hit[2]
        tiles = tuple((int(y), int(x)) for y, x in np.argwhere((graphic[..., channel] > 0.5) & walkable))
        memo[key] = (graphic, walkable, tiles)
        return tiles

    def _distance_to(self, walkable: np.ndarray, tiles: tuple[tuple[int, int], ...] | tuple[int, int]) -> np.ndarray:
        """Path distance from every cell to the nearest of ``tiles`` (one tile or a cluster)."""
        memo = self._tick_memo()
        try:
            mkey = ("teach_distance_to", id(walkable), tiles, getattr(self, "field_goals", None))
            hit = memo.get(mkey)
        except TypeError:  # unhashable tiles
            mkey, hit = None, None
        if hit is not None and hit[0] is walkable:
            return hit[1]
        cached = self._teach_distance_to(walkable, tiles)
        if mkey is not None:
            memo[mkey] = (walkable, cached)
        return cached

    def _teach_distance_to(self, walkable: np.ndarray, tiles) -> np.ndarray:
        if tiles and isinstance(tiles[0], (int, np.integer)):
            tiles = (tuple(tiles),)
        key = (walkable.tobytes(), tuple(sorted(tiles)))
        cached = self._teach_distance_cache.get(key)
        if cached is None:
            # 8-neighbour, no corner cutting => symmetric, so a map from a tile is a map to it.
            fn = self._distance_map
            shared_key = ("goal", getattr(fn, "__func__", fn), walkable.shape, key[0], key[1])
            cached = _shared.GOAL_FIELDS.get(shared_key)
            if cached is None:
                cached = _shared.GOAL_FIELDS.put(shared_key, _shared.frozen(
                    np.minimum.reduce([fn(walkable, tile) for tile in key[1]])))
            if len(self._teach_distance_cache) > 4096:
                self._teach_distance_cache.clear()
            self._teach_distance_cache[key] = cached
        return cached

    def _enemy_base_mask(self, graphic: np.ndarray) -> np.ndarray:
        cached = self._static_masks.get("teach_enemy_base")
        if cached is None:
            spawn = np.argwhere(graphic[..., SPAWN_ENEMY] > 0.5)
            origin = tuple(int(v) for v in spawn[0]) if len(spawn) else None
            cached = StrategicHeuristicV17._base_mask(origin, graphic.shape[:2])
            self._static_masks["teach_enemy_base"] = cached
        return cached


class StrategicHeuristicV4Teach(TeachLabeller, StrategicHeuristicV4):
    """V4's values as a memoryless, role-conditioned, motion-stable, set-valued action labeller."""

    def __init__(self, *, target_hunters: int | None = None, target_carriers: int | None = None, **kwargs):
        """target_hunters / target_carriers: opt-in composition labels (see _composition_goals);
        None leaves the composition entirely to the learner."""
        super().__init__(**kwargs)
        self.target_hunters = target_hunters
        self.target_carriers = target_carriers

    def _composition_targets(self, graphic: np.ndarray, states: np.ndarray, row_for: dict[str, int]) -> tuple[int, int]:
        return self.target_hunters or 0, self.target_carriers or 0

    # ------------------------------------------------------------------------------------ labels

    def teach(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, TeachLabel]:
        """Labels for every agent in ``obs`` (one team's observations, as ``act`` receives them)."""
        if not obs:
            return {}
        sample = next(iter(obs.values()))
        graphic, states, team_state = sample["graphic"], sample["agent_states"], sample["team_state"]

        # Episode boundary (remaining time jumps up), as in StrategicHeuristic.act.  Only static
        # caches (walls, storage components, distance maps) survive within an episode.
        time_left = float(team_state[2])
        if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
            self.reset()
        self._last_time_left = time_left
        self._tick += 1  # refreshes the per-tick battery/special scans
        self._memory.clear()  # no assignment hysteresis: the answer depends on this state only

        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}
        local_order = {name: j for j, name in enumerate(controlled)}
        # V4 values tasks on the wall channel alone; our own movement cannot enter the enemy base.
        walkable = graphic[..., WALL] < 0.5
        move_walkable = walkable & ~self._enemy_base_mask(graphic)
        self._teach_walkable = move_walkable
        shape = graphic.shape[:2]

        transforms = self._composition_goals(controlled, row_for, states, graphic, move_walkable)
        economic = [
            name for name in controlled
            if name not in transforms
            and not self._is_holding(states[row_for[name]]) and self._class_id(states[row_for[name]]) != HUNTER
        ]
        claims = self._economic_goals(economic, row_for, states, graphic, team_state, walkable)
        reservations: set[tuple[int, int]] = {tile for goals in claims.values() for tiles, _ in goals for tile in tiles}

        labels: dict[str, TeachLabel] = {}
        for name in controlled:
            state = states[row_for[name]]
            if name in transforms:
                goals = transforms[name]
            elif name in claims:
                goals = claims[name]
            elif self._is_holding(state):
                goals = self._storage_goals(graphic, state, states, team_state)
            else:
                target, kind = self._choose_target(
                    _CLASS_ROLE[self._class_id(state)], state, states, graphic, reservations,
                    local_order[name], team_state,
                )
                goals = () if target is None or kind == "transform" else (((tuple(target),), kind),)
                if target is not None and kind in {"battery", "special", "steal"}:
                    reservations.add(tuple(target))
            labels[name] = self._label(state, states, goals, move_walkable, shape)
        return labels

    # --------------------------------------------------------------------------------- decisions

    def _economic_goals(
        self,
        names: list[str],
        row_for: dict[str, int],
        states: np.ndarray,
        graphic: np.ndarray,
        team_state: np.ndarray,
        walkable: np.ndarray,
    ) -> dict[str, tuple[Goal, ...]]:
        """Each unit's best tasks under V4's value, minus the ones a clearly faster teammate wants.

        "Wants" is the teammate's own unfiltered choice, so exclusions never cascade."""
        tasks, pairs, distance_of = self._economic_candidates(names, row_for, states, graphic, team_state, walkable)
        by_unit: dict[str, list[tuple[float, int]]] = {}
        for value, name, task in pairs:
            by_unit.setdefault(name, []).append((value, task))
        speed = {name: 6.0 if self._class_id(states[row_for[name]]) == CARRIER else 4.0 for name in by_unit}

        def best(scored: list[tuple[float, int]]) -> list[int]:
            top = max(value for value, _ in scored)
            return [task for value, task in scored if value >= top - VALUE_BAND]

        wanted = {name: set(best(scored)) for name, scored in by_unit.items()}
        goals = {}
        for name, scored in by_unit.items():
            arrival = {task: distance_of[(name, task)] / speed[name] for _, task in scored}
            taken = {
                task
                for other, tasks_wanted in wanted.items() if other != name
                for task in tasks_wanted
                if task in arrival and distance_of[(other, task)] / speed[other] + EXCLUSION_MARGIN <= arrival[task]
            }
            free = [(value, task) for value, task in scored if task not in taken]
            if free:
                goals[name] = tuple(((tuple(tasks[task][0]),), tasks[task][1]) for task in best(free))
        return goals

    def _storage_goals(
        self, graphic: np.ndarray, state: np.ndarray, states: np.ndarray, team_state: np.ndarray
    ) -> tuple[Goal, ...]:
        """V3's storage choice by path distance, as whole clusters, keeping exact ties."""
        origin = self._to_pixel(state[:2], graphic.shape[:2])
        components = self._cached_components(graphic[..., STORAGE_ALLY] > 0.5)
        if not components:
            return ()
        available = self._fitting_storage_components(graphic, state, components)
        if not available:
            target, can_deposit = self._storage_target(graphic, state, origin)
            if target is None:
                return ()
            return (((tuple(target),), "deposit" if can_deposit else "wait_storage"),)
        costs = self._storage_component_costs(graphic, state, states, origin, team_state, components, available)
        best = min(cost for cost, _ in costs)
        return tuple((tuple(component), "deposit") for cost, component in costs if cost <= best + STORAGE_BAND)

    def _storage_distance(
        self, graphic: np.ndarray, origin: tuple[int, int], component: list[tuple[int, int]]
    ) -> float:
        # Path distance: straight-line distance can grow while a unit walks around a wall toward
        # the chosen store and shrink toward another, flipping the choice mid-trip.
        walkable = self._teach_walkable if self._teach_walkable is not None else graphic[..., WALL] < 0.5
        return float(self._distance_to(walkable, tuple(component))[origin])
