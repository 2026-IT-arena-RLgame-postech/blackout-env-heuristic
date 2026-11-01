"""V34: V33 for the balance patch (max 2 living Hunters per team, spawn-side walls; docs: reports/bp).

Post-patch matches are decided in the first ~20 s by the Hunter war.  Both sides' two race walkers reach the
sanctuary on the same tick (~4 s) and trade Hunter for Hunter; the side whose *next* Collector enters first then owns
the sanctuary and kills ~10 Collectors.  So the count of Collectors committed near the sanctuary decides, and the
Carrier made at 0.4 s is one walker fewer (it can never become a Hunter).  New knobs:

* ``carrier_delay`` (s, 0 = off): no Carrier before this match time, unless ``carrier_on_monopoly`` and we hold a
  Hunter monopoly (a Hunter of ours out, none of theirs out of base) -- the would-be Carrier joins the Hunter war.
* ``reserve_count`` may be 3 (V27's staged reserve walkers).
* ``cap_sync`` (off): the sanctuary transforms a Collector only when it *enters* the region and only while the team
  has fewer than 2 Hunters.  A reserve walker that steps in on the same tick as our race walkers is refused (they
  are still alive) and then stands inside as a Collector until it is killed -- with stacked walkers that loses the
  second race (and gives team B, whose units move after A's, a large edge).  With ``cap_sync`` at most
  ``2 - our Hunters`` of our Collectors may come within ``cap_enter`` tiles of the sanctuary per decision (nearest
  first); the others hold.  A Collector already inside without having transformed steps out to re-enter.
* ``camp_path`` (off): V20's camp danger zone and V21's Hunter-free routes use a disc of ``danger_tiles`` around each
  enemy Hunter.  After the patch the y = 18 wall row (x 5..16) and our base (closed to enemies) put our east exit
  (y 19-20) ~20 path tiles from a Hunter camping y = 17 that is 2 tiles away in a straight line, so the disc
  closes the one safe exit.  With ``camp_path`` the zone is the cells within ``danger_tiles`` *path* tiles of the
  Hunter on the map it can walk (walls and our base blocked).
* ``east_post`` (tiles, 0 = off): while we have 2 Hunters and the enemy none out of base, the Hunter nearest the
  enemy base's *east* exit (the y 19-20 corridor in team A's frame, which a Hunter camping y = 17 cannot reach
  quickly) holds a post ``east_post`` tiles into that corridor, engaging enemies within ``east_engage`` tiles.
* ``hunter_roles`` ("" = off; e.g. ``raid/raid``, ``south/raid``, ``south/east``, ``default/raid``): explicit jobs for our
  Hunters (first role to the Hunter best placed for it, ...).  ``default`` = the inherited behaviour (sanctuary
  sentry, spawn camp, chases), ``raid`` = hunt the enemy's battery supply: chase the enemy Collector/Carrier out of
  its base with the best (cargo bonus ``raid_value``) / path-distance score anywhere on the map, ``south`` / ``east`` =
  hold the enemy base's south (y = 17 row) / east (y 19-20 corridor) exit and strike within ``east_engage``.
  ``roles_when``: ``free`` = while the enemy has no Hunter out of base, ``always``.
* ``race_search`` (0 = off, else the max delay in decisions): when Collectors (ours or theirs) are within
  ``rs_trigger`` tiles of the Hunter sanctuary, the exact simulator is built from the observation and every
  combination of entry delays 0..race_search for our near Collectors (<= 4) is rolled out ``rs_horizon`` decisions
  against two opponent models (walk straight in / respect the Hunter cap); the combination with the best mean
  (our Hunters - theirs, then Collectors lost) is played: a delayed unit stands this decision.  Ties keep the
  inherited moves.
* ``storm_units`` (0 = off): after the opening (``storm_after`` s), while the enemy has more Hunters out of base than
  we have, up to this many empty Collectors (nearest the sanctuary first) are *committed* walkers to the Hunter
  sanctuary: no threat evasion, plain-map path.  A Collector that reaches the sanctuary turns into a Hunter on entry
  and trades with the enemy sentry next to it -- a Collector for a Hunter, which breaks the enemy's monopoly.

Defaults = V33 exactly.
"""

from __future__ import annotations

import numpy as np

from ..env.constants import MAX_HUNTERS, unit_index
from .any_angle import UNIT_RADIUS, AnyAngleLabel, pixel_position, world_heading
from .teach import _invalid
from .strategic import BATTERY, CARRIER, COLLECTOR, HUNTER, SITE_HUNTER, STORAGE_ENEMY
from .v17_planner import MATCH_SECONDS
from .v33_specials import StrategicHeuristicV33


def _grid_field(walk, sources):
    """Octile path distance (tiles) from a set of cells over ``walk`` (no corner cutting); inf where unreachable."""
    import heapq
    H_, W_ = walk.shape
    dist = np.full((H_, W_), np.inf)
    heap = []
    for q in sources:
        dist[q] = 0.0
        heap.append((0.0, q[0], q[1]))
    heapq.heapify(heap)
    r2 = 2 ** 0.5
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H_ and 0 <= nc < W_) or not walk[nr, nc]:
                continue
            if dr and dc:
                if not (walk[r + dr, c] and walk[r, c + dc]):
                    continue
                nd = d + r2
            else:
                nd = d + 1.0
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


class StrategicHeuristicV34(StrategicHeuristicV33):
    def __init__(self, *, carrier_delay: float = 0.0, carrier_on_monopoly: bool = True, cap_sync: bool = False,
                 cap_enter: float = 0.3, storm_units: int = 0, storm_after: float = 6.0, race_second: bool = False,
                 rs_radius: float = 1.5, rs_wait: float = 0.6, early_reserve: float = 0.0,
                 carrier_retreat: float = 0.0, camp_path: bool = False, east_post: float = 0.0,
                 east_engage: float = 3.0, hunter_roles: str = "", roles_when: str = "free", raid_value: float = 2.0,
                 raid_carrier: float = 1.0, raid_lead: float = 0.0, race_search: int = 0, rs_horizon: int = 10,
                 rs_trigger: float = 1.2, stage_by_name: bool = False, gate_split: bool = False,
                 gate_wait: float = 0.3, gate_tol: float = 0.4, gate_detour: float = 0.3, gate_commit: float = 4.0,
                 gate_breakout: int = 2, gate_return: float = 0.0, raid_idle: str = "", raid_sticky: float = 1.0, bait: float = 0.0,
                 bait_reach: float = 5.0, refill_reach: float = 0.0, refill_after: float = 6.0,
                 refill_safe: bool = True, refill_margin: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.refill_reach = float(refill_reach)
        self.refill_after = float(refill_after)
        self.refill_safe = bool(refill_safe)
        self.refill_margin = float(refill_margin)
        self.stage_by_name = bool(stage_by_name)
        self.gate_split = bool(gate_split)
        self.raid_idle = str(raid_idle)
        self.bait = float(bait)
        self.bait_reach = float(bait_reach)
        self.raid_sticky = float(raid_sticky)
        self._raid_target: dict[str, int] = {}
        self.gate_wait = float(gate_wait)
        self.gate_tol = float(gate_tol)
        self.gate_detour = float(gate_detour)
        self.gate_commit = float(gate_commit)
        self.gate_breakout = int(gate_breakout)
        self.gate_return = float(gate_return)
        self._gate_of: dict[str, int] = {}
        self._gate_cache: dict = {}
        self._stage_hold: set[str] = set()
        self.race_search = int(race_search)
        self.rs_horizon = int(rs_horizon)
        self.rs_trigger = float(rs_trigger)
        self.raid_lead = float(raid_lead)
        self._prev_enemy: dict[int, tuple[float, float]] = {}
        self.raid_carrier = float(raid_carrier)
        self.hunter_roles = [r for r in str(hunter_roles).replace("+", "/").split("/") if r]
        self.roles_when = str(roles_when)
        self.raid_value = float(raid_value)
        self.east_post = float(east_post)
        self.east_engage = float(east_engage)
        self.camp_path = bool(camp_path)
        self.carrier_retreat = float(carrier_retreat)
        self.early_reserve = float(early_reserve)
        self.race_second = bool(race_second)
        self.rs_radius = float(rs_radius)
        self.rs_wait = float(rs_wait)
        self._rs_since = None
        self.storm_units = int(storm_units)
        self.storm_after = float(storm_after)
        self.cap_sync = bool(cap_sync)
        self.cap_enter = float(cap_enter)
        self._cap_hold: set[str] = set()
        self._cap_out: dict[str, np.ndarray] = {}
        self.carrier_delay = float(carrier_delay)
        self.carrier_on_monopoly = bool(carrier_on_monopoly)

    def _composition_targets(self, graphic, states, row_for):
        hunters, carriers = super()._composition_targets(graphic, states, row_for)
        if carriers and self.carrier_delay > 0:
            ctx = self._teach_ctx
            if MATCH_SECONDS - ctx["remaining"] < self.carrier_delay:
                ours = sum(u.cls == HUNTER for u in ctx["own"])
                theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
                if not (self.carrier_on_monopoly and ours > 0 and theirs == 0):
                    carriers = 0
        return hunters, carriers

    # ------------------------------------------------------------------ refill

    def _refill(self, controlled, row_for, states, graphic, walkable, goals):
        """RL2 second-wave rule (reports/bp/NOTES.md): refill the Hunter quota *while an enemy Hunter is out*, also at
        1 v 1.  V21's composition drops every route that passes within ``danger_tiles`` of an enemy Hunter and only
        the last-step commit (``commit_reach`` tiles) overrides that, so a side that is down to 0-1 Hunters stays
        there (the refill rate at 1 v 1 is 0.2 per 100 ticks for B4 vs 0.9 for RL2).  Here, after ``refill_after`` s,
        the Collectors (no special item) nearest the sanctuary by exact path, within ``refill_reach`` tiles, are
        sent in and committed (no threat evasion, plain-map path) for as many Hunters as we are short of the quota,
        when (``refill_safe``) they get there ``refill_margin`` s before an enemy Hunter running straight at them
        could touch them."""
        ctx = self._teach_ctx
        if MATCH_SECONDS - ctx["remaining"] < self.refill_after:
            return
        enemies = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not enemies:
            return
        own = sum(u.cls == HUNTER for u in ctx["own"])
        tiles = tuple(sorted((int(y), int(x)) for y, x in np.argwhere((graphic[..., SITE_HUNTER] > 0.5) & walkable)))
        if not tiles:
            return
        tile_set = set(tiles)
        want = min(self.hunter_quota, MAX_HUNTERS) - own
        want -= sum(1 for g in goals.values() if any(kind == "transform" and set(t) & tile_set for t, kind in g))
        if want <= 0:
            return
        from .v22_teacher import CONTACT_RADIUS
        shape = graphic.shape[:2]
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        v_me = self._speed(ctx, COLLECTOR)
        v_them = self._speed(ctx, HUNTER, enemy=True)
        contact = CONTACT_RADIUS[HUNTER] * ctx["enemy_size"] + CONTACT_RADIUS[COLLECTOR] * ctx["own_size"]
        scored = []
        for name in controlled:
            state = states[row_for[name]]
            if self._class_id(state) != COLLECTOR or name in self.committed or float(state[5:9].max()) >= 0.5:
                continue
            if self._is_holding(state):
                continue
            here = pixel_position(state, shape)
            d = planner.distance(here)
            if not (0.0 < d <= self.refill_reach):
                continue
            if self.refill_safe:
                arrive = d / v_me + self.refill_margin
                caught = min((float(np.hypot(*np.subtract(pixel_position(states[e.row], shape), here))) - contact)
                             / (v_them + v_me) for e in enemies)
                if not arrive < caught:
                    continue
            scored.append((d, unit_index(name), name, state))
        for _, _, name, state in sorted(scored)[:want]:
            goals[name] = ((tiles, "transform"),)
            self.committed.append(name)
            self._committed_xy.add((float(state[0]), float(state[1])))
            self._count("refill")

    # ------------------------------------------------------------------ cap_sync

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        if self.refill_reach > 0:
            self._refill(controlled, row_for, states, graphic, walkable, goals)
        if self.storm_units > 0:
            self._storm(controlled, row_for, states, graphic, walkable, goals)
        if self.early_reserve > 0:
            self._early_reserve(controlled, row_for, states, graphic, walkable, goals)
        self._stage_hold = set()
        if self.stage_by_name and self._staged_xy:
            self._stage_holds(controlled, row_for, states, graphic, walkable)
        self._cap_hold, self._cap_out = set(), {}
        if not self.cap_sync:
            return goals
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return goals
        shape = graphic.shape[:2]
        tile_set = set(tiles)
        hunters = sum(self._class_id(states[row_for[n]]) == HUNTER for n in controlled)
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        cache = self.__dict__.setdefault("_cap_centres", {})  # the sanctuary tiles repeat tick after tick
        centre = cache.get(tiles)
        if centre is None:
            if len(cache) >= 64:
                cache.clear()
            centre = cache[tiles] = np.mean(np.asarray(tiles, np.float32), axis=0)
        near = []
        for n in controlled:
            st = states[row_for[n]]
            if self._class_id(st) != COLLECTOR or n not in goals:
                continue
            if not any(kind == "transform" and set(t) & tile_set for t, kind in goals[n]):
                continue
            p = pixel_position(st, shape)
            cell = (int(np.floor(p[0] + 0.5)), int(np.floor(p[1] + 0.5)))  # cell centres at integers
            xy = (float(st[0]), float(st[1]))
            if cell in tile_set:  # inside without having transformed: step out to re-enter
                away = np.asarray(p, np.float32) - centre
                norm = float(np.hypot(*away))
                self._cap_out[n] = world_heading(*(away / norm)) if norm > 1e-6 else np.array([0.0, 1.0], np.float32)
                continue
            d = planner.distance(p)
            if np.isfinite(d) and d <= self.cap_enter:
                near.append((d, self._rk(n), n))
        slots = max(0, MAX_HUNTERS - hunters)
        if self.race_second and near:
            # let the enemy's walkers transform first (then ours enter onto them: trades), at most rs_wait s
            ctx = self._teach_ctx
            now = MATCH_SECONDS - ctx["remaining"]
            theirs_h = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
            coming = any(e.cls == COLLECTOR and planner.distance(pixel_position(states[e.row], shape)) <= self.rs_radius
                         for e in ctx["enemies"])
            if theirs_h == 0 and coming:
                if self._rs_since is None:
                    self._rs_since = now
                if now - self._rs_since < self.rs_wait:
                    slots = 0
            else:
                self._rs_since = None
        for _, _, n in sorted(near)[slots:]:
            self._cap_hold.add(n)
        return goals

    # ------------------------------------------------------------------ gate_split

    def _labels(self, ctx, goals, move_walkable):
        labels = super()._labels(ctx, goals, move_walkable)
        if self.gate_split:
            self._split_gates(ctx, goals, move_walkable, labels)
        return labels

    def _gate_geometry(self, base, walk):
        key = (base.tobytes(), walk.tobytes())
        hit = self._gate_cache.get(key)
        if hit is not None:
            return hit
        H_, W_ = walk.shape
        cells = set()
        for r, c in np.argwhere(base):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if 0 <= q[0] < H_ and 0 <= q[1] < W_ and walk[q] and not base[q]:
                    cells.add((int(q[0]), int(q[1])))
        gates, seen = [], set()
        for c0 in sorted(cells):
            if c0 in seen:
                continue
            comp, stack = [], [c0]
            seen.add(c0)
            while stack:
                q = stack.pop()
                comp.append(q)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (q[0] + dr, q[1] + dc)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            gates.append(tuple(sorted(comp)))
        enemy_map = walk & ~base
        fields_enemy = [_grid_field(enemy_map, g) for g in gates]
        fields_own = [_grid_field(walk, g) for g in gates]
        outside = [_grid_field(enemy_map, g) for g in gates]  # gate -> anywhere outside the base
        hit = (gates, fields_enemy, fields_own, outside, enemy_map)
        if len(self._gate_cache) > 8:
            self._gate_cache.clear()
        self._gate_cache[key] = hit
        return hit

    def _split_gates(self, ctx, goals, walk, labels):
        """Leave the base through the exit the enemy Hunters reach latest (time margin), spread units leaving together
        over different exits, wait inside (enemies cannot enter) while every exit is covered."""
        base = self._cached_base_mask(ctx["ally_spawn"], ctx["shape"])
        walk = np.ascontiguousarray(walk, dtype=bool)
        gates, f_en, f_own, _, enemy_map = self._gate_geometry(base, walk)
        if len(gates) < 2:
            return
        hunters = [e.pos for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        alive = {u.name for u in ctx["own"]}
        self._gate_of = {n: g for n, g in self._gate_of.items() if n in alive}
        if not hunters:
            self._gate_of.clear()
            return
        h_time = [min(f[h] for h in hunters) / 6.0 for f in f_en]
        states, shape = ctx["states"], ctx["shape"]
        dist_base = self._gate_cache.get(("db", base.tobytes()))
        if dist_base is None:
            dist_base = self._gate_cache[("db", base.tobytes())] = _grid_field(walk, tuple(map(tuple, np.argwhere(base))))
        masks = []
        for g in gates:
            m = np.zeros_like(walk)
            for q in g:
                m[q] = True
            masks.append(m)
        leaving, used = [], [0] * len(gates)
        for unit in ctx["own"]:
            if unit.cls == HUNTER or unit.name not in labels or not labels[unit.name].valid:
                continue
            g_set = goals.get(unit.name, ())
            if not g_set:
                continue
            if not base[unit.pos]:
                g = self._gate_of.get(unit.name)
                if g is None:
                    if self.gate_return > 0 and dist_base[unit.pos] <= self.gate_return and any(
                            base[q] for tiles, _ in g_set for q in tiles):
                        self._return_gate(unit, g_set, gates, masks, h_time, f_own, walk, states, shape, labels)
                    continue
                if dist_base[unit.pos] > self.gate_commit:
                    del self._gate_of[unit.name]
                    continue
                # keep the committed exit: the other exits stay closed on its plan
                closed = np.zeros_like(walk)
                for k, m in enumerate(masks):
                    if k != g:
                        closed |= m
                lab = self._label(states[unit.row], states, g_set, walk & ~closed, shape)
                if lab.valid:
                    labels[unit.name] = lab
                    self._count("gate_keep")
                continue
            # inside the base: is the goal outside?  (closing every exit makes it unreachable)
            speed = 6.0 if unit.cls == CARRIER else 4.0
            opts = []
            for k in range(len(gates)):
                closed = np.zeros_like(walk)
                for j, m in enumerate(masks):
                    if j != k:
                        closed |= m
                lab = self._label(states[unit.row], states, g_set, walk & ~closed, shape)
                if not lab.valid:
                    continue
                margin = h_time[k] - f_own[k][unit.pos] / speed
                opts.append([margin, k, lab])
            if len(opts) < 1:
                continue
            leaving.append((unit.name, opts))
        if not leaving:
            return
        waiting = []
        for name, opts in leaving:
            best = max(o[0] for o in opts)
            # spread: among exits within gate_tol of the best margin, the one least used this decision
            near = [o for o in opts if o[0] >= best - self.gate_tol]
            near.sort(key=lambda o: (used[o[1]], -o[0]))
            choice = near[0]
            if choice[0] < self.gate_wait:
                waiting.append((name, opts))
                continue
            used[choice[1]] += 1
            labels[name] = choice[2]
            self._gate_of[name] = choice[1]
            self._count("gate_go")
        if waiting and len(waiting) < self.gate_breakout:
            for name, opts in waiting:
                labels[name] = _invalid(goals.get(name, ()))
                self._count("gate_wait")
        else:
            for name, opts in waiting:  # break out together, one per exit in turn
                opts.sort(key=lambda o: (used[o[1]], -o[0]))
                used[opts[0][1]] += 1
                labels[name] = opts[0][2]
                self._gate_of[name] = opts[0][1]
                self._count("gate_breakout")

    def _return_gate(self, unit, g_set, gates, masks, h_time, f_own, walk, states, shape, labels):
        """Heading home (a goal inside the base): come in through the exit we reach soonest before the Hunters."""
        speed = 6.0 if unit.cls == CARRIER else 4.0
        best = None
        for k in range(len(gates)):
            closed = np.zeros_like(walk)
            for j, m in enumerate(masks):
                if j != k:
                    closed |= m
            margin = h_time[k] - f_own[k][unit.pos] / speed
            if best is not None and margin <= best[0]:
                continue
            lab = self._label(states[unit.row], states, g_set, walk & ~closed, shape)
            if lab.valid:
                best = (margin, lab)
        if best is not None:
            labels[unit.name] = best[1]
            self._count("gate_return")

    def _stage_holds(self, controlled, row_for, states, graphic, walkable):
        """V27's reserve hold by unit name: stacked walkers share coordinates, so the xy test also froze the race
        walkers on the reserve's tile at the sanctuary edge until ``early_reserve`` ran out (both sides standing ~1 s)."""
        self._staged_xy = set()
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        shape = graphic.shape[:2]
        for n in self.staged:
            if n in row_for and n in controlled:
                if planner.distance(pixel_position(states[row_for[n]], shape)) <= self.reserve_stage:
                    self._stage_hold.add(n)

    def teach(self, obs):
        labels = super().teach(obs)
        if self.race_search > 0 and labels:
            try:
                self._race_search(obs, labels)
            except Exception:  # never let the lookahead break a match
                self._count("race_search_error")
        if self.carrier_retreat > 0:
            self._retreat_carriers(labels)
        if self.east_post > 0:
            self._post_east(labels)
        if self.hunter_roles:
            self._assign_roles(labels)
        if self.bait > 0:
            self._bait(labels)
        if self._cap_hold or self._cap_out or self._stage_hold:
            for n in list(labels):
                lab = labels[n]
                if n in self._stage_hold and n not in self._cap_out:
                    heading = np.zeros(2, dtype=np.float32)
                    self._count("stage_hold")
                elif n in self._cap_hold:
                    heading = np.zeros(2, dtype=np.float32)
                    self._count("cap_hold")
                elif n in self._cap_out:
                    heading = self._cap_out[n]
                    self._count("cap_out")
                else:
                    continue
                labels[n] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
        return labels


    # ------------------------------------------------------------------ storm

    def _storm(self, controlled, row_for, states, graphic, walkable, goals):
        ctx = self._teach_ctx
        if MATCH_SECONDS - ctx["remaining"] < self.storm_after:
            return
        ours = sum(u.cls == HUNTER for u in ctx["own"])
        theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
        if theirs <= ours:
            return
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return
        tile_set = set(tiles)
        shape = graphic.shape[:2]
        going = [n for n in controlled if n in goals
                 and any(kind == "transform" and set(t) & tile_set for t, kind in goals[n])]
        need = self.storm_units - len(going)
        if need <= 0:
            return
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        cands = []
        for n in controlled:
            st = states[row_for[n]]
            if n in going or self._class_id(st) != COLLECTOR or self._is_holding(st):
                continue
            d = planner.distance(pixel_position(st, shape))
            if np.isfinite(d):
                cands.append((d, self._rk(n), n))
        for _, _, n in sorted(cands)[:need]:
            goals[n] = ((tiles, "transform"),)
            if n not in self.committed:
                self.committed.append(n)
                st = states[row_for[n]]
                self._committed_xy.add((float(st[0]), float(st[1])))
            self._count("storm")

    # ------------------------------------------------------------------ early reserve

    def _early_reserve(self, controlled, row_for, states, graphic, walkable, goals):
        """Before ``early_reserve`` s (and while no enemy Hunter is out), the ``reserve_count`` empty Collectors
        nearest the sanctuary that have no goal from the composition walk there straight away and are staged
        (V27's hold at ``reserve_stage``) instead of picking up batteries on the way."""
        ctx = self._teach_ctx
        if MATCH_SECONDS - ctx["remaining"] >= self.early_reserve:
            return
        if any(e.cls == HUNTER and not e.in_base for e in ctx["enemies"]):
            return
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return
        tile_set = set(tiles)
        shape = graphic.shape[:2]
        staged = [n for n in self.staged if n in goals]
        need = self.reserve_count - len(staged)
        if need <= 0:
            return
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        cands = []
        for n in controlled:
            st = states[row_for[n]]
            if n in goals or self._class_id(st) != COLLECTOR:
                continue  # holding is fine: a battery picked up on the way is lost on the transform
            d = planner.distance(pixel_position(st, shape))
            if np.isfinite(d):
                cands.append((d, self._rk(n), n))
        for _, _, n in sorted(cands)[:need]:
            goals[n] = ((tiles, "transform"),)
            self.staged.append(n)
            st = states[row_for[n]]
            self._staged_xy.add((float(st[0]), float(st[1])))
            self._count("early_reserve")

    # ------------------------------------------------------------------ carrier retreat

    def _retreat_carriers(self, labels):
        """A Carrier outside our base runs home while an enemy Hunter (out of its base) is within
        ``carrier_retreat`` tiles: Hunters are as fast as Carriers and our base is closed to them."""
        ctx = self._teach_ctx
        hunters = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not hunters:
            return
        shape = ctx["shape"]
        base = self._cached_base_mask(ctx["ally_spawn"], shape)
        tiles = None
        for u in ctx["own"]:
            if u.cls != CARRIER or u.name not in labels or base[u.pos]:
                continue
            here = pixel_position(ctx["states"][u.row], shape)
            if min(np.hypot(here[0] - h.pos[0], here[1] - h.pos[1]) for h in hunters) > self.carrier_retreat:
                continue
            if tiles is None:
                tiles = tuple((int(r), int(c)) for r, c in np.argwhere(base & ctx["walkable"]))
            heading = self._path_heading(ctx["walkable"], tiles, here, UNIT_RADIUS[CARRIER])
            if heading is not None:
                lab = labels[u.name]
                labels[u.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
                self._count("carrier_retreat")

    # ------------------------------------------------------------------ camp_path

    def _near(self, shape, points, radius):
        if not self.camp_path or not points:
            return super()._near(shape, points, radius)
        ctx = self._teach_ctx
        base = self._cached_base_mask(ctx["ally_spawn"], tuple(shape))
        walk = ctx["walkable"] & ~base
        mask = np.zeros(tuple(shape), dtype=bool)
        for p in points:
            p = (int(p[0]), int(p[1]))
            if not walk[p]:
                mask |= super()._near(shape, [p], radius)
                continue
            mask |= self._distance_to(walk, p) <= radius
        return mask

    # ------------------------------------------------------------------ east post

    def _post_east(self, labels):
        ctx = self._teach_ctx
        hunters = [u for u in ctx["own"] if u.cls == HUNTER and u.name in labels]
        if len(hunters) < MAX_HUNTERS or any(e.cls == HUNTER and not e.in_base for e in ctx["enemies"]):
            return
        shape = ctx["shape"]
        er, ec = ctx["enemy_spawn"]
        # team A's frame: base x 1..4, y 19..22 (pixel rows 1..4, cols 1..4); east exit corridor rows 3..4 (y 19..20)
        depth = 4.0 + self.east_post  # columns into the corridor (x = 5 + east_post)
        if er < shape[0] // 2:  # enemy is team A (top-left in pixels)
            post = ((3, int(depth)), (4, int(depth)))
        else:  # enemy is team B: mirror (x, y) -> (y, x) => pixel (r, c) -> (23 - c, 23 - r)
            post = ((shape[0] - 1 - int(depth), shape[1] - 1 - 3), (shape[0] - 1 - int(depth), shape[1] - 1 - 4))
        walk = self._teach_walkable
        if not all(walk[p] for p in post):
            return
        dist = self._distance_to(walk, post)
        h = min(hunters, key=lambda u: (float(dist[u.pos]), unit_index(u.name)))
        here = pixel_position(ctx["states"][h.row], shape)
        prey = [e for e in ctx["enemies"] if e.cls != HUNTER and not e.in_base
                and np.hypot(here[0] - e.pos[0], here[1] - e.pos[1]) <= self.east_engage]
        if prey:
            return
        if float(dist[h.pos]) <= 0.5:
            heading = np.zeros(2, dtype=np.float32)
        else:
            heading = self._path_heading(walk, post, here, UNIT_RADIUS[HUNTER])
            if heading is None:
                return
        lab = labels[h.name]
        labels[h.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
        self._count("east_post")

    # ------------------------------------------------------------------ Hunter roles

    def _exit_post(self, which):
        """Pixel cells of the enemy base's south / east exit post (team A frame mirrored for team B)."""
        ctx = self._teach_ctx
        shape = ctx["shape"]
        er, _ = ctx["enemy_spawn"]
        if which == "south":
            a_cells = ((6, 3), (6, 4))   # pixel (row, col) of world (3,17), (4,17): just below the south exit
        else:
            d = int(4 + max(1.0, self.east_post or 2.0))
            a_cells = ((3, d), (4, d))   # world (d, 20), (d, 19): inside the east corridor
        if er < shape[0] // 2:
            return a_cells
        return tuple((shape[0] - 1 - c, shape[1] - 1 - r) for r, c in a_cells)

    def _assign_roles(self, labels):
        ctx = self._teach_ctx
        try:
            self._roles_body(labels, ctx)
        finally:
            shape = ctx["shape"]
            self._prev_enemy = {e.row: pixel_position(ctx["states"][e.row], shape) for e in ctx["enemies"]}

    def _roles_body(self, labels, ctx):
        if self.roles_when == "free" and any(e.cls == HUNTER and not e.in_base for e in ctx["enemies"]):
            return
        hunters = [u for u in ctx["own"] if u.cls == HUNTER and u.name in labels]
        if not hunters:
            return
        shape, states = ctx["shape"], ctx["states"]
        walk = self._teach_walkable
        prey = [e for e in ctx["enemies"] if e.cls != HUNTER and not e.in_base]
        free = list(hunters)
        claimed = set()
        idle_raiders = []
        for role in self.hunter_roles:
            if not free:
                break
            if role == "default":
                free.sort(key=lambda u: unit_index(u.name))
                free.pop(0)
                continue
            if role == "raid":
                best = None
                for e in prey:
                    if e.row in claimed:
                        continue
                    dist = self._distance_to(walk, (e.pos,))
                    cargo = self._is_holding(states[e.row])
                    for u in free:
                        d = float(dist[u.pos])
                        if not np.isfinite(d):
                            continue
                        score = d / ((self.raid_value if cargo else 1.0) * (self.raid_carrier if e.cls == CARRIER else 1.0))
                        if self._raid_target.get(u.name) == e.row:
                            score *= self.raid_sticky
                        if best is None or score < best[0]:
                            best = (score, u, e)
                if best is None:
                    if self.raid_idle == "gate":
                        idle_raiders.append(role)
                    continue
                _, u, e = best
                self._raid_target[u.name] = e.row
                claimed.add(e.row)
                here = pixel_position(states[u.row], shape)
                there = pixel_position(states[e.row], shape)
                heading = None
                gap = float(np.hypot(here[0] - there[0], here[1] - there[1]))
                prev = self._prev_enemy.get(e.row)
                if self.raid_lead > 0 and prev is not None and gap < 4.0:
                    # intercept: aim where the prey will be (its last-decision velocity x the steps to close the gap)
                    vel = (there[0] - prev[0], there[1] - prev[1])
                    if float(np.hypot(*vel)) < 0.5:  # not a respawn jump
                        steps = min(8.0, gap / 0.24) * self.raid_lead
                        aim = (there[0] + vel[0] * steps, there[1] + vel[1] * steps)
                        heading = world_heading(aim[0] - here[0], aim[1] - here[1])
                if heading is None and gap < 1.5:
                    heading = world_heading(there[0] - here[0], there[1] - here[1])
                if heading is None:
                    heading = self._path_heading(walk, (e.pos,), here, UNIT_RADIUS[HUNTER])
            elif role == "escort":
                # stay by our Carrier; charge an enemy Hunter that comes within east_engage tiles of it (a trade)
                carriers = [c for c in ctx["own"] if c.cls == CARRIER]
                if not carriers:
                    continue
                car = carriers[0]
                cpos = pixel_position(states[car.row], shape)
                threats = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base
                           and np.hypot(cpos[0] - e.pos[0], cpos[1] - e.pos[1]) <= self.east_engage]
                dist_c = self._distance_to(walk, (car.pos,))
                u = min(free, key=lambda v: (float(dist_c[v.pos]), unit_index(v.name)))
                here = pixel_position(states[u.row], shape)
                if threats:
                    t = min(threats, key=lambda e: np.hypot(cpos[0] - e.pos[0], cpos[1] - e.pos[1]))
                    there = pixel_position(states[t.row], shape)
                    heading = (world_heading(there[0] - here[0], there[1] - here[1])
                               if np.hypot(here[0] - there[0], here[1] - there[1]) < 1.5
                               else self._path_heading(walk, (t.pos,), here, UNIT_RADIUS[HUNTER]))
                elif float(dist_c[u.pos]) <= 1.5:
                    free.remove(u)  # close enough: keep its own label (hunting nearby prey)
                    continue
                else:
                    heading = self._path_heading(walk, (car.pos,), here, UNIT_RADIUS[HUNTER])
            elif role in ("south", "east", "storage", "field"):
                post = self._exit_post(role) if role in ("south", "east") else self._zone_post(role, prey)
                if not post:
                    continue
                if not all(walk[p] for p in post):
                    continue
                dist = self._distance_to(walk, post)
                u = min(free, key=lambda v: (float(dist[v.pos]), unit_index(v.name)))
                here = pixel_position(states[u.row], shape)
                near = [e for e in prey if np.hypot(here[0] - e.pos[0], here[1] - e.pos[1]) <= self.east_engage]
                if near:
                    free.remove(u)  # it is striking: keep its inherited chase label
                    continue
                heading = (np.zeros(2, dtype=np.float32) if float(dist[u.pos]) <= 0.5
                           else self._path_heading(walk, post, here, UNIT_RADIUS[HUNTER]))
            else:
                continue
            free.remove(u)
            if heading is None:
                continue
            lab = labels[u.name]
            labels[u.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
            self._count("role_" + role)
        if idle_raiders and free:
            self._idle_gates(labels, ctx, free[:len(idle_raiders)])

    def _bait(self, labels):
        """An enemy Hunter within ``bait`` tiles of our Carrier: our nearest empty Collector (out of base, within
        ``bait_reach`` path tiles of that Hunter) runs at it -- a nearer prey for the Hunter, a free death for us
        (instant respawn) -- so the Carrier keeps collecting."""
        ctx = self._teach_ctx
        shape, states = ctx["shape"], ctx["states"]
        walk = self._teach_walkable
        carriers = [u for u in ctx["own"] if u.cls == CARRIER]
        if not carriers:
            return
        base = self._cached_base_mask(ctx["ally_spawn"], shape)
        used = set()
        for car in carriers:
            cp = pixel_position(states[car.row], shape)
            threats = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base
                       and np.hypot(cp[0] - e.pos[0], cp[1] - e.pos[1]) <= self.bait]
            for h in sorted(threats, key=lambda e: np.hypot(cp[0] - e.pos[0], cp[1] - e.pos[1])):
                dist = self._distance_to(walk, (h.pos,))
                cands = [u for u in ctx["own"] if u.cls == COLLECTOR and u.name not in used and u.name in labels
                         and not base[u.pos] and not self._is_holding(states[u.row])
                         and float(dist[u.pos]) <= self.bait_reach]
                if not cands:
                    continue
                u = min(cands, key=lambda v: (float(dist[v.pos]), unit_index(v.name)))
                used.add(u.name)
                here = pixel_position(states[u.row], shape)
                there = pixel_position(states[h.row], shape)
                heading = (world_heading(there[0] - here[0], there[1] - here[1])
                           if np.hypot(here[0] - there[0], here[1] - there[1]) < 1.5
                           else self._path_heading(walk, (h.pos,), here, UNIT_RADIUS[COLLECTOR]))
                if heading is None:
                    continue
                lab = labels[u.name]
                labels[u.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
                self._count("bait")

    def _idle_gates(self, labels, ctx, units):
        """Raiders with no prey out of the enemy base wait on the enemy base's exits (one per exit): enemy units
        respawn inside it and must come out there."""
        shape, states = ctx["shape"], ctx["states"]
        walk = np.ascontiguousarray(self._teach_walkable, dtype=bool)
        ebase = self._cached_base_mask(ctx["enemy_spawn"], shape)
        gates = self._gate_geometry(ebase, walk)[0]
        if not gates:
            return
        posts = []
        for g in gates:
            cen = np.mean(np.asarray(g, np.float32), axis=0)
            posts.append(min(g, key=lambda q: (q[0] - cen[0]) ** 2 + (q[1] - cen[1]) ** 2))
        dist = [self._distance_to(walk & ~ebase, (p,)) for p in posts]
        left = list(range(len(posts)))
        for u in sorted(units, key=lambda v: unit_index(v.name)):
            if not left:
                break
            k = min(left, key=lambda j: float(dist[j][u.pos]))
            left.remove(k)
            here = pixel_position(states[u.row], shape)
            if float(dist[k][u.pos]) <= 0.5:
                heading = np.zeros(2, dtype=np.float32)
            else:
                heading = self._path_heading(walk & ~ebase, (posts[k],), here, UNIT_RADIUS[HUNTER])
                if heading is None:
                    continue
            lab = labels[u.name]
            labels[u.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
            self._count("raid_idle")

    def _zone_post(self, role, prey):
        """``storage``: the enemy storage cell nearest the enemy units out of base (their next deposit);
        ``field``: the cell with the most battery value within 2 tiles (the richest pickup area)."""
        ctx = self._teach_ctx
        graphic, walk = ctx["graphic"], self._teach_walkable
        if role == "storage":
            cells = np.argwhere((graphic[..., STORAGE_ENEMY] > 0.5) & walk)
            if not len(cells):
                return ()
            if prey:
                pts = np.array([e.pos for e in prey], np.float32)
                d = np.sqrt(((cells[:, None, :] - pts[None]) ** 2).sum(-1)).min(1)
                c = cells[int(np.argmin(d))]
            else:
                c = cells[0]
            return ((int(c[0]), int(c[1])),)
        bat = graphic[..., BATTERY].astype(np.float32)
        if bat.max() <= 0:
            return ()
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(bat, 2)
        dens = sliding_window_view(pad, (5, 5)).sum((-1, -2))
        dens = np.where(walk, dens, -1)
        r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
        return ((int(r), int(c)),)

    # ------------------------------------------------------------------ race search

    def _race_search(self, obs, labels):
        import itertools

        from ..sim import core
        from ..sim.state import SimState
        ctx = self._teach_ctx
        states = ctx["states"]
        names = list(labels)
        own_idx = {n: unit_index(n) for n in names}
        sim0 = SimState.from_obs(obs)
        pos = sim0.uf.copy()
        cls = sim0.ui[:, core.U_CLS].copy()
        centre = np.array([12.0, 12.0], np.float32)

        def edge(p):  # distance to the 2x2 sanctuary (world x, y in [11, 13))
            dx = max(11.0 - p[0], p[0] - 13.0, 0.0)
            dy = max(11.0 - p[1], p[1] - 13.0, 0.0)
            return float(np.hypot(dx, dy))

        ours = sorted(own_idx.values())
        team = 0 if ours[0] < 5 else 1
        mine = range(5 * team, 5 * team + 5)
        theirs = range(5 * (1 - team), 5 * (1 - team) + 5)
        near = [u for u in mine if cls[u] == core.COLLECTOR and 0.0 < edge(pos[u]) <= self.rs_trigger]
        enemy_near = [u for u in theirs if cls[u] == core.COLLECTOR and edge(pos[u]) <= self.rs_trigger + 0.5]
        if not near or not (enemy_near or any(cls[u] == core.HUNTER for u in theirs)):
            return
        near = sorted(near, key=lambda u: edge(pos[u]))[:4]
        name_of = {i: n for n, i in own_idx.items()}
        # inherited moves of our units this decision (world frame)
        base_move = np.zeros((10, 2), np.float32)
        for u in mine:
            n = name_of.get(u)
            if n is None:
                continue
            lab = labels[n]
            h = getattr(lab, "heading", None)
            if h is not None and lab.valid:
                base_move[u] = h
        H = self.rs_horizon

        def straight(p):
            v = centre - p
            n = float(np.hypot(*v))
            return v / n if n > 1e-6 else np.zeros(2, np.float32)

        def rollout(delays, model):
            sim = sim0.copy()
            for t in range(H):
                p = sim.uf
                c = sim.ui[:, core.U_CLS]
                mv = np.zeros((10, 2), np.float32)
                for u in mine:
                    if u in delays:
                        if t >= delays[u] and c[u] == core.COLLECTOR:
                            mv[u] = base_move[u] if t == 0 else straight(p[u])
                        elif c[u] == core.HUNTER:
                            mv[u] = base_move[u] if t == 0 else 0.0
                    elif t == 0:
                        mv[u] = base_move[u]
                their_h = sum(1 for u in theirs if c[u] == core.HUNTER)
                slots = MAX_HUNTERS - their_h
                for u in sorted(theirs, key=lambda u: edge(p[u])):
                    if c[u] == core.COLLECTOR and edge(p[u]) <= self.rs_trigger + 0.5 + 0.24 * t:
                        if model == 1 and edge(p[u]) <= 0.2:
                            if slots <= 0:
                                continue
                            slots -= 1
                        mv[u] = straight(p[u])
                    elif c[u] == core.HUNTER:
                        prey = [v for v in mine if c[v] != core.HUNTER]
                        if prey:
                            q = min(prey, key=lambda v: float(np.hypot(*(p[v] - p[u]))))
                            d = p[q] - p[u]
                            nn = float(np.hypot(*d))
                            if nn < 3.0:
                                mv[u] = d / max(nn, 1e-6)
                sim.step(mv)
            c = sim.ui[:, core.U_CLS]
            hun = sum(1 for u in mine if c[u] == core.HUNTER) - sum(1 for u in theirs if c[u] == core.HUNTER)
            # Collectors that died went home: far from the sanctuary now
            lost = sum(1 for u in near if edge(sim.uf[u]) > 6.0)
            return 10.0 * hun - lost

        best = None
        for combo in itertools.product(range(self.race_search + 1), repeat=len(near)):
            delays = dict(zip(near, combo))
            score = (rollout(delays, 0) + rollout(delays, 1)) / 2.0
            key = (score, -sum(combo))
            if best is None or key > best[0]:
                best = (key, delays)
        (score, _), delays = best
        base_score = None
        for u, d in delays.items():
            if d > 0:
                n = name_of[u]
                lab = labels[n]
                labels[n] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False,
                                          np.zeros(2, dtype=np.float32))
                self._count("race_search_hold")
        self._count("race_search")
