# Usage: PYTHONPATH=. python reports/p24_train/focus_probe.py <n seeds>  (vs V28, decisions 60-100)
# P24: does V32 (act teacher) flag any staged reserve unit at decisions 85-100?  Compare with V31.
import sys, collections, numpy as np
from blackout_env.heuristics import StrategicHeuristicV28
from blackout_env.heuristics.v32_four_hunters import V32
from blackout_env.sim.state import SimState, obs_from_raw
from blackout_env.train.distill.act_teacher import ActTeacher
from blackout_env.env.constants import unit_index
V31 = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5"
for name, spec in [("V31", V31), ("V32", V32)]:
    cnt = collections.Counter(); lab = collections.Counter(); n = 0
    for seed in range(4_000_000, 4_000_000 + int(sys.argv[1])):
        teacher = ActTeacher(spec); other = StrategicHeuristicV28(reserve_stage=0.5, trade_hold="advance")
        sim = SimState.from_seed(seed); n += 1
        for t in range(101):
            obs = obs_from_raw(sim.raw_state(), sim.raw_map())
            mv = np.zeros((10, 2), np.float32)
            teacher.teach({f"unit_{i}": obs[f"unit_{i}"] for i in range(5)})
            if 60 <= t <= 100:
                cnt[(t // 5 * 5, len(teacher.focus))] += 1
                if t >= 85:
                    for u in teacher.focus:
                        a = teacher.last_actions.get(u)
                        lab["move" if a is not None and np.abs(a).sum() > 0 else "zero/none"] += 1
            for nm, act in teacher.last_actions.items():
                mv[int(nm.split("_")[1])] = np.clip(act, -1, 1)
            for nm, act in other.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5, 10)}).items():
                mv[int(nm.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
            sim.step(mv)
    print(name, "games", n, "focus count by decision bucket:", sorted(cnt.items()), "labels 85-100:", dict(lab))
