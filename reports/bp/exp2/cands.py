"""Round-2 exploiter candidate builder.  python reports/bp/exp2/cands.py <set>  -> ';'-joined descriptors on stdout (names on stderr)."""
import sys
from blackout_env.heuristics.params import resolve_descriptor as R
from blackout_env.train.distill.ood_bp import OODBP

OOD = {n: d for n, d, _, _ in OODBP}
SENT = "solo_sentry=true,sentry_min_hunters=1,sentry_offset=1.5,sentry_radius=10,guard_fill=true"
BL = "B6,hunter_roles=," + SENT
RS = "B6,hunter_roles=raid/south,roles_when=always"
EX = "B6,hunter_roles=south/east,roles_when=always"
B6 = "B6"


def sw(t, a, b=B6):
    return f"switch:{t:g}:{R(a)}|{R(b) if not b.startswith('switch:') else b}"


SETS = {
    "A": {
        "bl6": sw(6, BL), "bl11": sw(11, BL), "bl14": sw(14, BL),
        "bl8_rs18": sw(8, BL, sw(18, RS)), "rs6_bl14": sw(6, RS, sw(14, BL)),
        "bl8_ex20": sw(8, BL, sw(20, EX)), "bl8_ex40": sw(8, BL, sw(40, EX)),
        "bl8_storm25": sw(8, BL, sw(25, "B6,storm_units=3,storm_after=4")),
        "noroles8": sw(8, "B6,hunter_roles="), "bl8_bait": sw(8, BL, "B6,bait=6"),
        "bl8_rs30": sw(8, BL, sw(30, RS)), "late_ex40": sw(40, B6, EX),
        "bl8_stealer": sw(8, BL, OOD["stealer"]),
        "ee8": sw(8, "B6,hunter_roles=east/east,roles_when=always"),
        "ss8": sw(8, "B6,hunter_roles=south/south,roles_when=always"),
        "rr8": sw(8, "B6,hunter_roles=raid/raid,roles_when=always,raid_value=4"),
    },
}

def b6k(extra):
    return "B6," + extra

SETS["R"] = {  # second-wave dispatch knobs, after the 6 s blockade phase
    "r_base": sw(6, BL),
    "r_reach30": sw(6, BL, b6k("refill_reach=30,refill_after=0,refill_safe=false")),
    "r_safe": sw(6, BL, b6k("refill_reach=30,refill_after=4,refill_safe=true,refill_margin=0")),
    "r_bold": sw(6, BL, b6k("refill_reach=20,refill_after=4,refill_safe=true,refill_margin=-1")),
    "r_free": sw(6, BL, b6k("refill_reach=30,refill_free=true,refill_safe=false,refill_after=2")),
    "r_storm3": sw(6, BL, b6k("storm_units=3,storm_after=4")),
    "r_storm2": sw(6, BL, b6k("storm_units=2,storm_after=5")),
    "r_res3": sw(6, BL, b6k("reserve_count=3")),
    "r_earlyres0": sw(6, BL, b6k("early_reserve=0")),
    "r_cm_neg": sw(6, BL, b6k("commit_margin=-1.0")),
    "r_cm_pos": sw(6, BL, b6k("commit_margin=0.25")),
    "r_quota1": sw(6, BL, b6k("hunter_quota=1")),
    "r_tfloor": sw(6, BL, b6k("trade_floor=0.05,trade_reach=2.0")),
    "r_raid_value": sw(6, BL, b6k("hunt_hh=0")),
}
SETS["T"] = {
    "bl5": sw(5, BL), "bl5.5": sw(5.5, BL), "bl6.5": sw(6.5, BL), "bl7": sw(7, BL),
    "bl6_off": sw(6, "B6,hunter_roles="),
    "bl6_rs30": sw(6, BL, sw(30, RS)),
    "bl6_rs20": sw(6, BL, sw(20, RS)),
    "bl6_rs45": sw(6, BL, sw(45, RS)),
    "bl6_rsE": sw(6, BL, sw(30, "B6,hunter_roles=raid/east,roles_when=always")),
    "bl6_rsB": sw(6, BL, sw(30, "B6,hunter_roles=raid/south,roles_when=always,raid_value=4,raid_carrier=1.5")),
    "bl6_plain": sw(6, "B6,hunter_roles=,sentry_offset=1.5"),
}

SETS["O"] = {  # opening-phase knobs inside the 6 s blockade phase (paired vs bl6)
    "o_base": sw(6, BL),
    "o_rstage1": sw(6, BL + ",reserve_stage=1.0"),
    "o_rstage2": sw(6, BL + ",reserve_stage=2.0"),
    "o_rcount1": sw(6, BL + ",reserve_count=1"),
    "o_rcount3": sw(6, BL + ",reserve_count=3"),
    "o_early0": sw(6, BL + ",early_reserve=0"),
    "o_early12": sw(6, BL + ",early_reserve=12"),
    "o_open6": sw(6, BL + ",opening_seconds=6"),
    "o_capenter": sw(6, BL + ",cap_enter=0.35"),
    "o_cm_neg": sw(6, BL + ",commit_margin=-1.0"),
    "o_cm_pos": sw(6, BL + ",commit_margin=0.2"),
    "o_trade2": sw(6, BL + ",trade_radius=2.0,trade_reach=2.0"),
    "o_cdelay": sw(6, BL + ",carrier_delay=30,carrier_on_monopoly=false"),
    "o_cdelay4": sw(6, BL + ",carrier_delay=4"),
    "o_byname": sw(6, BL + ",stage_by_name=true"),
    "o_noorder": sw(6, BL + ",race_order=false"),
}

SETS["S"] = {  # economy / specials knobs after the 6 s blockade phase (paired vs s_base)
    "s_base": sw(6, BL),
    "s_dspd8": sw(6, BL, b6k("w_dspd=8")),
    "s_bspd6": sw(6, BL, b6k("w_bspd=6")),
    "s_bsize15": sw(6, BL, b6k("w_bsize=1.5")),
    "s_dsize15": sw(6, BL, b6k("w_dsize=1.5")),
    "s_allsp": sw(6, BL, b6k("w_dspd=8,w_bspd=6,w_bsize=1.5,w_dsize=1.5")),
    "s_nosp": sw(6, BL, b6k("w_dspd=0,w_bspd=0,w_bsize=0,w_dsize=0")),
    "s_steal3": sw(6, BL, b6k("steal_weight=3.0")),
    "s_steal08": sw(6, BL, b6k("steal_weight=0.8")),
    "s_expo2": sw(6, BL, b6k("exposure_cost=2.0")),
    "s_expo03": sw(6, BL, b6k("exposure_cost=0.3")),
    "s_cfv10": sw(6, BL, b6k("carrier_field_value=10")),
    "s_cfv40": sw(6, BL, b6k("carrier_field_value=40")),
    "s_danger6": sw(6, BL, b6k("danger_hunter_radius=6,danger_weight=0.9,threat_radius=0.2")),
    "s_danger25": sw(6, BL, b6k("danger_hunter_radius=2.5,threat_radius=0.08")),
    "s_retreat": sw(6, BL, b6k("carrier_retreat=6")),
}

SETS["F"] = {  # final confirmation on fresh seeds: best singles + combos
    "s_base": sw(6, BL),
    "o_capenter": sw(6, BL + ",cap_enter=0.35"),
    "s_danger25": sw(6, BL, b6k("danger_hunter_radius=2.5,threat_radius=0.08")),
    "f_combo": sw(6, BL + ",cap_enter=0.35", b6k("danger_hunter_radius=2.5,threat_radius=0.08")),
    "f_combo_st": sw(6, BL + ",cap_enter=0.35", b6k("danger_hunter_radius=2.5,threat_radius=0.08,storm_units=2,storm_after=5")),
    "f_combo_d4": sw(6, BL + ",cap_enter=0.35", b6k("w_dspd=8,danger_hunter_radius=2.5,threat_radius=0.08")),
}

CAP = BL + ",cap_enter=0.35"
SETS["G"] = {  # round 3 vs rl10 it1050: X6 family variations
    "X6": sw(6, CAP),
    "g_cap05": sw(6, BL + ",cap_enter=0.5"),
    "g_cap025": sw(6, BL + ",cap_enter=0.25"),
    "g_cap8": sw(8, CAP),
    "g_cap4": sw(4.5, CAP),
    "g_cap_rs30": sw(6, CAP, sw(30, RS)),
    "g_cap_storm": sw(6, CAP, "B6,storm_units=2,storm_after=5"),
    "g_cap_bait": sw(6, CAP, "B6,bait=6"),
    "g_cap_steal": sw(6, CAP, "B6,steal_weight=3.0"),
    "g_cap_res3": sw(6, CAP + ",reserve_count=3"),
    "g_cap_q1": sw(6, CAP, "B6,hunter_quota=1"),
    "g_cap_cm": sw(6, CAP + ",commit_margin=-1.0"),
}

STEAL = OOD["stealer"]
SETS["H"] = {  # round 4 vs rl12 it600: economy / tempo exploiters (+ X6 opening combos)
    "X6": sw(6, CAP),
    "h_cap_steal": sw(6, CAP, STEAL),
    "h_cap_steal30": sw(6, CAP, sw(30, B6, STEAL)),
    "h_cap_pure": sw(6, CAP, OOD["pure_stealer"]),
    "h_cap_late": sw(6, CAP, OOD["late_burst"]),
    "h_cap_eco": sw(6, CAP, OOD["eco_boom"]),
    "h_cap_storm3": sw(6, CAP, "B6,storm_units=3,storm_after=4"),
    "h_cap_storm_bait": sw(6, CAP, "B6,storm_units=2,storm_after=5,bait=6"),
    "h_steal_rs": sw(20, STEAL, RS),
    "h_rs_steal": sw(6, RS, STEAL),
    "h_cap_v34c": sw(6, CAP, "V34c"),
    "h_cap_camp30": sw(6, CAP, sw(30, B6, "B6,hunter_roles=south/east,roles_when=always")),
}

if __name__ == "__main__":
    s = SETS[sys.argv[1]]
    sel = sys.argv[2].split(",") if len(sys.argv) > 2 else list(s)
    print(";".join(s[k] for k in sel))
    print(",".join(sel), file=sys.stderr)
