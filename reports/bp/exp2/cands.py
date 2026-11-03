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

if __name__ == "__main__":
    s = SETS[sys.argv[1]]
    sel = sys.argv[2].split(",") if len(sys.argv) > 2 else list(s)
    print(";".join(s[k] for k in sel))
    print(",".join(sel), file=sys.stderr)
