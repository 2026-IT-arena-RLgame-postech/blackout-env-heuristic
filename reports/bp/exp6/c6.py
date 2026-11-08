"""Round-6 candidate sets.  python reports/bp/exp6/c6.py <set> -> ';'-joined descriptors (names -> stderr)."""
import sys
from blackout_env.heuristics.params import resolve_descriptor as R

def X(extra, base="B6"):
    return R(base + ("," + extra if extra else "")).replace("strategic_v34:", "strategic_x6:", 1)

def sw(t, a, b="B6"):
    return f"switch:{t:g}:{a if a.startswith(('strategic_', 'switch:')) else R(a)}|{b if b.startswith(('strategic_', 'switch:')) else R(b)}"

SETS = {}
SETS["a"] = {
    "x_base": X(""),
    "as_always": X("assassin=30"),
    "as_ahead": X("assassin=30,assassin_when=ahead"),
    "as_behind": X("assassin=30,assassin_when=behind"),
    "as_near8": X("assassin=8"),
    "as_5_30": X("assassin=30,assassin_from=5"),
    "sh_4": X("shelter=4"),
    "sh_7": X("shelter=7"),
    "sh_10": X("shelter=10"),
    "as_sh7": X("assassin=30,shelter=7"),
}

if __name__ == "__main__":
    s = SETS[sys.argv[1]]
    print(";".join(s.values()))
    print(",".join(s), file=sys.stderr)
