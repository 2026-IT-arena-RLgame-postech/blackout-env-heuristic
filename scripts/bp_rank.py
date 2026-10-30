"""Quick rank of econ_stats dirs: per candidate mean margin vs each opponent (both sides, seed-paired vs --base)."""
import json, sys, glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from blackout_env.heuristics.params import resolve_descriptor, split_descriptor

def short(d, ref):
    if not d.startswith('strategic_'): return d
    p, kv = split_descriptor(d); r = split_descriptor(ref)[1]
    diff = {k: v for k, v in kv.items() if r.get(k) != v}
    return ','.join(f'{k}={v}' for k, v in diff.items()) or '(base)'

def main(dirs, base='V33bp'):
    ref = resolve_descriptor(base) if ':' not in base else base
    rows = [json.loads(l) for d in dirs for f in glob.glob(f'{d}/**/matches.jsonl', recursive=True) for l in open(f)]
    t = defaultdict(lambda: defaultdict(list))
    for r in rows: t[(r['ours'], r['opponent'])][r['seed']].append(r['margin'])
    opps = sorted({o for _, o in t})
    for o in opps:
        b = {s: np.mean(v) for s, v in t.get((ref, o), {}).items()}
        res = []
        for (c, oo), d in t.items():
            if oo != o: continue
            m = {s: np.mean(v) for s, v in d.items()}
            ks = [k for k in m if k in b]
            dd = np.array([m[k] - b[k] for k in ks]) if ks else np.array([np.nan])
            win = np.mean([np.sign(x) for v in d.values() for x in v]) / 2 + .5
            res.append((np.mean(list(m.values())), dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan, win, len(m), short(c, ref)))
        print(f'== vs {o}')
        for r in sorted(res, key=lambda x: -x[0]):
            print(f'  margin {r[0]:+6.1f}  paired {r[1]:+6.1f}±{r[2]:4.1f}  win {r[3]:.2f}  n={r[4]}  {r[5][:150]}')

if __name__ == '__main__':
    args = sys.argv[1:]
    base = 'V33bp'
    if args and args[0].startswith('--base='): base = args.pop(0)[7:]
    main(args, base)
