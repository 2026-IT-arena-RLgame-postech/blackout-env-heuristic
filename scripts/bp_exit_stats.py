"""Base-exit stats from traj_dump npz files: where our units die near our base (by route) and which exit they take."""
import glob, sys
import numpy as np
from collections import Counter


def route(x, y):  # team-A frame
    if x < 2.0 and y < 17.5: return 'W'
    if y >= 18.5 and x >= 4.8: return 'E'
    if y < 17.5: return 'S'
    return 'row18'


def main(path, ours_only=True):
    deaths, exits, n = Counter(), Counter(), 0
    for f in sorted(glob.glob(f'{path}/*.npz')):
        d = np.load(f)
        for b in range(len(d['seeds'])):
            cls, pos = d['cls'][b], d['pos'][b]
            T = (cls[:, 0] >= 0).sum()
            k = d['kills'][d['kills'][:, 0] == b]
            teams = [int(d['sides'][b])] if ours_only else [0, 1]
            n += 1
            for team in teams:
                mir = (lambda p: (float(p[1]), float(p[0]))) if team == 1 else (lambda p: (float(p[0]), float(p[1])))
                for _, t, a, v, vc, kc in k:
                    if v // 5 == team and kc == 1 and vc != 1:
                        x, y = mir(pos[t, v])
                        if x < 9 and y > 12: deaths[route(x, y)] += 1
                for u in range(5 * team, 5 * team + 5):
                    state = None
                    for tt in range(T):
                        x, y = mir(pos[tt, u])
                        inb = x < 5 and y >= 19
                        if state == 'out18' and route(x, y) in ('W', 'S'):
                            exits[route(x, y)] += 1; state = None
                        if state == 'base' and not inb:
                            if route(x, y) == 'E': exits['E'] += 1; state = None
                            else: state = 'out18'
                        if inb: state = 'base'
    print(f'{path}: games {n}  deaths near base by route {dict(deaths)}  exits {dict(exits)}')


if __name__ == '__main__':
    main(sys.argv[1], ours_only=len(sys.argv) < 3)
