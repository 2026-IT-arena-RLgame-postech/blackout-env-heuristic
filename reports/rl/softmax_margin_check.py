"""Softmax Collapse check (arXiv 2610.31642): how often is a policy's action softmax so confident that the
cross-entropy / log-prob gradient of the top action rounds to 0?

Plays greedy games (policy vs an opponent net) through evalfast, records every own-team logit row [33] the policy
produces, and reports the collapse margin m = -LSE_{j != top}(z_j - z_top) (so 1 - p_top ~ e^-m). Cliffs where
1 + e^-m == 1: bf16 ~5.5, fp16 ~8.3, fp32 ~16.6.
Usage: PYTHONPATH=. python reports/rl/softmax_margin_check.py --dev mps --seeds 4 ckpt [ckpt ...]"""
import argparse, json

import numpy as np
import torch

from blackout_env.train.distill import inference
from blackout_env.rl.evalfast import eval_batched

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--opp", default="net:checkpoints/distill/frozen/p24c_v321.pt")
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--dev", default="cpu")
ap.add_argument("--horizon", type=int, default=11000)
args = ap.parse_args()

REC = []
orig = inference._OwnQ.forward


def tapped(self, *a, **k):
    out = orig(self, *a, **k)
    if getattr(self, "_rec", False):
        lg = out[0] if isinstance(out, tuple) else out
        REC.append(lg.detach().float().reshape(-1, lg.shape[-1]).cpu())
    return out


inference._OwnQ.forward = tapped
_init = inference._OwnQ.__init__


def first_own(self, net):   # the first _OwnQ built per eval is the learner's (opponent NetPolicy loads after)
    _init(self, net)
    self._rec = not any(getattr(o, "_rec", False) for o in OWN)
    OWN.append(self)


OWN = []
inference._OwnQ.__init__ = first_own
import blackout_env.rl.evalfast as ef
ef._OwnQ = inference._OwnQ

CLIFF = {"bf16": 5.5, "fp16": 8.3, "fp32": 16.6}
res = {}
for ck in args.ckpts:
    REC.clear(); OWN.clear()
    r = eval_batched(ck, [args.opp], list(range(970000, 970000 + args.seeds)), args.dev, horizon=args.horizon,
                     graphs=False, fp16=False)
    z = torch.cat(REC).double()
    z = z - z.max(-1, keepdim=True).values
    top = z.argmax(-1)
    rest = z.scatter(-1, top.unsqueeze(-1), float("-inf"))
    m = (-torch.logsumexp(rest, -1)).numpy()
    row = {"rows": int(len(m)), "margin": r[args.opp]["margin"], "m_median": float(np.median(m)),
           "m_p90": float(np.percentile(m, 90)), "m_p99": float(np.percentile(m, 99)), "m_max": float(m.max()),
           **{f"frac>{k}": float((m > v).mean()) for k, v in CLIFF.items()},
           "stand_top_frac": float((top == z.shape[-1] - 1).float().mean())}
    res[ck] = row
    print(ck, json.dumps(row), flush=True)
