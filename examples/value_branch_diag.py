"""Is the PPO critic the bottleneck, or the game's own randomness?  Branch-rollout test (2026-11-05).

For N seeds, play learner (team A) vs a net opponent (team B) to time T (50 Hz), keep the games still running, read the
critic's V of team A's view, then branch every root K times (fresh special-spawn RNG per branch; the learner sampled or
greedy) and play each branch to the end (stop at exhaustion).  R = sign(lead) + lam * tanh(lead / scale) (PPO's reward).

Per T it reports
  var_R      total variance of R over all branches
  irreducible  E_root[Var(R | root)]  (what no critic can predict)
  explainable  Var_root(E[R | root])  (bias-corrected by the within / K term)
  critic_mse   E[(V - E[R|root])^2]   (bias-corrected)  and R2 = 1 - critic_mse / explainable
  lead_mse   the same for a critic-free baseline: PPO's reward of the current lead (no future)
so critic_mse << explainable means the critic already captures what is predictable; irreducible / var_R is the share of
outcome variance that is pure chance from that point.

    BLACKOUT_TPD=1 PYTHONPATH=. python examples/value_branch_diag.py --learner L.pt --critic L_critic.pt \
        --value VALUE.pt --opponent OPP.pt --roots 128 --k 16 --times 10,20,30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("BLACKOUT_TPD", "1")

from blackout_env.rl.grpo import LearnerPolicy  # noqa: E402
from blackout_env.sim.batch import N_AGENT_COLS, BatchRollout, BatchSim, NetPolicy, unpack_graphic  # noqa: E402
from blackout_env.sim.mapdata import H, W  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402
from blackout_env.train.distill.inference import _OwnQ, load_net  # noqa: E402
from blackout_env.value.model import REG_SCALE, load_value_net  # noqa: E402

LAM, SCALE = 0.2, 30.0
DPS = 50  # decisions per second at 50 Hz


def reward(lead):
    lead = np.asarray(lead, np.float64)
    return np.sign(lead) + LAM * np.tanh(lead / SCALE)


@torch.no_grad()
def critic_v(critic, pk, tm, am, dev):
    g = unpack_graphic(torch.from_numpy(pk).to(dev))
    t = torch.from_numpy(tm).to(dev)
    a = torch.from_numpy(am).to(dev)
    out = critic(g, t, a).float()
    h = out.shape[1] - 4
    lead = (t[:, 0] - t[:, 1]) * 100.0
    final = lead + out[:, h] / REG_SCALE * 100.0
    p = torch.softmax(out[:, h + 1:], -1)
    return ((p[:, 2] - p[:, 0]) + LAM * torch.tanh(final / SCALE)).cpu().numpy().astype(np.float64)


def make_learner(path, dev, sample):
    net = load_net(torch.load(path, map_location="cpu", weights_only=True)["policy_state"]).to(dev).eval()
    return LearnerPolicy(_OwnQ(net), dev, 10 ** 9 if sample else 0, fp16=dev.type == "cuda")


def play(pa, pb, roots, horizon, reseeds=None):
    bsim = BatchSim(roots)
    bsim.ticks_per_decision = 1
    res = BatchRollout(pa, pb, packed=True).run(None, horizon, bsim=bsim, reseeds=reseeds, stop_when_exhausted=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--learner", required=True)
    p.add_argument("--critic", required=True, help="<ckpt>_critic.pt (state dict)")
    p.add_argument("--value", required=True, help="the value net the critic was built from (architecture)")
    p.add_argument("--opponent", required=True, help="team B: net checkpoint (.pt) or a heuristic descriptor / alias")
    p.add_argument("--roots", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--times", default="10,20,30")
    p.add_argument("--seed0", type=int, default=80_000_000)
    p.add_argument("--modes", default="sample,greedy", help="learner in the branches: sampled (training) / greedy (eval)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="")
    p.add_argument("--save", default="", help="npz of the roots (learner views, T, mu, var) for examples/critic_bench.py")
    args = p.parse_args()
    dev = torch.device(args.device)
    critic = load_value_net(args.value, device=dev)
    critic.load_state_dict(torch.load(args.critic, map_location=dev, weights_only=True))
    critic.eval()
    if args.opponent.endswith(".pt"):
        opp = NetPolicy.from_checkpoint(args.opponent, device=args.device, fp16=dev.type == "cuda")
        opp_name = "net:" + args.opponent
    else:   # heuristic descriptor / alias: a fresh instance per rollout (per-game heuristic state restarts at the root)
        from blackout_env.rl.grpo import _heuristic_aliases
        from blackout_env.sim.batch import HeuristicPolicy

        opp = HeuristicPolicy(_heuristic_aliases().get(args.opponent, args.opponent), 0)
        opp_name = args.opponent
    seeds = list(range(args.seed0, args.seed0 + args.roots))
    rows = []
    keep = {"pk": [], "tm": [], "am": [], "T": [], "mu": [], "var_w": []}
    for mode in args.modes.split(","):
        learner = make_learner(args.learner, dev, mode == "sample")
        for T in [float(x) for x in args.times.split(",")]:
            res = play(learner, opp, [SimState.from_seed(s) for s in seeds], int(T * DPS),
                       reseeds=[s + 7 for s in seeds])
            bs = res.bsim
            stopped = res.exhausted if res.exhausted is not None else np.zeros(bs.n, bool)
            live = np.nonzero(~bs.over & ~stopped)[0]
            if len(live) < 4:
                print(f"{mode} T={T:g}: only {len(live)} games still running, skip", flush=True)
                continue
            pk = np.zeros((len(live), 2, H, W), np.int16)
            tm = np.zeros((len(live), 2, 4), np.float32)
            am = np.zeros((len(live), 2, 10, N_AGENT_COLS), np.float32)
            bs.render_packed(live, pk, tm, am)
            v = critic_v(critic, pk[:, 0], tm[:, 0], am[:, 0], dev)
            lead_now = (bs.scores[live, 0] - bs.scores[live, 1]).astype(np.float64)
            states = [bs.state(int(b)) for b in live]
            K = args.k
            broots = [s for s in states for _ in range(K)]
            bseeds = [int(args.seed0 * 3 + 1000 * i + j) for i in range(len(states)) for j in range(K)]
            br = play(learner, opp, broots, 425 * DPS, reseeds=bseeds)
            R = reward(br.scores[:, 0] - br.scores[:, 1]).reshape(len(states), K)
            mu, var_w = R.mean(1), R.var(1, ddof=1)
            if args.save and mode == "sample":
                keep["pk"].append(pk[:, 0]); keep["tm"].append(tm[:, 0]); keep["am"].append(am[:, 0])
                keep["T"].append(np.full(len(mu), T)); keep["mu"].append(mu); keep["var_w"].append(var_w)
            irr = float(var_w.mean())
            expl = float(mu.var(ddof=1) - irr / K)
            cm = float(np.mean((v - mu) ** 2) - irr / K)
            lm = float(np.mean((reward(lead_now) - mu) ** 2) - irr / K)
            row = dict(mode=mode, T=T, roots=len(states), k=K, var_R=float(R.var()), irreducible=irr, explainable=expl,
                       critic_mse=cm, critic_r2=1 - cm / expl if expl > 0 else float("nan"),
                       lead_mse=lm, lead_r2=1 - lm / expl if expl > 0 else float("nan"),
                       chance_share=irr / float(R.var()), corr_v_mu=float(np.corrcoef(v, mu)[0, 1]),
                       v_mean=float(v.mean()), mu_mean=float(mu.mean()))
            rows.append(row)
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    if args.save and keep["mu"]:
        np.savez(args.save, **{k: np.concatenate(v) for k, v in keep.items()}, k=args.k, opponent=opp_name)


if __name__ == "__main__":
    main()
