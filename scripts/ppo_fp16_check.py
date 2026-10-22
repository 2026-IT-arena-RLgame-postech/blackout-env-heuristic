"""fp16 vs fp32 update numerics on a saved batch: gradient agreement and (log-prob / value) differences for the critic
and the policy, with and without loss scaling.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/ppo_fp16_check.py /tmp/ppo_batch.npz
"""
import sys

import numpy as np
import torch
from torch.nn import functional as F

from blackout_env.rl.ppo import PPOConfig, PPOTrainer

d = dict(np.load(sys.argv[1]))
Fz = "checkpoints/distill/frozen"
cfg = PPOConfig(opponents=[f"net:{Fz}/p24c_v321.pt"], games=2, heur_workers=0)
tr = PPOTrainer(cfg)
dev = tr.dev
n = 1024
g, t, a, ret = (torch.from_numpy(d[k][:n]).to(dev) for k in ("g", "t", "a", "ret"))
act, adv = torch.from_numpy(d["act"][:n]).to(dev).long(), torch.from_numpy(d["adv"][:n]).to(dev)


def flat(ps):
    return torch.cat([p.grad.reshape(-1).float() for p in ps if p.grad is not None])


def cmp(name, ga, gb):
    cos = F.cosine_similarity(ga, gb, dim=0).item()
    print(f"{name:34s} cos={cos:.5f} rel_err={(ga - gb).norm().item() / ga.norm().item():.4f} |g|={ga.norm().item():.3e} vs {gb.norm().item():.3e}"
          f" nonfinite={int((~torch.isfinite(gb)).sum())} zeros={float((gb == 0).float().mean()):.3f}")


# critic
def critic_grad(fp16, scale):
    tr.cfg.critic_fp16 = fp16
    tr.critic.zero_grad(set_to_none=True)
    loss = F.smooth_l1_loss(tr._value(g, t, a), ret, beta=cfg.vf_huber)
    (loss * scale).backward()
    gr = flat(tr.critic.parameters()) / scale
    return gr, loss.item()


tr.critic.train()
g32, l32 = critic_grad(False, 1.0)
for sc in (1.0, 1024.0, 65536.0):
    g16, l16 = critic_grad(True, sc)
    cmp(f"critic fp16 scale={sc:g} (loss {l16:.5f} vs {l32:.5f})", g32, g16)
with torch.no_grad():
    tr.cfg.critic_fp16 = False; v32 = tr._value(g, t, a)
    tr.cfg.critic_fp16 = True; v16 = tr._value(g, t, a)
print("critic value fp16 vs fp32: max|d|=%.4f mean|d|=%.5f (std of v %.3f)" % ((v32 - v16).abs().max(), (v32 - v16).abs().mean(), v32.std()))


# policy
def policy_grad(fp16, scale):
    tr.cfg.update_fp16 = False
    with torch.no_grad():
        old = F.log_softmax(tr._logits(tr.own, g, t, a), -1).gather(-1, act.unsqueeze(-1)).squeeze(-1)
        ref = F.log_softmax(tr._logits(tr.ref, g, t, a), -1)
    # perturb so ratio != 1 (second-epoch regime)
    tr.cfg.update_fp16 = fp16
    tr.net.zero_grad(set_to_none=True)
    lp_all = F.log_softmax(tr._logits(tr.own, g, t, a), -1)
    lp = lp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(lp - old)
    A = adv.unsqueeze(-1)
    pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
    kl = (lp_all.exp() * (lp_all - ref)).sum(-1).mean()
    ((pg + cfg.beta * kl) * scale).backward()
    return flat(tr.net.parameters()) / scale


p32 = policy_grad(False, 1.0)
for sc in (1.0, 1024.0, 65536.0):
    cmp(f"policy fp16 scale={sc:g}", p32, policy_grad(True, sc))
with torch.no_grad():
    tr.cfg.update_fp16 = False; x = F.log_softmax(tr._logits(tr.own, g, t, a), -1)
    tr.cfg.update_fp16 = True; y = F.log_softmax(tr._logits(tr.own, g, t, a), -1)
print("policy dlogp mean %.5f max %.4f argmax agree %.5f" % ((x - y).abs().mean(), (x - y).abs().max(), (x.argmax(-1) == y.argmax(-1)).float().mean()))
