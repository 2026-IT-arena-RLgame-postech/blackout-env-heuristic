"""IQN checkpoint -> logits-head checkpoint (P5), the head fitted to the IQN head's outputs.

``MyModel.iqn_to_logits_state`` folds the IQN head analytically: ``in_scale`` = the mean tau
embedding at inference's 32 fixed taus.  That is exact for one tau, but the student reads the
*mean of 32 nonlinear* evaluations, so the fold is only close (D6, held-out eval rows: argmax
agreement 99.5%, max logit difference ~5).  This fits the logits head -- ``in_scale`` and the
value MLP, trunk frozen -- to the IQN head's fixed-tau mean logits on cached trunk features of
real rows (KL on the softmax plus a small squared-error term that keeps the logit scale, which
``tie_mean``/``arc_mean`` margins read), starting from the fold, and reports both on held-out rows.

  python -m blackout_env.train.distill.head_fit checkpoints/distill/d6/latest.pt \\
      --shards checkpoints/distill/d6/shards --eval-shards checkpoints/distill/d6/eval_shards \\
      --out checkpoints/distill/p5a/init_logits.pt

Stateless checkpoints only (a temporal one's head reads features that depend on the memory; use the
analytic fold).  CPU by default: a few hundred thousand unit rows, ~1-2 minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from blackout_env.model.my_model import N_VISION_TOKENS, MyModel
from blackout_env.train.distill.dataset import FIELDS, N_TEAM, decode_graphic, new_shards


def load_rows(root: Path, n: int, seed: int, newest: int | None = None) -> dict[str, np.ndarray]:
    """``n`` random rows (of the ``newest`` shards, if given) under ``root``."""
    paths = sorted(Path(root).glob("*.npz"))
    keep = {p.name for p in (paths[-newest:] if newest else paths)}
    parts = [rows for rows, _ in new_shards(Path(root), {p.name for p in paths} - keep)]
    rows = {k: np.concatenate([r[k] for r in parts]) for k in FIELDS}
    total = len(rows["labels"])
    idx = np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))
    return {k: v[idx] for k, v in rows.items()}


@torch.no_grad()
def own_unit_features(net: MyModel, rows: dict[str, np.ndarray], chunk: int = 512) -> torch.Tensor:
    """The head's input for the own team's units: [rows * 5, hidden]."""
    if net.is_temporal:
        raise ValueError("head_fit is for stateless nets")
    net.eval()
    device = next(net.parameters()).device
    out = []
    for start in range(0, len(rows["labels"]), chunk):
        graphic = torch.from_numpy(decode_graphic(rows["graphic"][start:start + chunk])).permute(0, 3, 1, 2).to(device)
        team_state = torch.from_numpy(rows["team_state"][start:start + chunk]).to(device)
        agent_states = torch.from_numpy(rows["agent_states"][start:start + chunk]).to(device)
        tokens = net._trunk(graphic, team_state, agent_states)
        out.append(tokens[:, N_VISION_TOKENS:N_VISION_TOKENS + N_TEAM].reshape(-1, net.hidden_size))
    return torch.cat(out)


@torch.no_grad()
def iqn_logits(iqn: MyModel, features: torch.Tensor, n_quantiles: int = 32, chunk: int = 4096) -> torch.Tensor:
    """The IQN head's mean over the fixed taus (inference.fixed_tau) for [N, hidden] features."""
    tau = ((torch.arange(n_quantiles, dtype=torch.float32, device=features.device) + 0.5) / n_quantiles)[None]
    return torch.cat([iqn.q_head(features[i:i + chunk][None], n_quantiles, tau)[0][0].mean(dim=1)
                      for i in range(0, len(features), chunk)])


@torch.no_grad()
def compare(head, features: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = head(features[None])[0]
    diff = (out - reference).abs()
    centred = (out - out.mean(-1, keepdim=True)) - (reference - reference.mean(-1, keepdim=True))
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "max_centred_diff": float(centred.abs().max()),
            "argmax_agree": float((out.argmax(-1) == reference.argmax(-1)).float().mean())}


def fit_head(net: MyModel, features: torch.Tensor, reference: torch.Tensor, steps: int = 1500, batch: int = 2048,
             lr: float = 3e-4, mse: float = 1e-3, seed: int = 0) -> None:
    """Fit ``net.q_head`` (a LogitsHead) in place so its logits on ``features`` match ``reference``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    head = net.q_head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    target_log_p = reference.log_softmax(-1)
    for _ in range(steps):
        idx = torch.randint(0, len(features), (batch,), generator=gen).to(features.device)
        out = head(features[idx][None])[0]
        loss = (F.kl_div(out.log_softmax(-1), target_log_p[idx], log_target=True, reduction="batchmean")
                + mse * (out - reference[idx]).pow(2).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("init", type=Path, help="an IQN-head checkpoint (policy_state)")
    p.add_argument("--shards", type=Path, required=True, help="rows to fit on (e.g. the run's own shards)")
    p.add_argument("--eval-shards", type=Path, required=True, help="held-out rows to report on")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=16384, help="fit rows (x5 units)")
    p.add_argument("--newest", type=int, default=60, help="draw fit rows from the newest this many shards")
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.init, map_location=args.device, weights_only=True)
    iqn = MyModel.from_state_dict(ckpt["policy_state"]).to(args.device).eval()
    if iqn.policy_head != "iqn":
        raise SystemExit(f"{args.init} already has a {iqn.policy_head} head")
    net = MyModel.from_state_dict(ckpt["policy_state"], policy_head="logits").to(args.device).eval()

    fit_x = own_unit_features(iqn, load_rows(args.shards, args.rows, args.seed, args.newest))
    eval_x = own_unit_features(iqn, load_rows(args.eval_shards, args.eval_rows, args.seed + 1))
    fit_y, eval_y = iqn_logits(iqn, fit_x), iqn_logits(iqn, eval_x)
    stats = {"fold": compare(net.q_head, eval_x, eval_y)}
    fit_head(net, fit_x, fit_y, steps=args.steps, seed=args.seed)
    stats["fit"] = compare(net.q_head, eval_x, eval_y)
    stats.update(fit_units=len(fit_x), eval_units=len(eval_x), steps=args.steps)
    print(json.dumps(stats, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    distill = dict(ckpt.get("distill", {}), policy_head="logits", head_fit={**stats, "source": str(args.init)})
    torch.save({"policy_state": {k: v.cpu() for k, v in net.state_dict().items()}, "distill": distill}, args.out)
    print(f"[head_fit] wrote {args.out}")


if __name__ == "__main__":
    main()
