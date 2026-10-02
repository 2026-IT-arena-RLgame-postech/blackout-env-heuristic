"""S1 baselines on hand features (features.py), fitted on the train split of the row cache.

  lead_only    ridge / logistic on the current lead alone
  lead_time    + time left and lead x time left
  hand_linear  every hand feature (score, clock, absorption clock, battery points on the floor,
               banked in each storage, carried by each team, a few interactions)
  hand_mlp     the same features through a 2x128 MLP with the value net's multi-task loss

The regression targets are the lead changes (``delta``, masked) and the remaining final change
(final - lead); the final-lead prediction is lead + that.  P(win) is a logistic fit on "won".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from blackout_env.value.data import SPLITS, unseen_group
from blackout_env.value.features import NAMES, hand_features
from blackout_env.value.metrics import evaluate

SETS = {"lead_only": ("lead",), "lead_time": ("lead", "time_left", "lead_x_time"), "hand_linear": NAMES}
CHUNK = 65_536


def cache_features(cache: dict) -> np.ndarray:
    n = len(cache["split"])
    out = np.zeros((n, len(NAMES)), np.float32)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        out[a:b] = hand_features(cache["graphic"][a:b], cache["team_state"][a:b], cache["agent_states"][a:b])
    return out


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    x1 = np.concatenate((x, np.ones((len(x), 1), x.dtype)), axis=1).astype(np.float64)
    a = x1.T @ x1 + lam * len(x1) * np.eye(x1.shape[1])
    a[-1, -1] -= lam * len(x1)  # no penalty on the intercept
    return np.linalg.solve(a, x1.T @ y.astype(np.float64))


def apply_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ w[:-1] + w[-1]


def logistic(x: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    xt = torch.from_numpy(np.concatenate((x, np.ones((len(x), 1), np.float32)), axis=1)).double()
    yt = torch.from_numpy(y.astype(np.float64))
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(xt @ w, yt) + 1e-6 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def fit_linear(x: np.ndarray, cache_rows: dict) -> dict:
    delta, mask = cache_rows["delta"], cache_rows["delta_mask"]
    lead = x[:, 0]
    models = {"delta": [ridge(x[mask[:, h]], delta[mask[:, h], h]) for h in range(delta.shape[1])],
              "final_rem": ridge(x, cache_rows["final"] - lead),
              "win": logistic(x, cache_rows["result"] == 1)}
    return models


def predict_linear(models: dict, x: np.ndarray) -> dict:
    lead = x[:, 0]
    return {"delta": np.stack([apply_linear(x, w) for w in models["delta"]], axis=1),
            "final": lead + apply_linear(x, models["final_rem"]),
            "p_win": 1.0 / (1.0 + np.exp(-apply_linear(x, models["win"])))}


class HandMLP(nn.Module):
    def __init__(self, n_in: int, n_horizons: int, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.out = nn.Linear(width, n_horizons + 1 + 3)

    def forward(self, x):
        return self.out(self.body(x))


def fit_mlp(x: np.ndarray, rows: dict, steps: int = 6000, batch: int = 2048, seed: int = 0) -> tuple:
    from blackout_env.value.model import value_loss

    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd)
    t = {k: torch.from_numpy(np.asarray(rows[k])) for k in ("delta", "delta_mask", "final", "result")}
    lead = torch.from_numpy(x[:, 0])
    net = HandMLP(x.shape[1], t["delta"].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(xt), (batch,), generator=g)
        out = net(xt[idx])
        loss, _ = value_loss(out, {k: v[idx] for k, v in t.items()}, lead[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, mu, sd


def predict_mlp(model: tuple, x: np.ndarray) -> dict:
    from blackout_env.value.model import decode_outputs

    net, mu, sd = model
    with torch.no_grad():
        out = net(torch.from_numpy((x - mu) / sd))
    return decode_outputs(out, torch.from_numpy(x[:, 0]))


def split_masks(cache: dict, matches: list[dict]) -> dict[str, np.ndarray]:
    split = np.asarray(cache["split"])
    groups = np.array([unseen_group(m) == "ood" for m in matches])
    ood = groups[np.asarray(cache["match"])]
    unseen = split == SPLITS.index("unseen")
    return {"heldout": split == SPLITS.index("heldout"), "unseen": unseen,
            "unseen_policy": unseen & ~ood, "unseen_ood": unseen & ood}


def eval_rows(cache: dict, mask: np.ndarray) -> dict:
    idx = np.flatnonzero(mask)
    return {k: np.asarray(cache[k][idx]) for k in ("delta", "delta_mask", "final", "result", "elapsed_s", "team_state")}


def save_models(path, linear: dict, mlp: tuple) -> None:
    net, mu, sd = mlp
    torch.save({"linear": {k: {kk: (np.stack(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
                           for k, v in linear.items()},
                "mlp_state": net.state_dict(), "mlp_in": len(mu), "mlp_horizons": net.out.out_features - 4,
                "mu": mu, "sd": sd}, path)


def load_models(path) -> tuple[dict, tuple]:
    blob = torch.load(path, weights_only=False)
    net = HandMLP(blob["mlp_in"], blob["mlp_horizons"])
    net.load_state_dict(blob["mlp_state"])
    linear = {k: {kk: (list(vv) if kk == "delta" else vv) for kk, vv in v.items()} for k, v in blob["linear"].items()}
    return linear, (net.eval(), blob["mu"], blob["sd"])


def predict_set(name: str, models: tuple, x: np.ndarray) -> dict:
    """Predictions of baseline ``name`` from all-feature rows ``x`` (cache_features / hand_features)."""
    linear, mlp = models
    if name == "hand_mlp":
        return predict_mlp(mlp, x)
    return predict_linear(linear[name], x[:, [NAMES.index(c) for c in SETS[name]]])


def run_baselines(cache: dict, matches: list[dict], log=print, save_path=None) -> dict:
    x = cache_features(cache)
    train = np.asarray(cache["split"]) == SPLITS.index("train")
    train_rows = eval_rows(cache, train)
    xtr = x[train]
    masks = split_masks(cache, matches)
    report, linear = {}, {}
    for name, cols in SETS.items():
        sel = [NAMES.index(c) for c in cols]
        if sel[0] != 0:
            raise ValueError("lead must be the first feature")
        models = linear[name] = fit_linear(xtr[:, sel], train_rows)
        report[name] = {split: evaluate(predict_linear(models, x[mask][:, sel]), eval_rows(cache, mask))
                        for split, mask in masks.items()}
        log(f"baseline {name} done")
    mlp = fit_mlp(xtr, train_rows)
    report["hand_mlp"] = {split: evaluate(predict_mlp(mlp, x[mask]), eval_rows(cache, mask)) for split, mask in masks.items()}
    log("baseline hand_mlp done")
    if save_path is not None:
        save_models(save_path, linear, mlp)
    return report
