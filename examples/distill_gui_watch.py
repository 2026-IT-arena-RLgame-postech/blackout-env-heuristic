"""Watch a distilled student play in the Unity window, e.g. while its run keeps training.

Reloads the checkpoint before every game (``latest.pt`` is replaced atomically by the trainer, so
each game shows the newest published student) and alternates sides.  Only reads the file: the
training run is not touched.

  python examples/distill_gui_watch.py checkpoints/distill/d2/latest.pt --opponent strategic_v17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from blackout_env import BlackOutEnv
from blackout_env.heuristics import POLICY_REGISTRY
from blackout_env.model.my_model import MyModel
from blackout_env.train.distill.inference import INFERENCE, make_student

try:
    from evaluate_checkpoint_vs_heuristic import play
except ImportError:
    from examples.evaluate_checkpoint_vs_heuristic import play


def load(path: Path, inference: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["policy_state"]
    net = MyModel(hidden_size=state["token_type_emb.weight"].shape[1])
    net.load_state_dict(state)
    return make_student(net, inference), ckpt.get("distill", {}).get("version"), ckpt.get("train_step_count")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--opponent", default="strategic_v17", choices=sorted(POLICY_REGISTRY))
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--inference", choices=INFERENCE, default="iqn")
    p.add_argument("--seed", type=int, default=None, help="default: random maps")
    p.add_argument("--build", default="build/mac/BlackOut.app")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = BlackOutEnv(args.build, time_scale=args.time_scale, no_graphics=False, unity_shaping=False)
    try:
        for game in range(args.games):
            student, version, step = load(args.checkpoint, args.inference)
            seed, swapped = int(rng.integers(1_000_000, 2**31)), bool(game % 2)
            print(f"[watch] game {game + 1}: student v{version} (step {step}) as team {'B' if swapped else 'A'} "
                  f"vs {args.opponent}, seed {seed}", flush=True)
            r = play(env, student, POLICY_REGISTRY[args.opponent](), swapped, seed, early_stop=True)
            outcome = "student wins" if r["winner"] == 0 else "opponent wins" if r["winner"] == 1 else "draw"
            print(f"[watch]   {outcome} {r['own']}-{r['opp']} ({r['steps']} steps)", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
