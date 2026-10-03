"""A distilled student (any distill checkpoint: K-bin, logits, angle head) against opponents, Unity window visible.

Usage:
    python examples/gui_student.py checkpoints/distill/frozen/p11_v22.pt strategic_v19
    python examples/gui_student.py checkpoints/distill/frozen/p11_v22.pt strategic_v19 strategic_v23 --speed 3 --seeds 2
"""

from __future__ import annotations

import argparse

import torch

from blackout_env import BlackOutEnv
from blackout_env.train.distill.collect import MatchTask, play_labelled_match
from blackout_env.train.distill.inference import load_net, make_student


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("opponents", nargs="+", help="opponents.py descriptors (heuristic id, ckpt:<path>, ...)")
    p.add_argument("--speed", type=float, default=3.0, help="Unity time scale")
    p.add_argument("--seeds", type=int, default=1, help="seeds per opponent, each played from both sides")
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    student = make_student(load_net(ckpt["policy_state"]), "iqn")
    bins = int(ckpt.get("distill", {}).get("action_bins", 8))
    print(f"student {args.checkpoint} (v{ckpt.get('distill', {}).get('version')}, {bins} bins)", flush=True)
    env = BlackOutEnv(args.build, time_scale=args.speed, no_graphics=args.headless, unity_shaping=False)
    try:
        for opp in args.opponents:
            for i in range(args.seeds):
                for swap in (False, True):
                    # beta 0: the student drives every tick; the teacher only labels (unused here)
                    task = MatchTask(seed=args.seed_base + i, swap=swap, opponent=opp, beta=0.0,
                                     teacher="act:strategic_v23", student=args.checkpoint, shard=None, tag="gui",
                                     inference="iqn", action_bins=bins, act_arc_bins=1)
                    _, s = play_labelled_match(env, task, student)
                    side = "B" if swap else "A"
                    print(f"vs {opp} seed {args.seed_base + i} side {side}: "
                          f"{s['learner_score']:.0f} - {s['opponent_score']:.0f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
