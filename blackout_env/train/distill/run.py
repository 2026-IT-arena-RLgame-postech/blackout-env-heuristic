"""Asynchronous DAgger distillation of a relabelling teacher into MyModel.

Collection and fitting run at the same time:

* **collector** -- ``--workers`` Unity processes play labelled matches in the background
  (collect.py), the student running on their CPUs.  In a match each learner unit follows the
  teacher with probability beta and the student otherwise; the teacher labels every state either
  way.  Each finished match lands as a shard on disk;
* **learner** -- this process fits the net (set-valued cross-entropy, loss.py) on the aggregate
  of every shard so far (the most recent ``--max-rows`` rows), on ``--device``.

They meet at two points:

* every ``--sync-every`` steps the learner collects finished matches, loads their shards into the
  replay ring, and queues more matches -- only while it has trained at least ``--replay-ratio``
  samples per collected row, so the (much faster) collector cannot flood the ring with rows the
  learner never sees;
* every ``--publish-every`` steps it atomically replaces ``latest.pt``; a worker reloads the
  student when it starts its next match.  Each publish is a new student *version*; beta decays
  per version, ``beta_start * beta_decay ** (version - 1)`` down to ``beta_min`` (version 0 = no
  student yet = teacher only, which also fills the ring up to ``--min-rows`` before fitting).
  Every ``--eval-every`` versions ``2 x --eval-seeds`` student-only matches against
  ``--eval-opponent`` on fixed seeds are queued ahead of collection; their states are labelled
  too.

``latest.pt`` holds ``policy_state`` = MyModel weights (load_my_policy_checkpoint and
examples/gui_match.py play it directly) plus what a restart needs: re-running with the same
``--out-dir`` resumes from it.

Example (smoke test)::

    python -m blackout_env.train.distill.run --build build/mac/BlackOut.app --out-dir checkpoints/distill/smoke \\
        --steps 120 --workers 2 --min-rows 2000 --publish-every 40 --eval-every 2 --eval-seeds 1 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from blackout_env.model.my_model import MyModel
from blackout_env.train.distill.collect import CLASS_COLS, TEACHERS, CollectorPool, MatchTask
from blackout_env.train.distill.dataset import GRAPHIC_SCALE, N_TEAM, DistillDataset
from blackout_env.train.distill.loss import set_valued_cross_entropy
from blackout_env.train.tb_logger import TBLogger

CLASS_NAMES = ("collector", "hunter", "carrier")
ROLE_KEYS = ("collectors", "hunters", "carriers")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--out-dir", type=Path, default=None, help="default: checkpoints/distill/<timestamp>")
    p.add_argument("--teacher", choices=sorted(TEACHERS), default="v17-teach")
    p.add_argument("--init", type=Path, default=None, help="start from these MyModel weights (any checkpoint with policy_state)")
    p.add_argument("--steps", type=int, default=100_000, help="total gradient steps")
    # collector
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--opponents", nargs="+", default=["mixture"], help="heuristic ids or 'mixture', cycled over matches")
    p.add_argument("--beta-start", type=float, default=1.0, help="beta of the first student version")
    p.add_argument("--beta-decay", type=float, default=0.9, help="per published version")
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--eval-opponent", default="strategic_v17")
    p.add_argument("--eval-seeds", type=int, default=3, help="student-only eval matches = 2 x this")
    p.add_argument("--seed", type=int, default=20260929)
    # synchronisation
    p.add_argument("--min-rows", type=int, default=20_000, help="teacher-only rows to collect before fitting")
    p.add_argument("--replay-ratio", type=float, default=2.0,
                   help="trained samples per collected row; below it no new matches are queued")
    p.add_argument("--sync-every", type=int, default=50, help="steps between replay syncs")
    p.add_argument("--publish-every", type=int, default=1000, help="steps between student versions")
    p.add_argument("--eval-every", type=int, default=5, help="versions between eval batches")
    p.add_argument("--snapshot-every", type=int, default=5, help="versions between kept v_NNNN.pt weights")
    p.add_argument("--log-every", type=int, default=200)
    # learner
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--z-loss", type=float, default=1e-4)
    p.add_argument("--class-weighting", choices=["none", "balanced"], default="none",
                   help="balanced: each unit class (Collector/Hunter/Carrier) gets equal total weight per batch, "
                   "so V17's 3-Hunter composition does not crowd out the one Collector's labels")
    p.add_argument("--max-rows", type=int, default=600_000, help="replay ring size (~8 KB per row in RAM)")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args(argv)


def beta_for(version: int, args) -> float:
    if version == 0:
        return 1.0
    return max(args.beta_min, args.beta_start * args.beta_decay ** (version - 1))


def wants_matches(trained_samples: int, collected_rows: int, dataset_rows: int, args) -> bool:
    """Queue more collection?  Always while filling up to --min-rows; afterwards only while the
    learner has used each collected row at least --replay-ratio times."""
    return dataset_rows < args.min_rows or trained_samples >= args.replay_ratio * collected_rows


def summarize(summaries: list[dict], prefix: str) -> dict[str, float]:
    margins = [s["learner_score"] - s["opponent_score"] for s in summaries]
    stats = {
        f"{prefix}/matches": len(summaries),
        f"{prefix}/mean_margin": float(np.mean(margins)),
        f"{prefix}/win_rate": float(np.mean([s["result"] == 1 for s in summaries])),
        f"{prefix}/teacher_frac": float(np.mean([s["teacher_frac"] for s in summaries])),
        f"{prefix}/rows": int(sum(s["rows"] for s in summaries)),
    }
    for role in ROLE_KEYS:  # the role ratio the learner's team actually played
        stats[f"{prefix}/{role}"] = float(np.mean([s[role] for s in summaries]))
    return stats


def class_weights(classes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, 5, 3] one-hot classes -> [B, 5] weights giving each present class equal total weight."""
    counts = (classes * valid.unsqueeze(-1)).sum(dim=(0, 1))  # [3]
    present = (counts > 0).sum().clamp_min(1)
    per_class = torch.where(counts > 0, valid.sum() / (present * counts.clamp_min(1)), torch.zeros_like(counts))
    return (classes * per_class).sum(dim=-1)


class Learner:
    """The net, its optimizer, and running training statistics (summed on-device: no per-step
    host sync)."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.net = MyModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self._bit_index = torch.arange(8, device=self.device)
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._totals = defaultdict(lambda: torch.zeros((), device=self.device))
        self._role_hits = torch.zeros(len(CLASS_NAMES), device=self.device)
        self._role_labels = torch.zeros(len(CLASS_NAMES), device=self.device)
        self._steps = 0

    def step(self, batch: dict[str, np.ndarray]) -> None:
        self.net.train()
        batch = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        graphic = batch["graphic"].permute(0, 3, 1, 2).float() / GRAPHIC_SCALE
        q_values, *_ = self.net(graphic, batch["team_state"], batch["agent_states"])
        logits = q_values[:, :N_TEAM]
        masks = ((batch["labels"].long().unsqueeze(-1) >> self._bit_index) & 1) == 1  # dataset.unpack_labels
        classes = batch["agent_states"][:, :N_TEAM, CLASS_COLS]
        valid = masks.any(dim=-1).float()
        weights = class_weights(classes, valid) if self.args.class_weighting == "balanced" else None
        loss, stats = set_valued_cross_entropy(logits, masks, weights, self.args.z_loss)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            for key, value in stats.items():
                self._totals[key] += value.detach()
            # accuracy per unit class: is each role's play being learned?
            hit = masks.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1).float() * valid
            self._role_hits += (hit.unsqueeze(-1) * classes).sum(dim=(0, 1))
            self._role_labels += (valid.unsqueeze(-1) * classes).sum(dim=(0, 1))
        self._steps += 1

    def pop_stats(self) -> dict[str, float]:
        """Means since the last call."""
        out = {f"train/{k}": float(v) / max(1, self._steps) for k, v in self._totals.items()}
        hits, labels = self._role_hits.tolist(), self._role_labels.tolist()
        for name, n_hit, n_label in zip(CLASS_NAMES, hits, labels):
            if n_label:
                out[f"train/accuracy_{name}"] = n_hit / n_label
                out[f"train/label_share_{name}"] = n_label / sum(labels)
        self._reset_stats()
        return out


class Run:
    """The learner's side of the loop: counters, checkpoints, queueing and logging."""

    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir or Path("checkpoints/distill") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.shards = self.out_dir / "shards"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.latest = self.out_dir / "latest.pt"
        self.learner = Learner(args)
        self.step = self.version = self.queued = self.collected_rows = 0
        self._restore()
        (self.out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
        self.dataset = DistillDataset(self.shards, args.max_rows)
        self.dataset.refresh()
        self.tb = TBLogger(self.out_dir / "tb")
        self.match_log = open(self.out_dir / "matches.jsonl", "a")
        self._collect_window: list[dict] = []
        self._evals: dict[int, list[dict]] = defaultdict(list)

    # ----------------------------------------------------------------------- checkpoints

    def _restore(self) -> None:
        if self.latest.exists():
            ckpt = torch.load(self.latest, map_location=self.learner.device, weights_only=True)
            self.learner.net.load_state_dict(ckpt["policy_state"])
            self.learner.optimizer.load_state_dict(ckpt["optimizer_state"])
            state = ckpt["distill"]
            self.step, self.version = ckpt["train_step_count"], state["version"]
            self.queued, self.collected_rows = state["queued"], state["collected_rows"]
            print(f"[distill] resuming at step {self.step}, student v{self.version}")
        elif self.args.init is not None:
            init = torch.load(self.args.init, map_location=self.learner.device, weights_only=True)
            self.learner.net.load_state_dict(init["policy_state"])
            self.publish()

    def publish(self) -> None:
        """A new student version: replace latest.pt atomically (workers may be reading it)."""
        self.version += 1
        payload = {
            "policy_state": self.learner.net.state_dict(),
            "optimizer_state": self.learner.optimizer.state_dict(),
            "train_step_count": self.step,
            "distill": {"version": self.version, "teacher": self.args.teacher, "queued": self.queued,
                        "collected_rows": self.collected_rows},
        }
        tmp = self.latest.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.latest)
        if self.version % self.args.snapshot_every == 0:
            torch.save({"policy_state": payload["policy_state"], "distill": payload["distill"]},
                       self.out_dir / f"v_{self.version:04d}.pt")

    # ------------------------------------------------------------------------- collection

    def _task(self, tag: str, seed: int, swap: bool, opponent: str, beta: float) -> MatchTask:
        index = self.queued
        self.queued += 1
        return MatchTask(seed=seed, swap=swap, opponent=opponent, beta=beta, teacher=self.args.teacher,
                         student=str(self.latest) if beta < 1.0 else None,
                         shard=str(self.shards / f"{index:06d}_{tag}_v{self.version:04d}.npz"),
                         tag=tag, version=self.version)

    def queue_eval(self, pool: CollectorPool) -> None:
        # the same seeds every time, so versions compare on equal maps
        for i in range(2 * self.args.eval_seeds):
            pool.submit(self._task("eval", self.args.seed - 1 - i // 2, bool(i % 2), self.args.eval_opponent, 0.0))

    def top_up(self, pool: CollectorPool) -> None:
        args = self.args
        while pool.in_flight < pool.workers and wants_matches(
                self.step * args.batch_size, self.collected_rows, self.dataset.size, args):
            n = self.queued
            pool.submit(self._task("collect", args.seed + n, bool(n % 2), args.opponents[n % len(args.opponents)],
                                   beta_for(self.version, args)))
            if self.dataset.size >= args.min_rows:
                break  # one at a time once fitting: the ratio is re-checked at the next sync

    def sync(self, pool: CollectorPool, timeout: float = 0.0) -> None:
        self._record(pool.poll(timeout))
        self.dataset.refresh()
        self.top_up(pool)

    def _record(self, summaries: list[dict]) -> None:
        for summary in summaries:
            self.match_log.write(json.dumps(summary) + "\n")
            self.collected_rows += summary["rows"]
            if summary["tag"] == "eval":
                self._log_eval(summary)
            else:
                self._collect_window.append(summary)
        self.match_log.flush()

    def _log_eval(self, summary: dict) -> None:
        group = self._evals[summary["version"]]
        group.append(summary)
        if len(group) < 2 * self.args.eval_seeds:
            return
        stats = summarize(group, "eval")
        self.tb.scalars("distill", stats, self.step)
        print(f"[distill] eval v{summary['version']} vs {self.args.eval_opponent}: margin {stats['eval/mean_margin']:+.1f} "
              f"win {stats['eval/win_rate']:.2f} H/Ca/C {stats['eval/hunters']:.1f}/{stats['eval/carriers']:.1f}/"
              f"{stats['eval/collectors']:.1f}", flush=True)
        del self._evals[summary["version"]]

    def log(self, started: float) -> None:
        stats = {"beta": beta_for(self.version, self.args), "version": self.version,
                 "dataset_rows": self.dataset.size, "collected_rows": self.collected_rows,
                 "replay_ratio": self.step * self.args.batch_size / max(1, self.collected_rows),
                 **self.learner.pop_stats()}
        if self._collect_window:
            stats.update(summarize(self._collect_window, "collect"))
            self._collect_window = []
        self.tb.scalars("distill", stats, self.step)
        collect = f" collect margin {stats['collect/mean_margin']:+.1f}" if "collect/matches" in stats else ""
        print(f"[distill] step {self.step} v{self.version} beta {stats['beta']:.2f} rows {self.dataset.size} "
              f"ratio {stats['replay_ratio']:.1f} | ce {stats['train/ce']:.3f} acc {stats['train/accuracy']:.3f}"
              f"{collect} | {time.time() - started:.0f}s", flush=True)

    # ------------------------------------------------------------------------------- loop

    def run(self) -> None:
        args = self.args
        rng = np.random.default_rng(args.seed + self.step)
        started = time.time()
        with CollectorPool(args.build, args.workers, args.time_scale) as pool:
            self.top_up(pool)
            while self.dataset.size < args.min_rows:  # teacher-only warm-up
                self.sync(pool, timeout=5.0)
            print(f"[distill] {self.dataset.size} rows, fitting", flush=True)
            while self.step < args.steps:
                self.learner.step(self.dataset.sample(args.batch_size, rng))
                self.step += 1
                if self.step % args.sync_every == 0:
                    self.sync(pool)
                if self.step % args.publish_every == 0:
                    self.publish()
                    if self.version % args.eval_every == 0:
                        self.queue_eval(pool)
                if self.step % args.log_every == 0:
                    self.log(started)
            self.publish()
            self._record(pool.drain())  # matches still running finish into the log (and shards)
        self.match_log.close()
        print(f"[distill] done: {self.latest} (v{self.version})")


def main(argv=None) -> None:
    Run(parse_args(argv)).run()


if __name__ == "__main__":
    main()
