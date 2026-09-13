"""
Learner process (see package docstring): the real qmix_trainer.QMIXTrainer, built with env=None
(the same mode offline_pretrain.py already relies on -- see QMIXTrainer's own docstring), fed by
actor.py's transitions instead of driving env.step() itself.

Everything QMIXTrainer already owns is reused completely unmodified: buffer_a/buffer_b.push(),
train_step() (n-step/gamma/PER-beta annealing, Double-DQN target, SPR loss, target/EMA updates,
TensorBoard, gradient clipping -- all of it), maybe_reset() (periodic shrink-and-perturb), save().
The only new logic here is the outer loop that used to be QMIXTrainer.collect_step(): draining
the transition queue into the buffers, deriving the shared bootstrap/epsilon-anchor flags actors
need (see shared.py), pacing train_step() calls against the configured replay ratio
(cfg.train_every / cfg.grad_steps_per_call) now that "one env step" arrives asynchronously from
many actors instead of one at a time from a single collect_step() call, and periodically writing
net/ema_net weights out for inference_server.py to pick up.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import torch

from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

from .shared import SharedState


def run_learner(
    cfg: QMIXConfig,
    transition_queue,
    shared: SharedState,
    total_env_steps: int,
    weights_path: str,
    weight_sync_interval: int,
    print_interval: int = 1000,
) -> None:
    trainer = QMIXTrainer(env=None, config=cfg)
    trainer._total_env_steps_hint = total_env_steps

    ckpt_dir = Path(trainer.cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_file = Path(weights_path)
    weights_file.parent.mkdir(parents=True, exist_ok=True)

    def sync_weights() -> None:
        # Write-then-rename: an inference_server.py reload racing a half-written file would
        # otherwise load a torn state_dict (see that module's maybe_reload try/except).
        tmp = weights_file.with_suffix(".tmp")
        torch.save({"net": trainer.net.state_dict(), "ema_net": trainer.ema_net.state_dict()}, tmp)
        tmp.replace(weights_file)

    sync_weights()  # give inference_server.py something to load before the first train_step

    steps_since_train = 0.0
    last_ckpt_bucket = 0
    last_print_step = 0
    t0 = time.time()
    recent_losses: list[float] = []

    try:
        while trainer.env_step_count < total_env_steps and not shared.stop.is_set():
            try:
                msg = transition_queue.get(timeout=0.5)
            except queue.Empty:
                trainer.env_step_count = shared.global_step.value
                continue

            drained = [msg]
            try:
                while True:
                    drained.append(transition_queue.get_nowait())
            except queue.Empty:
                pass

            for stream, graphic, team_state, agent_states, actions, reward, done in drained:
                buf = trainer.buffer_a if stream == "a" else trainer.buffer_b
                buf.push(graphic, team_state, agent_states, actions, reward, done)

            trainer.env_step_count = shared.global_step.value
            # Each real env.step() produces exactly one 'a' and one 'b' message (see actor.py) --
            # dividing by 2 recovers "env steps drained this round" from "messages drained".
            steps_since_train += len(drained) / 2.0

            bootstrapping = trainer._bootstrapping
            if trainer._was_bootstrapping and not bootstrapping:
                trainer._phase2_epsilon_anchor_step = trainer.env_step_count
                shared.phase2_anchor_step.value = trainer.env_step_count
                print(
                    f"[learner] buffers reached heuristic_fill_frac at env step {trainer.env_step_count} "
                    "-- switching actors to epsilon-mixed (model + heuristic) action selection"
                )
            trainer._was_bootstrapping = bootstrapping
            shared.bootstrapping.value = bootstrapping

            trainer.maybe_reset()

            while steps_since_train >= cfg.train_every:
                for _ in range(cfg.grad_steps_per_call):
                    loss = trainer.train_step()
                    if loss is not None:
                        recent_losses.append(loss)
                steps_since_train -= cfg.train_every

                if trainer.train_step_count % weight_sync_interval == 0:
                    sync_weights()

            ckpt_bucket = trainer.env_step_count // cfg.checkpoint_interval
            if ckpt_bucket > last_ckpt_bucket:
                trainer.save(ckpt_dir / f"step_{trainer.env_step_count}.pt")
                last_ckpt_bucket = ckpt_bucket

            if trainer.env_step_count - last_print_step >= print_interval:
                elapsed = time.time() - t0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
                print(
                    f"[learner] env_step={trainer.env_step_count} train_step={trainer.train_step_count} "
                    f"avg_loss={avg_loss:.4f} env_steps/s={trainer.env_step_count / max(elapsed, 1e-9):.1f} "
                    f"buffer_a={len(trainer.buffer_a)} buffer_b={len(trainer.buffer_b)}"
                )
                recent_losses.clear()
                last_print_step = trainer.env_step_count
    finally:
        trainer.save(ckpt_dir / "final.pt")
        trainer.tb.close()
        shared.stop.set()
