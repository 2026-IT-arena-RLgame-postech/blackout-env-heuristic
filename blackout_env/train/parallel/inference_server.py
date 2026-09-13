"""
Inference-server process (see package docstring): one dedicated GPU, one copy of net/ema_net,
answering every actor's action-selection request for this round in a single batched forward pass
instead of each actor paying its own tiny (B=2) forward-pass launch overhead.

This is the piece that turns "many small Unity instances" into the large, GPU-saturating batch
this workload otherwise never had -- qmix_trainer.py's own train_step docstring measured its
forward pass as launch-count-bound, not compute-bound, at batch=64; batching N actors' B=2
requests together (B = 2*N) spends the same fixed launch overhead on N times the useful work.

Owns net/ema_net independently from the learner's copies and periodically reloads them from
whatever the learner most recently wrote to `weights_path` (see learner.py's sync_weights) --
this is the ONLY channel of communication between the two, and it is intentionally one-way and
loosely synchronized (polled by mtime, not pushed): the resulting staleness (actions selected
against a slightly-out-of-date net) is bounded by weight_sync_interval gradient steps, no worse
than the staleness qmix_trainer.py's own self-play opponent (ema_net) or any off-policy replay
buffer already tolerates by design.
"""

from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import torch

from blackout_env.model.my_model import MyModel
from blackout_env.train.qmix_trainer import QMIXConfig, _own_team_rows

from .shared import put_until_stop


def run_inference_server(
    cfg: QMIXConfig,
    device: str,
    request_queue,
    response_queues: dict[int, object],
    stop_event,
    weights_path: str,
    max_batch: int = 64,
    poll_timeout: float = 0.05,
) -> None:
    torch_device = torch.device(device)
    model_kwargs = dict(
        hidden_size=cfg.hidden_size,
        n_items=cfg.n_items,
        n_classes=cfg.n_classes,
        team_state_size=cfg.team_state_size,
    )
    net = MyModel(**model_kwargs).to(torch_device).eval()
    ema_net = MyModel(**model_kwargs).to(torch_device).eval()

    weights_file = Path(weights_path)
    last_mtime = 0.0

    def maybe_reload() -> None:
        nonlocal last_mtime
        try:
            mtime = weights_file.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= last_mtime:
            return
        try:
            ckpt = torch.load(weights_file, map_location=torch_device, weights_only=True)
            net.load_state_dict(ckpt["net"])
            ema_net.load_state_dict(ckpt["ema_net"])
            last_mtime = mtime
        except Exception as e:  # noqa: BLE001 -- a torn read (learner mid-write) must not crash
            # inference; sync_weights() writes to a .tmp path and renames atomically precisely
            # to make this rare, but a concurrent read can still race a same-second rewrite.
            print(f"[inference_server] failed to reload weights ({e!r}) -- keeping previous copy")

    print(f"[inference_server] ready on {device}, watching {weights_path} for weight updates")
    with torch.no_grad():
        while not stop_event.is_set():
            maybe_reload()
            try:
                first = request_queue.get(timeout=poll_timeout)
            except queue.Empty:
                continue

            batch = [first]
            while len(batch) < max_batch:
                try:
                    batch.append(request_queue.get_nowait())
                except queue.Empty:
                    break

            actor_ids = [item[0] for item in batch]
            graphic = np.concatenate([item[1] for item in batch], axis=0)      # [2N, H, W, C]
            team_state = np.concatenate([item[2] for item in batch], axis=0)   # [2N, S]
            agent_states = np.concatenate([item[3] for item in batch], axis=0)  # [2N, 10, A]

            graphic_t = torch.tensor(graphic, dtype=torch.float32, device=torch_device).permute(0, 3, 1, 2)
            team_state_t = torch.tensor(team_state, dtype=torch.float32, device=torch_device)
            agent_states_t = torch.tensor(agent_states, dtype=torch.float32, device=torch_device)

            q_online, *_ = net(graphic_t, team_state_t, agent_states_t, n_quantiles=cfg.n_quantiles)
            q_ema, *_ = ema_net(graphic_t, team_state_t, agent_states_t, n_quantiles=cfg.n_quantiles)

            greedy_online = _own_team_rows(q_online, agent_states_t).argmax(dim=-1).cpu().numpy()  # [2N, N_TEAM]
            greedy_ema = _own_team_rows(q_ema, agent_states_t).argmax(dim=-1).cpu().numpy()

            for i, actor_id in enumerate(actor_ids):
                # At most 1 response is ever outstanding per actor (each actor blocks on its own
                # get() before sending its next request -- see actor.py), so this queue can only
                # ever back up if that specific actor has died; put_until_stop's timeout keeps a
                # dead actor from ever wedging this server (see shared.py for the general hang
                # this prevents).
                put_until_stop(
                    response_queues[actor_id],
                    (greedy_online[2 * i : 2 * i + 2], greedy_ema[2 * i : 2 * i + 2]),
                    stop_event,
                )

    # This process is a producer on every response queue -- see actor.py's matching call for why
    # skipping the default flush-on-exit behavior is required, not optional, once actors may have
    # already stopped reading.
    for q in response_queues.values():
        q.cancel_join_thread()
    print("[inference_server] stopped")
