"""
Actor / Inference-server / Learner training pipeline for a multi-GPU, many-core box (designed
for: 4x RTX 2080Ti 12GB + 28-core CPU + 64GB RAM), kept entirely separate from qmix_trainer.py
(the single-process Mac/MPS pipeline, unchanged and still the right tool for a one-GPU laptop).

Why this split instead of DDP or N independent single-process runs
--------------------------------------------------------------------
qmix_trainer.py's own docstrings already diagnose this workload's actual bottleneck: one process
alternates `env.step()` (blocks on a Unity gRPC round-trip) and `train_step()` (GPU) serially, so
the GPU sits idle during every env.step() and Unity sits idle during every train_step() -- and
train_step's forward pass is itself dominated by kernel-*launch* count, not FLOPs, at this
project's batch size (64). Two consequences follow directly:

  1. More GPU compute per gradient step (i.e. torch.distributed DDP averaging gradients across
     4 GPUs) doesn't address either problem -- it adds cross-GPU sync overhead to an already
     launch-bound, not compute-bound, forward/backward pass, and 2080Tis have no NVLink so that
     sync is plain PCIe. It also can't be retrofitted without rewriting most of QMIXTrainer's
     direct submodule access (self.net.attention.layers[i].gqa, ...), which breaks the moment
     `self.net` becomes a DistributedDataParallel wrapper.
  2. What the 28-core CPU actually enables is running many *parallel Unity instances* -- each
     headless instance costs roughly 1-2 cores, so 8-16 of them fit comfortably alongside a
     learner. That's a genuine, close-to-linear win on the real bottleneck (env-step throughput),
     which the existing single-process design (and DDP) both leave on the table.

So this package splits qmix_trainer.QMIXTrainer's two halves across three process roles instead:

  - actor.py            : N processes, each owning one BlackOutEnv (own Unity instance, own
                           HeuristicPolicyMixture pair). Runs the exact same select_actions /
                           select_actions_heuristic / bootstrap / epsilon-mixing / self-play-
                           opponent logic as qmix_trainer.QMIXTrainer.collect_step(), just with
                           the net/ema_net forward pass factored out to inference_server.py (see
                           below) and buffer.push() factored out to learner.py -- everything else
                           (which team is "online" this episode, heuristic-opponent episodes,
                           absorption-as-episode-boundary, per-episode heuristic reset) is
                           unchanged from that method. CPU/Unity-bound; this is what the box's 28
                           cores are for.
  - inference_server.py : ONE process per group, holding its own copy of net/ema_net on a
                           dedicated GPU. Batches every actor's pending action-selection request
                           from this round into a single forward pass (B = 2 * n_pending_actors)
                           instead of N tiny B=2 forward passes -- turning "many small Unity
                           instances" into the large, GPU-saturating batch this workload never
                           had a natural source of before. Refreshes its net/ema_net weights from
                           whatever the learner most recently wrote to disk (see weight_sync_
                           interval) -- a bounded staleness no worse than what the self-play
                           opponent (ema_net) or any replay-buffer-based off-policy method already
                           tolerates.
  - learner.py           : ONE process per group, holding the real QMIXTrainer (built with
                           env=None -- the same mode offline_pretrain.py already uses) and the
                           replay buffers. Drains actor transitions into buffer_a/buffer_b and
                           calls train_step() continuously, GPU-bound and otherwise never waiting
                           on Unity at all. Periodically checkpoints and writes its weights out
                           for inference_server.py to pick up.

A "group" is one inference_server + one learner (2 GPUs) + N actors (however many cores are
budgeted to it) -- see launch_group.py. launch_all.py runs 2 such groups side by side across all
4 GPUs (e.g. group 0 on GPUs 0-1, group 1 on GPUs 2-3, cores split between them) as two
independent experiments (different seeds/hyperparameters), the same "subprocess per independent
worker" convention collect_heuristic_dataset_parallel.py already uses elsewhere in this codebase.

shared.py holds the ONLY state that has to cross process boundaries beyond the queues below:
whether the learner's buffers are still in the phase-1 heuristic-fill bootstrap (actors need this
to decide whether to even bother calling the inference server) and the epsilon-anneal anchor step
(so phase 2 exploration starts at eps_start the same way qmix_trainer.py's does). Everything else
that QMIXTrainer already owns internally (n_step/gamma/PER-beta annealing, target/EMA updates,
losses, TensorBoard) stays put on the learner, untouched.

fake_env.py provides a `FakeBlackOutEnv` used only by --smoke-test, so the actor/inference/learner
wiring (queues, shared state, weight sync, checkpointing) can be validated end to end on any
machine (no Unity build, no CUDA) before trusting it on the real 4-GPU box.
"""
