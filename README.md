# blackout-env

PettingZoo Parallel environment wrapper for the BlackOut Unity ML-Agents game.

## Game Overview

BlackOut is a 2-team competitive game. Each team controls 5 units on a 24×24 grid map (fixed walls; the seed places storages, batteries and items), collecting **Batteries** and depositing them into storage to accumulate score. Every 20 seconds a **storage absorption** event permanently locks in battery score — but until then, enemies can raid your storage and steal items. First team to 100 points or the highest score after 7 minutes wins.

For full game rules see [docs/gameplay_en.md](docs/gameplay_en.md) / [docs/gameplay_ko.md](docs/gameplay_ko.md).

## Start here: the Run 11 80k checkpoint

This branch (`run11-80k`) packages our strongest model so far and everything needed to rebuild,
retrain and evaluate it. Clone it next to the Unity project's `run11-80k` branch:

```bash
git clone -b run11-80k https://github.com/cucumbersaurus/blackout-env.git
git clone -b run11-80k https://github.com/cucumbersaurus/blackout.git
cd blackout-env && git lfs pull        # fetches models/run11_step80k/step_80000.pt
```

| What | Where |
|---|---|
| Checkpoint, how it was trained, setup from scratch (KO) | [models/run11_step80k/README.md](models/run11_step80k/README.md) |
| One script for every stage: `build` → `collect` → `train` → `gui` / `measure` / `elo` | [models/run11_step80k/run11_pipeline.sh](models/run11_step80k/run11_pipeline.sh) |
| Baseline numbers, how to compare runs, known weaknesses, what was tried, next ideas (KO) | [docs/run11_research_baseline.md](docs/run11_research_baseline.md) |
| Map of all docs | [docs/README.md](docs/README.md) |

The parts worth understanding first are the three we changed most, not the network or the training
loop (those are replaceable):

| Area | Start with |
|---|---|
| **Observations (Unity ↔ Python)** — bit-packed 24×24 map + one shared state vector, instead of per-unit obs | Unity repo `Documentation/changes_since_team_version.md`, then [blackout_env/env/my_obs_preprocessor.py](blackout_env/env/my_obs_preprocessor.py) (docstring) and [docs/internals.md](docs/internals.md) |
| **Reward model** — Python reward v2 (per-unit potentials, fitted weights) replacing Unity's Ψ/Φ shaping | [docs/reward_v2_design.md](docs/reward_v2_design.md) (요약 first), [blackout_env/train/reward_v2.py](blackout_env/train/reward_v2.py); Unity side: `Documentation/reward_shaping.md` |
| **Heuristics** — V1–V19 rule-based policies: evaluation opponents and the demonstrators behind the training data | [docs/heuristic_policy_catalog_ko.md](docs/heuristic_policy_catalog_ko.md), [blackout_env/heuristics/](blackout_env/heuristics/) |

```bash
models/run11_step80k/run11_pipeline.sh build   # Unity player from ../blackout -> build/mac/BlackOut.app
models/run11_step80k/run11_pipeline.sh gui     # watch the 80k checkpoint play V4
```

The rest of this README covers the environment itself: installation, the observation/action
interface, and the competition API.

## Table of Contents

- [Getting Started](#getting-started)
  - [Local Installation](#local-installation)
  - [Docker (GPU Training)](#docker-gpu-training)
- [Usage](#usage)
- [Training](#training)
  - [Legacy online self-play trainer](#legacy-online-self-play-trainer)
  - [Multi-GPU / many-core training (experimental, unverified)](#multi-gpu--many-core-training-experimental-unverified)
- [Observation Space](#observation-space)
- [Competition](#competition)
  - [Observation](#observation)
  - [Action](#action)
  - [Step 1: Define your policy](#step-1-define-your-policy-policypy)
  - [Step 2: Save a checkpoint](#step-2-save-a-checkpoint)
  - [Step 3: Run a match](#step-3-run-a-match)
  - [Different architectures per team](#different-architectures-per-team)
  - [Implementing BaseModel directly (optional)](#implementing-basemodel-directly-optional)
  - [Running this repo's QMIX checkpoints](#running-this-repos-qmix-checkpoints)
- [Utilities](#utilities)

---

## Getting Started

### Local Installation

**Python 3.10.x required.** (`mlagents-envs 1.1.0` does not support 3.11+)

> **Note:** `mlagents-envs 1.1.0` declares a `pettingzoo==1.15.0` dependency that conflicts with
> blackout-env's requirement of `pettingzoo>=1.24.0`. Since `mlagents-envs` does not actually import
> pettingzoo at runtime, install it with `--no-deps` first, then install blackout-env normally.

#### Windows — uv

```powershell
# Install uv (if not already installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Run from the repo root (where pyproject.toml is)
uv venv blackout --python 3.10
blackout\Scripts\activate
uv pip install "mlagents-envs==1.1.0" --no-deps
uv pip install cloudpickle "grpcio>=1.11.0,<=1.48.2" "Pillow>=4.2.1" "protobuf>=3.6,<3.21" "pyyaml>=3.1.0" "gym>=0.21.0" "filelock>=3.4.0"
uv pip install .
```

#### Linux — uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run from the repo root (where pyproject.toml is)
uv venv blackout --python 3.10
source blackout/bin/activate
pip install "mlagents-envs==1.1.0" --no-deps
pip install cloudpickle "grpcio>=1.11.0,<=1.48.2" "Pillow>=4.2.1" "protobuf>=3.6,<3.21" "pyyaml>=3.1.0" "gym>=0.21.0" "filelock>=3.4.0"
pip install .
```

#### conda

```bash
# Run from the repo root (where pyproject.toml is)
conda create -n blackout python=3.10.12
conda activate blackout
pip install "mlagents-envs==1.1.0" --no-deps
pip install cloudpickle "grpcio>=1.11.0,<=1.48.2" "Pillow>=4.2.1" "protobuf>=3.6,<3.21" "pyyaml>=3.1.0" "gym>=0.21.0" "filelock>=3.4.0"
pip install .
```

Add the `fast` extra (`pip install ".[fast]"`) to get numba, which JIT-compiles the heuristic
policies' distance maps; data collection and evaluation against heuristics are much slower without it.

#### PyTorch

PyTorch is required for training and competition. Install it separately according to your CUDA version — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the right command.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124      # replace cu124 with your CUDA version
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # replace cu124 with your CUDA version
```

#### TensorBoard (training logs)

Plain `pip install tensorboard` (or `uv add tensorboard`) pulls in a newer `protobuf`/`grpcio`
than `mlagents-envs` allows and will silently break the Unity gRPC connection. Install it
pinned to the same ranges used above instead:

```bash
pip install tensorboard "protobuf>=3.6,<3.21" "grpcio>=1.11.0,<=1.48.2"
uv pip install tensorboard "protobuf>=3.6,<3.21" "grpcio>=1.11.0,<=1.48.2"
```

Do **not** add `tensorboard` to `pyproject.toml`'s `dependencies` — `mlagents-envs` and `torch`
are intentionally kept outside uv's tracked dependency graph (installed via `uv pip install`
above, not `uv add`), so running `uv add`/`uv sync` afterwards re-resolves the whole project
without knowing about those pins and will bump `protobuf`/`grpcio` right back and uninstall
`torch`/`mlagents-envs`/`gym`/`matplotlib` entirely (they get treated as untracked and removed
on sync). Always use `uv pip install <pkg>` for anything added after the initial setup above.

### Docker (GPU Training)

> **Note:** Docker manages the Python environment internally — no uv or conda needed on the host.

#### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU passthrough)

#### 1. Download and extract the Unity Linux build

Download `blackout_linux_build_x86_64.zip` from the [Releases page](../../releases) and extract it:

```bash
unzip blackout_linux_build_x86_64.zip -d ~/blackout_build
chmod +x ~/blackout_build/blackout_linux_build.x86_64
```

#### 2. Configure the build path

Open `docker-compose.yml` and set the Unity build path under `volumes:`:

```yaml
volumes:
  - /path/to/blackout_build:/unity_build:ro
```

#### 3. Build the image

```bash
docker compose build
```

#### 4. Run

Set `command:` in `docker-compose.yml` to your training script, then:

```bash
docker compose up -d blackout-trainer
docker compose logs -f blackout-trainer  # optional: stream training logs
```

Inside the container the build is at `/unity_build`:

```python
env = BlackOutEnv(
    env_path="/unity_build/blackout_linux_build.x86_64",
)
```

---

## Usage

`semantic_map_config.json` is the config file shared with Unity's StreamingAssets. A default copy is bundled with the package, so `semantic_config_path` is optional. Pass it explicitly only if you need to override the defaults.

`BlackOutEnv` itself only reads `n_items` and `n_classes`; the remaining keys are used by Unity
and by the legacy `ObsPreprocessor`. The bundled file looks like:

```json
{
    "resolution_scale": 4,
    "item_id_offset": 6,
    "n_items": 5,
    "n_classes": 3,
    "ids": {
        "empty": 0,
        "wall": 1,
        "ally_storage": 2,
        "enemy_storage": 3,
        "ally_unit": 4,
        "enemy_unit": 5
    }
}
```

```python
from blackout_env import BlackOutEnv, SemanticId, team_of, team_a_agents

env = BlackOutEnv(
    env_path="path/to/BlackOut.exe",  # None = connect to running Unity Editor
    # semantic_config_path defaults to the bundled config; override if needed:
    # semantic_config_path="path/to/semantic_map_config.json",
)

obs, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### Seeding

Pass `seed` to `reset()` to make map generation and item placement reproducible.
The seed is sent to Unity via a SideChannel before each episode begins, so `UnityEngine.Random` is initialized before `OnEpisodeBegin` runs.

```python
obs, infos = env.reset(seed=42)   # reproducible episode
obs, infos = env.reset(seed=42)   # identical map/item layout
obs, infos = env.reset(seed=99)   # different layout
obs, infos = env.reset()          # unseeded — random layout
```

---

## Training

Run 11 was trained **offline**: heuristic matches are recorded to disk first, then a QMIX learner
trains on that dataset while periodically playing V4 to add on-policy rows. The reward is
recomputed in Python (reward v2, `blackout_env/train/reward_v2.py`), not taken from Unity.

```bash
# 1. record heuristic matches (18 workers, ~6 min, ~57 GB for Run 11's 1M rows per stream)
python -m blackout_env.train.collect_heuristic_dataset_parallel --build build/mac/BlackOut.app \
    --steps 1000000 --workers 18 --noise-frac 0.1 --out datasets/my_dataset

# 2. train (Run 11's exact flags live in run11_pipeline.sh's `train` stage)
python -m blackout_env.train.offline_pretrain --dataset-dir datasets/my_dataset \
    --eval-build build/mac/BlackOut.app --reward v2-fitted --steps 200000 --compile ...
```

`python -m blackout_env.train.offline_pretrain --help` documents every flag. Checkpoints go to
`checkpoints/offline/<timestamp>/` (every 5k steps plus `final.pt`), TensorBoard logs to
`runs/offline/<timestamp>/`. The easiest way to reproduce or vary Run 11 is
`models/run11_step80k/run11_pipeline.sh`, whose settings can be overridden from the environment
(`DATASET=`, `TRAIN_STEPS=`, `DEVICE=`, ...).

Code map: `offline_pretrain.py` (entry point and loop) → `qmix_trainer.py` (`QMIXConfig`, losses,
BBF resets) → `model/my_model.py` (network). See [docs/internals.md](docs/internals.md).

### TensorBoard

Requires `tensorboard` — see [TensorBoard install](#tensorboard-training-logs) above. From the
repo root:

```bash
tensorboard --logdir runs
```

Each subdirectory under `--logdir` shows up as a separate run, so pointing it at `runs/offline`
compares all offline runs side by side.

Main tag groups for an offline run (see `train_step()` in `qmix_trainer.py` and
`blackout_env/train/tb_logger.py`):

| Tag prefix | Contents |
|---|---|
| `eval/*` | Periodic matches vs V4: win rate, mean score margin, idle/blocked rates, objective counts, stall breakdown, per-side splits. Noisy: read trends |
| `onpolicy/*` | Stats of the on-policy matches collected after each eval |
| `loss/*`, `bc/*`, `iqn/*`, `q_value/*` | Loss terms (IQN, SPR, BC), BC diagnostics, quantile spread, Q statistics |
| `grad_norm/*`, `weight_norm/*` | Per network part (`graphic_encoder`, `vector_encoder`, `attention_proj`, `attention_ffn`, `token_type_emb`, `spr_head`, `q_head`, `dist_mixer`, `spr_predictor`) |
| `schedule/*` | `n_step`, `gamma`, `per_beta`, `lr`, ... |
| `buffer_source_frac/*`, `buffer_rows/*`, `per_max_priority/*`, `batch_*` | Replay composition (dataset vs on-policy) and sampled-batch statistics |
| `attention_logit_rms/*`, `mixer_clamp_pressure/*`, `probe/*`, `td_error/mean` | Model-health diagnostics ([docs/design/](docs/design/) explains how to read them) |

### Legacy online self-play trainer

`python -m blackout_env.train.qmix_trainer --build path/to/BlackOut.app --steps 1000000` still runs
the original loop: self-play against an EMA copy of the network, with the reward summed on the
Unity side. It predates reward v2, the blocked penalty and the offline pipeline, so it **cannot
reproduce Run 11**; see `--help` and the `qmix_trainer.py` module docstring before using it.

### Multi-GPU / many-core training (experimental, unverified)

> **⚠️ Status: implemented but not yet run on real hardware, and not part of Run 11.** Like the
> legacy trainer above it learns from the Unity-side reward (no reward v2, no blocked penalty, no
> exhausted-match stop), so its results aren't comparable with Run 11.
>
> **Original note:** This was built and its
> multiprocessing wiring was smoke-tested (`--smoke-test`, a fake in-process env, no Unity/GPU)
> on a machine with no CUDA GPU and no Unity build available. It has **never been run against a
> real Unity build or a real GPU**, let alone the target 4-GPU box. Treat it as a starting point
> to validate, not a proven pipeline — start with a small `--num-actors` (3-4) and watch closely
> before trusting a long run to it. `qmix_trainer.py` above is unaffected and remains the
> verified, single-process pipeline.

`blackout_env/train/parallel/` is a separate training pipeline aimed at a multi-GPU, many-core
box (designed against: 4x RTX 2080Ti 12GB + 28-core CPU + 64GB RAM), instead of the single
GPU/single Unity instance `qmix_trainer.py` above assumes. It splits each of that trainer's roles
across processes instead of running them serially in one loop:

- **actor** (many processes, CPU/Unity-bound): each owns one headless Unity instance and runs
  the same self-play/heuristic-bootstrap/epsilon-mixing rollout logic as
  `QMIXTrainer.collect_step()`, minus the network forward pass.
- **inference server** (1 process per group, 1 GPU): batches every actor's pending action-
  selection request into a single forward pass instead of many tiny ones.
- **learner** (1 process per group, 1 GPU): the real `QMIXTrainer` (built with `env=None`, same
  mode `offline_pretrain.py` uses), fed by the actors' transitions instead of driving `env.step()`
  itself.

A "group" (1 inference GPU + 1 learner GPU + N actors) is one experiment; `launch_all.py` runs 2
groups side by side across all 4 GPUs as 2 independent experiments. See
`blackout_env/train/parallel/__init__.py` for the full design rationale (why this split instead
of DDP or N independent single-process runs).

```bash
# One group (2 GPUs: one for inference, one for learning)
python -m blackout_env.train.parallel.launch_group \
    --build build/linux/BlackOut.x86_64 --steps 2000000 --num-actors 10 \
    --infer-device cuda:0 --learn-device cuda:1

# Both groups at once, across all 4 GPUs (group 0 -> GPUs 0-1, group 1 -> GPUs 2-3)
python -m blackout_env.train.parallel.launch_all \
    --build build/linux/BlackOut.x86_64 --steps 2000000

# Smoke test: validates the multiprocessing wiring only (fake env, no Unity/GPU needed)
python -m blackout_env.train.parallel.launch_group --smoke-test --steps 2000 --num-actors 2 \
    --infer-device cpu --learn-device cpu
```

Known gaps: `--resume` is not yet wired into `launch_group.py` (it exits with an error telling
you so — resume with `qmix_trainer.py` instead, for now). Weight sync from learner to inference
server is file-based and polled (`--weight-sync-interval`, default every 50 gradient steps), so
actors act on a slightly stale net — bounded staleness, same order as the self-play EMA opponent
already tolerates, not a new correctness issue, but untested at scale.

---

## Observation Space

Each agent receives a dict observation with three keys — a semantic map, this agent's
team-level game state, and a table covering all 10 units. Unit **positions live only in
`agent_states`**, not in `graphic`.

| Key | Shape | Description |
|---|---|---|
| `"graphic"` | `float32[H × W × C]` | Per-team semantic map: tile-category one-hot + battery count + item one-hot |
| `"team_state"` | `float32[4]` | `[own_score, opp_score, episode_time_left, absorption_time_left]` |
| `"agent_states"` | `float32[10, 12]` | One row per unit (all 10, both teams), ordered unit_0~9 |

`C = 8 + 1 + (n_items - 1)` (item 0 is always the stackable battery, encoded as the scalar
channel 8; the remaining `n_items - 1` item types each get a one-hot channel).

### Graphic channels (`float32[H, W, C]`)

Channels 0-7 are a tile-category one-hot (binary 0.0/1.0), channel 8 is a battery-count
scalar (not one-hot), and the rest are per-item one-hot masks:

| Channel | Name | Value |
|---|---|---|
| 0 | void | 0.0 / 1.0 |
| 1 | wall | 0.0 / 1.0 |
| 2 | site_hunter | 0.0 / 1.0 |
| 3 | site_carrier | 0.0 / 1.0 |
| 4 | spawn_ally | 0.0 / 1.0 |
| 5 | spawn_enemy | 0.0 / 1.0 |
| 6 | storage_ally | 0.0 / 1.0 |
| 7 | storage_enemy | 0.0 / 1.0 |
| 8 | battery count | `count / 15` (scalar) |
| 9 | item_1 (BuffSpeed) | 0.0 / 1.0 |
| 10 | item_2 (DebuffSpeed) | 0.0 / 1.0 |
| 11 | item_3 (BuffSize) | 0.0 / 1.0 |
| 12 | item_4 (DebuffSize) | 0.0 / 1.0 |

`ally`/`enemy` channels (4↔5, 6↔7) are already flipped to the observing agent's own team
perspective. Only the channel *labels* are flipped: the grid, the unit row order and the action
frame stay in world coordinates, so Team B sees the map from the opposite corner. `MyModel`
mirrors Team B into Team A's frame first (`blackout_env/env/team_frame.py`) so one network plays
both sides. Unit positions are **not** part of
`graphic` — see `agent_states` below.

### `agent_states` row layout (`float32[10, 12]`)

Each row describes one of the 10 units (indices fixed: 0-4 = Team A, 5-9 = Team B), from
the observing agent's own team perspective:

| Offset | Length | Field | Notes |
|---|---|---|---|
| 0-1 | 2 | `pos_x`, `pos_y` | normalized to `[-1, 1]` |
| 2 | 1 | `team` | `+1.0` = ally, `-1.0` = enemy |
| 3-8 | 6 (`n_items+1`) | `holding_item` one-hot | index 0 = nothing, index 1 = battery (value = `count / 15`, not a flat 1.0), index 2+ = other item types |
| 9-11 | 3 (`n_classes`) | `class` one-hot | this unit's class |

### `team_state` (`float32[4]`)

`[own_score, opp_score, episode_time_left, absorption_time_left]` — the two scores are divided
by the target score (100) and reordered per team so index 0 is always "my score"; the two time values (both in `[0, 1]`)
are global and identical for both teams' observations.

### Items

Effects are active as long as the **item** sits in storage. Removing or stealing the item immediately cancels the effect.

| | Item | Effect | Target |
|---|---|---|---|
| <img src="docs/images/energy_2.png" width="48"> | **Battery** | Grants points equal to item amount on deposit; permanently locked in on absorption | — |
| <img src="docs/images/Feather.png" width="48"> | **BuffSpeed** | Speed +50% while this item is in allied storage | All ally units |
| <img src="docs/images/Slow.png" width="48"> | **DebuffSpeed** | Speed −90% while this item is in allied storage | Enemy Worker units only |
| <img src="docs/images/MushRoom.png" width="48"> | **BuffSize** | Size +50% while this item is in allied storage | All ally units |
| <img src="docs/images/MushRoomBad.png" width="48"> | **DebuffSize** | Size −30% while this item is in allied storage | All enemy units |

---

## Competition

Each participant submits two files:

1. **`policy.py`** — `nn.Module` implementation (model architecture)
2. **`checkpoint.pt`** — trained weights

### Observation

Each agent receives:

```python
obs[agent] = {
    "graphic": np.ndarray,       # float32[H, W, C] — semantic map, this agent's team perspective
    "team_state": np.ndarray,    # float32[4] — [own_score, opp_score, episode_time_left, absorption_time_left]
    "agent_states": np.ndarray,  # float32[10, 12] — one row per unit, all 10 units (both teams)
}
```

See [Observation Space](#observation-space) above for the full `graphic` channel table and
`agent_states` row layout — the schema is identical here, just scoped to what a policy
author needs to size their network's inputs:

```python
n_graphic_channels = 8 + 1 + (n_items - 1)          # e.g. 13 for n_items=5
agent_state_size    = 2 + 1 + (n_items + 1) + n_classes  # e.g. 12 for n_items=5, n_classes=3
team_state_size      = 4
```


### Action

`float32[2]` — `(dx, dy)` in `[-1, 1]` per agent.

(Our own model does not regress this directly: it scores 8 compass directions with a Q-network and
sends the chosen unit vector — see `blackout_env/model/my_policy.py`.)

### Step 1: Define your policy (`policy.py`)

Subclass `nn.Module` with `forward(graphic, team_state, agent_states) → action`:

```python
# policy.py
import torch
import torch.nn as nn

class MyPolicy(nn.Module):
    """
    Input:
        graphic      : (B, C, H, W)  float32  — CHW order (convert from env's (B,H,W,C) yourself)
        team_state   : (B, 4)        float32
        agent_states : (B, 10, 12)   float32  — flattened below
    Output:
        action : (B, 2)         float32  — (dx, dy) in [-1, 1]
    """
    def __init__(self, n_graphic_channels: int, agent_state_size: int, team_state_size: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_graphic_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        vector_size = team_state_size + 10 * agent_state_size
        self.mlp = nn.Sequential(
            nn.Linear(32 * 4 * 4 + vector_size, 256), nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(
        self, graphic: torch.Tensor, team_state: torch.Tensor, agent_states: torch.Tensor
    ) -> torch.Tensor:
        cnn_out = self.cnn(graphic).flatten(1)
        vec = torch.cat([team_state, agent_states.flatten(1)], dim=1)
        return self.mlp(torch.cat([vec, cnn_out], dim=1))
```

> **Note:** `graphic` arrives from the env as `(B, H, W, C)` — convert to `(B, C, H, W)`
> with `.permute(0, 3, 1, 2)` before feeding your CNN (see the `BaseModel` example below).
> `team_state` and `agent_states` need no reshaping beyond batching; the flatten above is
> just one way to fold `agent_states` into an MLP input — the concatenation scheme itself
> isn't fixed by the env, use whatever architecture suits your policy.

### Step 2: Save a checkpoint

```python
torch.save({"policy_state": model.state_dict()}, "checkpoint.pt")

# Or as a raw state dict (pass state_dict_key=None when loading)
torch.save(model.state_dict(), "checkpoint.pt")
```

### Step 3: Run a match

```python
import json
from blackout_env import BlackOutEnv, load_checkpoint, run_match, run_series
from policy import MyPolicy  # each participant's policy file

# derive observation sizes from config
cfg = json.load(open("semantic_map_config.json"))
n_items = cfg["n_items"]
n_classes = cfg["n_classes"]
n_graphic_channels = 8 + 1 + (n_items - 1)
agent_state_size = 2 + 1 + (n_items + 1) + n_classes
team_state_size = 4

# load models
model_a = load_checkpoint(
    MyPolicy,
    "team_a/checkpoint.pt",
    state_dict_key="policy_state",   # None if raw state dict
    device="cuda",
    n_graphic_channels=n_graphic_channels,
    agent_state_size=agent_state_size,
    team_state_size=team_state_size,
)
model_b = load_checkpoint(
    MyPolicy,
    "team_b/checkpoint.pt",
    state_dict_key="policy_state",
    device="cuda",
    n_graphic_channels=n_graphic_channels,
    agent_state_size=agent_state_size,
    team_state_size=team_state_size,
)

# create environment
env = BlackOutEnv(
    env_path="path/to/BlackOut.x86_64",
)

# single match
result = run_match(env, model_a, model_b)
print(f"Winner: {'A' if result.winner == 0 else 'B' if result.winner == 1 else 'Draw'}")

# best-of-10 series (sides swap each match)
series = run_series(env, model_a, model_b, n_matches=10)
print(f"A wins: {series.model_a_wins}, B wins: {series.model_b_wins}, Draws: {series.draws}")

env.close()
```

### Different architectures per team

Each participant can use a different model architecture. Load each team's `policy.py` dynamically:

```python
import importlib.util

def load_policy_class(policy_path: str, class_name: str = "MyPolicy"):
    spec = importlib.util.spec_from_file_location("policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

PolicyA = load_policy_class("team_a/policy.py")
PolicyB = load_policy_class("team_b/policy.py")

model_a = load_checkpoint(PolicyA, "team_a/checkpoint.pt", n_graphic_channels=..., agent_state_size=..., team_state_size=...)
model_b = load_checkpoint(PolicyB, "team_b/checkpoint.pt", n_graphic_channels=..., agent_state_size=..., team_state_size=...)
```

### Implementing BaseModel directly (optional)

For custom batching or inference logic, subclass `BaseModel` directly:

```python
from blackout_env import BaseModel
import numpy as np
import torch

class MyModel(BaseModel):
    def __init__(self, checkpoint_path: str):
        from policy import MyPolicy
        net = MyPolicy(n_graphic_channels=13, agent_state_size=12, team_state_size=4)
        ckpt = torch.load(checkpoint_path, weights_only=True)
        net.load_state_dict(ckpt["policy_state"])
        net.eval()
        self._net = net

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        agents = list(obs.keys())
        graphics = torch.tensor(
            np.stack([obs[a]["graphic"] for a in agents]), dtype=torch.float32
        ).permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)
        team_states = torch.tensor(
            np.stack([obs[a]["team_state"] for a in agents]), dtype=torch.float32
        )
        agent_states = torch.tensor(
            np.stack([obs[a]["agent_states"] for a in agents]), dtype=torch.float32
        )

        with torch.no_grad():
            actions = self._net(graphics, team_states, agent_states).clamp(-1, 1).numpy()
        return {agent: actions[i] for i, agent in enumerate(agents)}
```

> `load_checkpoint`/`CheckpointModel` (`model/loader.py`) implements this exact pattern
> internally. Only subclass `BaseModel` directly if you need custom preprocessing or
> ensembling.

### Running this repo's QMIX checkpoints

Checkpoints written by our trainer (e.g. `models/run11_step80k/step_80000.pt`) are Q-networks, not
action regressors, so load them with `load_my_policy_checkpoint` instead of `load_checkpoint`:

```python
from blackout_env import BlackOutEnv, StrategicHeuristicV4, load_my_policy_checkpoint, run_series

policy = load_my_policy_checkpoint("models/run11_step80k/step_80000.pt")
env = BlackOutEnv(env_path="build/mac/BlackOut.app", time_scale=20, unity_shaping=False)
series = run_series(env, policy, StrategicHeuristicV4(), n_matches=2, seeds=[404, 404])
```

---

## Utilities

```python
from blackout_env import team_of, team_a_agents, team_b_agents

# Split obs by team
a_obs = {k: v for k, v in obs.items() if team_of(k) == 0}

# Index into graphic by channel (see the channel table under Observation Space —
# there's currently no public constants class for these; MyObsPreprocessor's
# channel constants are internal to blackout_env.env.my_obs_preprocessor)
wall_mask     = graphic[:, :, 1]                  # wall
storage_ally  = graphic[:, :, 6]                  # storage_ally
battery_count = graphic[:, :, 8]                  # scalar, count / 15

# Unit info (position, team, held item, class) comes from agent_states, not graphic
ally_positions = agent_states[agent_states[:, 2] > 0, 0:2]
```

> `SemanticId` (still exported) is tied to the old, unused `ObsPreprocessor` channel scheme
> (`EMPTY`/`ALLY_UNIT`/`ENEMY_UNIT`/...) and does **not** match the channel layout `graphic`
> actually uses now — don't use it to index into the current `graphic` obs.
