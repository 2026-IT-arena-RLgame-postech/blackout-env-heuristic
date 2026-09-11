"""
Debug script: connect to the BlackOut Unity build, reset, take one step, and
plot the raw graphic observation (unit_0) — one image per channel, plus a
tile-category composite view.

Notes
-----
- no_graphics must be False for a standalone build: with graphics disabled,
  the RenderTextureSensor on MapObsAgent never renders and every channel
  comes back all-zero.
- The graphic obs is blank right at reset (MapObsAgent broadcasts its first
  map on the following step), so this script advances one step before
  plotting.

    python debug_plot_graphic_obs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blackout_env import BlackOutEnv

ROOT = Path(__file__).parent
ENV_PATH = str(ROOT.parent / "build" / "BlackOut.exe")
SEMANTIC_CONFIG = ROOT / "semantic_map_config.json"
OUT_CHANNELS = ROOT / "debug_graphic_obs_channels.png"
OUT_COMPOSITE = ROOT / "debug_graphic_obs_composite.png"
OUT_GRAYSCALE = ROOT / "debug_graphic_obs_grayscale.png"

with open(SEMANTIC_CONFIG) as f:
    cfg = json.load(f)

# Matches MyObsPreprocessor's channel layout (see blackout_env/env/my_obs_preprocessor.py):
# 8 mutually-exclusive tile-category channels (one-hot), then a battery-count scalar
# (not one-hot — can be nonzero alongside a base category), then one one-hot channel per
# non-battery item type. cfg["ids"]/cfg["item_id_offset"] describe the older, unrelated
# ObsPreprocessor channel scheme and don't apply here.
N_BASE_CHANNELS = 8
BASE_CHANNEL_NAMES = [
    "void", "wall", "site_hunter", "site_carrier",
    "spawn_ally", "spawn_enemy", "storage_ally", "storage_enemy",
]
CHANNEL_NAMES = BASE_CHANNEL_NAMES + ["battery"] + [
    f"item_{i}" for i in range(1, cfg["n_items"])
]
N_CHANNELS = len(CHANNEL_NAMES)  # == 8 + 1 + (n_items - 1)


def main() -> None:
    print(f"Connecting to Unity build: {ENV_PATH}")
    env = BlackOutEnv(
        env_path=ENV_PATH,
        semantic_config_path=SEMANTIC_CONFIG,
        no_graphics=False,
    )
    try:
        env.reset(seed=0)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
        graphic = obs["unit_0"]["graphic"]  # (H, W, C) float32 — channels 0-7 & 9+ are
                                             # binary one-hot, channel 8 (battery) is a scalar
        print(f"graphic obs: shape={graphic.shape} dtype={graphic.dtype} "
              f"min={graphic.min():.1f} max={graphic.max():.1f}")

        n_ch = graphic.shape[2]
        cols = 4
        rows = -(-n_ch // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for c in range(n_ch):
            ax = axes[c]
            ax.imshow(graphic[:, :, c], cmap="viridis", vmin=0, vmax=1, origin="upper")
            ax.set_title(f"{c}: {CHANNEL_NAMES[c]}  (sum={graphic[:, :, c].sum():.0f})")
            ax.axis("off")
        for c in range(n_ch, len(axes)):
            axes[c].axis("off")
        fig.suptitle("unit_0 graphic obs — after reset + 1 step, per channel")
        fig.tight_layout()
        fig.savefig(OUT_CHANNELS, dpi=120)
        print(f"Saved: {OUT_CHANNELS}")

        # Only channels 0..N_BASE_CHANNELS-1 (tile category) are a true one-hot / mutually
        # exclusive set — battery(8) and item_i(9+) are overlays that can be nonzero
        # *alongside* a base category (e.g. storage_ally=1 and battery=0.67 on the same
        # tile), so argmax-ing across all channels wouldn't recover a meaningful single
        # "ID". This composite/grayscale view is therefore just the tile-category layer;
        # battery/item presence is only visible in the per-channel grid above.
        base_composite = np.argmax(graphic[:, :, :N_BASE_CHANNELS], axis=2)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        im = ax2.imshow(base_composite, cmap="tab20", vmin=0, vmax=N_BASE_CHANNELS - 1, origin="upper")
        ax2.set_title("unit_0 tile-category composite (argmax of base channels 0-7)")
        ax2.axis("off")
        cbar = fig2.colorbar(im, ax=ax2, ticks=range(N_BASE_CHANNELS))
        cbar.ax.set_yticklabels([f"{i}:{CHANNEL_NAMES[i]}" for i in range(N_BASE_CHANNELS)])
        fig2.tight_layout()
        fig2.savefig(OUT_COMPOSITE, dpi=120)
        print(f"Saved: {OUT_COMPOSITE}")

        # Grayscale version of the same base-channel composite (no colorbar/labels).
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(base_composite, cmap="gray", vmin=0, vmax=N_BASE_CHANNELS - 1, origin="upper")
        ax3.set_title("unit_0 graphic obs — grayscale tile-category map")
        ax3.axis("off")
        fig3.tight_layout()
        fig3.savefig(OUT_GRAYSCALE, dpi=120)
        print(f"Saved: {OUT_GRAYSCALE}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
