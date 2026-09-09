"""
Debug script: connect to the BlackOut Unity build, reset, take one step, and
plot the raw graphic observation (unit_0) — one image per semantic channel,
plus a composite argmax view.

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

N_CHANNELS = cfg["item_id_offset"] + cfg["n_items"]
CHANNEL_NAMES = [""] * N_CHANNELS
for name, idx in cfg["ids"].items():
    CHANNEL_NAMES[idx] = name
for i in range(cfg["n_items"]):
    CHANNEL_NAMES[cfg["item_id_offset"] + i] = f"item_{i}"


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
        graphic = obs["unit_0"]["graphic"]  # (H, W, C) float32, binary per channel
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

        composite = np.argmax(graphic, axis=2)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        im = ax2.imshow(composite, cmap="tab20", vmin=0, vmax=n_ch - 1, origin="upper")
        ax2.set_title("unit_0 semantic composite (argmax channel)")
        ax2.axis("off")
        cbar = fig2.colorbar(im, ax=ax2, ticks=range(n_ch))
        cbar.ax.set_yticklabels([f"{i}:{CHANNEL_NAMES[i]}" for i in range(n_ch)])
        fig2.tight_layout()
        fig2.savefig(OUT_COMPOSITE, dpi=120)
        print(f"Saved: {OUT_COMPOSITE}")

        # Grayscale, no per-channel split: this is the single-channel semantic ID
        # map exactly as Unity encodes it (pixel value = id), just recovered from
        # the one-hot channels instead of the raw texture (id_map == argmax(graphic)).
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(composite, cmap="gray", vmin=0, vmax=n_ch - 1, origin="upper")
        ax3.set_title("unit_0 graphic obs — grayscale semantic ID map")
        ax3.axis("off")
        fig3.tight_layout()
        fig3.savefig(OUT_GRAYSCALE, dpi=120)
        print(f"Saved: {OUT_GRAYSCALE}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
