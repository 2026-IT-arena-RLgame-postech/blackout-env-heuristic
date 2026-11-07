"""Watch one match live (matplotlib window): our 50 Hz simulator (bit-exact with the official v1.0.2 rules), both players
on CPU.  A player is ``net:<checkpoint.pt>`` (PPO/distilled student) or a heuristic alias / descriptor (``B6``, ``V34c``).

    PYTHONPATH=. .venv/bin/python examples/watch_sim.py --a net:rl_bp14_1200.pt --b B6 --seed 7 --every 4

Team A (blue) is player ``--a``.  Collector = circle, Hunter = triangle, Carrier = square; a yellow edge = carrying an
item.  Items on the floor: battery = green dot, special = magenta star.  ``--every`` = sim ticks per drawn frame
(50 ticks = 1 s of game time); ``--fps`` caps the drawing rate.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("BLACKOUT_TPD", "1")

import numpy as np  # noqa: E402

from blackout_env.official.render import official_background  # noqa: E402
from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.batch import N_AGENT_COLS, BatchSim, HeuristicPolicy, NetPolicy  # noqa: E402
from blackout_env.sim.mapdata import CARRIER, COLLECTOR, HUNTER, H, W  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402


def make_player(spec: str, seed: int):
    if spec.startswith("net:"):
        return NetPolicy.from_checkpoint(spec[4:], device="cpu"), os.path.basename(spec[4:])
    from blackout_env.rl.grpo import _heuristic_aliases

    return HeuristicPolicy(_heuristic_aliases().get(spec, spec), seed), spec[:24]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--every", type=int, default=4, help="sim ticks per frame")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--max-s", type=float, default=425.0)
    p.add_argument("--video", default="", help="write an mp4 (ffmpeg) at --fps frames/s instead of a live window")
    p.add_argument("--dpi", type=int, default=100)
    a = p.parse_args()

    import matplotlib
    if a.video:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pa, na = make_player(a.a, a.seed)
    pb, nb = make_player(a.b, a.seed + 1)
    bs = BatchSim([SimState.from_seed(a.seed)])
    bs.ticks_per_decision = 1
    for pol in (pa, pb):
        pol.reset(1)
    packed = all(getattr(q, "accepts_packed", False) or not q.needs_inputs for q in (pa, pb))
    g = np.zeros((1, 2, H, W), np.int16) if packed else np.zeros((1, 2, 13, H, W), np.float32)
    bufs = (g, np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 10, N_AGENT_COLS), np.float32))
    need = pa.needs_inputs or pb.needs_inputs
    act = np.array([0])
    moves = np.zeros((1, 10, 2), np.float32)

    bg = official_background(bs.state(0).layout)
    cmap = plt.matplotlib.colors.ListedColormap(["#f4f1ea", "#3b3b3b", "#9ec5ff", "#ffb3b3"])
    if not a.video:
        plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8.4))
    writer = None
    if a.video:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=a.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        writer.setup(fig, a.video, dpi=a.dpi)
    ax.imshow(np.clip(bg, 0, 3), cmap=cmap, vmin=0, vmax=3, extent=(0, W, 0, H), interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_xticks([]); ax.set_yticks([])
    marker = {COLLECTOR: "o", HUNTER: "^", CARRIER: "s"}
    dyn = []
    t_last = time.perf_counter()
    tick = 0
    while not bs.over[0] and tick < a.max_s * 50:
        if need:
            (bs.render_packed if packed else bs.render)(act, *bufs)
        moves[0, :5] = pa.moves(bs, act, (0,), bufs)[0]
        moves[0, 5:] = pb.moves(bs, act, (1,), bufs)[1]
        bs.step(moves, act)
        tick += 1
        if tick % a.every:
            continue
        for d in dyn:
            d.remove()
        dyn = []
        it = bs.it[0]
        fl = np.nonzero((it[:, core.I_STATE] == core.GROUND) & (it[:, core.I_TX] >= 0))[0]
        bat = fl[it[fl, core.I_TYPE] == 0]
        spc = fl[it[fl, core.I_TYPE] != 0]
        if len(bat):
            dyn.append(ax.scatter(it[bat, core.I_TX] + 0.5, it[bat, core.I_TY] + 0.5, s=18, c="#2e9e44", zorder=2))
        if len(spc):
            dyn.append(ax.scatter(it[spc, core.I_TX] + 0.5, it[spc, core.I_TY] + 0.5, s=90, c="#c026d3", marker="*", zorder=2))
        uf, ui = bs.uf[0], bs.ui[0]
        for u in range(10):
            team = 0 if u < 5 else 1
            held = ui[u, core.U_HELD] > 0
            dyn.append(ax.scatter([uf[u, 0]], [uf[u, 1]], s=150, marker=marker.get(int(ui[u, core.U_CLS]), "o"),
                                  c="#1f5fd1" if team == 0 else "#d1342f", edgecolors="#ffd400" if held else "black",
                                  linewidths=2.5 if held else 0.8, zorder=3))
        sa, sb = bs.scores[0]
        ax.set_title(f"A (blue) {na}: {sa}    vs    B (red) {nb}: {sb}      t={tick / 50:5.1f}s", fontsize=10)
        if writer is not None:
            writer.grab_frame()
            continue
        fig.canvas.draw_idle()
        plt.pause(0.001)
        wait = 1.0 / a.fps - (time.perf_counter() - t_last)
        if wait > 0:
            time.sleep(wait)
        t_last = time.perf_counter()
    sa, sb = bs.scores[0]
    ax.set_title(f"FINAL  A {na}: {sa}  vs  B {nb}: {sb}   ({tick / 50:.1f}s, over={bool(bs.over[0])})", fontsize=10)
    print(f"final A {sa} B {sb} t={tick / 50:.1f}s")
    if writer is not None:
        for _ in range(int(2 * a.fps)):   # hold the final frame 2 s
            writer.grab_frame()
        writer.finish()
        print("wrote", a.video)
        return
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
