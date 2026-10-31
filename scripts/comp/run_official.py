"""The official competition runner (blackout_env.competition.run_competition, ist-tech-AI-games/blackout-env),
unchanged except the GPU it uses: MPS daemon and team containers on ``COMP_GPU`` (default 1) instead of 0, so it
can share a server whose GPU 0 is busy.  Run from the official checkout's venv:

    COMP_GPU=1 xvfb-run -a ~/official-v102/.venv/bin/python run_official.py --build ... --team-a ... --team-b ...
"""
import os
from dataclasses import dataclass

import blackout_env.competition.run_competition as rc
from blackout_env.competition import container

GPU = int(os.environ.get("COMP_GPU", "1"))


@dataclass
class Mps(container.MpsDaemon):
    device: int = GPU


@dataclass
class Team(container.TeamContainer):
    device: int = 0  # index inside the MPS daemon's view; the container gets the physical GPU below

    def start(self) -> None:
        orig = container.subprocess.run

        def run(cmd, *a, **k):  # --gpus device=<physical>
            if isinstance(cmd, list) and cmd[:2] == ["docker", "run"]:
                cmd = [f"device={GPU}" if c.startswith("device=") else c for c in cmd]
            return orig(cmd, *a, **k)

        container.subprocess.run = run
        try:
            super().start()
        finally:
            container.subprocess.run = orig


rc.MpsDaemon = Mps
rc.TeamContainer = Team

if __name__ == "__main__":
    rc.main()
