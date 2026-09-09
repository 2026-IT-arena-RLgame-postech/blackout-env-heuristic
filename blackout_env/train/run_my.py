"""
Run random-policy agents in the BlackOut environment.

Usage:
    python run_random.py                        # connect to Unity Editor (Play mode)
    python run_random.py --build path/to/build  # standalone build
    python run_random.py --episodes 3           # run multiple episodes
    python run_random.py --time-scale 50        # speed up headless runs (real-time = 1.0)
    python run_random.py --graphics             # show the Unity window (default: headless)
"""

import argparse

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.model.my_policy import MyPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        default=r"C:\Projects\blackout_rl\build\BlackOut.exe",
        help="Path to the Unity build executable",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Unity Time.timeScale; use 20-100 to speed up headless runs",
    )
    parser.add_argument(
        "--graphics",
        action="store_true",
        help="Show the Unity window instead of running headless",
    )
    args = parser.parse_args()

    env = BlackOutEnv(
        env_path=args.build,
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
    )

    policy = MyPolicy()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        steps = 0

        while env.agents:
            actions = policy.act(obs)
            obs, rewards, _, _, _ = env.step(actions)

            for agent, reward in rewards.items():
                total_rewards[agent] += reward
            steps += 1

        print(f"Episode {ep + 1} | Steps: {steps}")
        for agent, reward in total_rewards.items():
            print(f"  {agent}: {reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
