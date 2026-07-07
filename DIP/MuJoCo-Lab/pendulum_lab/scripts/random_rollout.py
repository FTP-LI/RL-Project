from __future__ import annotations

import argparse

from pendulum_lab.config import load_config
from pendulum_lab.envs.double_rotary_pendulum import DoubleRotaryPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random actions in the MuJoCo pendulum env.")
    parser.add_argument("--config", default="configs/double_rotary_pendulum.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rollout_config = config["random_rollout"]
    env = DoubleRotaryPendulumEnv(config)

    episodes = int(rollout_config["episodes"])
    seed = int(rollout_config["seed"])
    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(
            f"episode={episode:>3} "
            f"steps={steps:>4} "
            f"reward={total_reward:>9.3f} "
            f"terminated={terminated}"
        )

    env.close()


if __name__ == "__main__":
    main()

