from __future__ import annotations

import argparse

from pendulum_lab.config import load_config
from pendulum_lab.envs.double_rotary_pendulum import DoubleRotaryPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the MuJoCo model and env can load.")
    parser.add_argument("--config", default="configs/double_rotary_pendulum.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env = DoubleRotaryPendulumEnv(config)
    obs, _ = env.reset(seed=0)
    print(f"model: {env.model_path}")
    print(f"nq: {env.model.nq}, nv: {env.model.nv}, nu: {env.model.nu}")
    print(f"obs shape: {obs.shape}")
    env.close()


if __name__ == "__main__":
    main()

