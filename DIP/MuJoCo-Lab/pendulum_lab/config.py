from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("model_path", "env", "reward", "random_rollout"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    env = config["env"]
    for key in ("frame_skip", "episode_length", "torque_limit", "reset_noise", "terminate_angle"):
        if key not in env:
            raise ValueError(f"env.{key} is required")

    reward = config["reward"]
    for key in ("upright_weight", "velocity_weight", "action_weight", "alive_bonus"):
        if key not in reward:
            raise ValueError(f"reward.{key} is required")
