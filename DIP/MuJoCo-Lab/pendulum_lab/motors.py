from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pendulum_lab.utils.paths import resolve_project_path


def load_motor_profile(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}

    profile_path = resolve_project_path(path)
    with profile_path.open("r", encoding="utf-8") as file:
        profile = json.load(file)
    return profile


def get_sim_torque_limit(profile: dict[str, Any], fallback: float) -> float:
    simulation = profile.get("simulation", {})
    torque_limit = simulation.get("torque_limit_nm")
    if torque_limit is None:
        return fallback
    return float(torque_limit)

