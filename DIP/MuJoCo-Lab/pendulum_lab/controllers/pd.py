from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PDController:
    kp: float = 0.0
    kd: float = 0.1
    torque_limit: float = 2.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        yaw_vel = float(observation[6])
        torque = -self.kd * yaw_vel
        torque = float(np.clip(torque, -self.torque_limit, self.torque_limit))
        return np.array([torque], dtype=np.float32)

