from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pendulum_lab.motors import get_sim_torque_limit, load_motor_profile
from pendulum_lab.utils.paths import resolve_project_path


class DoubleRotaryPendulumEnv(gym.Env):
    """Gymnasium-style environment for a double rotary inverted pendulum."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config: dict[str, Any], render_mode: str | None = None) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is not installed. Run: python -m pip install -r requirements.txt") from exc

        self.mujoco = mujoco
        self.config = config
        self.env_config = config["env"]
        self.reward_config = config["reward"]
        self.motor_profile = load_motor_profile(config.get("motor_profile"))
        self.render_mode = render_mode
        self.model_path = resolve_project_path(config["model_path"])
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(self.env_config["frame_skip"])
        self.episode_length = int(self.env_config["episode_length"])
        self.torque_limit = get_sim_torque_limit(
            self.motor_profile,
            fallback=float(self.env_config["torque_limit"]),
        )
        self.reset_noise = float(self.env_config["reset_noise"])
        self.terminate_angle = float(self.env_config["terminate_angle"])
        self.step_count = 0
        self.viewer = None

        self.action_space = spaces.Box(
            low=np.array([-self.torque_limit], dtype=np.float32),
            high=np.array([self.torque_limit], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        noise = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=6)
        self.data.qpos[:] = np.array([0.0, noise[1], noise[2]], dtype=np.float64)
        self.data.qvel[:] = np.array([noise[3], noise[4], noise[5]], dtype=np.float64)
        self.mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        torque = float(np.clip(action[0], -self.torque_limit, self.torque_limit))
        self.data.ctrl[0] = torque

        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward = self._reward(torque)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.episode_length
        info = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "torque": torque,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self) -> np.ndarray:
        yaw, p1, p2 = self.data.qpos
        yaw_vel, p1_vel, p2_vel = self.data.qvel
        return np.array(
            [
                np.sin(yaw),
                np.cos(yaw),
                np.sin(p1),
                np.cos(p1),
                np.sin(p2),
                np.cos(p2),
                yaw_vel,
                p1_vel,
                p2_vel,
            ],
            dtype=np.float32,
        )

    def _reward(self, torque: float) -> float:
        _, p1, p2 = self.data.qpos
        velocities = self.data.qvel
        upright = np.cos(p1) + np.cos(p2)
        velocity_cost = float(np.sum(np.square(velocities)))
        action_cost = torque * torque
        return float(
            self.reward_config["alive_bonus"]
            + self.reward_config["upright_weight"] * upright
            - self.reward_config["velocity_weight"] * velocity_cost
            - self.reward_config["action_weight"] * action_cost
        )

    def _is_terminated(self) -> bool:
        _, p1, p2 = self.data.qpos
        return bool(abs(p1) > self.terminate_angle or abs(p2) > self.terminate_angle)


def make_env(config_path: str | Path, render_mode: str | None = None) -> DoubleRotaryPendulumEnv:
    from pendulum_lab.config import load_config

    config = load_config(config_path)
    return DoubleRotaryPendulumEnv(config=config, render_mode=render_mode)
