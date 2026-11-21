"""Native right-turn observations and the early yaw-deficit reward."""

import math

import torch


class RightTurnRollout:
    """Keep the established observed-gyro command mapping on the sampling device."""

    def __init__(self, env, joint_names, model_home):
        assert env.cfg.auto_reset and env.cfg.is_finite_horizon and env.step_dt == 0.02
        self.env = env
        robot = env.scene["robot"]
        ids, found = robot.find_joints(joint_names, preserve_order=True)
        assert found == joint_names
        default = robot.data.default_joint_pos[:, ids]
        self.home_delta = default - default.new_tensor(model_home)
        self.gyro_ema = default.new_zeros(env.num_envs)
        self.alpha = 1 - math.exp(-env.step_dt / 0.2)

    @torch.no_grad()
    def inputs(self, observations):
        """Call once per control. A reset row starts its EMA at its new observation."""
        actor = observations["actor"].clone()
        actor[:, 6:20].add_(self.home_delta)
        actor[:, 34:48].copy_(self.env.action_manager.action)
        command = self.env.command_manager.get_command("twist")
        actor[:, 48:51].copy_(command)
        gyro = actor[:, 2]
        filtered = self.gyro_ema + self.alpha * (gyro - self.gyro_ema)
        self.gyro_ema.copy_(torch.where(self.env.episode_length_buf == 0, gyro, filtered))
        requested_yaw = command[:, 2]
        base_yaw = (1.5 * requested_yaw).clamp(-1.0, 1.0)
        actor[:, 50] = (base_yaw + 0.75 * (requested_yaw - self.gyro_ema)).clamp(-1.0, 1.0)
        return actor, self.critic_inputs(observations)

    def critic_inputs(self, observations):
        """The tail-value call must leave the controller memory untouched."""
        critic = observations["critic"]
        remaining = 1 - self.env.episode_length_buf.to(critic.dtype) / self.env.max_episode_length
        return torch.cat((critic, remaining.unsqueeze(-1)), dim=-1)


def startup_yaw_deficit(env, startup_controls):
    """Score the current physical state in the native pre-reset reward phase."""
    command = env.command_manager.get_command("twist")
    yaw = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    target = -command[:, 2]
    right_turn = (target > 0) & (command[:, :2] == 0).all(dim=1)
    age = env.episode_length_buf
    active = right_turn & (age >= 1) & (age <= startup_controls)
    denominator = torch.where(right_turn, target, 1.)
    deficit = ((target + yaw) / denominator).clamp(0., 1.)
    return torch.where(active, -deficit, 0.)
