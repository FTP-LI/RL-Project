"""Device-side observation and sampling interface for native backward episodes."""

from typing import NamedTuple

import torch

from walk_backward_online_focus import BackwardStallFocus


class BackwardStep(NamedTuple):
    observations: object
    reward: torch.Tensor
    terminated: torch.Tensor
    timed_out: torch.Tensor
    completed_age: torch.Tensor
    forward_velocity: torch.Tensor
    actor_weight: torch.Tensor


class NativeBackwardRollout:
    def __init__(self, env, joint_names, model_home, policy_command, focus_plan):
        assert env.cfg.auto_reset and env.cfg.is_finite_horizon and env.step_dt == 0.02
        self.env = env
        self.robot = env.scene["robot"]
        joint_ids, found = self.robot.find_joints(joint_names, preserve_order=True)
        assert found == joint_names
        default_pos = self.robot.data.default_joint_pos[:, joint_ids]
        self.home_delta = default_pos - default_pos.new_tensor(model_home)
        self.policy_command = default_pos.new_tensor(policy_command)
        self.focus = BackwardStallFocus(
            env.num_envs, focus_plan["block_controls"], focus_plan["maximum_absolute_block_speed_m_s"],
            focus_plan["focus_stop_control"], focus_plan["focused_actor_loss_multiplier"], env.device,
        )

    @torch.no_grad()
    def inputs(self, observations):
        actor = observations["actor"].clone()
        assert actor.shape == (self.env.num_envs, 61)
        actor[:, 6:20].add_(self.home_delta)
        actor[:, 34:48].copy_(self.env.action_manager.action)
        actor[:, 48:51].copy_(self.policy_command)
        # The critic receives the external command and a finite-horizon clock.
        native_critic = observations["critic"]
        remaining = (1 - self.env.episode_length_buf / self.env.max_episode_length).to(native_critic.dtype)
        critic = torch.cat((native_critic, remaining.unsqueeze(-1)), dim=-1)
        return actor, critic

    @torch.no_grad()
    def advance(self, raw_action, focused):
        # mjlab resets finished rows inside step, so preserve the old episode clock.
        age = self.env.episode_length_buf.clone()
        weight = self.focus.actor_weights(age, focused)
        observations, reward, terminated, timed_out, _ = self.env.step(raw_action)
        velocity = self.robot.data.root_link_lin_vel_b[:, 0]
        self.focus.observe(velocity, age + 1, terminated | timed_out)
        return BackwardStep(observations, reward, terminated, timed_out, age + 1, velocity, weight)
