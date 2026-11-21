"""Vector interface for the forward yaw reward comparison."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from command_input_mapping import transform
from common_walk_startup import td
from forward_low_friction_yaw_reward import ForwardFeedback, target_with_exploration


class ForwardRollout(RslRlVecEnvWrapper):
    def __init__(self, env, route, draft, screen):
        super().__init__(env, clip_actions=None)
        walking = route['walking']
        names = walking['joint_names']
        robot = env.scene['robot']
        indices, joint_order = robot.find_joints(names, preserve_order=True)
        term = env.action_manager.get_term('joint_pos')
        if joint_order != names or list(term.target_names) != names:
            raise ValueError('Walking joint order changed')
        self.home = torch.tensor(walking['model_home'], device=self.device)
        self.limits = torch.tensor(walking['joint_limits'], device=self.device)
        self.home_delta = robot.data.default_joint_pos[:, indices] - self.home
        offset = torch.as_tensor(term._offset, device=self.device)
        torch.testing.assert_close(offset, self.home.expand_as(offset), atol=1e-7, rtol=0)
        self.command = torch.tensor(transform(draft['command']), device=self.device)
        self.left_hip, self.right_hip = (20 + names.index(side + '_hip_pitch') for side in ('left', 'right'))
        self.feedback = ForwardFeedback(screen, route['feedback']['maximum_yaw_input_rad_s'], self.device)
        self.correction_cap = route['feedback']['maximum_joint_correction_rad']
        self.last_features = torch.zeros((self.num_envs, 2), device=self.device)
        self.history = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.actor_inputs = None
        self.actor = None

    def adapt(self, observations):
        current = observations['actor'].clone()
        current[:, 6:20].add_(self.home_delta)
        current[:, 34:48].copy_(self.history)
        current[:, 48:51].copy_(self.command)
        observations['actor'] = current
        self.actor_inputs = current
        return observations

    def get_observations(self):
        return self.adapt(super().get_observations())

    def reset(self):
        self.history.zero_()
        self.last_features.zero_()
        observations, extras = super().reset()
        return self.adapt(observations), extras

    def target(self, raw, mean):
        inputs = self.actor_inputs
        feature = torch.stack((inputs[:, 2], inputs[:, self.left_hip] + inputs[:, self.right_hip]), dim=1)
        difference = torch.where(self.episode_length_buf[:, None] == 0, 0., feature - self.last_features)
        alternate = inputs.clone()
        alternate[:, 50] = self.feedback.commands(feature, difference)
        target = target_with_exploration(raw, mean, self.actor(td(alternate)),
                                         self.home, self.limits, self.correction_cap)
        return target, feature

    @torch.no_grad()
    def step(self, actions):
        # The sampler just passed the same actor observations through the
        # Gaussian. Reuse its cached mean for the nominal feedback branch.
        mean = self.actor.output_distribution_params[0]
        target, features = self.target(actions, mean)
        obs, reward, done, extras = super().step(target - self.home)
        self.last_features.copy_(features)
        self.history.copy_(actions)
        self.history.masked_fill_(done[:, None].bool(), 0.)
        return self.adapt(obs), reward, done, extras
