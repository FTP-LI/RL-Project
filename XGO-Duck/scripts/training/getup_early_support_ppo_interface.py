"""Share the released-policy latch across both PPO arms."""
from getup_early_support_reward import SupineReleaseLatch, early_support
import torch


def paired_reward(env, *, enabled, start_age, stop_age, tilt_limit_degrees):
    support = early_support(env, start_age=start_age, stop_age=stop_age,
                            tilt_limit_degrees=tilt_limit_degrees)
    return support if enabled else torch.zeros_like(support)


class SupportCriticObservations:
    def __init__(self, env, release_age, alignment):
        self.env = env
        self.latch = SupineReleaseLatch(env.num_envs, env.device, release_age, alignment)
        env._early_support_release = self.latch

    def before_action(self, observations):
        # The freshly returned observation is the next actor input. Reusing it
        # avoids advancing the noise stream or the previous-action history.
        assert observations['actor'].shape == (self.env.num_envs, 61)
        assert observations['critic'].shape == (self.env.num_envs, 75)
        self.latch.observe(observations['actor'], self.env.episode_length_buf)
        return {**observations,
                'critic': torch.cat((observations['critic'], self.latch.supine[:, None]), dim=1)}

    def step_wrapper(self, step):
        def step_with_critic(action):
            observations, reward, terminated, timed_out, info = step(action)
            return self.before_action(observations), reward, terminated, timed_out, info

        return step_with_critic
