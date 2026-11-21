"""Bridge the right-turn startup credit into the native reward manager."""

import torch

from walk_right_support_credit import RightSupportCredit


TERM = "right_support_startup_credit"


class SupportReward:
    def __init__(self, env, settings, amount, capture_controls):
        self.env = env
        self.clock = RightSupportCredit(env.num_envs, env.device, settings, amount)
        self.right = torch.tensor([0., 0., -.4], device=env.device)
        self.armed_control = None
        self.consumed_control = 0
        self.limit = capture_controls
        n = env.num_envs
        self.capture = {
            "age": torch.empty((self.limit, n), device=env.device, dtype=env.episode_length_buf.dtype),
            "gyro": torch.empty((self.limit, n), device=env.device),
            "yaw": torch.empty((self.limit, n), device=env.device),
            "contact": torch.empty((self.limit, n, 2), device=env.device, dtype=torch.bool),
            "terminal": torch.empty((self.limit, n), device=env.device, dtype=torch.bool),
            "credit": torch.empty((self.limit, n), device=env.device),
        }

    def attach(self, pipe):
        original = pipe.inputs

        def inputs(observations):
            actor, critic = original(observations)
            self.gyro = actor[:, 2]
            self.armed_control = self.env.common_step_counter+1
            return actor, critic

        pipe.inputs = inputs

    def compute(self):
        env = self.env
        control = env.common_step_counter
        assert control == self.armed_control and control == self.consumed_control+1
        self.consumed_control = control
        age = env.episode_length_buf
        desired = env.command_manager.get_command("twist")
        terminal = env.termination_manager.terminated | env.termination_manager.time_outs
        physical_yaw = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
        contact = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, 2, -1).any(dim=-1)
        credit = self.clock.advance(age, (desired == self.right).all(dim=1), terminal, self.gyro, physical_yaw, contact)
        if control <= self.limit:
            values = (age, self.gyro, physical_yaw, contact, terminal, credit)
            for buffer, row in zip(self.capture.values(), values, strict=True):
                buffer[control-1].copy_(row)
        return credit

    def arrays(self):
        assert self.limit <= self.consumed_control
        return {name: value.cpu().numpy() for name, value in self.capture.items()}


def credit_rate(env):
    # The manager integrates this rate over dt; the clock returns a time-integrated credit.
    return env.right_support_reward.compute()/env.step_dt
