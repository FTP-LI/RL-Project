"""Average the five solved contact samples that belong to a get-up control."""

import torch
import warp as wp

from getup_support_credit_device import SolvedFloorLoad


class GetupSupportCredit:
    def __init__(self, env, *, first, last, per_foot, total_foot, discount):
        assert (env.physics_dt, env.step_dt, env.cfg.decimation) == (.004, .02, 5)
        assert env.cfg.auto_reset and (env.max_episode_length, first, last) == (400, 124, 150)
        self.env, self.first, self.last = env, first, last
        self.normalization = sum(discount ** i for i in range(last - first))
        self.solved = SolvedFloorLoad(env, single_foot_minimum=per_foot, both_feet_minimum=total_foot)
        self.accumulated = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self.zero = self.accumulated.clone()
        self.delta = self.zero
        self.samples = 0
        self.readback_hook = None
        self.torch_stream = torch.cuda.current_stream(env.device)
        self.read_stream = wp.stream_from_torch(self.torch_stream)
        self.original_step, self.original_physics = env.step, env.sim.step
        self.age = -1
        self.capture = False
        # The original physics step still writes and integrates the motor
        # command. Read its solved contacts before the later forward refresh.
        env.step, env.sim.step = self.step, self.physics

    def step(self, action):
        self.age = int(self.env.common_step_counter) % self.env.max_episode_length
        self.capture = self.first <= self.age < self.last
        self.samples = 0
        if self.capture:
            self.accumulated.zero_()
        output = self.original_step(action)
        assert self.samples == (5 if self.capture else 0)
        return output

    def physics(self):
        self.original_physics()
        if not self.capture:
            return
        # Join the solver's Warp stream to the current Torch stream both
        # before reading the contact buffer and before the next solve.
        assert torch.cuda.current_stream(self.env.device) == self.torch_stream
        with wp.ScopedStream(self.read_stream, sync_enter=True, sync_exit=True):
            self.solved.capture()
            self.accumulated.add_(self.solved.torch_credit())
        self.samples += 1
        if self.readback_hook is not None:
            self.readback_hook(self.age, self.samples)

    def reward_rate(self, enabled):
        if not self.capture:
            assert self.samples == 0
            self.delta = self.zero
        else:
            assert self.samples == 5
            self.solved.check()
            release = self.env._early_support_release
            assert bool(release.latched.all())
            self.delta = torch.where(release.supine, self.accumulated / 5., 0.)
        # PPO already discounts across controls. Normalizing here makes 0.35
        # the largest discounted addition over the entire reward window.
        return (self.delta if enabled else self.zero) / (self.normalization * self.env.step_dt)


def support_credit_rate(env, *, enabled, **_):
    return env._support_transfer.reward_rate(enabled)
