"""Compute one causal right-turn startup credit using pre-reset simulation data."""

import torch


class RightSupportCredit:
    def __init__(self, worlds, device, settings, reward):
        self.first, self.last = settings["trigger_window_one_based"]
        self.gyro_limit = settings["trigger_abs_mean_observed_gyro_max_rad_s"]
        self.support_controls = settings["support_controls"]
        self.yaw_controls = settings["subsequent_yaw_controls"]
        self.yaw_threshold = settings["joint_physical_and_observed_yaw_min_rad_s"]
        self.deadline = settings["decision_deadline_control"]
        self.reward = reward
        self.cursor = 0
        self.trigger_sum = torch.zeros(worlds, device=device, dtype=torch.float64)
        self.trigger_samples = torch.zeros(worlds, device=device, dtype=torch.int32)
        self.eligible = torch.zeros(worlds, device=device, dtype=torch.bool)
        self.credited = torch.zeros_like(self.eligible)
        self.streak = torch.zeros((worlds, 2), device=device, dtype=torch.int32)
        self.yaw_buffer = torch.zeros((self.yaw_controls, worlds, 2), device=device)
        self.confirmations = torch.zeros((self.yaw_controls, worlds), device=device, dtype=torch.bool)
        self.yaw_sum = torch.zeros((worlds, 2), device=device)

    def advance(self, age, right_command, terminal, gyro_z, physical_yaw_z, contact):
        restarted = age == 1
        valid = right_command & ~terminal & torch.isfinite(gyro_z) & torch.isfinite(physical_yaw_z)
        self.trigger_sum.masked_fill_(restarted, 0.)
        self.trigger_samples.masked_fill_(restarted, 0)
        self.eligible &= valid & ~restarted
        self.credited &= ~restarted
        self.streak.masked_fill_(restarted[:, None], 0)
        self.yaw_sum.masked_fill_(restarted[:, None], 0.)

        window = valid & (age >= self.first) & (age <= self.last)
        self.trigger_sum += torch.where(window, gyro_z, 0.)
        self.trigger_samples += window
        count = self.last-self.first+1
        qualifies = (self.trigger_samples == count) & (self.trigger_sum.abs() <= self.gyro_limit*count)
        self.eligible |= (age == self.last) & qualifies & valid

        single = contact & (contact.sum(dim=1) == 1)[:, None] & valid[:, None]
        self.streak = (self.streak+1)*single
        slot = self.cursor
        delayed_support = self.confirmations[slot] & (age > self.yaw_controls)
        signed = torch.stack((-gyro_z, -physical_yaw_z), dim=1)
        self.yaw_sum += torch.where(valid[:, None], signed, 0.)
        self.yaw_sum -= torch.where((age > self.yaw_controls)[:, None], self.yaw_buffer[slot], 0.)
        self.yaw_buffer[slot].copy_(torch.where(valid[:, None], signed, 0.))
        # The stored confirmation precedes the current trailing yaw window by exactly one window.
        confirmation = self.eligible & (age >= self.last+self.support_controls) & (self.streak >= self.support_controls).any(dim=1)
        self.confirmations[slot].copy_(confirmation)
        self.cursor = (slot+1) % self.yaw_controls

        yaw_ok = (self.yaw_sum >= self.yaw_threshold*self.yaw_controls).all(dim=1)
        credit = valid & self.eligible & ~self.credited & delayed_support & yaw_ok & (age <= self.deadline)
        self.credited |= credit
        return credit.to(gyro_z.dtype)*self.reward
