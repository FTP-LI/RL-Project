"""Use the official right-turn actor only after the frozen startup check."""

import numpy as np
import torch

from check_walk_right_startup_routes import RightCommandActor


class EarlyRightActor(RightCommandActor):
    def __init__(self, official, base, desired, p, device, detector, cap, arm):
        super().__init__(official, base, base, desired, p["commands"], device)
        self.arm = arm
        self.device = device
        self.first = detector["first_observation_control"]
        self.last = detector["last_observation_control"]
        self.decision_control = detector["decision_control"]
        assert self.decision_control == self.last + 1
        self.gyro_limit = detector["absolute_mean_observed_gyro_max_rad_s"]
        self.ids = self.right_worlds.cpu().numpy()
        self.gyro = np.empty((self.last-self.first+1, len(self.ids)), np.float32)
        self.complete = np.ones(len(self.ids), bool)
        self.home = torch.tensor(p["model_home"], dtype=torch.float32, device=device)
        self.low, self.high = torch.tensor(p["joint_limits"], dtype=torch.float32, device=device).T
        self.cap = torch.tensor(cap, dtype=torch.float32, device=device)
        self.step = 0
        self.guarded = torch.zeros(len(self.ids), dtype=torch.bool, device=device)
        self.previous_target = None
        self.receipt = None

    def observe(self, obs, step, active):
        assert self.step < step
        self.step = step
        self.active = active[self.ids]
        if self.first <= step <= self.last:
            rate = obs[self.ids, 2]
            self.complete &= self.active & np.isfinite(rate)
            self.gyro[step-self.first] = np.where(np.isfinite(rate), rate, 0.)

    def forward(self, inputs):
        nominal = super().forward(inputs)
        if len(inputs["actor"]) != self.worlds or self.step < self.decision_control:
            return nominal
        if self.arm == "routed-parent":
            if self.step == self.decision_control:
                self.decide(nominal[self.right_worlds], None)
            return nominal
        # The shadow uses the same official inference and guard as the selected arm.
        alternate = self.official(inputs[self.other_ids])[self.right_rows_in_other_batch]
        if self.step == self.decision_control:
            self.decide(nominal[self.right_worlds], alternate)
        if self.arm == "shadow-official":
            return nominal
        assert self.arm == "early-official"
        active = torch.as_tensor(self.active, device=self.device)
        take = (self.guarded & active)[:, None]
        right = torch.where(take, alternate, nominal[self.right_worlds])
        return nominal.index_copy_(0, self.right_worlds, right)

    def decide(self, base, alternate):
        assert self.receipt is None
        mean = self.gyro.mean(axis=0, dtype=np.float64)
        trigger = self.complete & self.active & (np.abs(mean) <= self.gyro_limit)
        receipt = dict(control=self.step, right_worlds=self.ids.tolist(), mean_gyro_rad_s=mean.tolist(),
                       complete_window=self.complete.tolist(), active_at_decision=self.active.tolist(),
                       triggered_worlds=self.ids[trigger].tolist(), guarded_worlds=[], selected_worlds=[])
        if alternate is not None:
            previous = self.previous_target()
            assert previous.shape == alternate.shape == base.shape
            request = self.home + alternate
            target = request.clamp(self.low, self.high)
            normal = (self.home + base).clamp(self.low, self.high)
            delta = (target - previous).abs()
            too_fast = delta > self.cap + 1e-6
            limited = (request != target) & (normal != self.low) & (normal != self.high)
            finite = torch.isfinite(target).all(dim=1)
            self.guarded = torch.as_tensor(trigger, device=self.device) & finite & ~(too_fast | limited).any(dim=1)
            guarded = self.ids[self.guarded.cpu().numpy()].tolist()
            receipt.update(guarded_worlds=guarded, selected_worlds=guarded if self.arm == "early-official" else [],
                           candidates={str(self.ids[i]): dict(official_raw=alternate[i].cpu().tolist(),
                               previous_target=previous[i].cpu().tolist(), target_delta=delta[i].cpu().tolist(),
                               excessive_joints=too_fast[i].nonzero().flatten().cpu().tolist(),
                               newly_limited_joints=limited[i].nonzero().flatten().cpu().tolist(),
                               finite=bool(finite[i])) for i in np.flatnonzero(trigger)})
        self.receipt = receipt
