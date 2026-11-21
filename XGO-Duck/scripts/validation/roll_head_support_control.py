"""Candidate actuator overlay for the matched head-support pulse trial."""

import numpy as np
import torch

from roll_head_support_pulse import HeadSupportPulse
from roll_stand_walk_sequence import ROLL
from run_continuous_roll_projected_entry import ProjectedRollPolicy, baseline


PULSE_FIELDS = ('active', 'latched', 'previous', 'progress', 'remaining')


class HeadSupportRollPolicy(ProjectedRollPolicy):
    def __init__(self, spec, networks, homes, sim_home, limits):
        super().__init__(spec, networks, homes, sim_home, limits)
        candidate = spec['head_support_pulse']
        self.pulse = HeadSupportPulse(spec['worlds'], candidate['entry_degrees'], candidate['controls'])
        self.joint = spec['walking']['joint_names'].index(candidate['joint'])
        self.increment = candidate['target_increment_rad']
        self.pulse_enabled = False

    def next(self, step, obs, sequence):
        target, row = super().next(step, obs, sequence)
        # The trigger uses the same pre-step observation as the model.
        # Track it in all arms so the common prefix includes the trigger state.
        selected = self.pulse.advance(baseline.cpu(obs[:, 3:6]), row['phase_used'] == ROLL)
        enabled = self.pulse_enabled & selected
        original = row['applied_targets'][:, self.joint].copy()
        mask = torch.from_numpy(enabled).to(target.device)
        target[mask, self.joint] = torch.clamp(
            target[mask, self.joint] + self.increment, min=self.limits[self.joint, 0]
        )
        applied = baseline.cpu(target[:, self.joint])
        row['applied_targets'][:, self.joint] = applied
        row.update(
            pulse_enabled=np.full(len(selected), self.pulse_enabled, dtype=bool),
            pulse_selected=selected,
            pulse_original_head_roll=original,
            pulse_applied_delta_rad=applied - original,
        )
        # Carry the applied command into the slew logic. Actor history remains raw.
        self.previous = target.clone()
        return target, row


def pulse_snapshot_slots(slots, policy):
    return [*slots, *((f'controller.pulse.{field}', policy.pulse, field) for field in PULSE_FIELDS)]
