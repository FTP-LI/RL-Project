"""Restore the accepted gyro turn map at the continuous command boundary."""

import numpy as np
import torch

from routed_getup_sequence import FORWARD
from routed_seven_command_controller import SevenCommandSequenceController
from walking_command_controller import create_controller


class SevenCommandTurnFeedbackController(SevenCommandSequenceController):
    def __init__(self, n, device, settings, homes, limits, forward, *,
                 commands, groups, routes, ramp, **boundaries):
        super().__init__(n, device, settings, homes, limits, forward, commands=commands,
                         groups=groups, routes=routes, ramp=ramp, **boundaries)
        names = ('turn-left', 'turn-right')
        worlds = [world for name in names for world in groups[name]]
        assert worlds and all(commands[name][2] for name in names)
        self.turn_worlds = torch.as_tensor(worlds, device=device)
        self.turn_requests = np.asarray([commands[name] for name in names for _ in groups[name]], dtype=np.float32)
        self.turn_map = create_controller(len(worlds), .02)

    def advance(self, control, sensed):
        x, bridge = super().advance(control, sensed)
        if control >= self.forward_at:
            # The fixed walking evaluation initializes the EMA at the first
            # turn sample. Match its clock and feed the existing noisy gyro.
            worlds = self.turn_worlds
            yaw = self.turn_map.map(self.turn_requests, sensed[worlds].detach().cpu().numpy())[:, 2]
            reached = self.mode[worlds] == FORWARD
            x[worlds, 50] = torch.where(reached, torch.as_tensor(yaw, device=x.device), x[worlds, 50])
        return x, bridge
