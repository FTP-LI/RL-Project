"""Extend the continuous getup handoff to the seven qualified walking routes."""

import numpy as np
import torch

from command_input_mapping import transform
from routed_forward_entry_ramp import command_table
from routed_getup_sequence import FORWARD, GETUP, STAND, SequenceController, td


def balanced_groups(pose_codes, commands, seed):
    """Balance command counts globally and inside every requested pose."""
    pose_codes = np.asarray(pose_codes)
    random = np.random.default_rng(seed)
    chosen = np.empty(len(pose_codes), dtype=np.int8)
    offset = 0
    for code in np.unique(pose_codes):
        ids = random.permutation(np.flatnonzero(pose_codes == code))
        chosen[ids] = (offset + np.arange(len(ids))) % len(commands)
        offset += len(ids)
    return {name: np.flatnonzero(chosen == i).tolist() for i, name in enumerate(commands)}


class SevenCommandSequenceController(SequenceController):
    """Share the prefix and use explicit command groups at the walking handoff."""

    def __init__(self, n, device, settings, homes, limits, forward, *,
                 commands, groups, routes, ramp, **boundaries):
        super().__init__(n, device, settings, homes, limits, forward, **boundaries)
        assert set(commands) == set(groups) == set(routes) and routes['stand'] == 'base'
        assert sorted(i for ids in groups.values() for i in ids) == list(range(n))
        self.fixed_commands = torch.empty((n, 3), dtype=torch.float32, device=device)
        self.route_masks = {role: torch.zeros(n, dtype=torch.bool, device=device) for role in dict.fromkeys(routes.values())}
        self.forward_worlds = torch.zeros(n, dtype=torch.bool, device=device)
        self.forward_worlds[groups['forward']] = True
        for name, ids in groups.items():
            self.fixed_commands[ids] = torch.as_tensor(transform(commands[name]), device=device)
            self.route_masks[routes[name]][ids] = True
        assert routes['forward'] == 'official' and torch.equal(self.homes[STAND], self.homes[FORWARD])
        self.forward_ramp = torch.as_tensor(command_table(ramp, ramp['ramp_controls']), device=device)
        torch.testing.assert_close(self.forward_ramp[-1], self.forward_command, atol=0, rtol=0)

    def advance(self, control, sensed):
        x, bridge = super().advance(control, sensed)
        if control < self.forward_at:
            return x, bridge
        terminal = self.mode == FORWARD
        x[terminal, 48:51] = self.fixed_commands[terminal]
        elapsed = control - self.forward_at
        if elapsed < len(self.forward_ramp):
            x[terminal & self.forward_worlds, 48:51] = self.forward_ramp[elapsed]
        if elapsed == 0:
            carried = terminal & self.route_masks['base']
            self.history[FORWARD, carried] = self.history[STAND, carried]
            x[carried, 34:48] = self.history[FORWARD, carried]
        return x, bridge

    def infer(self, control, x, actors):
        if control < self.forward_at:
            return super().infer(control, x, actors)
        raw = torch.zeros_like(self.history[GETUP])
        yaw = torch.zeros_like(self.dwell, dtype=raw.dtype)
        alternate = raw.clone()
        getup, stand, terminal = (self.mode == role for role in (GETUP, STAND, FORWARD))
        if getup.any():
            raw[getup] = actors[GETUP](td('actor', x))[getup]
        if stand.any():
            raw[stand] = actors[STAND](td('actor', x[stand]))
        for role, mask in self.route_masks.items():
            selected = terminal & mask
            if selected.any():
                raw[selected] = actors[FORWARD][role](td('actor', x[selected]))
        forward = terminal & self.forward_worlds
        step = control - self.forward_at + 1
        if step > self.feedback.warmup:
            # A zero target correction outside forward uses the same
            # projected endpoint as the nominal action.
            other = terminal & ~self.forward_worlds
            alternate[other] = raw[other]
        if forward.any():
            ids = forward.nonzero().flatten().cpu().numpy()
            alternative_input = x[forward]
            adjusted = self.feedback.advance(alternative_input.cpu().numpy(), ids, step)
            if adjusted is not None:
                yaw[forward] = torch.as_tensor(adjusted, device=x.device)
                alternative_input[:, 50] = yaw[forward]
                alternate[forward] = actors[FORWARD]['official'](td('actor', alternative_input))
        for role, selected in ((GETUP, getup), (STAND, stand), (FORWARD, terminal)):
            self.history[role, selected] = raw[selected]
        return raw, yaw, alternate
