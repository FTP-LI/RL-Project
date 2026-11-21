"""Read floor load from the current solver and keep the reduction on CUDA."""

import mujoco
from mujoco_warp._src.support import contact_force_fn, vec5
import numpy as np
import torch
import warp as wp


CHANNELS = ('ground', 'head', 'left_foot', 'right_foot')


@wp.kernel
def reduce_solved_floor_load(
    cone: int, capacity: int, floor: int, worlds: int,
    count: wp.array[int], pairs: wp.array[wp.vec2i], world_ids: wp.array[int], kinds: wp.array[int],
    frame: wp.array[wp.mat33], friction: wp.array[vec5], dimensions: wp.array[int],
    addresses: wp.array2d[int], force: wp.array2d[float], njmax: int,
    sums: wp.array2d[float], invalid: wp.array[int],
):
    slot = wp.tid()
    size = count[0]
    if slot == 0 and (size < 0 or size >= capacity):
        wp.atomic_max(invalid, 0, 1)
    if slot >= size:
        return
    pair = pairs[slot]
    if (pair[0] == floor) == (pair[1] == floor):
        return
    geom = pair[0]
    if pair[0] == floor:
        geom = pair[1]
    world = world_ids[slot]
    if geom < 0 or geom >= kinds.shape[0] or world < 0 or world >= worlds:
        wp.atomic_max(invalid, 0, 1)
        return
    kind = kinds[geom]
    if kind < 0:
        return
    solved = contact_force_fn(cone, frame, friction, dimensions, addresses, force, njmax, count, world, slot, False)
    normal = solved[0]
    if not wp.isfinite(normal) or normal < -1.0e-5:
        wp.atomic_max(invalid, 0, 1)
        return
    normal = wp.max(normal, 0.0)
    wp.atomic_add(sums, world, 0, normal)
    if kind > 0:
        wp.atomic_add(sums, world, kind, normal)


@wp.kernel
def evaluate_weight_transfer(
    loads: wp.array2d[float], nominal_weight: float, single_foot_minimum: float, both_feet_minimum: float,
    credit: wp.array[float],
):
    world = wp.tid()
    head, left, right = loads[world, 1], loads[world, 2], loads[world, 3]
    feet = left + right
    support = wp.min(wp.min(left, right) / (single_foot_minimum * nominal_weight), 1.0)
    support *= wp.min(feet / (both_feet_minimum * nominal_weight), 1.0)
    share = 0.0
    if head + feet > 0.0:
        share = feet / (head + feet)
    credit[world] = support * share


class SolvedFloorLoad:
    """GPU buffers stay on the active Warp stream and are reused by the next read."""

    def __init__(self, env, *, single_foot_minimum, both_feet_minimum):
        self.sim = env.sim
        self.worlds = env.num_envs
        self.device = self.sim.wp_device
        model = self.sim.mj_model
        self.floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).item()
        trunk = model.body('robot/trunk_base').id
        assert model.body_rootid[trunk] == trunk
        kinds = np.where(model.body_rootid[model.geom_bodyid] == trunk, 0, -1)
        kinds[model.geom_bodyid == model.body('robot/jaw_soft').id] = 1
        for kind, side in enumerate(('left','right'), 2):
            kinds[model.geom(f'robot/{side}_foot_collision').id] = kind
        assert kinds[self.floor] == -1 and (kinds == 1).any()
        assert not np.asarray(model.opt.gravity[:2]).any()
        self.weight = float(-model.opt.gravity[2] * model.body_subtreemass[trunk])
        assert self.weight > 0.0 and 0.0 < single_foot_minimum <= both_feet_minimum
        self.fractions = [single_foot_minimum, both_feet_minimum]
        self.capacity = self.sim.wp_data.contact.geom.shape[0]
        self.kinds = wp.array(kinds.astype(np.int32), dtype=int, device=self.device)
        self.normal = wp.empty((self.worlds, len(CHANNELS)), dtype=float, device=self.device)
        self.credit = wp.empty(self.worlds, dtype=float, device=self.device)
        self.invalid = wp.zeros(1, dtype=int, device=self.device)

    def capture(self):
        """Call immediately after the physics step and before any forward refresh."""
        data, model = self.sim.wp_data, self.sim.wp_model
        contact = data.contact
        # One fused contact traversal avoids materializing the full 6D force buffer.
        with wp.ScopedDevice(self.device):
            self.normal.zero_()
            wp.launch(reduce_solved_floor_load, dim=self.capacity, inputs=[
                model.opt.cone, self.capacity, self.floor, self.worlds, data.nacon, contact.geom, contact.worldid,
                self.kinds, contact.frame, contact.friction, contact.dim, contact.efc_address, data.efc.force,
                data.njmax, self.normal, self.invalid,
            ], device=self.device)
            wp.launch(evaluate_weight_transfer, dim=self.worlds, inputs=[
                self.normal, self.weight, *self.fractions, self.credit,
            ], device=self.device)

    def check(self):
        """Synchronize at a diagnostic or policy guard point, never inside the reduction."""
        if int(self.invalid.numpy()[0]):
            raise RuntimeError('Solved contact buffer has invalid data or reached capacity')

    def torch_credit(self):
        """Return a view for a consumer using the same CUDA stream."""
        values = wp.to_torch(self.credit)
        assert isinstance(values, torch.Tensor)
        return values
