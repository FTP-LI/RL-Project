"""Read cached and post-forward contacts at the same control boundary."""
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mujoco_warp._src.types import ContactType

import roll_contact_clock_capture as clock
from roll_contact_endpoint import found_after_forward
from s1_reference_inputs import ROOT, sha
from record_io import write


SENSORS = ('robot_ground_contact', 'head_ground_contact', 'feet_ground_contact')
WIDTHS = (1, 1, 2)


def check_geom_layout(model):
    floor, kinds, _ = clock.contact_layout(model)
    terrain = int(model.geom_bodyid[floor])
    body, subtree, geom = (int(value) for value in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_XBODY,
                                                    mujoco.mjtObj.mjOBJ_GEOM))
    trunk, head = (model.body(name).id for name in ('robot/trunk_base', 'robot/jaw_soft'))
    left, right = (model.geom(f'robot/{side}_foot_collision').id for side in ('left', 'right'))
    targets = (('robot_ground_contact_trunk_base_found', subtree, trunk),
               ('head_ground_contact_jaw_soft_found', body, head),
               ('feet_ground_contact_left_foot_collision_found', geom, left),
               ('feet_ground_contact_right_foot_collision_found', geom, right))
    for name, kind, obj in targets:
        i = model.sensor(name).id
        assert int(model.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_CONTACT)
        assert (int(model.sensor_objtype[i]), int(model.sensor_objid[i])) == (kind, obj)
        assert (int(model.sensor_reftype[i]), int(model.sensor_refid[i]), int(model.sensor_dim[i])) == (body, terrain, 1)
    assert tuple(kinds[[left, right]]) == (2, 3)
    return floor, kinds, [item[0] for item in targets]


def constraint_counts(pairs, worlds, types, floor, kinds, n):
    active = (types & int(ContactType.CONSTRAINT)) != 0
    counts, _ = clock.reduce_contacts(pairs[active], worlds[active], np.zeros(int(active.sum())), floor, kinds, n)
    return counts


class EndpointReadback:
    def __init__(self, env, folder):
        self.env, self.folder = env, folder
        self.floor, self.kinds, self.found_slots = check_geom_layout(env.sim.mj_model)
        self.sensors = tuple(env.scene[name] for name in SENSORS)
        actual = [slot.sensor_name for sensor in self.sensors for slot in sensor._slots if slot.field_name == 'found']
        assert actual == self.found_slots, actual
        self.rows = []
        self.first_mismatches = {}
        self.paired_start_verified = False

    def capture(self):
        env = self.env
        k, substep = env.common_step_counter, env._sim_step_counter
        assert k == len(self.rows) and substep == 5*k
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(env.device)
        objects = tuple(sensor._cached_data for sensor in self.sensors)
        flags = tuple(sensor._cache_valid for sensor in self.sensors)
        current = [found_after_forward(sensor, width) for sensor, width in zip(self.sensors, WIDTHS, strict=True)]
        cached = [obj.found.reshape(env.num_envs, width, -1).any(dim=-1) if flag else fresh
                  for obj, width, flag, fresh in zip(objects, WIDTHS, flags, current, strict=True)]
        fresh, legacy = (clock.cpu(torch.cat(values, dim=1)) for values in (current, cached))
        raw = env.sim.wp_data
        count = int(raw.nacon.numpy()[0])
        assert 0 <= count < env.guard_slots
        pairs, worlds, types = (getattr(raw.contact, name).numpy()[:count] for name in ('geom', 'worldid', 'type'))
        points = constraint_counts(pairs, worlds, types, self.floor, self.kinds, env.num_envs)
        for i, name in enumerate(clock.LABELS):
            different = np.flatnonzero(fresh[:, i] != (points[:, i] > 0))
            if len(different) and name not in self.first_mismatches:
                world = int(different[0])
                self.first_mismatches[name] = dict(control=k, world=world, fresh=bool(fresh[world, i]),
                    points=int(points[world, i]), geometry_pairs=pairs[worlds == world].tolist(),
                    contact_types=types[worlds == world].tolist())
        assert all(sensor._cached_data is obj and sensor._cache_valid == flag
                   for sensor, obj, flag in zip(self.sensors, objects, flags, strict=True))
        assert (env.common_step_counter, env._sim_step_counter) == (k, substep)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(env.device))
        self.rows.append(dict(control=k, substep=substep, cached=legacy, fresh=fresh, constraint_points=points,
                              cache_valid=np.asarray(flags, dtype=bool), quaternion=clock.cpu(env.sim.data.qpos[:, 3:7])))

    def save(self, maximum):
        path = self.folder/'endpoint-contact.npz'
        np.savez_compressed(path, **{key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]})
        write(self.folder/'endpoint-contact.json', dict(sha256=sha(path), labels=clock.LABELS, controls=len(self.rows),
            complete=len(self.rows) == maximum, first_mismatches=self.first_mismatches,
            zero_matches_standalone=self.paired_start_verified,
            cache_counters_and_rng_unchanged_at_each_boundary=True))


def collect(plan, folder, state, save, startup=False):
    install, make_video = clock.install_guard, clock.make_video
    reader = None

    def install_with_readback(*args, **kwargs):
        guarded = install(*args, **kwargs)

        class EndpointEnv(guarded):
            def reset(self, *args, **kwargs):
                nonlocal reader
                result = super().reset(*args, **kwargs)
                if self._common_ready:
                    assert reader is None and self.common_step_counter == self._sim_step_counter == 0
                    reader = EndpointReadback(self, folder)
                    qpos, qvel = (clock.cpu(getattr(self.sim.data, name)) for name in ('qpos', 'qvel'))
                    reader.capture()
                    np.testing.assert_array_equal(qpos, clock.cpu(self.sim.data.qpos))
                    np.testing.assert_array_equal(qvel, clock.cpu(self.sim.data.qvel))
                    row = reader.rows[0]
                    np.testing.assert_array_equal(row['fresh'], row['constraint_points'] > 0)
                    if not startup:
                        with np.load(ROOT/plan['startup_endpoints']) as expected:
                            assert sorted(expected.files) == sorted(row)
                            for name, value in row.items():
                                np.testing.assert_array_equal(value, expected[name][0], err_msg=name)
                        reader.paired_start_verified = True
                return result

            def step(self, actions):
                result = super().step(actions)
                assert reader is not None
                reader.capture()
                return result

        return EndpointEnv

    def labeled_video(env, record, _):
        return make_video(env, record, f'Official fixed roll | seed {plan["seed"]} | endpoint readback')

    try:
        with patch.multiple(clock, install_guard=install_with_readback, make_video=labeled_video):
            clock.collect(plan, folder, state, save, startup=startup)
    finally:
        if reader is not None and reader.rows:
            reader.save(1 if startup else 1 + round(plan['criteria']['max_seconds']/plan['criteria']['step_dt']))
