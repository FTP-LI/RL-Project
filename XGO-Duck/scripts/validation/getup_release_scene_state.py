"""Include the pinned scene's fixed terrain and built-in sensor buffers."""
from types import SimpleNamespace

import torch
import warp as wp

from getup_release_fork_state import ReleaseSnapshot, clone, state_slots, tensor_fields


def classify_scene(scene):
    from mjlab.sensor.builtin_sensor import BuiltinSensor
    from mjlab.terrains.terrain_entity import TerrainEntity

    assert set(scene.entities) == {'terrain', 'robot'}, sorted(scene.entities)
    terrain = scene.entities['terrain']
    assert isinstance(terrain, TerrainEntity) and terrain is scene.terrain
    assert terrain.is_fixed_base and not terrain.is_articulated and not terrain.is_actuated
    assert terrain.cfg.terrain_type == 'plane'
    regular, builtin = {}, {}
    for name, sensor in scene.sensors.items():
        if isinstance(sensor, BuiltinSensor):
            builtin[name] = sensor
        else:
            assert type(sensor).__name__ in {'ContactSensor', 'TerrainHeightSensor'}, (name, type(sensor))
            regular[name] = sensor
    return terrain, regular, builtin


class SceneReleaseSnapshot(ReleaseSnapshot):
    def __init__(self, env, memory, maximum_warp_bytes):
        terrain, regular, builtin = classify_scene(env.scene)

        class RobotView:
            entities = {'robot': env.scene['robot']}
            sensors = regular

            def __getitem__(self, key):
                return self.entities[key]

        # The parent inventory has a robot-only interface. Extend its bindings
        # without modifying the live scene or replacing its data arrays.
        view = SimpleNamespace(**vars(env))
        view.scene = RobotView()
        slots, self.generators = state_slots(view, memory)
        self.slots = [(label, env if owner is view else owner, key) for label, owner, key in slots]
        assert {'dof_frictionloss', 'dof_damping'} <= env.sim.expanded_fields
        for index, actuator in enumerate(env.scene['robot'].actuators):
            self.slots.append((f'actuator.{index}._friction_fields_checked', actuator, '_friction_fields_checked'))
        self.slots.extend((f'terrain.data.{key}', terrain.data, key) for key in tensor_fields(terrain.data))
        self.slots.extend((f'terrain.{key}', terrain, key) for key in tensor_fields(terrain))
        for name, sensor in builtin.items():
            keys = set(tensor_fields(sensor)) | {'_cache_valid', '_cached_data'}
            self.slots.extend((f'sensor.{name}.{key}', sensor, key) for key in sorted(keys))
        assert len({label for label, _, _ in self.slots}) == len(self.slots)
        self.env, self.device = env, env.device
        self.cuda = torch.device(self.device).type == 'cuda'
        self.synchronize()
        values = [getattr(owner, field) for _, owner, field in self.slots]
        self.warp_bytes = sum(value.capacity for value in values if isinstance(value, wp.array))
        assert self.warp_bytes <= maximum_warp_bytes, self.warp_bytes
        self.values = []
        for value in values:
            if isinstance(value, wp.array):
                self.values.append(wp.clone(value) if value.size else value)
            else:
                self.values.append(clone(value))
        self.synchronize()
        self.rng = self.rng_state()
        self.expected = self.signatures(self.values, self.rng)
