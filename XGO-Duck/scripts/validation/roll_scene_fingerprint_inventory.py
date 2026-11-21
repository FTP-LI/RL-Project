"""Record plane terrain and builtin-sensor state alongside the roll inventory.

The existing robot inventory knows ContactSensor and TerrainHeightSensor. The
current scene also exposes BuiltinSensor wrappers for IMU and contacts. Keep
the original inventory unchanged, and explicitly read each wrapper's view and
cache without changing the live scene or triggering sensor computation.
"""
import torch
from mjlab.sensor.builtin_sensor import BuiltinSensor

from getup_release_fork_state import tensor_fields
from roll_plane_fingerprint_inventory import plane_roll_slots


class _SensorSceneView:
    def __init__(self, original):
        self._original = original
        self.sensors = {name: sensor for name, sensor in original.sensors.items()
                        if type(sensor) is not BuiltinSensor}

    def __getitem__(self, name):
        return self._original[name]

    def __getattr__(self, name):
        return getattr(self._original, name)


class _EnvironmentView:
    def __init__(self, original):
        self.__dict__.update(vars(original))
        self.__dict__['_sensor_view_original'] = original
        self.__dict__['scene'] = _SensorSceneView(original.scene)

    def __getattr__(self, name):
        return getattr(self.__dict__['_sensor_view_original'], name)


def scene_roll_slots(env, memory):
    sensors = env.scene.sensors
    if any(type(sensor).__name__ not in {'ContactSensor', 'TerrainHeightSensor', 'BuiltinSensor'}
           for sensor in sensors.values()):
        raise ValueError('Unexpected sensor type')
    builtins = {name: sensor for name, sensor in sensors.items() if type(sensor) is BuiltinSensor}
    if not builtins or any(type(sensor).__name__ == 'BuiltinSensor' and type(sensor) is not BuiltinSensor
                           for sensor in sensors.values()):
        raise ValueError('Builtin sensor set missing or has unexpected class')
    for sensor in builtins.values():
        if not isinstance(sensor._data_view, torch.Tensor):
            raise ValueError('Builtin sensor view is not initialized')
    slots, generators = plane_roll_slots(_EnvironmentView(env), memory)
    for name, sensor in sorted(builtins.items()):
        fields = set(tensor_fields(sensor)) | {'_data_view', '_cache_valid', '_cached_data'}
        for field in sorted(fields):
            if hasattr(sensor, field):
                slots.append((f'sensor.{name}.{field}', sensor, field))
    names = [name for name, _, _ in slots]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate sensor fingerprint field')
    if any(f'sensor.{name}._data_view' not in names for name in builtins):
        raise ValueError('Builtin sensor data view missing')
    return slots, generators
