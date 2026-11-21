"""Extend the existing read-only roll fingerprint inventory for plane terrain.

The previous inventory assumes that the robot is the only scene entity. mjlab
also inserts the configured ground plane as ``terrain``. This adapter keeps the
original robot checks, adds terrain origins and initialized terrain data, and
rejects other entity layouts. It is for fingerprint reads, not state restore.
"""
import torch

import roll_entry_snapshot
from getup_release_fork_state import tensor_fields


class _SceneView:
    def __init__(self, original):
        self._original = original
        self.entities = {'robot': original.entities['robot']}

    def __getitem__(self, name):
        return self._original[name]

    def __getattr__(self, name):
        return getattr(self._original, name)


class _EnvironmentView:
    def __init__(self, original):
        # Keep vars(env) and all original object references visible to the old
        # inventory without changing the live environment or its entity map.
        self.__dict__.update(vars(original))
        self.__dict__['_fingerprint_original'] = original
        self.__dict__['scene'] = _SceneView(original.scene)

    def __getattr__(self, name):
        return getattr(self.__dict__['_fingerprint_original'], name)


def plane_roll_slots(env, memory):
    scene = env.scene
    if set(scene.entities) != {'robot', 'terrain'}:
        raise ValueError('Expected only robot and plane terrain entities')
    terrain = scene.entities['terrain']
    if (scene.terrain is not terrain or type(terrain).__name__ != 'TerrainEntity'
            or terrain.cfg.terrain_type != 'plane' or terrain._actuators
            or terrain._flat_patches or terrain._flat_patch_radii):
        raise ValueError('Unexpected terrain type or mutable terrain components')
    if not isinstance(terrain.env_origins, torch.Tensor) or tuple(terrain.env_origins.shape) != (env.num_envs, 3):
        raise ValueError('Terrain environment origins missing or wrong shape')
    if not hasattr(terrain, 'data') or not hasattr(terrain, 'indexing'):
        raise ValueError('Terrain must be initialized before fingerprinting')

    slots, generators = roll_entry_snapshot.roll_slots(_EnvironmentView(env), memory)
    for name in sorted(set(tensor_fields(terrain)) | {'terrain_origins', '_flat_patches', '_flat_patch_radii'}):
        if hasattr(terrain, name):
            slots.append((f'terrain.{name}', terrain, name))
    for name in sorted(tensor_fields(terrain.data)):
        slots.append((f'terrain.data.{name}', terrain.data, name))
    for name in sorted(tensor_fields(terrain.indexing)):
        slots.append((f'terrain.indexing.{name}', terrain.indexing, name))
    if hasattr(scene, '_default_env_origins'):
        slots.append(('scene._default_env_origins', scene, '_default_env_origins'))
    names = [label for label, _, _ in slots]
    if 'terrain.env_origins' not in names or len(set(names)) != len(names):
        raise ValueError('Terrain fingerprint inventory missing or duplicated')
    return slots, generators
