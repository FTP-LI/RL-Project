"""Verify builtin sensor fingerprint wiring using CPU objects."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.sensor.builtin_sensor import BuiltinSensor
import roll_entry_snapshot
from getup_release_fork_state import signature
from roll_scene_fingerprint_hook import scene_inventory
from verify_roll_plane_fingerprint_inventory import Scene, TerrainEntity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/scene-inventory-fixture.json'


class ContactSensor:
    pass


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    scene = Scene(TerrainEntity())
    sensor = BuiltinSensor.from_existing('robot/imu_ang_vel')
    sensor._data_view = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
    scene.sensors = {'feet_ground_contact': ContactSensor(), 'robot/imu_ang_vel': sensor}
    env = SimpleNamespace(scene=scene, num_envs=2)
    calls = []
    def old(view, _):
        assert set(view.scene.entities) == {'robot'}
        assert set(view.scene.sensors) == {'feet_ground_contact'}
        calls.append(True)
        return [('env.num_envs', view, 'num_envs')], {}
    with patch.object(roll_entry_snapshot, 'roll_slots', old):
        with scene_inventory():
            slots, _ = roll_entry_snapshot.roll_slots(env, None)
            values = {name: signature(getattr(owner, field)) for name, owner, field in slots}
            assert 'terrain.env_origins' in values and 'sensor.robot/imu_ang_vel._data_view' in values
            assert 'sensor.robot/imu_ang_vel._cache_valid' in values and calls == [True]
            sensor._data_view[0, 0] = 7
            changed = {name: signature(getattr(owner, field)) for name, owner, field in slots}
            assert values['sensor.robot/imu_ang_vel._data_view'] != changed['sensor.robot/imu_ang_vel._data_view']
            assert set(scene.sensors) == {'feet_ground_contact', 'robot/imu_ang_vel'}
            rejected = []
            sensor._data_view = None
            try:
                roll_entry_snapshot.roll_slots(env, None)
            except ValueError:
                rejected.append('uninitialized-builtin')
            else:
                raise AssertionError('Uninitialized sensor accepted')
            sensor._data_view = torch.zeros((2, 3))
            scene.sensors['unexpected'] = object()
            try:
                roll_entry_snapshot.roll_slots(env, None)
            except ValueError:
                rejected.append('unexpected-type')
            else:
                raise AssertionError('Unexpected sensor accepted')
        assert roll_entry_snapshot.roll_slots is old
    assert not torch.cuda.is_initialized()
    files = [Path(__file__), ROOT / 'scripts/validation/roll_scene_fingerprint_inventory.py',
             ROOT / 'scripts/validation/roll_scene_fingerprint_hook.py',
             ROOT / 'records/reports/roll-matched-factor-preflight-20261116/scene-sensors-cpu.json']
    result = {'status': 'passed-cpu-scene-inventory-fixture-only', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'sensor_data_change_detected': True, 'terrain_included': True, 'original_scene_unmodified': True,
        'invalid_sensors_rejected': rejected, 'bindings_restored': True,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'scope': 'Synthetic CPU state only; real initialized sensor and CUDA inventory remain unverified.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
