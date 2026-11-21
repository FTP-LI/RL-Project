"""Verify the terrain fingerprint adapter with isolated CPU fixtures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import roll_entry_snapshot
from getup_release_fork_state import signature
from roll_plane_fingerprint_inventory import plane_roll_slots

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/terrain-inventory-fixture.json'


class TerrainEntity:
    def __init__(self):
        self.cfg = SimpleNamespace(terrain_type='plane')
        self.env_origins = torch.zeros((2, 3))
        self.terrain_origins = None
        self._actuators = []
        self._flat_patches = {}
        self._flat_patch_radii = {}
        self.data = SimpleNamespace(default_root_state=torch.tensor([[1.0]]))
        self.indexing = SimpleNamespace(body_ids=torch.tensor([0]))


class Scene:
    def __init__(self, terrain):
        self.entities = {'robot': object(), 'terrain': terrain}
        self.terrain = terrain
        self._default_env_origins = torch.zeros((2, 3))

    def __getitem__(self, name):
        return self.entities[name]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUTPUT.exists() and not torch.cuda.is_initialized()
    terrain = TerrainEntity()
    scene = Scene(terrain)
    env = SimpleNamespace(scene=scene, num_envs=2, _roulade_fixture=torch.tensor([3.0]))
    calls = []
    def old_inventory(view, memory):
        assert set(view.scene.entities) == {'robot'} and view.scene['robot'] is scene.entities['robot']
        assert vars(view)['_roulade_fixture'] is env._roulade_fixture
        calls.append(True)
        return [('env._roulade_fixture', view, '_roulade_fixture')], {}
    with patch.object(roll_entry_snapshot, 'roll_slots', side_effect=old_inventory):
        slots, generators = plane_roll_slots(env, None)
        labels = {name for name, _, _ in slots}
        assert {'terrain.env_origins', 'terrain.data.default_root_state',
                'terrain.indexing.body_ids', 'scene._default_env_origins'} <= labels
        assert generators == {} and set(scene.entities) == {'robot', 'terrain'} and calls == [True]
        values = {name: signature(getattr(owner, field)) for name, owner, field in slots}
        terrain.env_origins[0, 0] = 1
        changed = {name: signature(getattr(owner, field)) for name, owner, field in slots}
        assert values['terrain.env_origins'] != changed['terrain.env_origins']
        rejected = []
        for name, change, restore in [
            ('extra-entity', lambda: scene.entities.update(extra=object()), lambda: scene.entities.pop('extra')),
            ('wrong-origins', lambda: setattr(terrain, 'env_origins', torch.zeros((1, 3))),
             lambda: setattr(terrain, 'env_origins', torch.zeros((2, 3)))),
            ('flat-patches', lambda: terrain._flat_patches.update(extra=torch.zeros(1)),
             lambda: terrain._flat_patches.clear()),
            ('wrong-terrain', lambda: setattr(terrain.cfg, 'terrain_type', 'generator'),
             lambda: setattr(terrain.cfg, 'terrain_type', 'plane')),
        ]:
            change()
            try:
                plane_roll_slots(env, None)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Invalid terrain accepted: ' + name)
            finally:
                restore()
    assert not torch.cuda.is_initialized()
    files = [Path(__file__), ROOT / 'scripts/validation/roll_plane_fingerprint_inventory.py',
             ROOT / 'scripts/validation/roll_entry_snapshot.py',
             ROOT / 'records/reports/roll-matched-factor-preflight-20261116/terrain-scene-cpu.json']
    result = {'status': 'passed-cpu-plane-inventory-fixture-only',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'terrain_fields': sorted(x for x in labels if x.startswith('terrain.')),
        'origin_change_detected': True, 'original_scene_unmodified': True, 'invalid_layouts_rejected': rejected,
        'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'scope': 'CPU fixture only; initialized real CUDA terrain inventory remains unverified.'}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
