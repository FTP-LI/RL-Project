"""Check adapter binding, delegation and restoration on CPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import roll_entry_snapshot
import roll_plane_fingerprint_inventory as plane
from roll_plane_fingerprint_hook import plane_inventory
from verify_roll_plane_fingerprint_inventory import Scene, TerrainEntity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/terrain-hook-fixture.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    env = SimpleNamespace(scene=Scene(TerrainEntity()), num_envs=2)
    calls = []
    def original(view, _):
        calls.append(sorted(view.scene.entities))
        return [('env.num_envs', view, 'num_envs')], {}
    real_module = plane.roll_entry_snapshot
    with patch.object(roll_entry_snapshot, 'roll_slots', original):
        with plane_inventory():
            slots, _ = roll_entry_snapshot.roll_slots(env, None)
            assert calls == [['robot']] and 'terrain.env_origins' in [label for label, _, _ in slots]
        assert roll_entry_snapshot.roll_slots is original and plane.roll_entry_snapshot is real_module
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), ROOT / 'scripts/validation/roll_plane_fingerprint_hook.py',
               ROOT / 'scripts/validation/roll_plane_fingerprint_inventory.py',
               ROOT / 'scripts/validation/verify_roll_plane_fingerprint_inventory.py',
               ROOT / 'records/reports/roll-matched-factor-preflight-20261116/terrain-inventory-fixture.json']
    result = {'status': 'passed-cpu-plane-hook-fixture-only', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'original_inventory_called_once': True, 'terrain_included': True, 'bindings_restored': True,
        'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'scope': 'CPU fixture; actual CUDA capture and paired comparison remain unverified.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
