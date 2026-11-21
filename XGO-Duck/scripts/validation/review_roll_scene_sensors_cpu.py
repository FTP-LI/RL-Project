"""List sensors in the actual nominal CPU scene before simulator creation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
sys.path.insert(0, str(ROOT / 'scripts/training'))
PLAN = ROOT / 'configs/experiments/roll_factor_fp_plane_zero_nominal_s1242_20261116.json'
OUTPUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/scene-sensors-cpu.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch
    from mjlab.scene import Scene
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen

    assert not OUTPUT.exists() and not torch.cuda.is_initialized()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    cfg = screen.parameter_configuration(route.configuration,
        {'factors': {'foot_friction': 1.0, 'vin_range': 7.7}})(plan)
    scene = Scene(cfg.scene, device='cpu')
    allowed = {'ContactSensor', 'TerrainHeightSensor'}
    sensors = {name: {'type': type(sensor).__name__, 'module': type(sensor).__module__,
        'fields_before_initialize': sorted(vars(sensor)),
        'tensor_fields_before_initialize': sorted(k for k, v in vars(sensor).items() if isinstance(v, torch.Tensor)),
        'accepted_by_previous_inventory': type(sensor).__name__ in allowed}
        for name, sensor in sorted(scene.sensors.items())}
    assert sensors and set(scene.entities) == {'robot', 'terrain'} and not torch.cuda.is_initialized()
    rejected = [name for name, row in sensors.items() if not row['accepted_by_previous_inventory']]
    assert rejected
    result = {'status': 'actual-cpu-scene-sensors-reviewed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'sensors': sensors, 'rejected_by_previous_inventory': rejected,
        'scope': 'Actual configured CPU Scene before simulator initialization; previous GPU stop did not save its rejected sensor name.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), PLAN]},
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'dynamic_execution_authorized': False}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
