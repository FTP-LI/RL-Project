"""Inspect the real nominal Scene construction without creating a simulator."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
sys.path.insert(0, str(ROOT / 'scripts/training'))
OUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/terrain-scene-cpu.json'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_logged_zero_nominal_s1242_20261116.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch
    from mjlab.scene import Scene
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen

    assert not OUT.exists() and not torch.cuda.is_initialized()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    cfg = screen.parameter_configuration(route.configuration,
        {'factors': {'foot_friction': 1.0, 'vin_range': 7.7}})(plan)
    assert cfg.scene.num_envs == 128 and cfg.scene.terrain.terrain_type == 'plane'
    scene = Scene(cfg.scene, device='cpu')
    entities = sorted(scene.entities)
    terrain = scene.terrain
    assert entities == ['robot', 'terrain'] and terrain is scene.entities['terrain']
    origins = terrain.env_origins.detach().cpu().contiguous().numpy()
    assert origins.shape == (128, 3)
    tensor_fields = sorted(k for k, value in vars(terrain).items() if isinstance(value, torch.Tensor))
    assert 'env_origins' in tensor_fields and not torch.cuda.is_initialized()
    report = {'status': 'actual-cpu-scene-construction-reviewed',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'entities': entities,
        'terrain_type': type(terrain).__name__, 'terrain_configuration': cfg.scene.terrain.terrain_type,
        'terrain_is_scene_entity': True, 'terrain_tensor_fields_before_initialize': tensor_fields,
        'terrain_object_fields_before_initialize': sorted(vars(terrain)),
        'origin_shape': list(origins.shape), 'origins_sha256': hashlib.sha256(origins.tobytes()).hexdigest(),
        'origin_min': float(origins.min()), 'origin_max': float(origins.max()),
        'scope': 'Actual CPU Scene construction only; initialized runtime terrain state still needs inspection.',
        'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in [Path(__file__), PLAN]},
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'dynamic_execution_authorized': False}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
