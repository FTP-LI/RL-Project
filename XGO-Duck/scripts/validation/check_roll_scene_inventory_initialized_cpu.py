"""Check the scene fingerprint on an initialized two-world CPU environment.

Physics-step entry points are blocked before construction. This is a wiring
diagnostic only; it is not a rollout or evidence for the GPU factor arms.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
sys.path.insert(0, str(ROOT / 'scripts/training'))
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_plane_zero_nominal_s1242_20261116.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import mujoco
    import mujoco_warp as mjwarp
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim import Simulation
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen
    from roll_factor_start_fingerprint import capture
    from roll_scene_fingerprint_hook import scene_inventory

    result_path = REPORT / 'initialized-scene-inventory-cpu.json'
    fingerprint = REPORT / 'initialized-scene-fingerprint-cpu.json'
    assert not result_path.exists() and not fingerprint.exists() and not torch.cuda.is_initialized()
    attempts = []
    env = None
    def no_step(*args, **kwargs):
        attempts.append('step-attempt-blocked')
        raise RuntimeError('Physics step is not permitted in this check')
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'physical_steps': 0,
              'learning_updates': 0, 'hardware_io': False, 'worlds': 2,
              'scope': 'Initialized CPU wiring check only; does not verify actual CUDA or factor matching.'}
    try:
        plan = deepcopy(json.loads(PLAN.read_text(encoding='utf-8')))
        plan['worlds'] = plan['roll']['episodes'] = 2
        cfg = screen.parameter_configuration(route.configuration,
            {'factors': {'foot_friction': 1.0, 'vin_range': 7.7}})(plan)
        with patch.object(Simulation, 'step', side_effect=no_step), \
             patch.object(mjwarp, 'step', side_effect=no_step), \
             patch.object(mujoco, 'mj_step', side_effect=no_step):
            print('CPU_SCENE_INITIALIZING', flush=True)
            env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
            env.seed(plan['seed'])
            env.reset()
            assert (env.common_step_counter, env._sim_step_counter) == (0, 0)
            memory = SimpleNamespace(previous=torch.zeros((2, 14)), history=torch.zeros((2, 14)), previous_features=None)
            with scene_inventory():
                capture_result = capture(env, memory, fingerprint, 'cpu-scene-wiring-only', PLAN)
            saved = json.loads(fingerprint.read_text(encoding='utf-8'))
            assert any(name.startswith('terrain.data.') for name in saved['fields'])
            assert any(name.startswith('sensor.robot/imu_ang_vel.') for name in saved['fields'])
            assert not attempts and not torch.cuda.is_initialized()
            result.update(status='passed-initialized-cpu-scene-inventory',
                entity_names=sorted(env.scene.entities), sensor_types={name: type(item).__name__
                    for name, item in sorted(env.scene.sensors.items())},
                field_count=capture_result['field_count'], fingerprint_sha256=sha(fingerprint),
                counters=saved['counters'], rng_unchanged=saved['rng_unchanged'])
    except Exception as exc:
        result.update(status='stopped-initialized-cpu-scene-review', error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        result['blocked_step_attempts'] = len(attempts)
        result['cuda_initialized'] = bool(torch.cuda.is_initialized())
        result['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), PLAN,
            ROOT / 'scripts/validation/roll_scene_fingerprint_inventory.py',
            ROOT / 'scripts/validation/roll_scene_fingerprint_hook.py']}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'passed-initialized-cpu-scene-inventory' else 2


if __name__ == '__main__':
    raise SystemExit(main())
