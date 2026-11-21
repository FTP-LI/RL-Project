"""Verify numeric zero-step capture on an initialized CPU environment."""
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
sys.path.insert(0, str(ROOT/'scripts/validation'))
sys.path.insert(0, str(ROOT/'scripts/training'))
REPORT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'
OUT = REPORT/'raw-capture-cpu'
PLAN = ROOT/'configs/experiments/roll_factor_fp_scene_zero_nominal_s1242_20261116.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim import Simulation
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen
    from roll_factor_start_fingerprint import capture
    from roll_friction_raw_capture import record_after_capture
    from roll_scene_fingerprint_hook import scene_inventory

    assert not OUT.exists() and not torch.cuda.is_initialized()
    OUT.mkdir()
    blocked, env = [], None
    def no_step(*args, **kwargs):
        blocked.append(True)
        raise RuntimeError('Physics step disabled during raw CPU verification')
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'physical_steps': 0,
              'learning_updates': 0, 'hardware_io': False, 'worlds': 2}
    try:
        plan = deepcopy(json.loads(PLAN.read_text(encoding='utf-8')))
        plan['worlds'] = plan['roll']['episodes'] = 2
        cfg = screen.parameter_configuration(route.configuration,
            {'factors': {'foot_friction': 1.0, 'vin_range': 7.7}})(plan)
        with patch.object(Simulation, 'step', side_effect=no_step), \
             patch.object(mjwarp, 'step', side_effect=no_step), \
             patch.object(mujoco, 'mj_step', side_effect=no_step):
            env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
            env.seed(plan['seed'])
            env.reset()
            memory = SimpleNamespace(previous=torch.zeros((2, 14)), history=torch.zeros((2, 14)), previous_features=None)
            with scene_inventory():
                record_after_capture(capture, env, memory, OUT/'start-fingerprints.json', 'cpu-raw-only', PLAN)
            metadata = json.loads((OUT/'selected-zero-numeric.json').read_text(encoding='utf-8'))
            with np.load(OUT/'selected-zero-numeric.npz', allow_pickle=False) as archive:
                assert set(archive.files) == set(metadata['fields'])
                assert len(archive.files) == 34
            assert not blocked and not torch.cuda.is_initialized()
            assert metadata['arrays_sha256'] == sha(OUT/'selected-zero-numeric.npz')
            result.update(status='passed-initialized-cpu-numeric-capture', fields=len(metadata['fields']),
                arrays_sha256=metadata['arrays_sha256'], fingerprint_sha256=metadata['fingerprint_sha256'],
                rng_unchanged=metadata['rng_unchanged'])
    except Exception as exc:
        result.update(status='stopped-cpu-numeric-capture-review', error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        result['blocked_step_attempts'] = len(blocked)
        result['cuda_initialized'] = bool(torch.cuda.is_initialized())
        files = [Path(__file__), ROOT/'scripts/validation/roll_friction_raw_capture.py',
                 ROOT/'scripts/validation/roll_scene_fingerprint_inventory.py',
                 REPORT/'friction-cache-identity-review.json', PLAN]
        result['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
        (OUT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'passed-initialized-cpu-numeric-capture' else 2


if __name__ == '__main__':
    raise SystemExit(main())
