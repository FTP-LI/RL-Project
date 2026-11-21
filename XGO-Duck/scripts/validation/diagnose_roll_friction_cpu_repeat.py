"""Compare two nominal CPU resets and one low-friction reset without steps.

This separates same-configuration repeat variability from factor changes in a
small CPU environment. It cannot prove the cause of earlier GPU differences.
"""
import argparse
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
OUT = REPORT / 'cpu-friction-repeat-review'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_scene_zero_nominal_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def child():
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim import Simulation
    import roll_entry_snapshot
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen
    from roll_factor_start_fingerprint import capture
    from roll_scene_fingerprint_hook import scene_inventory

    prior = read(REPORT / 'friction-low-scene-zero-comparison.json')
    assert prior['status'] == 'review-required-internal-state-differences'
    assert not torch.cuda.is_initialized()
    selected = set(prior['changed_fingerprint_fields'])
    plan = deepcopy(read(PLAN))
    plan['worlds'] = plan['roll']['episodes'] = 2
    attempts, rows = [], []
    def no_step(*args, **kwargs):
        attempts.append('blocked')
        raise RuntimeError('Physical steps are disabled during this CPU review')
    with patch.object(Simulation, 'step', side_effect=no_step), \
         patch.object(mjwarp, 'step', side_effect=no_step), \
         patch.object(mujoco, 'mj_step', side_effect=no_step):
        for label, friction in [('nominal-a', 1.0), ('nominal-b', 1.0), ('friction-low', 0.8)]:
            project = read(ROOT / 'project.json')
            assert project['goal_execution']['status'] == 'active'
            assert not project['goal_execution']['pause_requested_by_user']
            assert not project['staged_training_request']['pause_requested']
            print('CPU_ZERO_CASE', label, flush=True)
            write(OUT / 'status.json', {'status': 'running', 'case': label, 'physical_steps': 0, 'learning_updates': 0})
            cfg = screen.parameter_configuration(route.configuration,
                {'factors': {'foot_friction': friction, 'vin_range': 7.7}})(plan)
            env = None
            try:
                env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
                env.seed(plan['seed'])
                obs, _ = env.reset()
                assert (env.common_step_counter, env._sim_step_counter) == (0, 0)
                memory = SimpleNamespace(previous=torch.zeros((2, 14)), history=torch.zeros((2, 14)), previous_features=None)
                with scene_inventory():
                    fingerprint = capture(env, memory, OUT/(label+'-fingerprint.json'), 'cpu-'+label, PLAN)
                    slots, _ = roll_entry_snapshot.roll_slots(env, memory)
                raw = {}
                for name, owner, field in slots:
                    if name not in selected:
                        continue
                    value = getattr(owner, field)
                    if isinstance(value, torch.Tensor):
                        raw[name] = value.detach().cpu().numpy().copy()
                    elif isinstance(value, wp.array):
                        raw[name] = value.numpy().copy()
                raw['startup.qpos'] = env.sim.data.qpos.detach().cpu().numpy().copy()
                raw['startup.qvel'] = env.sim.data.qvel.detach().cpu().numpy().copy()
                raw['startup.obs'] = obs['actor'].detach().cpu().numpy().copy()
                with (OUT/(label+'-selected.npz')).open('xb') as stream:
                    np.savez_compressed(stream, **raw)
                rows.append({'case': label, 'friction': friction, 'fields': fingerprint['field_count'],
                    'raw_fields': sorted(raw), 'fingerprint_sha256': sha(OUT/(label+'-fingerprint.json')),
                    'selected_sha256': sha(OUT/(label+'-selected.npz'))})
            finally:
                if env is not None:
                    env.close()
    assert not attempts and not torch.cuda.is_initialized()

    def compare(left, right):
        first = read(OUT/(left+'-fingerprint.json'))
        second = read(OUT/(right+'-fingerprint.json'))
        assert set(first['fields']) == set(second['fields'])
        changed = sorted(k for k in first['fields'] if first['fields'][k] != second['fields'][k])
        values = {}
        with np.load(OUT/(left+'-selected.npz'), allow_pickle=False) as a, \
             np.load(OUT/(right+'-selected.npz'), allow_pickle=False) as b:
            for name in sorted(set(a.files) & set(b.files)):
                x, y = a[name], b[name]
                if x.shape != y.shape or x.dtype.kind not in 'biuf' or y.dtype.kind not in 'biuf':
                    continue
                finite = np.isfinite(x) & np.isfinite(y)
                difference = np.abs(x.astype(np.float64) - y.astype(np.float64))
                values[name] = {'equal': bool(np.array_equal(x, y)),
                    'nonfinite_count': int(finite.size - np.count_nonzero(finite)),
                    'maximum_finite_absolute_difference': float(difference[finite].max()) if finite.any() else None}
        return {'changed_fields': changed, 'rng_equal': first['rng'] == second['rng'], 'selected_values': values}
    result = {'status': 'completed-cpu-repeat-diagnostic', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'cases': rows, 'same_factor_repeat': compare('nominal-a', 'nominal-b'),
        'low_friction_comparison': compare('nominal-a', 'friction-low'),
        'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False, 'hardware_io': False,
        'blocked_step_attempts': len(attempts), 'dynamic_execution_authorized': False,
        'scope': 'Two CPU worlds; results do not identify cause or size of saved GPU differences.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), PLAN,
            ROOT/'scripts/validation/roll_scene_fingerprint_inventory.py',
            REPORT/'friction-low-scene-zero-comparison.json']}}
    write(OUT/'analysis.json', result)
    write(OUT/'status.json', {'status': 'completed', 'physical_steps': 0, 'learning_updates': 0})
    print(json.dumps({'status': result['status'],
        'same_factor_changed_fields': len(result['same_factor_repeat']['changed_fields']),
        'low_friction_changed_fields': len(result['low_friction_comparison']['changed_fields'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        try:
            child()
            return 0
        except Exception as exc:
            write(OUT/'failure.json', {'status': 'stopped-review-required', 'error_type': type(exc).__name__,
                'error': str(exc), 'traceback': traceback.format_exc(), 'automatic_retry': False})
            write(OUT/'status.json', {'status': 'stopped-review-required', 'physical_steps': 0, 'learning_updates': 0})
            print(traceback.format_exc(), flush=True)
            return 2
    from recorded_process import run_recorded
    assert not OUT.exists()
    OUT.mkdir()
    receipt = run_recorded([sys.executable, '-u', str(Path(__file__).resolve()), '--child'], cwd=ROOT,
        log_path=OUT/'execution.log', receipt_path=OUT/'execution.json', max_wall_seconds=180)
    print(json.dumps({'status': receipt['status'], 'exit_code': receipt['exit_code'],
                      'log_sha256': receipt['log_sha256']}), flush=True)
    return 0 if receipt['status'] == 'completed' and receipt['exit_code'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
