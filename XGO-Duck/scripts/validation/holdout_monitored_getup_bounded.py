"""New held-out run after an observer failure, without resuming partial data.

The original three models and scoring are unchanged. The new seed is reserved,
the observer tolerates only a bounded missing-source interval, and 2,816
environments leave headroom below the measured 3,072-environment memory peak.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import os
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

import numpy as np
import torch
import evaluate_monitored_getup_pair as base
import getup_shared_release as shared
import holdout_monitored_getup_pair as previous
import verify_getup_supine_evaluation as verifier
from getup_shared_release_terrain import cpu_scene_readback
from native_roll_observation_bounded import NativeRollObservationBounded
from record_io import write
from s1_reference_inputs import read, sha

PLAN = ROOT / 'configs/experiments/getup_monitored_pair_holdout_bounded_s4268.json'
OUT = ROOT / 'records/reports/getup-monitored-pair-holdout-s4268'
STAGE = 'observed-getup-holdout-bounded'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    if PLAN.exists() or OUT.exists():
        raise FileExistsError('Preserve the previous attempt')
    failure = read(previous.OUT / 'entry-result.json')
    io = read(previous.OUT / 'native-io-diagnosis.json')
    life = read(previous.OUT / 'native-observer-lifecycle.json')
    project = read(ROOT / 'project.json')
    registered = project['goal_execution']['monitored_getup_holdout_recovery']
    if (failure['status'] != 'stopped-review-required' or failure['protocol_sha256'] != sha(previous.PLAN)
            or io['status'] != 'passed-bounded-observer-tests' or life['status'] != 'passed-actual-observer-lifecycle'
            or registered['observer_lifecycle_sha256'] != sha(previous.OUT / 'native-observer-lifecycle.json')
            or io['adapter_sha256'] != sha(ROOT / 'scripts/training/native_run_observer_bounded.py')
            or life['bridge_sha256'] != sha(ROOT / 'scripts/training/native_roll_observation_bounded.py')):
        raise ValueError('The prior stop or bounded observer checks do not match')
    if any('_s4268_' in p.name for p in (ROOT / 'records/evaluations').iterdir()):
        raise ValueError('Reserved seed has already been used')
    spec = deepcopy(read(previous.PLAN))
    shared.assert_sources(spec)
    p = spec['environment']
    n, seed = 2816, 4268
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(
        np.repeat(np.arange(4), [264, 263, 1144, 1144]))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_monitor-pair_holdout-bounded_e{n}_s{seed}_{stamp}'
    p.update(num_envs=n, seed=seed, stage=STAGE, run_id=run_id,
             split=dict(pose_codes=codes.tolist(), groups=groups),
             maximum_control_transitions=n * p['maximum_control_steps'],
             video_worlds={pose: groups[pose][0] for pose in shared.POSES})
    for policy in spec['policies'].values():
        policy['seed'] = seed
    for path in (Path(__file__).resolve(), ROOT / 'scripts/training/native_run_observer_bounded.py',
                 ROOT / 'scripts/training/native_roll_observation_bounded.py',
                 ROOT / 'scripts/validation/check_native_observer_bounded.py',
                 ROOT / 'scripts/validation/check_native_observer_bounded_lifecycle.py'):
        spec['source_sha256'][rel(path)] = sha(path)
    spec['references'].update({rel(path): sha(path) for path in (previous.PLAN, previous.OUT / 'entry-result.json',
        previous.OUT / 'native-io-diagnosis.json', previous.OUT / 'native-observer-lifecycle.json')})
    spec.update(stage=STAGE,
                cameras={role: {pose: groups[pose][0] for pose in shared.POSES} for role in spec['branches']},
                maximum_control_transitions=n * spec['maximum_unique_controls'],
                heldout_used=True, holdout_for_tuning=False,
                heldout_scope='New reserved seed after observer stop; no tuning or automatic promotion.',
                observation_change='Retry only a missing exact status source for at most 0.5 s; all existing freshness and stop guards remain.',
                capacity_reason=registered['reason'])
    shared.assert_sources(spec)
    scene = cpu_scene_readback(p, spec)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN), scene=scene,
          seed=seed, num_envs=n, heldout_used=True, holdout_for_tuning=False,
          observer_lifecycle_sha256=sha(previous.OUT / 'native-observer-lifecycle.json'),
          physical_substeps=0, learning_updates=0))
    project = read(ROOT / 'project.json')
    task = project['staged_training_request']
    if (task['pause_requested'] or task['simulation_running'] or task['learning_running']
            or project['goal_execution']['status'] != 'active'):
        raise RuntimeError('Project is paused or another run is active')
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Open the current native view, run the new bounded holdout once, then independently verify it.')
    project['goal_execution']['monitored_getup_holdout_recovery'].update(status='frozen-cpu-zero-step-passed',
        protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run_id)
    write(ROOT / 'project.json', project)
    print('BOUNDED_HOLDOUT_FROZEN', run_id, sha(PLAN), flush=True)


def allowed(protocol_hash):
    try:
        project = read(ROOT / 'project.json')
        task, goal = project['staged_training_request'], project['goal_execution']
        return (goal['status'] == 'active' and not goal.get('pause_requested_by_user', False)
                and not task['pause_requested'] and not task['learning_running']
                and task['current_stage'] == STAGE and task['current_protocol_frozen']
                and task['current_protocol'] == rel(PLAN) and sha(PLAN) == protocol_hash)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def running(value):
    project = read(ROOT / 'project.json')
    task = project['staged_training_request']
    item = project['goal_execution']['monitored_getup_holdout_recovery']
    if value:
        if not allowed(sha(PLAN)) or task['simulation_running']:
            raise RuntimeError('Project does not allow the held-out evaluation')
        item['owner_pid'] = os.getpid()
    elif item.get('owner_pid') != os.getpid():
        raise RuntimeError('Held-out evaluation owner changed')
    task['simulation_running'] = value
    item['status'] = 'running' if value else 'stopped-review-required'
    write(ROOT / 'project.json', project)


def run():
    with patch.object(previous, 'PLAN', PLAN), patch.object(previous, 'OUT', OUT), \
         patch.object(previous, 'STAGE', STAGE), patch.object(previous, 'allowed', allowed), \
         patch.object(previous, 'running', running), \
         patch.object(base, 'NativeRollObservation', NativeRollObservationBounded):
        previous.run()


def verify():
    spec = read(PLAN)
    if sha(PLAN) != read(OUT / 'preflight.json')['protocol_sha256'] or spec['heldout_used'] is not True:
        raise RuntimeError('The held-out protocol differs from preflight')
    shared.assert_sources(spec)
    record = ROOT / 'records/evaluations' / spec['environment']['run_id']
    state = read(record / 'evaluation.json')
    if state['status'] != 'collected-review-required' or state['heldout_used'] is not True:
        raise RuntimeError('Held-out sampling was not completed and labelled')

    def select(roles, reports, comparisons, minimum_gain):
        return dict(original_gates={role: bool(reports[role]['all_motion_gates_passed']) for role in roles},
                    model_promotion_allowed=False,
                    explanation='Reserved holdout is reported without automatic replacement or tuning.')

    with patch.object(verifier, 'select', select):
        result = verifier.verify(spec, record, state, minimum_gain=0)
    result.update(heldout_used=True, holdout_for_tuning=False,
                  heldout_scope='New reserved seed; no tuning or automatic promotion.')
    write(OUT / 'analysis.json', dict(result, protocol_sha256=sha(PLAN)))
    project = read(ROOT / 'project.json')
    project['goal_execution']['monitored_getup_holdout_recovery']['status'] = 'independently-verified'
    write(ROOT / 'project.json', project)
    print('BOUNDED_HOLDOUT_VERIFIED', {role: arm['combined_successes'] for role, arm in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    modes = {'freeze': freeze, 'run': run, 'verify': verify}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('Usage: holdout_monitored_getup_bounded.py {freeze|run|verify}')
    modes[sys.argv[1]]()
