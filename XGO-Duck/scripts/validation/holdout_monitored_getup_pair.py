"""One reserved held-out comparison of the previously reviewed PPO pair.

Reuse the unmodified observed shared-release runner. Only the seed, sample
count, run budget, and held-out bookkeeping change; the three frozen models,
physical scene, original action gates, and full-episode recordings remain the
same. Results are not used to tune this candidate or promote it automatically.
"""
from __future__ import annotations

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
import verify_getup_supine_evaluation as verifier
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import read, sha

PLAN = ROOT / 'configs/experiments/getup_monitored_pair_holdout_s4267.json'
DEVELOPMENT = base.PLAN
DEV_OUT = base.OUT
OUT = ROOT / 'records/reports/getup-monitored-pair-holdout-s4267'
STAGE = 'observed-getup-holdout'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    if PLAN.exists() or OUT.exists():
        raise FileExistsError('The held-out attempt must not overwrite a previous one')
    development = read(DEV_OUT / 'analysis.json')
    registered = read(ROOT / 'project.json')['goal_execution']['monitored_getup_holdout']
    if (development['status'] != 'passed-independent-verification'
            or development['protocol_sha256'] != sha(DEVELOPMENT)
            or not development['arms']['candidate']['all_motion_gates_passed']
            or registered['development_analysis_sha256'] != sha(DEV_OUT / 'analysis.json')
            or registered['holdout_for_tuning'] is not False):
        raise ValueError('A passing, frozen development result is required before holdout')
    if any('_s4267_' in p.name for p in (ROOT / 'records/evaluations').iterdir()):
        raise ValueError('Held-out seed has already been used')
    spec = deepcopy(read(DEVELOPMENT))
    shared.assert_sources(spec)
    p = spec['environment']
    n, seed = 3072, 4267
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(
        np.repeat(np.arange(4), [288, 287, 1248, 1248]))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_monitor-pair_holdout_e{n}_s{seed}_{stamp}'
    p.update(num_envs=n, seed=seed, stage=STAGE, run_id=run_id,
             split=dict(pose_codes=codes.tolist(), groups=groups),
             maximum_control_transitions=n * p['maximum_control_steps'],
             video_worlds={pose: groups[pose][0] for pose in shared.POSES})
    for model in spec['policies'].values():
        model['seed'] = seed
    spec['source_sha256'][rel(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    spec['references'].update({rel(path): sha(path) for path in (
        DEVELOPMENT, DEV_OUT / 'analysis.json', DEV_OUT / 'preflight.json',
        DEV_OUT / 'actual-right-review/manifest.json')})
    spec.update(stage=STAGE,
                cameras={role: {pose: groups[pose][0] for pose in shared.POSES} for role in spec['branches']},
                maximum_control_transitions=n * spec['maximum_unique_controls'],
                maximum_warp_snapshot_bytes=1280 * 1024 * 1024,
                maximum_sampling_wall_seconds=1500, heldout_used=True,
                heldout_scope='One independent reserved seed; do not tune the candidate using this result.',
                choice='Report original action gates and matched outcomes; do not automatically promote a model.')
    shared.assert_sources(spec)
    scene = cpu_scene_readback(p, spec)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          development_analysis_sha256=sha(DEV_OUT / 'analysis.json'), scene=scene, seed=seed, num_envs=n,
          heldout_used=True, holdout_for_tuning=False, physical_substeps=0, learning_updates=0))
    project = read(ROOT / 'project.json')
    task = project['staged_training_request']
    if (task['pause_requested'] or task['simulation_running'] or task['learning_running']
            or project['goal_execution']['status'] != 'active'):
        raise RuntimeError('Project is paused or another run is active')
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Open the current native view, run the bounded holdout once, and independently verify it.')
    project['goal_execution']['monitored_getup_holdout'].update(status='frozen-cpu-zero-step-passed',
        protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run_id)
    write(ROOT / 'project.json', project)
    print('GETUP_HOLDOUT_FROZEN', run_id, sha(PLAN), flush=True)


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
    item = project['goal_execution']['monitored_getup_holdout']
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
    # The legacy runner labels every evaluation as development. Correct only
    # this process's saved status metadata before writing; physics is untouched.
    original = base.early.evaluation.run

    def mark_heldout(spec, record, scene, check_active):
        writer = base.early.evaluation.write

        def write_marked(path, value):
            if path == record / 'evaluation.json':
                value['heldout_used'] = True
            return writer(path, value)

        with patch.object(base.early.evaluation, 'write', write_marked):
            return original(spec, record, scene, check_active)

    with patch.object(base, 'PLAN', PLAN), patch.object(base, 'OUT', OUT), patch.object(base, 'STAGE', STAGE), \
         patch.object(base, 'allowed', allowed), patch.object(base, 'running', running), \
         patch.object(base.early.evaluation, 'run', mark_heldout):
        base.run()


def verify():
    spec = read(PLAN)
    if sha(PLAN) != read(OUT / 'preflight.json')['protocol_sha256'] or spec['heldout_used'] is not True:
        raise RuntimeError('The reserved protocol differs from preflight')
    shared.assert_sources(spec)
    record = ROOT / 'records/evaluations' / spec['environment']['run_id']
    state = read(record / 'evaluation.json')
    if state['status'] != 'collected-review-required' or state['heldout_used'] is not True:
        raise RuntimeError('Held-out sampling was not completed and labelled')

    def select(roles, reports, comparisons, minimum_gain):
        return dict(original_gates={role: bool(reports[role]['all_motion_gates_passed']) for role in roles},
                    model_promotion_allowed=False,
                    explanation='Reserved holdout is reported without automatic model replacement or tuning.')

    with patch.object(verifier, 'select', select):
        result = verifier.verify(spec, record, state, minimum_gain=0)
    # The shared historical verifier hardcodes development bookkeeping. The
    # current protocol and saved acquisition status both identify this holdout.
    result.update(heldout_used=True, holdout_for_tuning=False,
                  heldout_scope='Independent reserved seed; no automatic promotion or tuning.')
    write(OUT / 'analysis.json', dict(result, protocol_sha256=sha(PLAN)))
    project = read(ROOT / 'project.json')
    project['goal_execution']['monitored_getup_holdout']['status'] = 'independently-verified'
    write(ROOT / 'project.json', project)
    print('GETUP_HOLDOUT_VERIFIED', {role: arm['combined_successes'] for role, arm in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    modes = {'freeze': freeze, 'run': run, 'verify': verify}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('Usage: holdout_monitored_getup_pair.py {freeze|run|verify}')
    modes[sys.argv[1]]()
