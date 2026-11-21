"""Verify matched forward PPO starts below the measured local GPU ceiling."""

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
from mjlab.utils.torch import configure_torch_backends

import repair_forward_yaw_zero_single_reset as single
import run_forward_yaw_reward_zero as zero


D, ROOT, PROJECT, REPORT = zero.D, zero.ROOT, zero.PROJECT, zero.REPORT
PLAN = ROOT / 'configs/experiments/forward_low_friction_yaw_reward_memory_admission.json'
ENTRY = Path(__file__).resolve()


def freeze():
    project, prior = D.read(PROJECT), D.read(single.PLAN)
    task, state = project['staged_training_request'], project['goal_execution'][zero.check.KEY]
    failure_path = ROOT / state['zero_evidence']
    failure = D.read(failure_path)
    D.require(D.pair.trial.allowed(project, zero.STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] == 'cuda-zero-review' and not PLAN.exists()
              and D.sha(single.PLAN) == state['zero_protocol_sha256'] == failure['protocol_sha256']
              and D.sha(failure_path) == state['zero_evidence_sha256'] and prior['worlds'] == 4096
              and failure['learning_updates'] == failure['physics_substeps'] == 0
              and 'Observed GPU memory ceiling: 15760.0/16384.0 MiB' in failure['error'],
              'Expected the measured 4,096-world memory ceiling')
    inputs = failure_path.parent / 'control-initial-state.npz'
    with np.load(inputs, allow_pickle=False) as data:
        D.require(data['actor'].shape == (4096, 61)
                  and np.all(data['foot_sliding'] == np.float32(.8)), 'The recorded zero-step layout differs')
    prior['sources'].update({D.rel(f): D.sha(f) for f in (ENTRY, single.PLAN, failure_path, inputs)})
    prior.update(worlds=3072, startup_processes='One fresh CUDA process per arm',
                 higher_world_count_evidence=D.rel(failure_path), higher_world_count=4096)
    zero.verify_sources(prior)
    D.put(PLAN, prior)
    task.update(current_protocol=D.rel(PLAN), current_evidence=D.rel(failure_path),
                next_gate='Compare two isolated 3,072-world CUDA starts under the measured GPU memory ceiling.')
    state.update(status='cuda-zero-ready', zero_protocol=D.rel(PLAN), zero_protocol_sha256=D.sha(PLAN))
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_MEMORY_ADMISSION_FROZEN', D.sha(PLAN), flush=True)


def checked(status):
    project, plan = D.read(PROJECT), D.read(PLAN)
    state, task = project['goal_execution'][zero.check.KEY], project['staged_training_request']
    D.require(D.pair.trial.allowed(project, zero.STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] == status and state['zero_protocol_sha256'] == D.sha(PLAN),
              'CUDA initialization authorization changed')
    zero.verify_sources(plan)
    return project, plan, state


def worker(role):
    _, plan, state = checked('cuda-zero-running')
    D.require(state['worker_parent_pid'] == os.getppid() and state['active_zero_role'] == role,
              'Use the recorded CUDA initialization coordinator')
    output = ROOT / state['zero_record']
    route, draft = (D.read(ROOT / plan[key]) for key in ('route', 'draft'))
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = zero.MemoryWatch(draft['capacity']['gpu_memory_fraction_max'])
    try:
        with watch.active(), patch.object(zero, 'ForwardRollout', single.SingleResetForwardRollout):
            zero.capture_zero(plan, route, draft, role, output, output / 'common-parent.pt', watch)
    finally:
        D.put(output / f'{role}-resources.json', zero.resources(watch))


def run():
    project, plan, state = checked('cuda-zero-ready')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    out = REPORT / f'cuda-zero-memory-{stamp}'
    out.mkdir()
    outcome = {'protocol_sha256': D.sha(PLAN), 'worlds': plan['worlds'], 'physics_substeps': 0,
               'learning_updates': 0, 'hardware_io': False}
    path = out / 'pair-check.json'
    try:
        for name in ('control', 'candidate'):
            project = D.read(PROJECT)
            D.require(D.pair.trial.allowed(project, zero.STAGE), 'Pause before the next CUDA initialization')
            run_state = project['goal_execution'][zero.check.KEY]
            run_state.update(status='cuda-zero-running', zero_record=D.rel(out),
                             worker_parent_pid=os.getpid(), active_zero_role=name)
            project['staged_training_request']['current_evidence'] = D.rel(out)
            D.pair.trial.write(PROJECT, project)
            cmd = [sys.executable, '-B', '-u', str(ENTRY), 'worker', name]
            with (out / f'{name}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND ' + subprocess.list2cmdline(cmd) + '\n')
                log.flush()
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(out / 'control-initial-state.npz', allow_pickle=False) as a, \
                np.load(out / 'candidate-initial-state.npz', allow_pickle=False) as b:
            zero.compare_arrays({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
        parent = out / 'common-parent.pt'
        outcome.update(status='passed-matched-cuda-zero', parent=D.rel(parent), parent_sha256=D.sha(parent),
            matched_physical_state_observations_and_rng=True, initial_learner_exact=True,
            resources={role: D.read(out / f'{role}-resources.json') for role in ('control', 'candidate')},
            receipts={role: D.sha(out / f'{role}.json') for role in ('control', 'candidate')})
        print('FORWARD_YAW_MATCHED_CUDA_ZERO', D.rel(path),
              [outcome['resources'][k]['peak_used_mib'] for k in ('control', 'candidate')], flush=True)
    except BaseException as error:
        outcome.update(status='review-required', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        D.put(path, outcome)
        current = D.read(PROJECT)
        mark = current['goal_execution'][zero.check.KEY]
        mark.pop('active_zero_role', None)
        mark.pop('worker_parent_pid', None)
        passed = outcome['status'] == 'passed-matched-cuda-zero'
        mark.update(status='cuda-zero-passed' if passed else 'cuda-zero-review',
                    zero_evidence=D.rel(path), zero_evidence_sha256=D.sha(path))
        current['staged_training_request'].update(current_evidence=D.rel(path),
            next_gate='Run measured forward PPO capacity tests.' if passed else 'Review the isolated CUDA start.')
        D.pair.trial.write(PROJECT, current)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'run', 'worker'))
    parser.add_argument('role', choices=('control', 'candidate'), nargs='?')
    args = parser.parse_args()
    if args.operation == 'worker':
        worker(args.role)
    else:
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (freeze if args.operation == 'freeze' else run)()
