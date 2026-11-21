"""Run the frozen nominal/randomized short PPO pair.

The learner, parent checkpoint, observations, rewards and action history are
the existing getup path.  The only candidate change is applied after the
shared configuration is built: foot friction is sampled in [1.0, 1.3] and
the actuator supply range is [7.4, 7.7] V.  This keeps the physical change
visible in one small function and makes the control arm an exact nominal
comparison.
"""

from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'scripts/validation'))
sys.path.insert(0, str(ROOT / 'scripts/training'))

from getup_recovery_retention_pair import SCENE
from record_io import write
from s1_reference_inputs import read, sha
import run_getup_recovery_retention as retention
import run_getup_supine_training as supine


PLAN = ROOT / 'configs/experiments/parameter_randomized_getup_training.json'
PARENT = ROOT / 'configs/experiments/getup_recovery_retention.json'
OUT = ROOT / 'records/reports/parameter-randomized-training'


def memory_row():
    free, total = torch.cuda.mem_get_info()
    return dict(at=time.time(), free_mib=int(free / 2**20), total_mib=int(total / 2**20),
                used_mib=int((total - free) / 2**20), fraction=float((total - free) / total))


class MemoryWatch:
    def __init__(self, limit):
        self.limit = limit
        self.rows = []
        self.stop = threading.Event()
        self.error = None

    def sample(self):
        row = memory_row()
        self.rows.append(row)
        if row['fraction'] > self.limit:
            raise RuntimeError(f'GPU memory ceiling exceeded: {row}')

    def monitor(self):
        while not self.stop.wait(1.0):
            try:
                self.sample()
            except BaseException as error:
                self.error = error
                return

    @contextmanager
    def active(self):
        self.sample()
        thread = threading.Thread(target=self.monitor, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self.stop.set()
            thread.join()

    def check(self):
        if self.error is not None:
            raise self.error


def arm_plan(spec, parent, role):
    p = json.loads(json.dumps(parent['training']['candidate']))
    p.update(
        arm=role,
        stage=spec['stage'],
        seed=spec['training_seed'],
        num_envs=spec['num_envs'],
        training_updates=spec['short_updates'],
        run_id=f'duck_getup_parameter-randomized_{role}_s{spec["training_seed"]}_{time.strftime("%Y%m%dT%H%M%S")}',
        scope='Frozen 8-update parameter-randomized getup pair; original learner and gates retained',
    )
    p['split']['pose_codes'] = [0, 1, 2, 3] * (spec['num_envs'] // 4)
    p['learner']['minibatches'] = 4
    return p


def prepare_for_arm(original, ranges):
    def prepare(plan):
        cfg, display, metadata = original(plan)
        # The deliberate experiment edit: only these two physical ranges vary.
        cfg.env.events['foot_friction'].params['ranges'] = tuple(ranges['foot_friction'])
        actuator = cfg.env.scene.entities['robot'].articulation.actuators[0]
        actuator.vin_range = tuple(ranges['vin_range'])
        display.update(parameter_arm=ranges, native_live_window=False,
                       recorder='ActionVideo; complete source clip and periodic wall snapshots')
        return cfg, display, metadata
    return prepare


def run_arm(spec, parent, role, ranges):
    p = arm_plan(spec, parent, role)
    record = ROOT / 'records/experiments' / p['run_id']
    zero_record = OUT / f'{role}-zero'
    if record.exists() or zero_record.exists():
        raise FileExistsError(f'preserve existing run: {record} {zero_record}')
    protocol_sha = sha(PLAN)
    original_prepare = supine.prepare
    randomized_prepare = prepare_for_arm(original_prepare, ranges)
    memory = MemoryWatch(spec['gates']['gpu_memory_fraction_max'])

    # The inherited startup audit assumes every expanded field is uniform.
    # Domain randomization deliberately makes only geom_friction vary per
    # environment, so keep the archived nominal representative for that one
    # field while retaining the full randomized expanded-fields archive.
    import common_scene_contract as scene_contract
    original_nominal_primary = scene_contract.nominal_primary

    def randomization_aware_primary(fields, n):
        retained = {key: value for key, value in fields.items() if key != 'geom_friction'}
        result = original_nominal_primary(retained, n)
        with np.load(ROOT / 'records/reports/common-plant-screen/startup/getup/nominal-primary-fields.npz') as data:
            result['geom_friction'] = data['geom_friction'].copy()
        return result

    def check_active():
        memory.check()
        project = read(ROOT / 'project.json')
        if project['staged_training_request'].get('pause_requested', False):
            raise RuntimeError('pause requested by project state')

    write(ROOT / 'records/training-current.json', dict(
        run_id=p['run_id'], record=record.relative_to(ROOT).as_posix(),
        purpose='parameter-randomized-getup-short-pair', arm=role,
        protocol=protocol_sha, pid=os.getpid()))
    started = time.time()
    try:
        with patch.object(supine, 'prepare', randomized_prepare), \
             patch.object(scene_contract, 'nominal_primary', randomization_aware_primary), memory.active():
            # zero_step writes the exact paired startup state used by train.
            supine.zero_step(p, zero_record, SCENE)
            memory.sample()
            limit = dict(rollout_controls=spec['rollout_controls'], controls=spec['rollout_controls'] * spec['short_updates'],
                         transitions=spec['transitions_per_arm'], substeps=spec['rollout_controls'] * spec['short_updates'] * 5,
                         wall_seconds_per_arm=1200)
            state = retention.run_with_retention(supine.train, p, record, SCENE, zero_record, protocol_sha, limit, check_active)
            memory.check()
        result = dict(status='completed-review-required', arm=role, run_id=p['run_id'],
                      record=record.relative_to(ROOT).as_posix(), protocol_sha256=protocol_sha,
                      actual_control_transitions=state['controls'] * spec['num_envs'],
                      actual_physics_substeps=state['substeps'], completed_updates=state['completed_updates'],
                      learning_metrics=record.joinpath('learning-metrics.jsonl').relative_to(ROOT).as_posix(),
                      video_glob=[path.relative_to(ROOT).as_posix() for path in record.glob('*.mp4')],
                      memory=memory.rows, elapsed_seconds=time.time() - started)
    except BaseException as error:
        result = dict(status='stopped-review-required', arm=role, run_id=p['run_id'],
                      record=record.relative_to(ROOT).as_posix(), protocol_sha256=protocol_sha,
                      error=repr(error), memory=memory.rows, elapsed_seconds=time.time() - started)
        record.mkdir(parents=True, exist_ok=True)
        write(record / 'parameter-randomized-stop.json', result)
        raise
    finally:
        write(OUT / f'{role}-result.json', result if 'result' in locals() else dict(status='failed-start'))
    return result


def main():
    spec = read(PLAN)
    parent = read(ROOT / spec['parent_protocol'])
    assert spec['status'] == 'frozen-before-learning'
    assert spec['short_updates'] == 8 and spec['transitions_per_arm'] == 196608
    assert spec['learning_updates_authorized_by_draft'] == 0
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for role, ranges in (('candidate', spec['candidate_randomization']), ('control', spec['control_nominal'])):
        print('PARAM_RANDOM_TRAIN_START', role, flush=True)
        result = run_arm(spec, parent, role, ranges)
        results.append(result)
        write(OUT / 'progress.json', dict(status='running', completed=len(results), total=2, results=results))
        print('PARAM_RANDOM_TRAIN_DONE', role, result['completed_updates'], flush=True)
    summary = dict(status='completed-review-required', protocol_sha256=sha(PLAN), arms=results,
                   total_transitions=sum(item.get('actual_control_transitions', 0) for item in results),
                   learning_updates=spec['short_updates'] * len(results))
    write(OUT / 'analysis.json', summary)
    write(OUT / 'progress.json', dict(status='completed', completed=2, total=2, results=results))
    print('PARAM_RANDOM_TRAIN_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
