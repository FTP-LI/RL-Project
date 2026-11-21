"""Repair native JSON readback for a new matched upright-band run."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_nominal_supine_upright_band as original


ROOT, PROJECT = original.ROOT, original.PROJECT
read, rel, sha, write = original.read, original.rel, original.sha, original.write
KEY = original.KEY + '_observation_recovery'
STAGE = 'nominal-supine-upright-band-observed-pair'
PLAN = original.PLAN.with_name(KEY + '.json')
OUT = original.OUT.parent / 'observed-training'
ENTRY = Path(__file__).resolve()


class JsonRunObserver(original.RunObserver):
    def publish(self, record, snapshot):
        # A Python tuple in the PPO metrics is stored as a JSON array.
        # Compare in JSON representation while retaining the existing native
        # checks for exact bytes, progress, timestamps and media provenance.
        canonical = json.loads(json.dumps(snapshot, allow_nan=False))
        return super().publish(record, canonical)


def reproduce_readback(record, plan):
    state = read(record / 'state.json')
    assert state['status'] == 'stopped-review-required'
    assert state['controls'] == 24 and state['substeps'] == 120 and state['completed_updates'] == 1
    metric, = [json.loads(line) for line in (record / 'learning-metrics.jsonl').read_text().splitlines()]
    assert metric['active_policy_samples'] == metric['actor_gradient_steps'] == 0
    assert metric['critic_steps_this_run'] == 120 and metric['controls'] == 24
    observer_error = "PPO snapshot does not match the written run observation"
    assert state['error'] == f"ValueError({observer_error!r})"

    # Reuse the interrupted update and its original rendered frame. Recreate
    # the in-memory tuple which JSON converted when the update was first saved.
    runtime_row = dict(metric, replay_drift_before_rad=tuple(metric['replay_drift_before_rad']))
    n, updates = plan['environments'], plan['updates_per_arm']
    temporary_root = ROOT / 'records/experiments'
    with TemporaryDirectory(prefix='upright-observer-test-', dir=temporary_root) as folder:
        trial = Path(folder)
        probe = dict(state, status='training', error=None, clips=[])
        assert len(probe['images']) == 1
        frame = probe['images'][0]['file']
        copyfile(record / frame, trial / frame)
        write(trial / 'state.json', probe)
        with patch.object(original.live, 'PAGE', trial / 'observation.html'):
            snapshot = original.live.publish(trial, probe, [runtime_row], n, updates,
                                             {'peak_mib': 0})
        observers = [cls(probe['run_id'], OUT, lambda: True, environments=n, planned_updates=updates)
                     for cls in (original.RunObserver, JsonRunObserver)]
        for observer in observers:
            observer.run_folder = trial
        try:
            observers[0].publish(trial, snapshot)
        except ValueError as error:
            assert str(error) == observer_error
        else:
            raise AssertionError('The original observer did not reproduce the interruption')
        corrected = observers[1]
        corrected.publish(trial, snapshot)
        assert corrected.state['rows'] == [metric]
        assert corrected.state['transitions'] == 24*n
        assert corrected.state['trainer_observation_sha256'] == sha(trial / 'observation.json')
        assert sha(temporary_root / corrected.state['frame']) == sha(record / frame)
    return dict(field='replay_drift_before_rad', failed_before=observer_error, corrected_readback=True,
                original_controls=24, original_batched_substeps=120,
                original_transitions=24*n, original_ppo_updates=1, original_actor_gradient_steps=0,
                original_critic_gradient_steps=120, source_frame_sha256=sha(record / frame),
                new_physical_substeps=0, new_learning_updates=0)


def freeze():
    assert not OUT.exists() and not PLAN.exists()
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    old = goal[original.KEY]
    previous = read(original.PLAN, old['protocol_sha256'])
    assert original.parent.allowed(project, previous['training_stage']) and not task['simulation_running']
    assert old['status'] == 'review-required' and set(old['results']) == {'control'}
    assert not old['results']['control']['complete'] and KEY not in goal
    original.sources_checked(previous)
    stopped = ROOT / old['results']['control']['record']
    zero = ROOT / old['zero']
    admission = read(zero, old['zero_sha256'])
    assert admission['protocol_sha256'] == sha(original.PLAN)
    assert admission['physics_steps'] == admission['learning_updates'] == 0
    checkpoint = stopped / 'stopped-partial.pt'
    assert checkpoint.is_file()
    for role in original.ROLES:
        saved = zero.parent / role / 'matched-state.json'
        assert sha(saved) == admission['proof'][role]['matched_state_sha256']
    assert sha(stopped / 'matched-state.json') == admission['proof']['control']['matched_state_sha256']

    repair = reproduce_readback(stopped, previous)
    observer_preflight = original.parent.preflight()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    runs = {role: f'duck-getup-upright-band-{role}-observed-s{previous["training_seed"]}-{stamp}'
            for role in original.ROLES}
    plan = dict(previous, training_stage=STAGE, runs=runs, observer_preflight=observer_preflight,
                previous_protocol=rel(original.PLAN), previous_zero=rel(zero),
                observation_change='Canonicalize only the native JSON comparison. Both arms start from the original shared parent.')
    paths = (ENTRY, original.PLAN, zero, checkpoint, stopped / 'state.json', stopped / 'learning-metrics.jsonl',
             ROOT / old['results']['control']['result'], Path(original.parent.preflight.__code__.co_filename).resolve())
    plan['integrity'] = dict(previous['integrity'], **{rel(path):sha(path) for path in paths})
    for role in original.ROLES:
        saved = zero.parent / role / 'matched-state.json'
        plan['integrity'][rel(saved)] = sha(saved)
        output = OUT / 'cuda-zero' / role
        output.mkdir(parents=True)
        copyfile(saved, output / saved.name)
    write(PLAN, plan)
    reused = OUT / 'cuda-zero' / 'matched.json'
    write(reused, dict(admission, protocol_sha256=sha(PLAN), reused_prior_cuda_proof=rel(zero),
                       reused_prior_cuda_proof_sha256=sha(zero), per_arm_runtime_signature_required=True))
    receipt = OUT / 'observation-repair.json'
    write(receipt, dict(repair, status='passed-saved-native-readback', protocol_sha256=sha(PLAN),
                        stopped_partial=rel(checkpoint), stopped_partial_sha256=sha(checkpoint),
                        reused_cuda_zero=rel(reused), observer=observer_preflight))
    goal[KEY] = dict(status='paired-training-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
                     zero=rel(reused), zero_sha256=sha(reused), runs=runs, preflight=rel(receipt),
                     previous_interrupted_run=rel(stopped), previous_protocol_sha256=sha(original.PLAN),
                     stopped_first_run=repair, hardware_io=False)
    old.update(status='closed-native-json-observation-interruption', observation_recovery=rel(receipt),
               actual_controls=24, actual_batched_substeps=120, completed_updates=1, actor_gradient_steps=0)
    info = dict(current_protocol_execution='nominal-supine-upright-band-observation-repaired',
                current_evidence=rel(receipt))
    project.update(info)
    task.update(info, current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Verify both per-arm CUDA signatures, then run the equal-budget pair and shared-state evaluation.')
    write(PROJECT, project)
    print('UPRIGHT_BAND_NATIVE_REPAIR', sha(PLAN), runs, flush=True)


def train(role):
    with (patch.object(original, 'KEY', KEY), patch.object(original, 'OUT', OUT),
          patch.object(original, 'PLAN', PLAN), patch.object(original, 'RunObserver', JsonRunObserver)):
        original.train(role)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'train'))
    parser.add_argument('role', choices=original.ROLES, nargs='?')
    args = parser.parse_args()
    assert (args.role is not None) == (args.action == 'train')
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'freeze':
            freeze()
        else:
            train(args.role)


if __name__ == '__main__':
    main()
