"""Freeze, visibly train and independently score one bounded getup PPO pair."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

import getup_shared_release as shared
import getup_supine_evaluation as evaluation
from getup_anchor_same_release import model_metadata, visible_log
from getup_shared_release_terrain import cpu_scene_readback
from prepare_getup_supine_training import DRAFT, OUT, training_plans
from record_io import write
from run_getup_supine_training import check_matched_start, train, zero_step
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_getup_supine_evaluation import verify as verify_evaluation
from verify_getup_supine_training import verify as verify_training


PLAN = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
PRIOR = ROOT/'configs/experiments/getup_anchor_same_release.json'
PROJECT = ROOT/'project.json'
SCENE = OUT/'scene-static-fields.npz'
PROGRESS = OUT/'coordinator.json'
STAGE = 'S3-supine-weighted-learning'
ARMS = ('control', 'candidate')


def checked(spec):
    eval_spec = spec['evaluation']
    shared.assert_sources(eval_spec)
    for plan in spec['training'].values():
        for base, key in ((WORK, 'training_source_sha256'), (Path('/'), 'library_source_sha256')):
            for filename, checksum in plan[key].items():
                assert sha(base/filename) == checksum, filename
        assert sha(ROOT/plan['parent_checkpoint']) == plan['parent_checkpoint_sha256']
        assert sha(ROOT/plan['probe_dataset']) == plan['probe_sha256']


def freeze():
    assert not PLAN.exists() and not PROGRESS.exists(), 'Frozen or closed protocols cannot be repeated'
    assert not torch.cuda.is_initialized()
    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and not draft['learning_started']
    checks = {name: OUT/f'{name}-preflight.json' for name in ('checkpoint', 'optimizer', 'scene')}
    cpu = {name: read(path) for name, path in checks.items()}
    for proof in cpu.values():
        assert proof['draft_sha256'] == sha(DRAFT) and not proof['cuda_initialized']
        assert proof['physical_substeps'] == 0 and proof['status'].startswith('passed-cpu-')
        for filename, checksum in proof.get('source_sha256', {}).items():
            assert sha(ROOT/filename) == checksum, filename
    assert cpu['scene']['static_fields_sha256'] == sha(SCENE)
    assert cpu['scene']['script_sha256'] == sha(Path(sys.modules[training_plans.__module__].__file__))
    assert cpu['scene']['checkpoint_preflight_sha256'] == sha(checks['checkpoint'])
    assert cpu['scene']['optimizer_preflight_sha256'] == sha(checks['optimizer'])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    historical, training = training_plans(draft)
    proposed = draft['proposed_first_pair']
    train_seed = proposed['training_seed']
    for role, p in training.items():
        p['run_id'] = f'duck_getup_supine-weighted_{role}_s{train_seed}_{stamp}'
    assert not any('getup' in d.name and f'_s{train_seed}_' in d.name for d in (ROOT/'records/experiments').iterdir())
    controls = proposed['rollout_control_steps']*proposed['maximum_updates_per_arm']
    transitions = proposed['num_envs']*controls
    assert controls == 1200 and transitions == proposed['maximum_learning_transitions_per_arm']
    assert 2*transitions == proposed['maximum_learning_transitions_total']

    prior = read(PRIOR)
    shared.assert_sources(prior)
    prior_analysis = read(ROOT/draft['evidence']['evaluation'])
    prior_record = ROOT/prior_analysis['record']
    prior_state = read(prior_record/'evaluation.json')
    assert sha(prior_record/'protocol.json') == sha(PRIOR)
    replay = verify_evaluation(prior, prior_record, prior_state, minimum_gain=6)
    for role, proof in replay['arms'].items():
        assert proof['combined_successes'] == prior_analysis['arms'][role]['combined_successes']
        assert proof['all_motion_gates_passed'] == prior_analysis['arms'][role]['all_motion_gates_passed']
    assert replay['selection']['selected_for_independent_confirmation'] is None
    p = deepcopy(prior['environment'])
    roles = draft['evaluation']['arms']
    parent = draft['parent']
    n, seed = draft['evaluation']['worlds_per_arm'], proposed['evaluation_development_seed']
    allocation = list(draft['evaluation']['requested_pose_counts_prone_supine_left_right'])
    assert n == sum(allocation) == 320 and allocation == [32, 32, 128, 128]
    for folder in ('records/experiments', 'records/evaluations'):
        assert not any(f'_s{seed}_' in d.name for d in (ROOT/folder).iterdir())
    allocation[1] -= 1
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), allocation))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    p.update(seed=seed, num_envs=n, run_id=f'duck_getup_supine-weighted_paired_s{seed}_{stamp}', stage=STAGE,
             split=dict(pose_codes=codes.tolist(), groups=groups), model=parent['model'],
             model_sha256=parent['model_sha256'])
    cameras = {role: {'supine': groups['supine'][0]} for role in roles}
    cameras[roles[-1]] = {pose: groups[pose][0] for pose in shared.POSES}
    assert prior['maximum_control_transitions'] == draft['evaluation']['maximum_control_transitions']
    for name in ('criteria', 'success_rate_min', 'actual_release_minimum', 'handoff_settings',
                 'model_home', 'simulation_home', 'joint_limits', 'maximum_control_steps'):
        assert p[name] == prior['environment'][name], name
    assert not p['neck_home_enabled'] and not p['handoff_enabled']
    sources = dict(prior['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('check_getup_supine_parent.py', 'check_getup_supine_ppo.py'):
        filename = f'scripts/validation/{name}'
        sources[filename] = sha(ROOT/filename)
    references = dict(prior['references'])
    training_refs = [ROOT/historical[key] for key in ('probe_dataset', 'anchor_data', 'anchor_cells')]
    reference_paths = [DRAFT, PRIOR, SCENE, *checks.values(), *training_refs,
                       *(ROOT/name for name in draft['evidence'].values()), ROOT/parent['training_protocol']]
    references.update({path.relative_to(ROOT).as_posix(): sha(path) for path in reference_paths})
    eval_spec = dict(prior, stage=STAGE, environment=p, branches=roles, cameras=cameras,
                     policies={roles[0]: dict(model=p['model'], model_sha256=p['model_sha256'], seed=seed)},
                     source_sha256=sources, references=references, model_interface=model_metadata(p, parent))
    assert eval_spec['model_interface'] == prior['model_interface']
    spec = dict(status='frozen-before-physical-steps', stage=STAGE, training=training,
                training_limit=dict(controls=controls, substeps=5*controls, transitions=transitions,
                                    rollout_controls=proposed['rollout_control_steps'], wall_seconds_per_arm=900),
                evaluation=eval_spec, actual_supine_minimum_gain=6,
                selection_order=[roles[1], roles[0], roles[2]],
                hardware_io=False, heldout_used=False, automatic_retry=False, automatic_extension=False)
    checked(spec)
    snapshot = cpu_scene_readback(p, eval_spec)
    log = io.StringIO()
    with visible_log(log):
        print('STDOUT')
        print('STDERR', file=sys.stderr)
    assert log.getvalue().splitlines() == ['STDOUT', 'STDERR'] and not torch.cuda.is_initialized()
    assert len(sources) >= len(prior['source_sha256'])
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          source_count=len(sources), scene_snapshot=snapshot, reference_verifier_replay={
              name: arm['combined_successes'] for name, arm in replay['arms'].items()},
          training_worlds=proposed['num_envs'], evaluation_worlds=n, output_streams_verified=True,
          physical_substeps=0, cuda_initialized=False, heldout_used=False))
    print('SUPINE_PAIR_FROZEN', sha(PLAN), 'training', train_seed, 'evaluation', seed, flush=True)


def available():
    state = read(PROJECT)['staged_training_request']
    assert state['current_stage'] == STAGE and state['current_protocol_frozen'] and not state['pause_requested']
    assert ROOT/state['current_protocol'] == PLAN


def project_running(simulation, learning):
    project = read(PROJECT)
    project['staged_training_request'].update(simulation_running=simulation, learning_running=learning)
    write(PROJECT, project)


def run_verifier(option):
    command = [sys.executable, '-B', str(Path(__file__)), option]
    result = subprocess.run(command, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end='', flush=True)
    result.check_returncode()


def run():
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Run in the original visible VS Code task terminal'
    available()
    spec, pre = read(PLAN), read(OUT/'preflight.json')
    protocol_sha = sha(PLAN)
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == protocol_sha
    assert not PROGRESS.exists(), 'Closed protocols cannot restart'
    checked(spec)
    state = dict(status='running', phase='gpu-zero-step', pid=os.getpid(), protocol_sha256=protocol_sha,
                 started_at=datetime.now(timezone.utc).isoformat(), visible_tty=True,
                 training_records={role: f'records/experiments/{p["run_id"]}' for role, p in spec['training'].items()},
                 evaluation_record=f'records/evaluations/{spec["evaluation"]["environment"]["run_id"]}')

    def publish(**fields):
        state.update(fields)
        write(PROGRESS, state)

    publish()
    write(OUT/'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(), visible_tty=True))
    project_running(True, False)
    try:
        with (OUT/'execution.log').open('x', buffering=1) as log, visible_log(log):
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            shutil.copy2(PLAN, OUT/'protocol.json')
            for name in spec['evaluation']['source_sha256']:
                target = OUT/'sources'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, target)
            for role in ARMS:
                available()
                publish(phase=f'gpu-zero-{role}')
                print('SUPINE_GPU_ZERO', role, flush=True)
                zero_step(spec['training'][role], OUT/'zero-step'/role, SCENE)
            check_matched_start(OUT/'zero-step'/'candidate', OUT/'zero-step'/'control')
            write(OUT/'zero-step/verification.json', dict(status='passed', identical_reset_and_rng=True,
                  parent_networks_and_adam_exact=True, physical_substeps=0, protocol_sha256=protocol_sha))
            print('SUPINE_GPU_ZERO_MATCHED', flush=True)
            project_running(True, True)
            for role in ARMS:
                available()
                publish(phase=f'training-{role}')
                record = ROOT/state['training_records'][role]
                write(ROOT/'records/training-current.json', dict(path=str(record), record=str(record),
                      run_id=spec['training'][role]['run_id'], pid=os.getpid(), purpose='supine-only-actor-weight-paired-ppo'))
                train(spec['training'][role], record, SCENE, OUT/'zero-step'/role,
                      protocol_sha, spec['training_limit'], available)
            project_running(True, False)
            publish(phase='independent-training-verification')
            run_verifier('--verify-training')
            available()
            bound = evaluation.bind(spec, OUT, protocol_sha)
            record = ROOT/state['evaluation_record']
            publish(phase='shared-release-evaluation')
            evaluation.run(bound, record, SCENE, available)
            publish(phase='independent-evaluation-verification')
            run_verifier('--verify-evaluation')
            proof = read(OUT/'analysis.json')
            publish(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'),
                    selection=proof['selection'])
    except BaseException as error:
        publish(status='stopped-review-required', error=repr(error))
        raise
    finally:
        publish(ended_at=datetime.now(timezone.utc).isoformat())
        project_running(False, False)


def verify(phase):
    spec, coordinator = read(PLAN), read(PROGRESS)
    assert coordinator['protocol_sha256'] == sha(PLAN)
    assert sha(OUT/'protocol.json') == sha(PLAN)
    checked(spec)
    if phase == 'training':
        assert coordinator['phase'] == 'independent-training-verification'
        zero = read(OUT/'zero-step/verification.json')
        assert zero['status'] == 'passed' and zero['protocol_sha256'] == sha(PLAN)
        check_matched_start(OUT/'zero-step'/'candidate', OUT/'zero-step'/'control')
        for role in ARMS:
            verify_training(spec['training'][role], OUT, sha(PLAN), spec['training_limit'])
    else:
        assert coordinator['phase'] == 'independent-evaluation-verification'
        record = ROOT/coordinator['evaluation_record']
        state, bound = read(record/'evaluation.json'), read(record/'protocol.json')
        assert state['status'] == 'collected-review-required'
        assert bound == evaluation.bind(spec, OUT, sha(PLAN))
        shared.assert_sources(bound)
        result = verify_evaluation(bound, record, state, spec['actual_supine_minimum_gain'])
        result.update(protocol_sha256=sha(PLAN), evaluation_protocol_sha256=sha(record/'protocol.json'),
                      training={role: read(OUT/f'{role}-training-verification.json') for role in ARMS})
        write(OUT/'analysis.json', result)
        state.update(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'))
        write(record/'evaluation.json', state)
        print('SUPINE_PAIR_VERIFIED', {role: arm['combined_successes'] for role, arm in result['arms'].items()},
              result['selection'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify-training', action='store_true')
    mode.add_argument('--verify-evaluation', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify_training or args.verify_evaluation:
        verify('training' if args.verify_training else 'evaluation')
    else:
        run()
