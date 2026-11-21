"""Freeze, train and score a bounded recovery-retention pair in one visible task."""
import argparse
from copy import deepcopy
from dataclasses import asdict
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
import run_getup_supine_training as rollout
from common_scene_contract import compare_arrays, model_arrays
from evaluate_roll_reset_pair import differences
from getup_anchor_same_release import model_metadata, visible_log
from getup_shared_release_terrain import cpu_scene_readback
from getup_supine_training_pair import checked, project_running
from getup_teacher_anchor_train import prepare
from prepare_getup_recovery_retention import DRAFT, OUT, PAIR
from record_io import write
from review_getup_recovery_retention import groups as recovery_groups
from run_getup_recovery_retention import run_with_retention, verify_retention
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_getup_supine_evaluation import paired, verify as verify_evaluation
from verify_getup_supine_training import verify as verify_training


PLAN = ROOT/'configs/experiments/getup_recovery_retention.json'
PREVIOUS = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
SCENE = PAIR/'scene-static-fields.npz'
PROGRESS = OUT/'coordinator.json'
PROJECT = ROOT/'project.json'
STAGE = 'S3-recovery-retention'
ARMS = ('control', 'candidate')
ROLES = ('fixed-weighted', 'unretained-continuation', 'recovery-retention')


def decide(result, roles, minimum):
    parent, control, candidate = roles
    release, requested, chosen = recovery_groups(result['arms'][candidate])
    for role in (parent, control):
        role_release, role_requested, _ = recovery_groups(result['arms'][role])
        np.testing.assert_array_equal(release, role_release)
        np.testing.assert_array_equal(requested, role_requested)
    _, _, reference = recovery_groups(result['arms'][control])
    _, _, inherited = recovery_groups(result['arms'][parent])
    right = paired(reference, chosen, release == 'right-side')
    requested_supine = paired(reference, chosen, requested == 'supine')
    actual_supine = paired(inherited, chosen, release == 'supine')
    checks = dict(all_original_gates=result['arms'][candidate]['all_motion_gates_passed'],
                  actual_right_gain_against_control=right['net'],
                  requested_supine_gain_against_control=requested_supine['net'],
                  actual_supine_gain_against_fixed_weighted=actual_supine['net'])
    checks['passes_admission'] = bool(
        checks['all_original_gates'] and right['net'] >= minimum['actual_right']
        and requested_supine['net'] >= minimum['requested_supine'] and actual_supine['net'] >= 0
    )
    selected = next((role for role, passes in (
        (control, result['arms'][control]['all_motion_gates_passed']),
        (parent, result['arms'][parent]['all_motion_gates_passed']),
        (candidate, checks['passes_admission']),
    ) if passes), None)
    return dict(candidate=checks, selected_for_independent_confirmation=selected,
                actual_right=right, requested_supine=requested_supine, actual_supine=actual_supine)


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not PROGRESS.exists(), 'Frozen or closed protocols cannot restart'
    assert not torch.cuda.is_initialized()
    draft, prior = read(DRAFT), read(PREVIOUS)
    assert draft['status'] == 'draft-not-executable'
    checked(prior)
    preflight_path = OUT/'optimizer-preflight.json'
    prep_path = OUT/'replay-preflight.json'
    cpu, prep = read(preflight_path), read(prep_path)
    assert cpu['draft_sha256'] == prep['draft_sha256'] == sha(DRAFT)
    assert cpu['status'] == 'passed-cpu-full-ppo-fixture' and cpu['replay_preflight_sha256'] == sha(prep_path)
    assert cpu['physical_substeps'] == cpu['saved_optimizer_updates'] == 0 and not cpu['heldout_used']
    assert not cpu['cuda_initialized']
    for name, checksum in cpu['source_sha256'].items():
        assert sha(ROOT/name) == checksum, name
    budget = draft['proposed_budget']
    seed, n, steps = budget['training_seed'], budget['environments'], budget['maximum_ppo_updates_per_arm']
    length = budget['rollout_controls']
    assert (seed, n, steps, length) == (46, 512, 50, 24)
    assert 2*n*steps*length == budget['maximum_learning_transitions_total']
    eval_seed = budget['fresh_development_evaluation_seed']
    assert not any('getup' in d.name and f'_s{seed}_' in d.name for d in (ROOT/'records/experiments').iterdir())
    for folder in ('records/experiments', 'records/evaluations'):
        assert not any(f'_s{eval_seed}_' in d.name for d in (ROOT/folder).iterdir())
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    parent = read(PAIR/'candidate-training-verification.json')
    assert parent['status'] == 'passed' and parent['protocol_sha256'] == sha(PREVIOUS)
    assert parent['checkpoint_sha256'] == draft['starting_checkpoint_sha256']
    assert parent['model_sha256'] == draft['starting_model_sha256']
    historical = prior['training']['candidate']
    common = deepcopy(historical)
    codes = np.random.default_rng(10_000+seed).permutation(np.repeat(np.arange(4), 128))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    common.update(
        seed=seed, model=draft['starting_model'], model_sha256=draft['starting_model_sha256'],
        parent_checkpoint=draft['starting_checkpoint'], parent_checkpoint_sha256=draft['starting_checkpoint_sha256'],
        split=dict(pose_codes=codes.tolist(), groups=groups), anchor_seed=54_000+seed,
        retention_data=(OUT/'parent-replay.npz').relative_to(ROOT).as_posix(),
        retention_cells=(OUT/'sampling-cells.json').relative_to(ROOT).as_posix(),
        retention_seed=cpu['retention_seed'], retention_samples_per_cell=prep['samples_per_cell'],
        inherited_adam_steps=cpu['inherited_adam_steps'],
    )
    common['learner']['minibatch_seed'] = 5_500+seed
    assert common['focus_weight'] == 2. and common['anchor_weight'] == .01
    assert common['num_envs'] == n and common['training_updates'] == steps
    weights = {'control': 0., 'candidate': cpu['calibration']['chosen']}
    assert 0. < weights['candidate'] <= .02 and cpu['calibration']['chosen_conservative_ratio'] <= .10
    plans = {role: dict(deepcopy(common), arm=role, retention_weight=weight,
                        run_id=f'duck_getup_recovery-retention_{role}_s{seed}_{stamp}') for role, weight in weights.items()}
    assert {key for key in plans['control'] if plans['control'][key] != plans['candidate'][key]} == {
        'arm', 'run_id', 'retention_weight',
    }
    cfg, display, meta = prepare(plans['control'])
    other, _, other_meta = prepare(plans['candidate'])
    old_cfg, _, old_meta = prepare(historical)
    assert meta == other_meta == old_meta and not differences(asdict(cfg.env), asdict(other.env))
    old_cfg.env.seed = cfg.env.seed
    old_cfg.env.events['set_ground_state'].params['pose_codes'] = cfg.env.events['set_ground_state'].params['pose_codes']
    assert not differences(asdict(cfg.env), asdict(old_cfg.env))
    assert cfg.env.scene.num_envs == n and not display['native_live_window']
    with np.load(SCENE) as fields:
        compare_arrays(model_arrays(Scene(cfg.env.scene, device='cpu').compile()), {k: fields[k] for k in fields.files})

    analysis_path = ROOT/draft['source_evaluation']
    analysis = read(analysis_path)
    old_record = ROOT/analysis['record']
    old_spec, old_state = read(old_record/'protocol.json'), read(old_record/'evaluation.json')
    assert sha(old_record/'protocol.json') == analysis['evaluation_protocol_sha256']
    assert old_state['status'] == 'completed' and old_state['analysis_sha256'] == sha(analysis_path)
    old_verified = verify_evaluation(old_spec, old_record, old_state, minimum_gain=0)
    scores = {name: arm['combined_successes'] for name, arm in old_verified['arms'].items()}
    assert scores == {name: arm['combined_successes'] for name, arm in analysis['arms'].items()}
    minimum = dict(actual_right=2, requested_supine=1)
    retrospective = decide(old_verified, old_spec['branches'], minimum)
    assert retrospective['selected_for_independent_confirmation'] is None
    assert retrospective['candidate']['actual_right_gain_against_control'] == 1
    assert retrospective['candidate']['requested_supine_gain_against_control'] == 4
    assert retrospective['candidate']['actual_supine_gain_against_fixed_weighted'] == 10

    previous_eval = prior['evaluation']
    p = deepcopy(previous_eval['environment'])
    m = p['num_envs']
    allocation = [32, 31, 128, 128]  # World zero is the fixed supine camera.
    codes = np.concatenate(([1], np.random.default_rng(eval_seed).permutation(np.repeat(np.arange(4), allocation))))
    assert len(codes) == m == 320
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    p.update(seed=eval_seed, stage=STAGE, model=common['model'], model_sha256=common['model_sha256'],
             split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_recovery-retention_paired_s{eval_seed}_{stamp}')
    assert previous_eval['maximum_control_transitions'] == budget['maximum_evaluation_transitions_total']
    cameras = {role: {'supine': groups['supine'][0]} for role in ROLES}
    cameras[ROLES[-1]] = {pose: groups[pose][0] for pose in shared.POSES}
    sources = dict(previous_eval['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    name = 'scripts/validation/check_getup_recovery_retention_ppo.py'
    sources[name] = sha(ROOT/name)
    references = dict(previous_eval['references'])
    references.update({path.relative_to(ROOT).as_posix(): sha(path) for path in (
        DRAFT, PREVIOUS, SCENE, prep_path, preflight_path, analysis_path, ROOT/draft['retention_audit'],
        OUT/'parent-replay.npz', OUT/'sampling-cells.json', PAIR/'candidate-training-verification.json',
    )})
    eval_spec = dict(previous_eval, stage=STAGE, environment=p, branches=list(ROLES), cameras=cameras,
                     policies={ROLES[0]: dict(model=p['model'], model_sha256=p['model_sha256'], seed=eval_seed)},
                     source_sha256=sources, references=references,
                     model_interface=model_metadata(previous_eval['environment'], p))
    assert eval_spec['model_interface'] == previous_eval['model_interface']
    spec = dict(status='frozen-before-physical-steps', stage=STAGE, training=plans, evaluation=eval_spec,
                training_limit=deepcopy(prior['training_limit']), minimum_retention_gain=minimum,
                selection_order=[ROLES[1], ROLES[0], ROLES[2]], hardware_io=False, heldout_used=False,
                automatic_retry=False, automatic_extension=False)
    checked(spec)
    snapshot = cpu_scene_readback(p, eval_spec)
    log = io.StringIO()
    with visible_log(log):
        print('STDOUT')
        print('STDERR', file=sys.stderr)
    assert log.getvalue().splitlines() == ['STDOUT', 'STDERR'] and not torch.cuda.is_initialized()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          source_count=len(sources), scene_snapshot=snapshot, original_scores_replayed=scores,
          retention_coefficient=weights['candidate'], training_worlds=n, evaluation_worlds=m,
          physical_substeps=0, output_streams_verified=True, cuda_initialized=False, heldout_used=False))
    print('RETENTION_PAIR_FROZEN', sha(PLAN), 'weight', weights['candidate'], 'seeds', seed, eval_seed, flush=True)


def available():
    state = read(PROJECT)['staged_training_request']
    assert state['current_stage'] == STAGE and state['current_protocol_frozen'] and not state['pause_requested']
    assert ROOT/state['current_protocol'] == PLAN


def run_verifier(option):
    command = [sys.executable, '-B', str(Path(__file__)), option]
    result = subprocess.run(command, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end='', flush=True)
    result.check_returncode()


def run():
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original visible VS Code terminal'
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
                print('RETENTION_GPU_ZERO', role, flush=True)
                run_with_retention(rollout.zero_step, spec['training'][role], OUT/'zero-step'/role, SCENE)
            rollout.check_matched_start(OUT/'zero-step'/'candidate', OUT/'zero-step'/'control')
            write(OUT/'zero-step/verification.json', dict(status='passed', identical_reset_and_rng=True,
                  parent_networks_and_adam_exact=True, physical_substeps=0, protocol_sha256=protocol_sha))
            print('RETENTION_GPU_ZERO_MATCHED', flush=True)
            project_running(True, True)
            for role in ARMS:
                available()
                publish(phase=f'training-{role}')
                record = ROOT/state['training_records'][role]
                p = spec['training'][role]
                write(ROOT/'records/training-current.json', dict(path=str(record), record=str(record),
                      run_id=p['run_id'], pid=os.getpid(), purpose='paired-parent-action-retention-ppo'))
                print('RETENTION_PPO', role, 'coefficient', p['retention_weight'], flush=True)
                run_with_retention(rollout.train, p, record, SCENE, OUT/'zero-step'/role,
                                   protocol_sha, spec['training_limit'], available)
            project_running(True, False)
            publish(phase='independent-training-verification')
            run_verifier('--verify-training')
            available()
            bound = evaluation.bind(spec, OUT, protocol_sha)
            publish(phase='shared-release-evaluation')
            evaluation.run(bound, ROOT/state['evaluation_record'], SCENE, available)
            publish(phase='independent-evaluation-verification')
            run_verifier('--verify-evaluation')
            result = read(OUT/'analysis.json')
            publish(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'),
                    selection=result['selection'])
    except BaseException as error:
        publish(status='stopped-review-required', error=repr(error))
        raise
    finally:
        publish(ended_at=datetime.now(timezone.utc).isoformat())
        project_running(False, False)


def verify(phase):
    spec, coordinator = read(PLAN), read(PROGRESS)
    digest = sha(PLAN)
    assert coordinator['protocol_sha256'] == digest == sha(OUT/'protocol.json')
    checked(spec)
    if phase == 'training':
        assert coordinator['phase'] == 'independent-training-verification'
        zero = read(OUT/'zero-step/verification.json')
        assert zero['status'] == 'passed' and zero['protocol_sha256'] == digest
        rollout.check_matched_start(OUT/'zero-step'/'candidate', OUT/'zero-step'/'control')
        streams = {}
        for role in ARMS:
            p = spec['training'][role]
            proof = verify_training(p, OUT, digest, spec['training_limit'])
            proof['retention'] = verify_retention(p, proof)
            streams[role] = proof['retention']['sample_hashes']
            write(OUT/f'{role}-training-verification.json', proof)
        assert streams['control'] == streams['candidate']
        print('RETENTION_REPLAY_STREAMS_MATCHED', len(streams['control']), 'updates', flush=True)
    else:
        assert coordinator['phase'] == 'independent-evaluation-verification'
        record = ROOT/coordinator['evaluation_record']
        state, bound = read(record/'evaluation.json'), read(record/'protocol.json')
        assert state['status'] == 'collected-review-required'
        assert bound == evaluation.bind(spec, OUT, digest)
        shared.assert_sources(bound)
        result = verify_evaluation(bound, record, state, minimum_gain=0)
        result['selection'] = decide(result, bound['branches'], spec['minimum_retention_gain'])
        result.update(protocol_sha256=digest, evaluation_protocol_sha256=sha(record/'protocol.json'),
                      training={role: read(OUT/f'{role}-training-verification.json') for role in ARMS})
        write(OUT/'analysis.json', result)
        state.update(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'))
        write(record/'evaluation.json', state)
        print('RETENTION_PAIR_VERIFIED', {role: arm['combined_successes'] for role, arm in result['arms'].items()},
              result['selection']['candidate'], flush=True)


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
