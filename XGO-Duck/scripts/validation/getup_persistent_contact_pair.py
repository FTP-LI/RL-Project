"""Freeze and execute the matched persistent-contact PPO comparison.

The parent collectors and recovery gates are kept as archived. This study
changes the privileged value input and switches one training reward; both
policies have the same teacher anchor. Independent evaluation does not read
the training clock or reward, so it uses the unchanged recovery evaluator.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from s1_reference_inputs import WORK, read, sha
from record_io import write
import getup_teacher_anchor_pair as parent
from getup_persistent_contact_train import kinematic_stable, match_start, prepare, run, zero_step
from getup_late_feasibility_pair import candidate_admitted, cpu_selection_checks, decision

PLAN = ROOT / 'configs/experiments/getup_persistent_contact_pair.json'
OUT = ROOT / 'records/reports/getup-persistent-contact/pair'
STAGE = 'S3-persistent-contact-pair-execution'
ARMS = ('control', 'candidate')
EVAL_ARMS = ('control', 'original', 'candidate')


def freeze():
    import mujoco
    from mjlab.scene import Scene
    from mjlab.managers import reward_manager
    from common_scene_contract import compare_arrays, model_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_neck_home_order import check as processor_check
    from getup_release_coverage import cpu_check
    from verify_getup_persistent_contact_train import verify_clock

    assert not PLAN.exists() and not OUT.exists(), 'Keep frozen attempts immutable'
    torch.set_num_threads(4)
    value_path = OUT.parent / 'value/cpu-reciprocal-review/analysis.json'
    value = read(value_path)
    value_protocol = ROOT / 'configs/experiments/getup_persistent_value.json'
    plan = read(value_protocol)
    assert value['both_value_gates_passed'] and value['actor_bitwise_unchanged']
    assert value['actual_gpu_selected_reset_verified'] and value['critic_updates'] == 50
    assert value['protocol_sha256'] == sha(value_protocol)
    value_record = ROOT / value['record']
    data = value_record / 'value-data.npz'
    checkpoint = value_record / 'critic-final.pt'
    assert sha(data) == value['dataset_sha256'] and sha(checkpoint) == value['critic_sha256']
    warm = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert {int(s['step']) for s in warm['critic_optimizer_state_dict']['state'].values()} == {50}
    assert warm['reward_scale'] == value['reward_scale']

    ppo_path = OUT.parent / 'ppo-preflight.json'
    ppo = read(ppo_path)
    assert ppo['status'] == 'passed-cpu-real-optimizer-fixture'
    assert ppo['same_reward_actor_critic_adam_bitwise_equal'] and ppo['changed_reward_reaches_actor_and_critic']
    assert ppo['admission_sha256'] == sha(value_path) and ppo['source_critic_sha256'] == sha(checkpoint)
    draft_path = ROOT / 'configs/experiments/getup_persistent_contact_draft.json'
    draft = read(draft_path)
    parent_path = ROOT / 'configs/experiments/getup_teacher_anchor_pair.json'
    archived = read(parent_path)
    old = archived['training_arms']['candidate']
    anchor_data, anchor_cells = (ROOT / old[name] for name in ('anchor_data', 'anchor_cells'))
    anchor_preparation = anchor_data.with_name('preparation.json')
    anchor = read(anchor_preparation)
    assert sha(anchor_data) == ppo['replay_sha256'] == anchor['data_sha256']
    assert sha(anchor_cells) == anchor['cells_sha256']
    assert plan['seed'] == draft['paired_actor_training_seed'] == 44
    assert plan['contact_threshold_seconds'] == draft['candidate_contact_threshold_seconds'] == 1.2
    assert draft['candidate_reward_weight'] == -2 and draft['both_anchor_weights'] == ppo['both_anchor_weight'] == .01
    plan.update(status='frozen-before-learning', stage=STAGE, actor_frozen=False, critic_updates_at_start=50,
                critic_checkpoint=str(checkpoint.relative_to(ROOT)), critic_sha256=value['critic_sha256'],
                reward_scale=warm['reward_scale'], probe_dataset=str(data.relative_to(ROOT)),
                probe_sha256=value['dataset_sha256'], learner=ppo['learner'],
                anchor_data=old['anchor_data'], anchor_cells=old['anchor_cells'], anchor_weight=.01,
                anchor_seed=54401, tripwire=deepcopy(old['tripwire']), training_updates=50,
                control_steps_per_update=24, maximum_learning_control_steps=1200,
                forced_video_boundaries=[0, 800], video_seconds=8.,
                scope='One bounded matched persistent-contact PPO pilot; separate evaluation required')
    # Remove the old value-admission execution metadata from the training plan.
    for name in ('actor_updates', 'critic_updates', 'learning_updates', 'minibatch_seed', 'frozen_at',
                 'candidate_reward_enabled', 'reference_sha256', 'critic_initialization', 'reset_test'):
        plan.pop(name, None)
    assert plan['initial_std'] == plan['learner']['initial_std'] == .10

    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    for record in (ROOT / 'records/experiments').glob('duck_getup_persistent-contact_*'):
        assert '_s44_' not in record.name, 'Seed already used for persistent-contact PPO'
    training = {}
    for role in ARMS:
        training[role] = dict(plan, arm=role, persistent_penalty_enabled=role == 'candidate',
                              run_id=f'duck_getup_persistent-contact_{role}_s44_{stamp}',
                              matching_control=f'duck_getup_persistent-contact_control_s44_{stamp}')
    cfg, display, meta = prepare(training['control'])
    different = differences(asdict(cfg), asdict(prepare(training['candidate'])[0]))
    assert different == [dict(path='.env.rewards.late_head_leg_contact.params.enabled',
                             control='False', candidate='True')], different
    assert cfg.env.auto_reset and cfg.env.is_finite_horizon and cfg.env.episode_length_s == 8
    assert not display['native_live_window']
    model = Scene(cfg.env.scene, device='cpu').compile()
    fields = model_arrays(model)
    cpu_value_fields = OUT.parent / 'value/cpu-scene-fields.npz'
    with np.load(cpu_value_fields) as before:
        compare_arrays(fields, {name: before[name] for name in before.files})
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + name)
           for name in meta['joint_names'].split(',')]
    addresses = model.jnt_qposadr[ids].tolist()
    ranges = model.jnt_range[ids]
    np.testing.assert_allclose(ranges, plan['joint_limits'], atol=1e-7, rtol=0)
    with np.load(data) as saved:
        watch_ids = sorted(set(range(32)) & set(plan['split']['train_ids']))
        assert watch_ids and not set(watch_ids) & set(plan['split']['validation_ids'])
        reference = float(kinematic_stable(torch.from_numpy(saved['qpos'][349:, watch_ids]),
                         torch.from_numpy(saved['qvel'][349:, watch_ids]), addresses,
                         torch.tensor(ranges)).float().mean())
        pre = saved['critic_obs'][..., 74]
        post = np.concatenate((pre[1:], np.zeros_like(pre[:1])))
        trace = dict(contact=saved['contact'], eligible=saved['eligible'], ticks=saved['ticks'],
                     potential=saved['potential'], pre_age_input=pre, post_age_input=post)
    clock_checks = {}
    for role, enabled in [('control', False), ('candidate', True)]:
        cost = trace['potential'] if enabled else np.zeros_like(trace['potential'])
        clock_checks[role] = verify_clock(dict(trace, cost=cost, manager_rate=-2 * cost), enabled)
    assert clock_checks['control']['charged_samples'] == 0 < clock_checks['candidate']['charged_samples']
    for arm in training.values():
        arm.update(qpos_addresses=addresses, watch_reference_stable_fraction=reference,
                   watch_reference_indices=watch_ids, watch_stable_min=max(.5, reference - .25))

    seed = draft['proposed_evaluation_seed']
    assert seed == 1050 and seed != plan['seed']
    for folder in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT / folder).iterdir())
    template = deepcopy(archived['arms']['original'])
    codes = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [32, 32, 128, 128]))
    names = ('prone', 'supine', 'left-side', 'right-side')
    groups = {name: np.flatnonzero(codes == i).tolist() for i, name in enumerate(names)}
    template.update(stage=STAGE, seed=seed, split=dict(pose_codes=codes.tolist(), groups=groups),
                    video_worlds={name: worlds[0] for name, worlds in groups.items()},
                    video_title='PERSISTENT CONTACT', progress_label='PERSISTENT_CONTACT_EVAL',
                    handoff_enabled=False, neck_home_enabled=False)
    arms = {role: dict(template, arm=role, run_id=f'duck_getup_persistent-contact-eval_{role}_s{seed}_{stamp}')
            for role in EVAL_ARMS}
    ecfg, _, _ = parent.evaluation_prepare(arms['original'])
    compare_arrays(fields, model_arrays(Scene(ecfg.env.scene, device='cpu').compile()))
    processor = processor_check(arms['original'], ecfg)
    numeric = cpu_check(dict(roles={'getup': template, 'standing': dict(template, model=template['standing_model'],
                                 model_sha256=template['standing_model_sha256'])}))

    sources = {}
    for manifest in (archived['source_sha256'], plan['source_sha256'], ppo['source_sha256']):
        for name, digest in manifest.items():
            assert name not in sources or sources[name] == digest, name
            sources[name] = digest
    for name in ('scripts/training/getup_persistent_contact_train.py',
                 'scripts/validation/verify_getup_persistent_contact_train.py',
                 'scripts/validation/verify_getup_persistent_value_reciprocal.py',
                 'scripts/validation/getup_persistent_contact_pair.py'):
        sources[name] = sha(ROOT / name)
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, name
    library = dict(plan['library_source_sha256'])
    library[reward_manager.__file__] = sha(Path(reward_manager.__file__))
    for arm in [*training.values(), *arms.values()]:
        arm.update(source_sha256=sources, library_source_sha256=library)
        for base, key in ((WORK, 'training_source_sha256'), (Path('/'), 'library_source_sha256')):
            for name, digest in arm[key].items():
                assert sha(base / name) == digest, name

    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    for role in ARMS:
        (OUT / role).mkdir()
    np.savez_compressed(OUT / 'cpu-scene-fields.npz', **fields)
    refs = {str(path.relative_to(ROOT)): sha(path) for path in
            (value_path, value_protocol, cpu_value_fields, ppo_path, draft_path, parent_path,
             anchor_data, anchor_cells, anchor_preparation, OUT / 'cpu-scene-fields.npz')}
    m = dict(status='frozen-before-learning', stage=STAGE, stamp=stamp, training_arms=training, arms=arms,
             source_sha256=sources, reference_sha256=refs, training_seed=44, evaluation_seed=seed,
             maximum_learning_transitions=2457600, maximum_evaluation_transitions=528000,
             heldout_used=False, hardware_io=False, automatic_retry=False, automatic_extension=False,
             candidate_budget_rule='Use verified completed control updates; if unmatched or drift fails, close without candidate eval',
             selection_rule='Official or safe control passes all gates, or safe matched candidate passes all gates and improves final holds by2',
             stop_rule='Close on runtime failures, severe KL, sustained posture loss, no completed episode, or drift beyond .05RMS/.25max',
             scope='Bounded development study; complete S2-S6 scope remains open')
    write(PLAN, m)
    write(OUT / 'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), source_count=len(sources),
          training_config_difference=different, full_physical_fields_exact=True, processor=processor,
          model_import=numeric, selection=cpu_selection_checks(), clock_checks_on_saved_data=clock_checks,
          watch_reference=reference, watch_min=training['control']['watch_stable_min'],
          checkpoint_scale_exact=True, cuda_initialized=False, physical_steps=0, learning_updates=0))
    print('PERSISTENT_CONTACT_PAIR_FROZEN', sha(PLAN), len(sources), 'sources; max50 updates per arm', flush=True)


def checked(resolve_models=True):
    # Bind the frozen collector/checker to this study only in its dedicated
    # child process. Historical plans and sources stay byte-for-byte intact.
    parent.PLAN, parent.OUT = PLAN, OUT
    return parent.checked(resolve_models)


def project_state(**values):
    project = read(ROOT / 'project.json')
    project['staged_training_request'].update(values)
    write(ROOT / 'project.json', project)


def available_stage():
    stage = read(ROOT / 'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen']
    assert not stage['pause_requested'], 'Paused by user'


def verify_zero():
    zero = OUT / 'zero-step'
    match_start(zero / 'candidate', zero / 'control')
    for role in ARMS:
        startup = read(zero / role / 'startup.json')
        interface = read(zero / role / 'reward-interface.json')
        assert startup['status'] == 'passed' and startup['physical_steps'] == 0
        assert startup['actual_critic_width'] == 76 and startup['actor_width'] == 61
        assert interface['enabled'] is (role == 'candidate')
    write(zero / 'verification.json', dict(status='passed', physics_steps=0, actor_updates=0, critic_updates=0,
          matched_observations_state_expanded_scene_rng_models_and_adam=True, protocol_sha256=sha(PLAN)))
    print('PERSISTENT_CONTACT_PAIRED_GPU_START_PASSED 0physics', flush=True)


def execute():
    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    available_stage()
    checked(False)
    coord = OUT / 'coordinator.json'
    assert not coord.exists(), 'No retry or extension'
    status = dict(status='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write(coord, status)

    def step(flag, role):
        available_stage()
        status['phase'] = f'{flag[2:]}-{role}'
        write(coord, status)
        subprocess.run([sys.executable, '-B', str(Path(__file__)), flag, role], cwd=WORK, check=True)

    def train(role):
        project_state(learning_running=True, simulation_running=True)
        step('--train', role)
        project_state(learning_running=False, simulation_running=False)
        step('--verify-train', role)
        return read(OUT / role / 'training-verification.json')

    def evaluate(role):
        project_state(simulation_running=True)
        step('--evaluate', role)
        project_state(simulation_running=False)
        step('--verify-eval', role)
        return read(OUT / f'{role}-analysis.json')

    try:
        for role in ARMS:
            step('--zero', role)
        verify_zero()
        cp = train('control')
        training, proofs = {'control': cp}, {}
        choice = dict(action='close', reason='Control stopped before a full episode or made no actor update')
        if cp['actual_control_transitions'] >= 409600 and cp['actor_moved']:
            proofs = {'control': evaluate('control'), 'original': evaluate('original')}
            control = proofs['control']
            eligible_control = dict(control, all_motion_gates_passed=
                                    control['all_motion_gates_passed'] and not cp['drift_tripwire_triggered'])
            choice = decision(eligible_control, proofs['original'])
            write(OUT / 'selection.json', choice)
            if choice['action'] == 'candidate':
                ap = training['candidate'] = train('candidate')
                matched = cp['rollout_updates'] == ap['rollout_updates']
                if matched and ap['actor_moved'] and not ap['drift_tripwire_triggered']:
                    proofs['candidate'] = evaluate('candidate')
                admitted = 'candidate' in proofs and candidate_admitted(control, proofs['candidate'])
                choice = dict(action='select' if admitted else 'close', selected='candidate' if admitted else None,
                              candidate_admitted=admitted, actual_budgets_matched=matched,
                              required_successes=choice['required_successes'],
                              candidate_eval_ran='candidate' in proofs,
                              reason='Frozen functional, relative, equal-budget and drift gates')
        write(OUT / 'selection.json', choice)
        result = dict(status='passed-integrity', selection=choice,
                      timely_counts={role: p['overall']['timely_holds'] for role, p in proofs.items()},
                      final_hold_counts={role: p['end_hold']['overall']['timely_holds'] for role, p in proofs.items()},
                      actual_learning_transitions=sum(p['actual_control_transitions'] for p in training.values()),
                      actual_evaluation_transitions=len(proofs) * 176000, protocol_sha256=sha(PLAN),
                      heldout_used=False, hardware_io=False, automatic_extension=False,
                      scope='Bounded development comparison; reliability remains a separate gate')
        write(OUT / 'analysis.json', result)
        status.update(status='completed-review-required', phase='closed')
        print('PERSISTENT_CONTACT_PAIR_CLOSED', json.dumps(result), flush=True)
    except BaseException as error:
        status.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(coord, status)
        project_state(learning_running=False, simulation_running=False)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--freeze', action='store_true')
    for name in ('zero', 'train', 'verify-train'):
        modes.add_argument('--' + name, choices=ARMS)
    for name in ('evaluate', 'verify-eval'):
        modes.add_argument('--' + name, choices=EVAL_ARMS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    if not any(vars(args).values()):
        execute()
        return
    available_stage()
    coord = read(OUT / 'coordinator.json')
    assert coord['status'] == 'running' and coord['pid'] == os.getppid(), 'Use the single visible coordinator'
    plan = checked(False)
    if args.zero:
        zero_step(plan['training_arms'][args.zero], OUT)
    elif args.train:
        assert read(OUT / 'zero-step/verification.json')['status'] == 'passed'
        run(plan['training_arms'][args.train], PLAN, OUT)
    elif args.verify_train:
        from verify_getup_persistent_contact_train import verify
        verify(plan['training_arms'][args.verify_train], OUT)
    elif args.evaluate:
        parent.evaluate(args.evaluate)
    else:
        import verify_getup_teacher_anchor_eval as verifier
        verifier.PLAN, verifier.OUT = PLAN, OUT
        verifier.verify(args.verify_eval)


if __name__ == '__main__':
    main()
