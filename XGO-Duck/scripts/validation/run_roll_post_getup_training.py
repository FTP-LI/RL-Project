"""Qualify and run a paired PPO curriculum from physical roll-entry states."""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_roll_friction_paired_reset import arrays, capture
from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
import roll_friction_ppo as learner
from roll_friction_paired import collect
from roll_post_getup_reset import add_curriculum
from run_nominal_skill_composition import config as nominal_config
from run_roll_ppo_capacity_bounded import BoundedNativePPO, read_status
from run_roll_ppo_capacity_live import SynchronousAuthorization


KEY = 'roll_post_getup_training'
STAGE = 'nominal-roll-post-getup-start-training'
PROJECT = ROOT / 'project.json'
PLAN = ROOT / 'configs/experiments/roll_post_getup_start_training.json'
REPORT = ROOT / 'records/reports/nominal-skill-composition/post-getup-roll-curriculum'
ROLES = ('official-standing-reset', 'post-getup-start-curriculum')
BASE_PLAN = ROOT / 'configs/experiments/roll_friction_paired_training.json'
CAPACITY = ROOT / 'records/reports/roll-friction-ppo-capacity-live/phase-2-selection.json'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def user_allows(project):
    task, goal = project['staged_training_request'], project['goal_execution']
    return (goal['status'] == 'active' and not goal['pause_requested_by_user']
            and not task['pause_requested'] and not task['deployment_authorized_now'])


def bank_from_nominal(project):
    from mjlab.terrains.terrain_entity import TerrainEntity

    goal = project['goal_execution']
    previous = goal['nominal_roll_neck_descent_pair']
    audit = ROOT / previous['physical_comparison']
    archived = read(audit, previous['physical_comparison_sha256'])['proposed_training_start_distribution']
    assert archived['source_seed'] == 1256 and archived['excluded_from_archive'] == [1257, 2256, 2246]
    source = goal['nominal_skill_composition_contact_load']
    ledger_path = ROOT / source['collection']
    ledger = read(ledger_path, source['collection_sha256'])
    protocol_path = ledger_path.with_name('protocol.json')
    protocol = read(protocol_path, source['protocol_sha256'])
    assert protocol['environment']['seed'] == 1256
    archive_file = ROOT / archived['artifact']
    assert sha(archive_file) == archived['artifact_sha256']
    saved = arrays(archive_file)
    terrain = nominal_config(protocol).scene.terrain
    assert terrain.terrain_type == 'plane' and terrain.num_envs == protocol['composition']['worlds']
    all_origins = TerrainEntity._compute_env_origins_grid(SimpleNamespace(_device='cpu'),
        terrain.num_envs, terrain.env_spacing).numpy()
    reset_file = ledger_path.with_name('initial.npz')
    assert sha(reset_file) == ledger['initial_sha256']
    with np.load(reset_file, allow_pickle=False) as data:
        start_offset = np.abs(data['qpos'][:, :2] - all_origins[:, :2])
    world = saved['world']
    assert len(world) == archived['worlds'] == len(np.unique(world))
    local = saved.pop('qpos')
    local[:, :3] -= all_origins[world]
    # The reset's configured horizontal jitter is +/-0.50 m. Walking during
    # get-up can go farther; archived starts must stay inside the source tile.
    assert np.max(start_offset) <= .50 and np.max(np.abs(local[:, :2])) < .5 * terrain.env_spacing
    assert np.max(np.abs(np.linalg.norm(local[:, 3:7], axis=-1) - 1.)) < 2e-7
    assert np.all((local[:, 2] > .13) & (local[:, 2] < .15))

    control = archived['archived_control']
    boundary, = (row for row in ledger['boundaries'] if row['control'] == control)
    entry_path = ledger_path.with_name(boundary['file'])
    assert sha(entry_path) == boundary['sha256'] == archived['entry_target_sha256']
    from nominal_skill_composition_controller import ROLLING
    with np.load(entry_path, allow_pickle=False) as data:
        assert np.all(data['mode'][world] == ROLLING)
        observed = data['actor_input'][world]
        np.testing.assert_array_equal(observed[:, 34:], 0.)
        np.testing.assert_array_equal(data['previous_target'][world], saved['prior_applied_target'])
    joints = protocol['walking']['joint_names']
    assert ledger['qpos_joint_addresses'] == list(range(7, 21)) and len(joints) == 14
    positions = observed[:, 6:20] + np.asarray(protocol['policy_home']['official-forward-roll'])
    position_error = float(np.max(np.abs(positions - local[:, 7:21])))
    assert position_error < protocol['environment']['common_actor_noise']['joint_pos'] + 1e-6
    lo, hi = np.asarray(protocol['environment']['joint_limits']).T
    assert np.all(saved['prior_applied_target'] >= lo) and np.all(saved['prior_applied_target'] <= hi)
    bank = REPORT / 'local-roll-entry-states.npz'
    with bank.open('xb') as file:
        np.savez_compressed(file, local_qpos=local, **saved)
    proof = dict(source_seed=1256, archived_control=control, worlds=len(world), source_spacing_m=terrain.env_spacing,
        maximum_source_reset_xy_offset_m=float(start_offset.max()), maximum_local_xy_m=float(np.abs(local[:, :2]).max()),
        measured_height_m=[float(local[:, 2].min()), float(local[:, 2].max())],
        maximum_source_actor_joint_position_error_rad=position_error, source_actor_history_and_commands_zero=True,
        bank=rel(bank), bank_sha256=sha(bank), untouched_development_and_confirmation_seeds=archived['excluded_from_archive'])
    write(REPORT / 'entry-state-audit.json', proof)
    return proof, joints, (audit, archive_file, protocol_path, ledger_path, reset_file, entry_path)


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    assert user_allows(project) and task['current_stage'] == STAGE + '-preparation'
    assert not (task['learning_running'] or task['simulation_running']) and KEY not in goal
    assert not (PLAN.exists() or torch.cuda.is_initialized())
    assert not REPORT.exists() or not any(REPORT.iterdir())
    base, capacity = read(BASE_PLAN), read(CAPACITY)
    n = capacity['provisional_worlds']
    assert n == 4608 and capacity['eligible'][str(n)]
    REPORT.mkdir(exist_ok=True)
    proof, joints, evidence = bank_from_nominal(project)
    recipe = dict(base['recipe'], seed=1258)
    checks = dict(base['training_checks'], late_control_steps=[336, 686], clip_start_controls=[0])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    arms = {role: dict(run_id=f'duck-roll-{role}-e{n}-s1258-{stamp}', environments=n, updates=30,
        transitions=n * recipe['rollout_controls'] * 30, extra_checkpoints=[15, 24],
        friction_role='official-distribution') for role in ROLES}
    files = (BASE_PLAN, CAPACITY, REPORT / 'entry-state-audit.json', ROOT / proof['bank'], *evidence,
        Path(__file__).resolve(), ROOT / 'scripts/training/roll_post_getup_reset.py',
        ROOT / 'scripts/training/roll_friction_ppo.py', ROOT / 'scripts/training/roll_friction_paired.py',
        ROOT / 'scripts/training/native_ppo_observation.py', ROOT / 'scripts/training/native_ppo_viewer_bounded.py',
        Path(capture.__code__.co_filename).resolve(), Path(read_status.__code__.co_filename).resolve(),
        Path(SynchronousAuthorization.check.__code__.co_filename).resolve(),
        ROOT / 'scripts/training/run_roll_post_getup_training.sh',
        ROOT / recipe['parent_checkpoint'], ROOT / recipe['official_onnx'])
    plan = dict(scope='formal-paired-training', stage=STAGE, arms=arms, recipe=recipe, training_checks=checks,
        curriculum=dict(bank=proof['bank'], joint_names=joints, stride=4, source_seed=1256),
        development_seed=1257, confirmation_seeds=[2256, 2246], hardware_io=False,
        source_sha256={rel(path): sha(path) for path in files}, installed_sha256=base['installed_sha256'])
    assert plan['source_sha256'][recipe['parent_checkpoint']] == recipe['parent_sha256']
    assert plan['source_sha256'][recipe['official_onnx']] == recipe['official_sha256']
    write(PLAN, plan)
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), reports=rel(REPORT))
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='post-getup-training-frozen', current_evidence=rel(REPORT / 'entry-state-audit.json'))
    write(PROJECT, project)
    print('ROLL_POST_GETUP_TRAINING_FROZEN', sha(PLAN), flush=True)


def checked():
    project = read(PROJECT)
    task = project['staged_training_request']
    state = project['goal_execution'][KEY]
    assert user_allows(project) and task['current_stage'] == STAGE and task['current_protocol'] == rel(PLAN)
    plan = read(PLAN, state['protocol_sha256'])
    for path, expected in plan['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in plan['installed_sha256'].items():
        assert sha(Path(path)) == expected, path
    return project, task, state, plan


@contextmanager
def configured(plan, role):
    prepare = learner.prepare
    def with_course(spec, arm, metadata):
        cfg = prepare(spec, arm, metadata)
        if role == ROLES[1]:
            course = plan['curriculum']
            add_curriculum(cfg.env, course['bank'], course['stride'], course['joint_names'])
        return cfg
    with patch.object(learner, 'prepare', with_course):
        yield


def zero(role):
    _, task, state, plan = checked()
    assert state['status'] == 'zero-ready' and not task['learning_running'] and not task['simulation_running']
    folder = REPORT / role
    folder.mkdir(exist_ok=True)
    destination = folder / 'zero'
    destination.mkdir()
    setup = learner.initial_setup

    def check_reset(raw, wrapped, runner, parent, session, metadata, arm, record):
        setup(raw, wrapped, runner, parent, session, metadata, arm, record)
        robot = raw.scene['robot']
        data = raw.sim.data
        staged = robot.data.joint_pos_target.clone()
        if role == ROLES[1]:
            bank = raw.roll_entry_states
            ids = torch.where(bank.last_case >= 0)[0]
            cases = bank.last_case[ids]
            assert len(ids) == raw.num_envs // plan['curriculum']['stride']
            origin = raw.scene.env_origins[ids]
            expected = bank.qpos[cases].clone()
            expected[:, :3] += origin
            torch.testing.assert_close(data.qpos[ids], expected, rtol=0, atol=0)
            torch.testing.assert_close(data.qvel[ids], bank.qvel[cases], rtol=0, atol=0)
            torch.testing.assert_close(staged[ids[:, None], bank.joints], bank.prior_target[cases], rtol=0, atol=0)
            assert bool((raw._roulade_supported_accum[ids] == 0).all() and (~raw._roulade_head_latch[ids]).all())
            assert len(torch.unique(cases)) == len(bank.qpos)
            with (record / 'course-cases.npz').open('xb') as file:
                np.savez_compressed(file, environments=ids.cpu().numpy(), bank_rows=cases.cpu().numpy())
        obs = wrapped.get_observations()
        with torch.no_grad():
            action = runner.alg.actor(obs)
        # The native learner retains the three twist command channels. The
        # continuous-route archive sends zeros at skill entry; both learners
        # must see the same native commands and zero raw action history.
        assert raw.observation_manager.active_terms['actor'][-3:] == ['command', 'head_command', 'body_command']
        assert bool((obs['actor'][:, 34:48] == 0).all() and (obs['actor'][:, 51:] == 0).all())
        initial = data.ctrl.clone()
        raw.action_manager.process_action(action)
        raw.action_manager.apply_action()
        raw.scene.write_data_to_sim()
        assert bool(torch.isfinite(data.ctrl).all()) and raw.common_step_counter == raw._sim_step_counter == 0
        with (record / 'zero-motor.npz').open('xb') as file:
            np.savez_compressed(file, previous_target=staged.cpu().numpy(), native_twist=obs['actor'][:, 48:51].cpu().numpy(),
                pre_first_action=initial.cpu().numpy(), first_motor_input=data.ctrl.cpu().numpy())

    with configured(plan, role), patch.object(learner, 'initial_setup', check_reset):
        peak_mib = capture(plan, plan['arms'][role], destination)
    zero_proof = read(destination / 'zero-step.json')
    write(destination / 'receipt.json', dict(role=role, protocol_sha256=state['protocol_sha256'], peak_mib=peak_mib,
        startup_sha256=zero_proof['startup_inputs_sha256'], first_motor_sha256=sha(destination / 'zero-motor.npz'),
        physics_substeps=0, learning_updates=0))
    print('ROLL_POST_GETUP_ACTUAL_CUDA_ZERO', role, peak_mib, flush=True)


def match_zero():
    project, task, state, plan = checked()
    assert state['status'] == 'zero-ready'
    folders = [REPORT / role / 'zero' for role in ROLES]
    receipts = [read(folder / 'receipt.json') for folder in folders]
    assert all(r['protocol_sha256'] == state['protocol_sha256'] for r in receipts)
    paths = [folder / 'startup-inputs.npz' for folder in folders]
    a, b = (arrays(path) for path in paths)
    for path, evidence in zip(paths, receipts, strict=True):
        assert sha(path) == evidence['startup_sha256']
    n = plan['arms'][ROLES[1]]['environments']
    stride = plan['curriculum']['stride']
    changed = np.arange(n) % stride == 0
    for key in ('cpu_rng', 'gpu_rng', 'sliding_mu'):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for key in ('qpos', 'qvel', 'actor', 'policy'):
        np.testing.assert_array_equal(a[key][~changed[:len(a[key])]], b[key][~changed[:len(a[key])]], err_msg=key)
    motor_paths = [folder / 'zero-motor.npz' for folder in folders]
    for path, evidence in zip(motor_paths, receipts, strict=True):
        assert sha(path) == evidence['first_motor_sha256']
    motor_a, motor_b = (arrays(path) for path in motor_paths)
    np.testing.assert_array_equal(motor_a['native_twist'], motor_b['native_twist'])
    assert all(np.isfinite(motor['first_motor_input']).all() for motor in (motor_a, motor_b))
    assignments = arrays(folders[1] / 'course-cases.npz')
    np.testing.assert_array_equal(assignments['environments'], np.flatnonzero(changed))
    assert not np.array_equal(a['qpos'][changed], b['qpos'][changed])
    peak = max(r['peak_mib'] for r in receipts)
    output = REPORT / 'matched-reset.json'
    write(output, dict(status='matched-cuda-zero', protocol_sha256=state['protocol_sha256'], worlds_per_arm=n,
        archived_worlds=int(changed.sum()), unchanged_worlds=int((~changed).sum()),
        distinct_archived_states=len(np.unique(assignments['bank_rows'])), identical_foot_friction_and_rng=True,
        identical_native_twist=True, native_twist_nonzero_worlds=int(np.any(motor_a['native_twist'] != 0., axis=1).sum()),
        maximum_full_device_memory_mib=peak, startup_sha256_by_role=dict(zip(ROLES, (sha(p) for p in paths), strict=True)),
        physics_substeps=0, learning_updates=0))
    state.update(status='training-ready', matched_zero=rel(output), matched_zero_sha256=sha(output))
    task.update(current_protocol_execution='post-getup-training-cuda-zero-verified', current_evidence=rel(output))
    write(PROJECT, project)
    print('ROLL_POST_GETUP_MATCHED_CUDA_ZERO', sha(output), peak, flush=True)


def train(role):
    project, task, state, plan = checked()
    assert (state['status'] == ('training-ready' if role == ROLES[0] else 'candidate-ready')
            and not task['learning_running'] and not task['simulation_running'])
    proof = read(ROOT / state['matched_zero'], state['matched_zero_sha256'])
    arm = plan['arms'][role]
    folder = REPORT / role
    startup = folder / 'zero' / 'startup-inputs.npz'
    assert sha(startup) == proof['startup_sha256_by_role'][role]
    record = ROOT / 'records/experiments' / arm['run_id']
    record.mkdir()
    write(record / 'protocol-arm.json', dict(arm, protocol_sha256=state['protocol_sha256'],
        curriculum=plan['curriculum'] if role == ROLES[1] else None))

    def allowed():
        p = read_status(PROJECT, reader=read)
        s, g = p['staged_training_request'], p['goal_execution'][KEY]
        return (user_allows(p) and s['current_stage'] == STAGE and s['current_protocol'] == rel(PLAN)
                and g['status'] == 'training' and g['active_arm'] == role and g['protocol_sha256'] == sha(PLAN))

    state.update(status='training', active_arm=role)
    task.update(learning_running=True, simulation_running=True,
        current_protocol_execution='post-getup-paired-ppo-live', current_evidence=rel(folder))
    write(PROJECT, project)
    authorization = SynchronousAuthorization(allowed).start()
    observer = BoundedNativePPO(arm['run_id'], folder, authorization.check, environments=arm['environments'],
        planned_updates=arm['updates'], purpose='滚翻训练 · ' + ('原始站立重置' if role == ROLES[0] else '起身后启动状态课程'))
    ending = dict(status='stopped-review-required', role=role)
    try:
        observer.start()
        with configured(plan, role):
            evidence = collect(plan, arm, record, observer, startup)
        assert evidence['status'] == 'collected-review-required' and evidence['transitions'] == arm['transitions']
        ending = dict(status=evidence['status'], role=role, result=rel(record / 'online-result.json'),
            result_sha256=sha(record / 'online-result.json'))
    except BaseException as error:
        ending['reason'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            authorization.close()
            ending['authorization_monitor'] = authorization.summary()
            observer.close(ending)
        finally:
            write(folder / 'training-receipt.json', ending)
            project = read(PROJECT)
            task, state = project['staged_training_request'], project['goal_execution'][KEY]
            success = ending['status'] == 'collected-review-required'
            state.update(status=('candidate-ready' if role == ROLES[0] else 'trained-review-required') if success
                else 'stopped-review-required', active_arm=None)
            task.update(learning_running=False, simulation_running=False,
                current_protocol_execution='post-getup-paired-ppo-review', current_evidence=rel(folder / 'training-receipt.json'))
            write(PROJECT, project)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'zero', 'match-zero', 'train'))
    parser.add_argument('role', nargs='?', choices=ROLES)
    args = parser.parse_args()
    if args.action in ('zero', 'train'):
        if args.role is None:
            parser.error('Specify the arm')
        return {'zero': zero, 'train': train}[args.action](args.role)
    return {'freeze': freeze, 'match-zero': match_zero}[args.action]()


if __name__ == '__main__':
    main()
