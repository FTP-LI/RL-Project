"""Compare CUDA integration steps after the same official roll-entry snapshot."""

from argparse import ArgumentParser
from contextlib import ExitStack
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
import warp as wp

import evaluate_roll_post_getup_training as comparison


driver, recorder, verifier = comparison.driver, comparison.recorder, comparison.verifier
ROOT, PROJECT = comparison.ROOT, comparison.PROJECT
read, rel, sha, write = comparison.read, comparison.rel, comparison.sha, comparison.write
KEY = 'nominal_roll_timestep_closed_loop'
STAGE = 'nominal-roll-timestep-closed-loop'
PLAN = ROOT / 'configs/experiments/nominal_roll_timestep_closed_loop.json'
REPORT = ROOT / 'records/reports/nominal-skill-composition/closed-loop-timestep'
ROLES = ('reference-4ms-first', 'reference-4ms-repeat', 'solver-2ms-during-roll')
LABELS = dict(zip(('shared', *ROLES), ('共同释放与起身', '4ms参照一', '4ms参照二', '滚翻段2ms求解'), strict=True))
MOTOR_PERIOD = .004
CONTROL_PERIOD = .02
AUDIT_CONTROLS = 3
CLOCK_TOLERANCE = 2e-6


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    design = goal[KEY]
    assert driver.parent.allowed(project, STAGE + '-preparation') and not task['simulation_running']
    assert design['status'] == 'design-ready' and not PLAN.exists() and not REPORT.exists()
    clock = goal['nominal_roll_timestep_cuda_clock']
    bench = read(ROOT / clock['evidence'], clock['evidence_sha256'])
    assert clock['status'] == 'reviewed' and bench['solver_passes'] == 9
    assert all(mode['input_unchanged_within_every_interval'] for mode in bench['modes'])
    prior = goal[comparison.KEY]
    old = read(ROOT / prior['result'], prior['result_sha256'])
    plan = read(ROOT / prior['protocol'], prior['protocol_sha256'])
    previous_actor_comparison = plan.pop('roll_policy_comparison')
    c, seven = plan['composition'], plan['seven_command']
    start, stop = c['roll_start_control'], c['roll_deadline_control']
    assert start == design['fork_control'] == 600 and stop == design['fine_solver_end_control_exclusive'] == 1050
    seed, worlds = plan['environment']['seed'], c['worlds']
    assert seed == old['seed'] == design['seed'] == 1257 and worlds == old['worlds'] == design['matched_worlds'] == 896
    assert design['reserved_seeds'] == previous_actor_comparison['confirmation_seeds'] and seed not in design['reserved_seeds']
    assert c['physics_period_s'] == MOTOR_PERIOD and c['control_period_s'] == CONTROL_PERIOD
    assert plan['standing_pair']['common_snapshot_control'] == start
    assert plan['roll_route']['minimum_roll_completion'] == 812 and seven['minimum_overall'] == 852
    assert all(len(ids) == 128 and seven['minimum_by_command'][name] == 122 for name, ids in seven['groups'].items())
    saved = ROOT / plan['standing_pair']['archived_initial']
    reset = saved.with_name('reset.npz')
    assert sha(reset) == read(saved)['reset_sha256']
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-solver-2ms-vs-4ms-s{seed}-{now}'
    controls = start + len(ROLES) * (c['maximum_controls'] - start)
    plan['environment']['run_id'] = run
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', maximum_unique_controls=controls,
        maximum_control_transitions=controls * worlds, maximum_physics_substeps=controls * 5,
        solver_comparison=dict(roles=list(ROLES), audited_controls_per_arm=AUDIT_CONTROLS,
            fine_solver_interval=[start, stop], motor_update_s=MOTOR_PERIOD, control_update_s=CONTROL_PERIOD,
            solver_seconds_by_role={r: MOTOR_PERIOD / (2 if r == ROLES[-1] else 1) for r in ROLES},
            maximum_solver_launches=controls * 5 + (stop - start) * 5,
            reserved_seeds=design['reserved_seeds'], integrated_acceptance_unchanged=True, learning_updates=0))
    for record, field in ((prior, 'protocol'), (prior, 'result'), (clock, 'protocol'), (clock, 'evidence')):
        plan['references'][record[field]] = record[field + '_sha256']
    plan['references'].update({rel(p): sha(p) for p in (saved, reset, Path(__file__).resolve())})
    driver.parent.base.shared.assert_sources(plan)
    observer = driver.parent.preflight()
    REPORT.mkdir()
    write(PLAN, plan)
    preflight = REPORT / 'preflight.json'
    write(preflight, dict(protocol_sha256=sha(PLAN), observer=observer, physical_substeps=0,
        maximum_solver_launches=plan['solver_comparison']['maximum_solver_launches']))
    design.update(status='sample-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run, preflight=rel(preflight))
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='roll-timestep-closed-loop-frozen', current_evidence=rel(preflight))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])
    write(PROJECT, project)
    print('ROLL_TIMESTEP_CLOSED_LOOP_FROZEN', sha(PLAN), run, flush=True)


class SolverClock:
    def __init__(self, env, folder, plan):
        self.env, self.folder = env, folder
        self.sim = env.sim
        self.solve = env.sim.step
        self.environment_step = env.step
        self.graph = self.sim.step_graph
        self.start, self.stop = plan['solver_comparison']['fine_solver_interval']
        self.limit = plan['solver_comparison']['maximum_solver_launches']
        self.role, self.factor, self.executed = 'shared', 1, 0
        self.rows = {'shared': self.new_arm()}
        self.audit = None
        self.check_model_step(1)

    def new_arm(self):
        return dict(motor_ticks=0, solver_launches=0, start_time_s=float(self.sim.data.time[0].item()), controls=[])

    def check_model_step(self, factor):
        assert self.sim.use_cuda_graph and self.graph is self.sim.step_graph and self.graph is not None
        assert self.sim.cfg.mujoco.timestep == self.sim.mj_model.opt.timestep == self.env.physics_dt == MOTOR_PERIOD
        assert self.env.cfg.decimation == 5 and self.env.step_dt == CONTROL_PERIOD
        ts = self.sim.wp_model.opt.timestep
        np.testing.assert_array_equal(ts.numpy(), np.full(ts.shape, MOTOR_PERIOD / factor, dtype=np.float32))

    def select(self, factor):
        if factor != self.factor:
            with wp.ScopedDevice(self.sim.wp_device):
                self.sim.wp_model.opt.timestep.fill_(MOTOR_PERIOD / factor)
            self.factor = factor
        self.check_model_step(factor)

    def leave(self):
        self.rows[self.role]['end_time_s'] = float(self.sim.data.time[0].item())
        self.select(1)

    def enter(self, role):
        self.role = role
        self.rows[role] = self.new_arm()
        self.select(2 if role == ROLES[-1] else 1)

    def step(self):
        arm = self.rows[self.role]
        audit = self.audit
        if audit is not None:
            begin = self.sim.data.time.clone()
            motor = self.sim.data.ctrl.clone()
            times = []
        assert self.executed + self.factor <= self.limit
        for part in range(1, self.factor + 1):
            self.solve()
            self.executed += 1
            arm['solver_launches'] += 1
            if audit is not None:
                elapsed = self.sim.data.time.clone() - begin
                torch.testing.assert_close(elapsed, torch.full_like(elapsed, part * MOTOR_PERIOD / self.factor),
                    atol=CLOCK_TOLERANCE, rtol=0)
                torch.testing.assert_close(self.sim.data.ctrl.clone(), motor, atol=0, rtol=0)
                times.append(float(elapsed[0].item()))
                audit['events'].append('solver')
        arm['motor_ticks'] += 1
        if audit is not None:
            audit['elapsed_solver_s_world_0'].append(times)

    def control_step(self, action):
        k = self.env.common_step_counter
        if self.role == ROLES[-1] and k == self.stop:
            # Restore the native integrator before subsequent standing and commands.
            self.select(1)
        if self.role == 'shared' or not self.start <= k < self.start + AUDIT_CONTROLS:
            return self.environment_step(action)
        env = self.env
        actuator, = env.scene['robot'].actuators
        assert actuator.cfg.delay_min_lag == actuator.cfg.delay_max_lag == 1
        entry = dict(control=k, events=[], elapsed_solver_s_world_0=[])
        begin = self.sim.data.time.clone()
        self.audit = entry
        calls = (
            (env.action_manager, 'process_action', 'action'),
            (env.action_manager, 'apply_action', 'apply'),
            (env.scene, 'write_data_to_sim', 'write'),
            (actuator, 'apply_delay', 'delay'),
            (actuator, 'compute', 'motor'),
            (env.scene, 'update', 'scene'),
            (env.sim, 'forward', 'forward'),
            (env.sim, 'sense', 'sense'),
            (env.observation_manager, 'compute', 'history'),
        )

        def record(method, label):
            def call(*args, **kwargs):
                entry['events'].append(label)
                if label == 'history':
                    assert kwargs['update_history'] is True
                if label == 'scene':
                    assert kwargs['dt'] == MOTOR_PERIOD
                return method(*args, **kwargs)
            return call

        try:
            with ExitStack() as stack:
                for obj, method, label in calls:
                    stack.enter_context(patch.object(obj, method, record(getattr(obj, method), label)))
                result = self.environment_step(action)
        finally:
            self.audit = None
        elapsed = self.sim.data.time.clone() - begin
        torch.testing.assert_close(elapsed, torch.full_like(elapsed, CONTROL_PERIOD), atol=5 * CLOCK_TOLERANCE, rtol=0)
        entry['elapsed_control_s_world_0'] = float(elapsed[0].item())
        check_control(entry, self.factor)
        assert env.common_step_counter == k + 1 and env._sim_step_counter == 5 * (k + 1)
        arm = self.rows[self.role]
        arm['controls'].append(entry)
        if len(arm['controls']) == AUDIT_CONTROLS:
            path = self.folder / self.role / 'closed-loop-clock.json'
            write(path, dict(role=self.role, controls=arm['controls'], worlds=self.env.num_envs,
                checked_every_world_clock=True, unchanged_motor_input_between_solver_launches=True,
                motor_delay_ticks=1, host_simulation_step_s=MOTOR_PERIOD, actor_observation_step_s=CONTROL_PERIOD))
            arm['audit_sha256'] = sha(path)
            print('ROLL_TIMESTEP_CLOCK_VERIFIED', self.role, k + 1, arm['audit_sha256'], flush=True)
        return result


def check_control(entry, factor):
    tick = ['apply', 'write', 'delay', 'motor', *(['solver'] * factor), 'scene']
    assert entry['events'] == ['action', *(tick * 5), 'forward', 'sense', 'history'], entry['events']
    offsets = [part * MOTOR_PERIOD / factor for part in range(1, factor + 1)]
    np.testing.assert_allclose(entry['elapsed_solver_s_world_0'], [offsets] * 5, atol=CLOCK_TOLERANCE, rtol=0)
    assert abs(entry['elapsed_control_s_world_0'] - CONTROL_PERIOD) <= 5 * CLOCK_TOLERANCE


class CollectorLedger(dict):
    def __missing__(self, key):
        # The frozen collector uses an old spelling only in its final assertion.
        if key == 'physical_substeps':
            return self['physics_substeps']
        raise KeyError(key)


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = SolverClock(env, folder, plan)
    base_fork = recorder.StandingFork
    roles = iter(ROLES)
    publish = observer.publish
    counters = CollectorLedger(ledger)

    class TimestepFork(base_fork):
        def restore(self):
            clock.leave()
            state = super().restore()
            assert state['physical']['counters'] == [clock.start, 5 * clock.start]
            clock.enter(next(roles))
            return state

    def publish_solver(path, state):
        publish(path, dict(state, purpose=f'滚翻闭环积分对照｜{LABELS[state["phase"]]}',
            solver_launches=clock.executed))

    try:
        with (patch.multiple(recorder, ROLES=ROLES, LABELS=LABELS, CANDIDATE=None, StandingFork=TimestepFork),
              patch.object(env, 'step', clock.control_step), patch.object(env.sim, 'step', clock.step),
              patch.object(observer, 'publish', publish_solver)):
            recorder.collect(env, plan, actors, folder, observer, watch, counters)
    finally:
        ledger.update(counters, solver_launches=clock.executed, solver_roles=clock.rows)
        clock.leave()


def score():
    project, plan = driver.checked('score-pending')
    state = project['goal_execution'][KEY]
    folder = ROOT / plan['record']
    collection = folder / 'collection.json'
    ledger = read(collection, state['collection_sha256'])
    assert ledger['status'] == 'collected' and ledger['resets'] == 1 and ledger['actor_weights_unchanged']
    assert not (ledger['learning_updates'] or ledger['hardware_io'] or ledger['reserved_seed_used'])
    assert sha(folder / 'protocol.json') == sha(PLAN)
    assert sha(folder / 'zero/proof.json') == ledger['process_local_zero_sha256']
    start, stop = plan['solver_comparison']['fine_solver_interval']
    assert ledger['prefix']['controls'] == start and tuple(ledger['branches']) == ROLES
    worlds = plan['composition']['worlds']
    fixed = ledger['solver_roles']
    assert fixed['shared']['solver_launches'] == fixed['shared']['motor_ticks'] == start * 5
    assert ledger['controls'] == start + sum(x['controls'] - start for x in ledger['branches'].values())
    assert ledger['transitions'] == worlds * ledger['controls'] <= plan['maximum_control_transitions']
    assert ledger['physics_substeps'] == sum(x['motor_ticks'] for x in fixed.values()) == 5 * ledger['controls']
    assert ledger['solver_launches'] == sum(x['solver_launches'] for x in fixed.values()) <= plan['solver_comparison']['maximum_solver_launches']
    fork = read(folder / 'fork.json', ledger['fork_sha256'])
    initial_path, poses_path = folder / 'initial.npz', folder / 'release-poses.npz'
    assert sha(initial_path) == ledger['initial_sha256'] and sha(poses_path) == ledger['release_sha256']
    with np.load(initial_path, allow_pickle=False) as saved, np.load(poses_path, allow_pickle=False) as release:
        initial, poses = dict(saved), release['poses']
    prefix = comparison.read_trace(folder / 'shared', ledger['prefix']['chunks'], 1, start, comparison.EVIDENCE_FIELDS, sha)
    common = dict(ledger['prefix'], qpos_joint_addresses=ledger['qpos_joint_addresses'], camera_command=ledger['camera_command'])
    outcomes, population, first = {}, {}, {}
    with patch.object(verifier, 'standing_checks', comparison.branch_first):
        for role in ROLES:
            row = ledger['branches'][role]
            factor = 2 if role == ROLES[-1] else 1
            assert fixed[role]['motor_ticks'] == 5 * (row['controls'] - start)
            extra = 5 * (min(stop, row['controls']) - start) * (factor - 1)
            assert fixed[role]['solver_launches'] == fixed[role]['motor_ticks'] + extra
            assert fixed[role]['start_time_s'] == fixed['shared']['end_time_s']
            clock = read(folder / role / 'closed-loop-clock.json', fixed[role]['audit_sha256'])
            assert clock['role'] == role and clock['worlds'] == worlds and clock['motor_delay_ticks'] == 1
            assert clock['checked_every_world_clock'] and clock['unchanged_motor_input_between_solver_launches']
            assert clock['controls'] == fixed[role]['controls']
            assert [x['control'] for x in clock['controls']] == list(range(start, start + AUDIT_CONTROLS))
            for item in clock['controls']:
                check_control(item, factor)
            outcomes[role], population[role], first[role] = verifier.verify_branch(
                plan, folder / role, row, prefix, common, initial, fork)
            result = outcomes[role]
            result['candidate'] = role == ROLES[-1]
            result['roll_policy_admission'] = result.pop('standing_policy')
            result['roll_by_actual_release_pose'] = {
                pose: dict(roll_complete=int(population[role]['roll'][poses == pose].sum()), worlds=int((poses == pose).sum()))
                for pose in driver.parent.POSES}
    for role in ROLES[1:]:
        for field in ('observed', 'actor_input', 'previous_target', 'mode', 'raw', 'actual_target'):
            np.testing.assert_array_equal(first[role][field], first[ROLES[0]][field], err_msg=field)
        np.testing.assert_array_equal(first[role]['motor'][0], first[ROLES[0]]['motor'][0], err_msg='first motor input')
    comparisons = {}
    for a, b in ((ROLES[1], ROLES[0]), (ROLES[-1], ROLES[0]), (ROLES[-1], ROLES[1])):
        summary = verifier.compare(population[a], population[b], plan['seven_command']['groups'])
        summary['roll_net_by_actual_release_pose'] = {
            pose: outcomes[a]['roll_by_actual_release_pose'][pose]['roll_complete']
                  - outcomes[b]['roll_by_actual_release_pose'][pose]['roll_complete']
            for pose in driver.parent.POSES}
        comparisons[f'{a}-vs-{b}'] = summary
    output = REPORT / 'analysis.json'
    assert not output.exists()
    write(output, dict(protocol_sha256=sha(PLAN), collection_sha256=sha(collection), seed=plan['environment']['seed'],
        worlds=worlds, results=outcomes, comparisons=comparisons, verified_snapshot=fork['physical'],
        first_policy_inputs_output_and_motor_exact=True, verified_control_clocks=fixed, learning_updates=0,
        hardware_io=False, reserved_seed_used=False, controls=ledger['controls'], transitions=ledger['transitions'],
        solver_launches=ledger['solver_launches'], integrator_scene_steps=ledger['solver_launches'] * worlds))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output),
        candidate_passed_original_route=outcomes[ROLES[-1]]['passed_development_gate'])
    task = project['staged_training_request']
    task.update(current_protocol_execution='roll-timestep-closed-loop-reviewed', current_evidence=rel(output))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])
    write(PROJECT, project)
    print('ROLL_TIMESTEP_CLOSED_LOOP_SCORED', sha(output), {r: x['roll_complete'] for r, x in outcomes.items()}, flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'sample', 'score'))
    action = cli.parse_args().action
    with ((Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(),
          patch.multiple(driver, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT, ROLES=ROLES, collect=collect),
          patch.object(driver.parent, 'PLAN', PLAN), patch.object(comparison.feedback, 'components', driver.isolated_components)):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze': freeze, 'sample': driver.sample, 'score': score}[action]()


if __name__ == '__main__':
    main()
