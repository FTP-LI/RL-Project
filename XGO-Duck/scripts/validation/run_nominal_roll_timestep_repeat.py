"""Repeat the fixed 2 ms route from one actual CUDA release snapshot."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_roll_closed_loop_timestep as original
from review_nominal_roll_closed_loop_timestep import throughput
from roll_stand_walk_sequence import ROLL
from standup_metrics import projected_gravity


comparison, driver, recorder, verifier = original.comparison, original.driver, original.recorder, original.verifier
parent = driver.parent
ROOT, PROJECT = original.ROOT, original.PROJECT
read, rel, sha, write = original.read, original.rel, original.sha, original.write
KEY, STAGE = 'nominal_roll_timestep_repeat', 'nominal-roll-timestep-repeat'
PLAN = ROOT / 'configs/experiments' / (KEY + '.json')
DESIGN = PLAN.with_name(PLAN.stem + '_design.json')
REPORT = original.REPORT.with_name('closed-loop-timestep-repeat')
ROLES = ('fixed-policy-first', 'fixed-policy-repeat')
LABELS = dict(zip(('shared', *ROLES), ('共同跌倒与释放', '固定策略第一遍', '固定策略第二遍'), strict=True))
ENTRY = Path(__file__).resolve()


def freeze():
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    design = read(DESIGN)
    assert design['stage']==STAGE
    assert KEY not in goal and parent.allowed(project, design['preparation_from']) and not task['simulation_running']
    readiness, prior = (goal[key] for key in ('nominal_timestep_forward_readiness', original.KEY))
    assert readiness['status'] == prior['status'] == 'reviewed'
    source = ROOT / prior['protocol']
    old_score = ROOT / prior['result']
    old_record = ROOT / prior['collection']
    handoff = ROOT / readiness['evidence']
    plan = read(source, prior['protocol_sha256'])
    old = read(old_score, prior['result_sha256'])
    assert sha(old_record) == old['collection_sha256'] == prior['collection_sha256']
    reviewed = read(handoff, readiness['evidence_sha256'])
    assert reviewed['prior_forward_entrants'] == 120 and reviewed['prior_forward_minimum'] == 122
    d, seven = plan['composition'], plan['seven_command']
    release, begin, stop, end = (d[k] for k in ('release_control','roll_start_control','roll_deadline_control','maximum_controls'))
    assert release == design['shared_release_control'] == plan['dual_route']['release_counter']
    assert [begin,stop] == design['fine_solver_interval'] == plan['solver_comparison']['fine_solver_interval']
    assert end == design['maximum_branch_control'] and d['command_entry_control'] == design['command_entry_control']
    assert d['control_period_s'] == design['policy_period_seconds'] == original.CONTROL_PERIOD
    assert d['physics_period_s'] == design['motor_period_seconds'] == original.MOTOR_PERIOD
    assert design['physics_solver_seconds_during_roll'] == original.MOTOR_PERIOD / 2
    seed, worlds = plan['environment']['seed'], d['worlds']
    assert seed == old['seed'] == design['development_seed'] and worlds == old['worlds'] == design['worlds']
    assert design['reserved_seeds'] == plan['solver_comparison']['reserved_seeds'] and seed not in design['reserved_seeds']
    assert tuple(design['roles']) == ROLES and not (design['hardware_io'] or design['learning_updates'])
    assert design['minimum_roll_completion'] == plan['roll_route']['minimum_roll_completion']
    assert design['minimum_total_commands'] == seven['minimum_overall']
    assert all(len(ids)==128 and seven['minimum_by_command'][name]==design['minimum_each_command']
               for name, ids in seven['groups'].items())
    controls = release + len(ROLES)*(end-release)
    motor = 5*controls
    solver = motor + 5*len(ROLES)*(stop-begin)
    assert controls == design['maximum_batched_controls'] and motor == design['maximum_motor_intervals']
    assert solver == design['maximum_solver_launches'] and controls*worlds == design['maximum_control_transitions']
    zero = ROOT / plan['standing_pair']['archived_initial']
    reset = zero.with_name('reset.npz')
    assert sha(reset) == read(zero)['reset_sha256']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-fixed-policy-2ms-repeat-s{seed}-{stamp}'
    plan['environment']['run_id'] = run
    plan['standing_pair']['common_snapshot_control'] = release
    plan['solver_comparison'].update(roles=list(ROLES), maximum_solver_launches=solver,
                                     solver_seconds_by_role={r:original.MOTOR_PERIOD/2 for r in ROLES})
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=controls,
                maximum_control_transitions=controls*worlds,maximum_physics_substeps=motor,repeat_design=design)
    sources = (ENTRY, DESIGN, source, old_score, old_record, handoff, zero, reset,
               *(Path(module.__file__).resolve() for module in (original, comparison, driver, recorder, verifier)))
    plan['references'].update({rel(path):sha(path) for path in sources})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    preflight = REPORT/'preflight.json'
    write(preflight,dict(status='prepared',protocol_sha256=sha(PLAN),observer=observer,
        maximum_batched_controls=controls,maximum_solver_launches=solver,
        physical_substeps=0,learning_updates=0,hardware_io=False))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(preflight),
                     run_id=run,hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(preflight))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Compare both arms under the original roll, entry and seven-command functional criteria.')
    write(PROJECT,project)
    print('ROLL_TIMESTEP_REPEAT_FROZEN',sha(PLAN),run,flush=True)


class RepeatClock(original.SolverClock):
    def enter(self, role):
        # Release is before rolling, so every arm begins on the 4 ms clock.
        self.role = role
        self.rows[role] = self.new_arm()
        self.check_model_step(1)

    def control_step(self, action):
        k = self.env.common_step_counter
        if self.role != 'shared':
            if k == self.start:
                self.select(2)
            elif k == self.stop:
                self.select(1)
        return super().control_step(action)


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = RepeatClock(env,folder,plan)
    release = plan['standing_pair']['common_snapshot_control']
    base_fork = recorder.StandingFork
    roles = iter(ROLES)
    publish = observer.publish
    counters = original.CollectorLedger(ledger)

    class ReleaseFork(base_fork):
        def __init__(self, plant, controller, actor, memory):
            # Initialize the pose latches from the actual release observation.
            # No sampled action or physics step is applied before the fork.
            assert plant.common_step_counter==release and actor is actors['getup']
            assert actor.calls==0 and actor.right is None and actor.parent.supine_at_release is None
            actor(parent.base.shared.td('actor',plant.obs_buf['actor']))
            actor.calls = 0
            super().__init__(plant,controller,actor,memory)

        def restore(self):
            clock.leave()
            state = super().restore()
            assert state['physical']['counters']==[release,5*release]
            clock.enter(next(roles))
            return state

    def publish_repeat(path, state):
        publish(path,dict(state,purpose=f'2ms连续滚翻重复评估｜{LABELS[state["phase"]]}',solver_launches=clock.executed))

    try:
        with (
            patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=ReleaseFork),
            patch.object(env,'step',clock.control_step),
            patch.object(env.sim,'step',clock.step),
            patch.object(observer,'publish',publish_repeat),
        ):
            recorder.collect(env,plan,actors,folder,observer,watch,counters)
    finally:
        ledger.update(counters,solver_launches=clock.executed,solver_roles=clock.rows)
        clock.leave()


def score():
    project, plan = driver.checked('score-pending')
    goal, task = project['goal_execution'], project['staged_training_request']
    node, folder = goal[KEY], ROOT/plan['record']
    path = folder/'collection.json'
    raw = read(path,node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not (raw['learning_updates'] or raw['hardware_io'] or raw['reserved_seed_used'])
    assert sha(folder/'protocol.json')==sha(PLAN)
    assert sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    release = plan['standing_pair']['common_snapshot_control']
    begin, stop = plan['solver_comparison']['fine_solver_interval']
    assert raw['prefix']['controls']==release and tuple(raw['branches'])==ROLES
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[release,5*release]
    assert len(fork['fields'])==fork['physical']['fields']
    assert raw['controls']==release+sum(x['controls']-release for x in raw['branches'].values())
    worlds, clocks = plan['composition']['worlds'], raw['solver_roles']
    assert raw['transitions']==worlds*raw['controls']<=plan['maximum_control_transitions']
    assert raw['physics_substeps']==sum(x['motor_ticks'] for x in clocks.values())==5*raw['controls']
    assert raw['solver_launches']==sum(x['solver_launches'] for x in clocks.values())<=plan['solver_comparison']['maximum_solver_launches']
    assert clocks['shared']['solver_launches']==clocks['shared']['motor_ticks']==5*release
    initial_path, poses_path = folder/'initial.npz', folder/'release-poses.npz'
    assert sha(initial_path)==raw['initial_sha256'] and sha(poses_path)==raw['release_sha256']
    with np.load(initial_path,allow_pickle=False) as saved, np.load(poses_path,allow_pickle=False) as actual:
        initial, poses = dict(saved), actual['poses']
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,release,comparison.EVIDENCE_FIELDS,sha)
    outcomes, populations, first, rotations = {}, {}, {}, {}
    markers = (release,124,141,begin,stop)

    def read_branch(plan, path, record, trajectory):
        role = path.name
        recordings = trajectory['quaternion'][np.asarray(markers)-1].astype(np.float64)
        projected_gravity(recordings)
        rotations[role] = recordings
        entered = trajectory['roll_phase'][begin-1]==ROLL
        assert not (entered & (trajectory['mode'][begin-1]!=verifier.STAND)).any()
        marker, = (item for item in record['boundaries'] if item['control']==release)
        boundary = path/marker['file']
        assert sha(boundary)==marker['sha256']
        with np.load(boundary,allow_pickle=False) as saved:
            first[role] = dict(saved)
        result, _ = comparison.branch_first(plan,path,record,trajectory)
        assert result['worlds_entered']==int(entered.sum())
        result['roll_entry_worlds'] = np.flatnonzero(entered).tolist()
        return result, first[role]

    offset = release
    view = read(folder/'state.json')
    assert view['run_id']==raw['run_id'] and view['controls']==raw['controls']
    with patch.object(verifier,'standing_checks',read_branch):
        for role in ROLES:
            branch, clock = raw['branches'][role], clocks[role]
            assert clock['motor_ticks']==5*(branch['controls']-release)
            assert clock['solver_launches']==clock['motor_ticks']+5*(min(stop,branch['controls'])-begin)
            assert clock['start_time_s']==clocks['shared']['end_time_s']
            measured = read(folder/role/'closed-loop-clock.json',clock['audit_sha256'])
            assert measured['role']==role and measured['worlds']==worlds and measured['motor_delay_ticks']==1
            assert measured['checked_every_world_clock'] and measured['unchanged_motor_input_between_solver_launches']
            assert measured['controls']==clock['controls']
            assert [x['control'] for x in measured['controls']]==list(range(begin,begin+original.AUDIT_CONTROLS))
            for row in measured['controls']:
                original.check_control(row,2)
            outcomes[role], populations[role], _ = verifier.verify_branch(
                plan,folder/role,branch,prefix,common,initial,fork)
            outcomes[role].update(solver_step_s=original.MOTOR_PERIOD/2,
                                  sampled_throughput=throughput(view['updates'],worlds,offset+begin-release,offset+stop-release))
            entry = outcomes[role].pop('standing_policy')
            populations[role]['roll_entry'] = np.isin(np.arange(worlds),entry.pop('roll_entry_worlds'))
            outcomes[role].update(first_roll_action=entry,roll_entry=int(populations[role]['roll_entry'].sum()))
            offset += branch['controls']-release
    assert offset==raw['controls']
    a, b = ROLES
    for field in ('observed','actor_input','previous_target','mode','raw','actual_target'):
        np.testing.assert_array_equal(first[a][field],first[b][field],err_msg=field)
    np.testing.assert_array_equal(first[a]['motor'][0],first[b]['motor'][0],err_msg='first delayed motor input')
    x,y = rotations[a],rotations[b]
    cosine = abs(np.sum(x*y,axis=-1))/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
    distance = np.rad2deg(2*np.arccos(np.clip(cosine,0.,1.)))
    groups = plan['seven_command']['groups']
    paired = verifier.compare(populations[b],populations[a],groups)
    assigned = np.asarray(groups['forward'])
    for role in ROLES:
        bank = populations[role]
        outcomes[role].update(forward_roll_entry=int(bank['roll_entry'][assigned].sum()),
                              roll_by_actual_release_pose={
                                  pose:dict(worlds=int((poses==pose).sum()),roll_complete=int(bank['roll'][poses==pose].sum()))
                                  for pose in parent.POSES})
    differs = {}
    if populations[a]['command_admission'] is not None and populations[b]['command_admission'] is not None:
        for failed, passed in ((a,b),(b,a)):
            x,y = populations[failed],populations[passed]
            differing = (~x['command_admission'] & y['command_admission'])[assigned]
            differs[failed] = dict(
                before_roll=assigned[differing & ~x['roll_entry'][assigned]].tolist(),
                during_roll=assigned[differing & x['roll_entry'][assigned] & ~x['roll'][assigned]].tolist(),
                after_roll=assigned[differing & x['roll'][assigned]].tolist(),
            )
    qualified = all(row['passed_development_gate'] for row in outcomes.values())
    result = REPORT/'analysis.json'
    assert not result.exists()
    write(result,dict(status='verified-fixed-policy-repeats',protocol_sha256=sha(PLAN),collection_sha256=sha(path),
        seed=plan['environment']['seed'],worlds=worlds,shared_release_control=release,verified_snapshot=fork['physical'],
        controls=raw['controls'],transitions=raw['transitions'],motor_intervals=raw['physics_substeps'],
        solver_launches=raw['solver_launches'],peak_gpu_mib=max(x['used_mib'] for x in raw['gpu_samples']),
        first_policy_inputs_output_and_motor_exact=True,same_policy_weights=True,
        paired_root_rotation_deg=[dict(control=k,median=float(np.median(d)),p95=float(np.quantile(d,.95)))
                                 for k,d in zip(markers,distance,strict=True)],
        results=outcomes,paired_repeat_minus_first=paired,forward_disagreement_stage_by_failed_arm=differs,
        advance_to_independent_confirmation=qualified,reserved_seed_used=False,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',result=rel(result),result_sha256=sha(result),qualified=qualified)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(result))
    project.update(info)
    task.update(info,next_gate='Use the paired stage disagreements to diagnose the remaining original route gate.'
                if not qualified else 'Freeze the original nominal confirmation route using reserved seed 2256.')
    write(PROJECT,project)
    print('ROLL_TIMESTEP_REPEAT_SCORED',sha(result),
          {r:(x['roll_complete'],x['command_admission']) for r,x in outcomes.items()},qualified,flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with (
        (Path.home()/'XGO-Duck/training.lock').open('a') as lock,
        torch.inference_mode(),
        patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
        patch.object(parent,'PLAN',PLAN),
        patch.object(comparison.feedback,'components',driver.isolated_components),
    ):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__=='__main__':
    main()
