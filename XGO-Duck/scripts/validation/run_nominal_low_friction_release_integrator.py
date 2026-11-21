"""Compare the low-friction release integrator with a matched CUDA state."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import fcntl
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_low_friction_roll_window as window
from review_nominal_low_friction_failure_recurrence import KEY as RECURRENCE


base, driver, parent = window.base, window.driver, window.parent
original, recorder, comparison, verifier = window.original, window.recorder, window.comparison, window.verifier
ROOT, PROJECT, read, rel, sha, write = window.ROOT, window.PROJECT, window.read, window.rel, window.sha, window.write
KEY = 'nominal_low_friction_release_integrator'
STAGE = 'nominal-low-friction-release-integrator'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = window.REPORT.with_name('low-friction-release-integrator')
ROLES = ('reference-4ms-first','reference-4ms-repeat','release-2ms-candidate')
LABELS = dict(zip(('shared',*ROLES),('共同跌倒与释放','4ms起身参照一','4ms起身参照二','首1.2秒2ms起身候选'),strict=True))
MOTOR_TICKS = window.MOTOR_TICKS


def schedule(plan, role):
    comparison_plan = plan['solver_comparison']
    start, stop = comparison_plan['fine_solver_interval']
    roll, switch, end = comparison_plan['roll_clock_controls']
    changes = {roll:4,switch:2,end:1}
    if role == ROLES[-1]:
        changes = {start:2,stop:1,**changes}
    return changes


def expected_launches(controls, plan, role):
    start, stop = plan['solver_comparison']['fine_solver_interval']
    roll, switch, end = plan['solver_comparison']['roll_clock_controls']
    early = max(0,min(controls,stop)-start) if role==ROLES[-1] else 0
    rolling = max(0,min(controls,end)-roll)
    first_roll = max(0,min(controls,switch)-roll)
    return MOTOR_TICKS*(controls-start+early+rolling+2*first_roll)


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'],project['goal_execution']
    assert parent.allowed(project,'nominal-low-friction-failure-recurrence') and not task['simulation_running']
    assert KEY not in goal and not REPORT.exists() and not PLAN.exists() and not torch.cuda.is_initialized()
    source, recurrence = goal[window.KEY],goal[RECURRENCE]
    assert source['status']==recurrence['status']=='reviewed' and source['preferred'] is None
    evidence = read(ROOT/recurrence['result'],recurrence['result_sha256'])
    assert evidence['status']=='verified-same-scene-bank-readback' and not evidence['reserved_seed_used']
    assert len(evidence['acquisitions'])==7
    left = evidence['left_entries']
    assert [left[k]['by_stage']['no-roll-entry'] for k in ('two_ms','first_window_one_ms')]==[6,6]
    plan = read(ROOT/source['protocol'],source['protocol_sha256'])
    source_analysis = read(ROOT/source['result'],source['result_sha256'])
    assert [source_analysis['results'][r]['roll_complete'] for r in window.ROLES]==[853,862,870]
    d, seven = plan['composition'],plan['seven_command']
    start,stop = d['release_control'],d['release_control']+60
    roll,switch = plan['solver_comparison']['candidate_early_solver_interval']
    end,limit = d['roll_deadline_control'],d['maximum_controls']
    assert (start,stop,roll,switch,end,limit)==(100,160,600,660,1050,2125)
    assert (d['physics_period_s'],d['control_period_s'],d['foot_friction'])==(.004,.02,.8)
    seed,worlds = plan['environment']['seed'],d['worlds']
    assert (seed,worlds,d['existing_low_friction_reserved_seed'])==(1257,896,2246)
    assert (plan['roll_route']['minimum_roll_completion'],seven['minimum_overall'])==(812,852)
    assert len(seven['groups'])==7 and all(len(ids)==128 and seven['minimum_by_command'][k]==122
                                           for k,ids in seven['groups'].items())
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-release-clock-friction080-s{seed}-{stamp}'
    plan['environment']['run_id'] = run
    plan['standing_pair'].update(common_snapshot_control=start,roles=list(ROLES))
    plan['solver_comparison'] = dict(roles=list(ROLES),fine_solver_interval=[start,stop],
        seconds_during_release_window=dict(zip(ROLES,(.004,.004,.002),strict=True)),
        roll_clock_controls=[roll,switch,end],roll_seconds=[.001,.002],motor_update_s=.004,control_update_s=.02,
        audited_controls_per_arm=original.AUDIT_CONTROLS,integrated_acceptance_unchanged=True)
    launches = MOTOR_TICKS*start+sum(expected_launches(limit,plan,r) for r in ROLES)
    plan['solver_comparison']['maximum_solver_launches'] = launches
    plan['roll_window'] = dict(plan['roll_window'],promotion='Shared by all arms')
    controls = start+len(ROLES)*(limit-start)
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=controls,
        maximum_physics_substeps=MOTOR_TICKS*controls,maximum_control_transitions=controls*worlds,
        release_integrator=dict(reserved_seed=2246,archived_recurring_worlds=list(evidence['recurring_supine_failures']),
            actor_weights_and_policy_unchanged=True,promotion='Require the original complete route and at least two extra '
            'left-assigned scenes reaching roll compared with each 4 ms repeat. A comparable passing reference '
            'requires the original route on both 4 ms repeats. Independent confirmation is separate.'))
    for record in (source,recurrence):
        for field in ('protocol','result'):
            plan['references'][record[field]]=record[field+'_sha256']
    for path in (Path(__file__).resolve(),Path(window.__file__).resolve()):
        plan['references'][rel(path)] = sha(path)
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),observer=observer,physical_substeps=0,worlds=worlds,
        fork_control=start,maximum_batched_controls=controls,maximum_solver_launches=launches,
        maximum_scene_transitions=controls*worlds,learning_updates=0))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,
        preflight=rel(receipt),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Run the matched release branches and verify the original complete route and early recovery counters.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_RELEASE_INTEGRATOR_FROZEN',sha(PLAN),run,flush=True)


class ReleaseClock(original.SolverClock):
    def __init__(self, env, folder, plan):
        super().__init__(env,folder,plan)
        self.changes = {role:schedule(plan,role) for role in ROLES}

    def enter(self, role):
        self.role = role
        self.rows[role] = dict(self.new_arm(),transitions=[])
        self.check_model_step(1)

    def control_step(self, action):
        control = self.env.common_step_counter
        if self.role == 'shared':
            return super().control_step(action)
        if control==self.start:
            self.early_started = monotonic()
        if control in self.changes[self.role]:
            before = self.factor
            self.select(self.changes[self.role][control])
            self.rows[self.role]['transitions'].append(dict(control=control,prior_factor=before,
                new_factor=self.factor,engine_clock_s=float(self.sim.data.time[0].item())))
        result = super().control_step(action)
        if control+1==self.stop:
            self.rows[self.role]['early_window_wall_s'] = monotonic()-self.early_started
        return result


def collect(env, plan, actors, folder, observer, watch, ledger):
    fields = ROOT/plan['friction_development']['startup_fields']
    assert env.common_step_counter==env._sim_step_counter==0
    with np.load(fields,allow_pickle=False) as expected:
        for name in sorted(base.RESET_FIELDS):
            np.testing.assert_array_equal(parent.cpu(getattr(env.sim.model,name)),expected['expanded_'+name],err_msg=name)
    ledger['actual_slide_friction'] = plan['composition']['foot_friction']
    clock = ReleaseClock(env,folder,plan)
    assigned = iter(ROLES)
    base_fork = recorder.StandingFork
    publish = observer.publish
    counters = original.CollectorLedger(ledger)

    class SharedRelease(base_fork):
        def __init__(self, plant, controller, actor, memory):
            # Read the routing latch from the common observation before the first
            # physical action. Include it in all three restored branch states.
            assert plant.common_step_counter==clock.start
            assert actor.calls==0 and actor.right is None and actor.parent.supine_at_release is None
            actor(parent.base.shared.td('actor',plant.obs_buf['actor']))
            actor.calls = 0
            super().__init__(plant,controller,actor,memory)
            path = folder/'release-routing.npz'
            np.savez_compressed(path,gravity=recorder.cpu(actor.first_gravity),right=recorder.cpu(actor.right),
                latched_supine=recorder.cpu(actor.parent.supine_at_release))
            counters['release_routing_sha256'] = sha(path)

        def restore(self):
            clock.leave()
            state = super().restore()
            assert state['physical']['counters']==[clock.start,MOTOR_TICKS*clock.start]
            clock.enter(next(assigned))
            return state

    def publish_release(path,state):
        publish(path,dict(state,purpose=f'低摩擦起身步长对照｜{LABELS[state["phase"]]}',solver_launches=clock.executed))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=SharedRelease),
              patch.object(env,'step',clock.control_step),patch.object(env.sim,'step',clock.step),
              patch.object(observer,'publish',publish_release)):
            recorder.collect(env,plan,actors,folder,observer,watch,counters)
    finally:
        clock.leave()
        ledger.update(counters,solver_launches=clock.executed,solver_roles=clock.rows)


def check_timing(folder, plan, role, reached, raw):
    clock = raw['solver_roles'][role]
    start = plan['composition']['release_control']
    worlds = plan['composition']['worlds']
    origin = raw['solver_roles']['shared']['end_time_s']
    assert clock['start_time_s']==origin and clock['motor_ticks']==MOTOR_TICKS*(reached-start)
    assert clock['solver_launches']==expected_launches(reached,plan,role)
    record = read(folder/role/'closed-loop-clock.json',clock['audit_sha256'])
    assert record['role']==role and record['worlds']==worlds and record['motor_delay_ticks']==1
    assert record['checked_every_world_clock'] and record['unchanged_motor_input_between_solver_launches']
    assert record['controls']==clock['controls']
    assert [row['control'] for row in clock['controls']]==list(range(start,start+original.AUDIT_CONTROLS))
    changes = schedule(plan,role)
    for row in clock['controls']:
        original.check_control(row,changes.get(start,1))
    events = clock['transitions']
    assert [event['control'] for event in events]==[k for k in changes if k<reached]
    factor = 1
    for event in events:
        k = event['control']
        assert event['prior_factor']==factor and event['new_factor']==changes[k]
        factor = changes[k]
        np.testing.assert_allclose(event['engine_clock_s'],origin+(k-start)*.02,atol=1e-5,rtol=0)
    window_time = clock['early_window_wall_s']
    assert window_time > 0
    return dict(solver_launches=clock['solver_launches'],early_window_wall_s=window_time,
        early_scene_transitions_per_second=60*worlds/window_time,actual_switches=events)


def first_release_target(folder, record, start, routed_gravity):
    row, = (row for row in record['boundaries'] if row['control']==start)
    path = folder/row['file']
    assert sha(path)==row['sha256']
    with np.load(path,allow_pickle=False) as data:
        first = {name:data[name] for name in ('observed','mode','actor_input','raw','previous_target','actual_target')}
        first['first_delayed_motor'] = data['motor'][0]
    np.testing.assert_array_equal(first['actor_input'][:,3:6],routed_gravity)
    return first


def early_orientation(path, record, actual_supine, entered):
    rows = {name:[] for name in ('quaternion','foot_contact','head_contact')}
    # Release trace samples are indexed by the endpoint of each 20 ms control.
    for part in record['chunks']:
        a,b = part['first'],part['last']
        if a>175:
            break
        source = path/part['file']
        assert sha(source)==part['sha256'] and a>=101
        with np.load(source,allow_pickle=False) as data:
            for name in rows:
                rows[name].append(data[name][:min(b,175)-a+1])
    q,feet,head = (np.concatenate(value) for value in rows.values())
    assert len(q)==75
    q = q.astype(np.float64)
    q /= np.linalg.norm(q,axis=-1,keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1.-2.*(q[...,1]**2+q[...,2]**2),-1.,1.)))
    upright = feet.all(axis=-1)&~head&(tilt<=15.)
    dwell = (upright[:-2]&upright[1:-1]&upright[2:]).any(axis=0)
    return dict(actual_supine=int(actual_supine.sum()),
        actual_supine_entered_roll=int((actual_supine&entered).sum()),
        actual_supine_three_sample_upright_by_175=int((actual_supine&dwell).sum()),
        actual_supine_tilt_at_150_median_deg=float(np.median(tilt[49,actual_supine])))


def score():
    project, plan = driver.checked('score-pending')
    goal, task = project['goal_execution'],project['staged_training_request']
    node, folder = goal[KEY],ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert raw['actual_slide_friction']==.8 and not (raw['hardware_io'] or raw['learning_updates'] or raw['reserved_seed_used'])
    assert sha(folder/'protocol.json')==sha(PLAN) and sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    start,worlds = plan['composition']['release_control'],plan['composition']['worlds']
    clocks = raw['solver_roles']
    assert raw['prefix']['controls']==start and tuple(raw['branches'])==ROLES
    assert raw['controls']==start+sum(row['controls']-start for row in raw['branches'].values())
    assert raw['transitions']==raw['controls']*worlds<=plan['maximum_control_transitions']
    assert raw['physics_substeps']==MOTOR_TICKS*raw['controls']==sum(x['motor_ticks'] for x in clocks.values())
    assert raw['solver_launches']==sum(x['solver_launches'] for x in clocks.values())<=plan['solver_comparison']['maximum_solver_launches']
    assert clocks['shared']['solver_launches']==clocks['shared']['motor_ticks']==MOTOR_TICKS*start
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[start,MOTOR_TICKS*start]
    assert fork['physical']['fields']==len(fork['fields']) and fork['fields']==sorted(fork['field_signatures'])
    initial_path, release_path, routing = (folder/name for name in ('initial.npz','release-poses.npz','release-routing.npz'))
    assert sha(initial_path)==raw['initial_sha256'] and sha(release_path)==raw['release_sha256']
    assert sha(routing)==raw['release_routing_sha256']
    with (np.load(initial_path,allow_pickle=False) as stored,np.load(release_path,allow_pickle=False) as poses,
          np.load(routing,allow_pickle=False) as routed):
        initial, actual_supine, gravity = dict(stored),poses['poses']=='supine',routed['gravity']
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,start,comparison.EVIDENCE_FIELDS,sha)
    groups = plan['seven_command']['groups']
    recurring = plan['release_integrator']['archived_recurring_worlds']
    first = None
    outcomes, populations, roll = {},{},{}
    for role in ROLES:
        branch = raw['branches'][role]
        arm = folder/role
        timing = check_timing(folder,plan,role,branch['controls'],raw)
        action = first_release_target(arm,branch,start,gravity)
        if first is None:
            first = action
        for key in first:
            np.testing.assert_array_equal(action[key],first[key],err_msg=f'{role}/{key}')
        with patch.object(verifier,'standing_checks',comparison.branch_first):
            scorecard,populations[role],roll_action = verifier.verify_branch(plan,arm,branch,prefix,common,initial,fork)
        stitched = [dict(part,file=f'{name}/{part["file"]}') for name,segment in (
            ('shared',raw['prefix']),(role,branch)) for part in segment['chunks']]
        _,entered,passed,reasons,peak,_ = window.replay(plan,folder,dict(chunks=stitched))
        np.testing.assert_array_equal(passed,populations[role]['roll'])
        np.testing.assert_array_equal(entered,roll_action['mode']==comparison.ROLLING)
        populations[role]['roll_entry'] = entered
        scorecard.update(timing=timing,early_orientation=early_orientation(arm,branch,actual_supine,entered),
            roll_entry=int(entered.sum()),entry_by_command={cmd:int(entered[ids].sum()) for cmd,ids in groups.items()},
            roll_reasons=dict(Counter(reasons)),recurring_worlds={k:dict(roll_entry=bool(entered[int(k)]),
                reason=reasons[int(k)],peak_supported_deg=float(peak[1,int(k)])) for k in recurring})
        outcomes[role] = scorecard
        roll[role] = dict(passed=passed,reason=reasons,peak_supported_deg=peak[1])
    pairs = {f'{a}-vs-{b}':dict(terminal=verifier.compare(populations[a],populations[b],groups),
        roll=window.matched_effect(roll[a],roll[b],np.ones(worlds,bool)))
        for a,b in ((ROLES[1],ROLES[0]),(ROLES[-1],ROLES[0]),(ROLES[-1],ROLES[1]))}
    candidates = [outcomes[r]['passed_development_gate'] for r in ROLES]
    left = [outcomes[r]['entry_by_command']['turn-left'] for r in ROLES]
    preferred = ROLES[-1] if candidates[-1] and left[-1]>=max(left[:2])+2 else ROLES[0] if all(candidates[:2]) else None
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='verified-matched-release-integrator',protocol_sha256=sha(PLAN),
        collection_sha256=node['collection_sha256'],seed=plan['environment']['seed'],worlds=worlds,
        verified_fork=fork['physical'],identical_first_action_and_first_delayed_motor=True,
        transitions=raw['transitions'],solver_launches=raw['solver_launches'],results=outcomes,pairs=pairs,
        peak_gpu_mib=max(x['used_mib'] for x in raw['gpu_samples']),preferred_for_independent_check=preferred,
        learning_updates=0,hardware_io=False,reserved_seed_used=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),preferred=preferred)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    task.update(info,next_gate='Freeze independent low-friction confirmation with the selected route.' if preferred else
        'Use new matched early orientation and contact evidence to select a targeted recovery change.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_RELEASE_INTEGRATOR_SCORED',sha(output),
        {role:(row['roll_entry'],row['roll_complete'],row['complete']) for role,row in outcomes.items()},preferred,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.multiple(parent,config=base.configured,PLAN=PLAN),
          patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()

