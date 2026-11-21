"""Test a bounded right hip and knee residual in a common early get-up state."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_low_friction_release_integrator as source
import run_nominal_supine_foot_authority as short_check
from review_nominal_low_friction_right_support_signal import mask
from routed_seven_command_score import recovery_and_standing
from run_nominal_roll_lateral_disturbance import ImpulseClock
from verify_nominal_low_friction_release_integrator_readback import reconstruct_clock
from routed_getup_sequence import GETUP


base,driver,parent,recorder,original,comparison = (
    source.base,source.driver,source.parent,source.recorder,source.original,source.comparison)
ROOT,PROJECT,read,rel,sha,write = source.ROOT,source.PROJECT,source.read,source.rel,source.sha,source.write
KEY = 'nominal_low_friction_right_support_trial'
STAGE = 'nominal-low-friction-right-support-trial'
PRIOR = 'nominal-low-friction-right-support-position'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = source.REPORT.with_name('low-friction-right-support-trial')
ROLES = ('original-first','original-repeat','right-hip-knee-candidate')
LABELS = dict(zip(('shared',*ROLES),('共同跌倒和起身','原起身参照一','原起身参照二','右髋膝位置候选'),strict=True))


def freeze():
    project = read(PROJECT)
    task,goal = project['staged_training_request'],project['goal_execution']
    assert parent.allowed(project,PRIOR) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    low,signal,position = (goal[k] for k in (source.KEY,'nominal_low_friction_right_support_signal',
                                          'nominal_low_friction_right_support_position'))
    assert all(x['status']=='reviewed' for x in (low,signal,position))
    plan = read(ROOT/low['protocol'],low['protocol_sha256'])
    marker = read(ROOT/signal['protocol'],signal['protocol_sha256'])
    motor_proof = read(ROOT/position['result'],position['result_sha256'])
    assert marker['control']==136 and motor_proof['recommended_position_offset_rad']==position['position_offset_rad']==.02
    assert marker['tilt_min_deg']==20. and marker['tilt_max_deg']==60.
    d,p = plan['composition'],plan['environment']
    split,end = marker['control'],d['roll_start_control']
    lo,hi = plan['solver_comparison']['fine_solver_interval']
    assert (lo,split,hi,end,d['getup_to_stand_control'])==(100,136,160,600,550)
    assert (p['seed'],d['foot_friction'],d['worlds'],d['existing_low_friction_reserved_seed'])==(1257,.8,896,2246)
    count = split+len(ROLES)*(end-split)
    fine = split-lo+len(ROLES)*(hi-split)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-right-support-friction080-s1257-{stamp}'
    p['run_id'] = run
    plan['standing_pair'].update(common_snapshot_control=split,roles=list(ROLES))
    plan['solver_comparison'] = dict(fine_solver_interval=[lo,hi],maximum_solver_launches=5*(count+fine),roles=list(ROLES))
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=count,
        maximum_physics_substeps=5*count,maximum_control_transitions=count*d['worlds'],
        right_support_trial=dict(decision_control=split,end_control=end,begin=lo,end_fine=hi,
            pulse_controls=4,joints=['right_hip_pitch','right_knee'],residual_rad=[-.02,.02],
            minimum_actual_torque_delta_nm=.005,minimum_both_joints_direction_share=.8,minimum_left_gain_vs_both=2,
            signal=signal['protocol'],signal_sha256=signal['protocol_sha256'],
            score_scope='Original get-up with hold and original G600 roll entry. Independent complete route follows eligible candidates.',
            reserved_seed=2246,automatic_retry=False,learning_updates=0,hardware_io=False))
    for record in (low,signal,position):
        for field in ('protocol','result'):
            plan['references'][record[field]] = record[field+'_sha256']
    entry = Path(__file__).resolve()
    plan['references'][rel(entry)] = sha(entry)
    for name in ('run_nominal_roll_lateral_disturbance.py','run_nominal_supine_foot_authority.py',
                 'verify_nominal_low_friction_release_integrator_readback.py'):
        path = entry.with_name(name)
        plan['references'][rel(path)] = sha(path)
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),observer=observer,zero_physical_steps=0,
        fixed_budget_controls=count,maximum_scene_transitions=count*d['worlds'],maximum_batched_solver_launches=5*(count+fine)))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,preflight=rel(receipt),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Compare the measured two-joint action and the original get-up and roll-entry scores from the same G136 snapshot.')
    project.update(info)
    write(PROJECT,project)
    print('RIGHT_SUPPORT_TRIAL_FROZEN',sha(PLAN),run,flush=True)


def collect(env,plan,actors,folder,observer,watch,ledger):
    assert env.common_step_counter==env._sim_step_counter==0
    with np.load(ROOT/plan['friction_development']['startup_fields'],allow_pickle=False) as f:
        for name in sorted(base.RESET_FIELDS):
            np.testing.assert_array_equal(parent.cpu(getattr(env.sim.model,name)),f['expanded_'+name],err_msg=name)
    study = plan['right_support_trial']
    pattern = read(ROOT/study['signal'],study['signal_sha256'])
    start,end = study['decision_control'],study['end_control']
    short = dict(plan,composition={**plan['composition'],'maximum_controls':end})
    ctrl,fork = recorder.Composition,recorder.StandingFork
    clock = ImpulseClock(env,folder,plan)
    counters = original.CollectorLedger(ledger)
    order = iter(ROLES)
    route = actors['getup']
    role = 'shared'
    names = plan['walking']['joint_names']
    indexes = [names.index(name) for name in study['joints']]
    joint = torch.tensor(indexes,device=env.device)
    limits = torch.tensor(np.asarray(plan['environment']['joint_limits'])[indexes],device=env.device,dtype=torch.float32)
    value = torch.tensor(study['residual_rad'],device=env.device,dtype=torch.float32)
    before,applied = [],[]

    class RightTargets(ctrl):
        def observe_roll(self,k,frame):
            super().observe_roll(k,frame)
            if k==start:
                gravity = parent.cpu(env.obs_buf['actor'][:,3:6])
                latched = parent.cpu(route.parent.supine_at_release&~route.right)
                self.eligible = mask(gravity,latched&frame['active'],pattern)
                self.world = torch.as_tensor(np.flatnonzero(self.eligible),device=env.device)[:,None]
                assert self.eligible.any()
                dest = folder/'trigger.npz'
                np.savez_compressed(dest,gravity=gravity,latched_supine=latched,active=frame['active'],eligible=self.eligible)
                counters['trigger_sha256'] = sha(dest)

        def apply(self,term,previous,control,action):
            super().apply(term,previous,control,action)
            if role!=ROLES[-1] or not start<=control<start+study['pulse_controls']:
                return
            assert np.all(action.mode[self.eligible]==GETUP)
            original_target = term._processed_actions[self.world,joint]
            minimum,maximum = limits.unbind(-1)
            proposed = (original_target+value).clamp(minimum,maximum)
            chosen = torch.where((original_target>=minimum)&(original_target<=maximum),proposed,original_target)
            for field in ('requested_target','projected_destination','_processed_actions','previous_target'):
                getattr(term,field)[self.world,joint] = chosen
            before.append(parent.cpu(original_target))
            applied.append(parent.cpu(chosen))

    class SameSupportState(fork):
        def restore(self):
            nonlocal role
            clock.finish_arm()
            clock.select(2)
            state = super().restore()
            assert state['physical']['counters']==[start,5*start]
            role = next(order)
            clock.enter(role)
            return state

    publish = observer.publish

    def status(path,state):
        publish(path,dict(state,purpose=f'低摩擦右髋膝起身对照｜{LABELS[state["phase"]]}',solver_launches=clock.executed))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=SameSupportState,Composition=RightTargets),
              patch.object(env,'step',clock.control_step),patch.object(env.sim,'step',clock.step),
              patch.object(observer,'publish',status)):
            recorder.collect(env,short,actors,folder,observer,watch,counters)
    finally:
        if before:
            archive = folder/ROLES[-1]/'right-support-targets.npz'
            np.savez_compressed(archive,before=np.stack(before),applied=np.stack(applied))
            counters['target_archive_sha256'] = sha(archive)
        clock.leave()
        ledger.update(counters,solver_launches=clock.executed,solver_roles=clock.rows)


def score():
    project,plan = driver.checked('score-pending')
    goal,task = project['goal_execution'],project['staged_training_request']
    node,folder = goal[KEY],ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not any(raw[k] for k in ('hardware_io','learning_updates','reserved_seed_used'))
    assert raw['transitions']==plan['maximum_control_transitions'] and raw['solver_launches']==plan['solver_comparison']['maximum_solver_launches']
    assert sha(folder/'protocol.json')==sha(PLAN) and sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    settings = plan['right_support_trial']
    start,end,beg,fin = (settings[k] for k in ('decision_control','end_control','begin','end_fine'))
    pattern = read(ROOT/settings['signal'],settings['signal_sha256'])
    short = dict(plan,foot_authority=dict(decision_control=start,end_control=end))
    worlds = plan['composition']['worlds']
    assert raw['prefix']['controls']==start and tuple(raw['branches'])==ROLES
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[start,5*start] and fork['physical']['fields']==len(fork['fields'])
    assert sha(folder/'trigger.npz')==raw['trigger_sha256'] and sha(folder/'initial.npz')==raw['initial_sha256']
    with np.load(folder/'trigger.npz',allow_pickle=False) as data:
        trigger = dict(data)
    np.testing.assert_array_equal(trigger['eligible'],mask(trigger['gravity'],trigger['latched_supine']&trigger['active'],pattern))
    chosen = trigger['eligible']
    sides = plan['seven_command']['groups']
    ind = [plan['walking']['joint_names'].index(name) for name in settings['joints']]
    directions = np.sign(settings['residual_rad'])
    fields = comparison.EVIDENCE_FIELDS
    common = comparison.read_trace(folder/'shared',raw['prefix']['chunks'],1,start,fields,sha)
    with np.load(folder/'initial.npz',allow_pickle=False) as data:
        initial = dict(data)
    scores,enters,firsts = {},{},{}
    _,shared_time = reconstruct_clock(0,0,start,{beg:2})
    _,end_time = reconstruct_clock(0,0,end,{beg:2,fin:1})
    assert raw['solver_roles']['shared']['end_time_s']==shared_time
    for role in ROLES:
        arm = raw['branches'][role]
        assert arm['controls']==end
        timing = raw['solver_roles'][role]
        assert timing['start_time_s']==shared_time and timing['end_time_s']==end_time
        assert timing['motor_ticks']==5*(end-start) and timing['solver_launches']==5*(end-start+fin-start)
        marker = read(folder/role/'closed-loop-clock.json',timing['audit_sha256'])
        assert marker['checked_every_world_clock'] and marker['unchanged_motor_input_between_solver_launches']
        assert marker['motor_delay_ticks']==1 and [x['control'] for x in marker['controls']]==list(range(start,start+3))
        for tick in marker['controls']:
            original.check_control(tick,2)
        segment,first,entry,clip = short_check.verify_arm(short,folder,role,dict(arm,camera_command=raw['camera_command']),
                                                       fork,common,initial)
        enters[role],firsts[role] = entry,first
        trace = {name:np.concatenate((common[name],segment[name])) for name in fields}
        joints = trace['qpos'][...,raw['qpos_joint_addresses']]
        limits = np.asarray(plan['environment']['joint_limits'])
        tol = np.deg2rad(plan['environment']['applied_joint_limit_tolerance_deg'])
        env = dict(trace,foot_contact=trace['original_feet'],joint_positions=joints,
                   joint_envelope_ok=((joints>=limits[:,0]-tol)&(joints<=limits[:,1]+tol)).all(axis=-1))
        _,_,getup = recovery_and_standing(plan,folder,trace,initial,env)
        scores[role] = dict(getup=getup,roll_entry=int(entry.sum()),by_command={c:int(entry[ids].sum()) for c,ids in sides.items()},
                           triggered_entry=int((entry&chosen).sum()),video=clip)
    no,again,candidate = [firsts[r] for r in ROLES]
    np.testing.assert_array_equal(no['observed'][:,3:6],trigger['gravity'])
    for other in (again,candidate):
        for name in ('observed','actor_input','mode','raw','previous_target'):
            np.testing.assert_array_equal(other[name],no[name])
        np.testing.assert_array_equal(other['motor'][0],no['motor'][0])
    np.testing.assert_array_equal(again['actual_target'],no['actual_target'])
    area = np.zeros((worlds,14),bool)
    area[np.ix_(chosen,ind)] = True
    np.testing.assert_array_equal(candidate['actual_target'][~area],no['actual_target'][~area])
    np.testing.assert_array_equal(candidate['actual_target'],candidate['expected_articulation_target'])
    file = folder/ROLES[-1]/'right-support-targets.npz'
    assert sha(file)==raw['target_archive_sha256']
    with np.load(file,allow_pickle=False) as data:
        prior,updated = data['before'],data['applied']
    assert prior.shape==updated.shape==(settings['pulse_controls'],int(chosen.sum()),2)
    np.testing.assert_array_equal(updated[0],candidate['actual_target'][np.ix_(chosen,ind)])
    np.testing.assert_array_equal(prior[0],no['actual_target'][np.ix_(chosen,ind)])
    assert (np.abs(updated-prior)<=.02000011).all() and (((updated-prior)*directions)>=0).all()
    slot = lambda row:row['motor'][1][np.ix_(chosen,ind)]
    before = slot(no)
    after = slot(candidate)
    measured = np.minimum((after-before)*directions,(after-slot(again))*directions)
    direction_hit = (measured>=settings['minimum_actual_torque_delta_nm']).all(axis=1)
    advances = (direction_hit.mean()>=settings['minimum_both_joints_direction_share']
        and scores[ROLES[-1]]['getup']['functional_coverage_gate']
        and scores[ROLES[-1]]['roll_entry']>max(scores[r]['roll_entry'] for r in ROLES[:2])
        and scores[ROLES[-1]]['by_command']['turn-left']>=max(scores[r]['by_command']['turn-left'] for r in ROLES[:2])
            +settings['minimum_left_gain_vs_both'])
    pairs = {}
    for role in ROLES[:2]:
        new,old = enters[ROLES[-1]],enters[role]
        pairs[role] = dict(gain=np.flatnonzero(new&~old).tolist(),loss=np.flatnonzero(~new&old).tolist(),
            gain_among_triggered=np.flatnonzero(new&~old&chosen).tolist(),loss_among_triggered=np.flatnonzero(~new&old&chosen).tolist())
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='verified-right-hip-knee-position-trial',protocol_sha256=sha(PLAN),collection_sha256=node['collection_sha256'],
        roles=scores,transitions=raw['transitions'],solver_launches=raw['solver_launches'],fork=fork['physical'],
        initial_policy_and_delayed_motor_exact=True,chosen=int(chosen.sum()),both_joint_measured_direction_count=int(direction_hit.sum()),
        both_joint_minimum_direction_deltas_nm=measured.tolist(),triggered_world_ids=np.flatnonzero(chosen).tolist(),
        paired_entry=pairs,advance_to_full_low_friction=bool(advances),
        peak_gpu_mib=max(x['used_mib'] for x in raw['gpu_samples']),learning_updates=0,hardware_io=False,reserved_seed_used=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),full_route_admitted=bool(advances))
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    project.update(info)
    task.update(info,next_gate='Run a full low-friction development route before opening the independent 2246 confirmation.'
        if advances else 'Compare the actual early right-joint action, per-scene support response, and unrecovered groups before selecting the next low-friction trial.')
    write(PROJECT,project)
    print('RIGHT_SUPPORT_TRIAL_SCORED',sha(output),
        {r:(s['roll_entry'],s['by_command']['turn-left']) for r,s in scores.items()},int(direction_hit.sum()),bool(advances),flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    args = cli.parse_args()
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.multiple(parent,config=base.configured,PLAN=PLAN),
          patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[args.action]()
