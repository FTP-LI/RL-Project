"""Measure actual 4 ms support and motor torque during the fixed-policy roll comparison."""

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_roll_endpoint_progress_evaluation as endpoint
from native_nominal_baseline import verify_sources
from record_io import write
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


KEY = 'roll_head_recontact_load_probe'
STAGE = 'roll-head-recontact-load-probe'
ENTRY = Path(__file__).resolve()
PLANS = endpoint.PLANS.with_name('head-recontact-load-probe')
MASTER = PLANS/'master.json'
REPORT = endpoint.REPORT.with_name('head-recontact-load-probe')
SOURCE_MASTER = endpoint.MASTER
ROLE_NAMES, CASES = endpoint.ROLES, endpoint.CASES
FIRST, LAST = 63, 100
SUBSTEPS = 5


class LoadObserver(RecordedObserver):
    def start(self):
        self.state['purpose'] = '滚翻头部再接触载荷与电机力矩｜固定策略'
        super().start()


def freeze():
    from verify_roll_contact_clock import check_alignment_fixture

    project = read(P)
    task, goals = project['staged_training_request'], project['goal_execution']
    loc, earlier = (goals[key] for key in ('roll_endpoint_support_localization','roll_endpoint_progress_evaluation'))
    require(endpoint.previous.direct.pair.trial.allowed(project,'roll-endpoint-support-localization')
            and loc['status']=='verified-archived-development-roll-loss' and earlier['status']=='reviewed'
            and not (task['learning_running'] or task['simulation_running'] or PLANS.exists() or REPORT.exists()),
            'The saved support localization is incomplete or the project is not idle')
    origin, result = read(SOURCE_MASTER), ROOT/loc['result']
    original = read(result)
    require(sha(SOURCE_MASTER)==earlier['master_sha256'] and sha(result)==loc['result_sha256']
            and original['replayed_episodes']==768 and not origin['reserved_seed_used'] and origin['reserved_seed']==2246,
            'The archived localization, source case or seed split differs')
    for case in CASES:
        for rows in original['seeds'][case]['matched_rows'].values():
            require(all(FIRST < row['target_position_marks']['90']['control']
                        and row['first_geometric_crossing_control'] <= LAST for row in rows),
                    'The measurement window does not cover the archived roll interval')
    check_alignment_fixture()
    dependencies = (ENTRY,result,SOURCE_MASTER,Path(endpoint.__file__),ENTRY.with_name('roll_contact_clock_capture.py'),
                    ENTRY.with_name('verify_roll_checkpoint_selection.py'))
    evidence = {rel(path):sha(path) for path in dependencies}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    protocols = {}
    for case in CASES:
        source = SOURCE_MASTER.with_name(case+'.json')
        plan = read(source)
        require(sha(source)==origin['protocols'][case] and plan['initial_stand_controls']==50
                and plan['maximum_controls']==500 and plan['worlds']==128 and plan['minimum_roll_completion']==116,
                'Archived action gate, population or timing differs')
        plan['frozen_sha256'].update(evidence | {rel(source):sha(source)})
        out = REPORT/case
        plan.update(stage=STAGE, run_id=f'duck_roll_head_recontact_load_s{plan["seed"]}_{stamp}',
                    guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='Original fixed-actor roll comparison with a read-only 4 ms contact and motor measurement',
                    force_capture=dict(first_control=FIRST,last_control=LAST,capture_seconds=[FIRST*.02,LAST*.02],
                        maximum_batched_substeps=plan['maximum_controls']*SUBSTEPS,load_threshold_nominal_weight_fraction=.01))
        verify_sources(plan)
        protocols[case] = plan
    REPORT.mkdir()
    PLANS.mkdir()
    admission = REPORT/'admission.json'
    put(admission,dict(observation=preflight(), source_sha256=evidence, original_gate=116, first_control=FIRST,
                       last_control=LAST, added_physics_for_measurement=0))
    hashes = {}
    for case,plan in protocols.items():
        out, path = REPORT/case, PLANS/(case+'.json')
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,plan)
        hashes[case] = sha(path)
        put(out/'windows-entry.json',dict(protocol_sha256=hashes[case],evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=list(CASES),protocols=hashes,physical_transition_budget=origin['physical_transition_budget'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),admission=rel(admission),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='roll-head-recontact-load-ready',current_evidence=rel(admission),
                next_gate='Measure actual normal load and joint actuator torque at every 4 ms substep through the original roll.')
    write(P,project)
    print('ROLL_HEAD_RECONTACT_LOAD_FROZEN',sha(MASTER),flush=True)


def instrument(factory, base, may_continue):
    from roll_contact_clock_capture import ContactClock

    original = factory(base,may_continue)

    def collect(env, observation, plan, folder, *args):
        import mujoco

        clock = ContactClock(env,plan['force_capture'])
        model = env.sim.mj_model
        joint_names = plan['walking']['joint_names']
        require(np.all(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT), 'Actuator transmission is not a joint motor')
        joint_ids = np.asarray(model.actuator_trnid)[:,0]
        order = [model.joint(int(joint)).name.rsplit('/',1)[-1] for joint in joint_ids]
        require(len(order)==len(joint_names) and set(order)==set(joint_names), 'Compiled joint motor order differs')
        motors = [order.index(name) for name in joint_names]
        dofs = model.jnt_dofadr[joint_ids[motors]].tolist()
        motor_index = base.torch.as_tensor(motors,device=env.device)
        dof_index = base.torch.as_tensor(dofs,device=env.device)
        physics_step = env.sim.step
        shared = plan['checkpoint_selection']['plant']['shared_snapshot_control']*SUBSTEPS
        position = -1

        def flush():
            if not clock.rows:
                return
            branch = folder/'clips'/ROLE_NAMES[position]
            clock.save(branch)
            stamp = branch/'contact-clock.json'
            write(stamp,dict(read(stamp), joint_names=joint_names,
                motor_control_clock='prior to each physics step', joint_actuator_torque_clock='after the same solver step'))

        def step():
            substep = env._sim_step_counter
            if substep == shared+1:
                expected = shared if position<0 else clock.maximum
                require(clock.completed==expected and position+1<len(ROLE_NAMES), 'Unexpected roll snapshot rewind')
                flush()
                clock.rows.clear()
                position += 1
                clock.completed = shared
            capture = clock.first < substep <= clock.last
            if capture:
                require(position>=0, 'Force window overlaps the shared prefix')
                motor = base.cpu(env.sim.data.ctrl[:,motor_index])
            clock.record(physics_step)
            if capture:
                clock.rows[-1].update(motor_control=motor,joint_actuator_torque_nm=base.cpu(env.sim.data.qfrc_actuator[:,dof_index]))

        try:
            with patch.object(env.sim,'step',step):
                result = original(env,observation,plan,folder,*args)
            require(position==len(ROLE_NAMES)-1 and clock.completed==clock.maximum, 'A branch or its physical record is missing')
            return result
        finally:
            flush()

    return collect


@contextmanager
def scope():
    paths = dict(KEY=KEY,STAGE=STAGE,PLANS=PLANS,MASTER=MASTER,REPORT=REPORT,ENTRY=ENTRY)
    with patch.multiple(endpoint,**paths,RollEvaluationView=LoadObserver), endpoint.stage_bindings():
        factory = endpoint.previous.make_collector
        with patch.object(endpoint.previous,'make_collector',lambda base,guard:instrument(factory,base,guard)):
            yield


def measured_phase(plan, raw, role, receipt, paths):
    folder = raw/'clips'/role
    spec = plan['force_capture']
    boundary = plan['checkpoint_selection']['plant']['shared_snapshot_control']
    assert FIRST > boundary and spec['first_control']==FIRST and spec['last_control']==LAST
    trace = folder/'trace.npz'
    archive = folder/'contact-clock.npz'
    meta_path = folder/'contact-clock.json'
    meta = read(meta_path)
    paths.update({rel(path):sha(path) for path in (trace,archive,meta_path)})
    require(meta['complete'] and meta['joint_names']==plan['walking']['joint_names']
            and meta['labels']==['ground','head','left_foot','right_foot']
            and paths[rel(archive)]==meta['sha256'] and paths[rel(trace)]==receipt['branches'][role]['trace_sha256']['trace.npz'],
            'Roll load or trained actor pose record differs')
    with np.load(trace,allow_pickle=False) as stored:
        keys = ('qpos','ground_contact','phase_next')
        state = {key:stored[key][FIRST-boundary:LAST-boundary+1] for key in keys}
        start = plan['initial_stand_controls']-boundary
        prefix = stored['qpos'][start:FIRST-boundary+1,:,3:7]
    with np.load(archive,allow_pickle=False) as stored:
        clock = dict(stored)
    np.testing.assert_array_equal(clock['substep'],np.arange(FIRST*SUBSTEPS+1,LAST*SUBSTEPS+1))
    np.testing.assert_array_equal(clock['before'][::SUBSTEPS],state['qpos'][:-1,:,3:7])
    np.testing.assert_array_equal(clock['after'][SUBSTEPS-1::SUBSTEPS],state['qpos'][1:,:,3:7])
    np.testing.assert_array_equal(clock['before'][1:],clock['after'][:-1])
    sample_count, worlds = ((LAST-FIRST)*SUBSTEPS,plan['worlds'])
    force, torque, motor = (clock[key] for key in ('normal_force_n','joint_actuator_torque_nm','motor_control'))
    require(force.shape==(sample_count,worlds,4) and torque.shape==motor.shape==(sample_count,worlds,14)
            and all(np.isfinite(value).all() for value in (force,torque,motor)) and (force>=0).all(), 'Invalid load or torque samples')
    require(meta['force_threshold_n']==meta['nominal_weight_n']*spec['load_threshold_nominal_weight_fraction'], 'Load threshold changed')
    source = np.concatenate((prefix,clock['after']))
    phi = np.rad2deg(progress(dict(quaternion=source,ground_contact=np.ones(source.shape[:2],dtype=bool)))[0])
    phi = phi[len(prefix)-1:]
    phase_covered = (phi[0] <= 180.) & (phi.max(axis=0) >= 270.)
    require(phase_covered.all(), 'The measurement window misses part of the measured 180–270 degree phase')
    overlap = np.maximum(np.minimum(phi[1:],270.)-np.maximum(phi[:-1],180.),0.)
    gap = (~state['ground_contact'][1:]) & (state['phase_next'][:-1]==ROLL)
    gap_angle = overlap*np.repeat(gap,SUBSTEPS,axis=0)
    loaded = force[...,0] > meta['force_threshold_n']
    head = force[...,1] > meta['force_threshold_n']
    head_joint = meta['joint_names'].index('head_pitch')
    phase = overlap > 0
    values = dict(endpoint_excluded_angle_deg=gap_angle.sum(axis=0),
        endpoint_excluded_angle_with_load_deg=(gap_angle*loaded).sum(axis=0),
        endpoint_excluded_angle_without_load_deg=(gap_angle*~loaded).sum(axis=0),
        endpoint_excluded_angle_with_head_load_deg=(gap_angle*head).sum(axis=0),
        phase_head_impulse_ns=(force[...,1]*phase).sum(axis=0)*.004,
        peak_abs_head_pitch_actuator_torque_nm=np.where(phase,np.abs(torque[...,head_joint]),0.).max(axis=0),
        peak_abs_head_pitch_motor_control=np.where(phase,np.abs(motor[...,head_joint]),0.).max(axis=0))
    np.testing.assert_allclose(values['endpoint_excluded_angle_with_load_deg']+values['endpoint_excluded_angle_without_load_deg'],
                               values['endpoint_excluded_angle_deg'],atol=1e-9,rtol=0)
    return values


def summarize():
    from locate_roll_endpoint_support_loss import cohort_stats

    project = read(P)
    goals = project['goal_execution']
    state, original = (goals[key] for key in (KEY,'roll_endpoint_support_localization'))
    old, proof = ROOT/original['result'], REPORT/'comparison.json'
    require(state['status']=='reviewed' and sha(proof)==state['result_sha256'] and sha(old)==original['result_sha256'],
            'The matched prospective scores or archived selection are incomplete')
    cases, results, hashes = read(proof)['arms'], {}, {rel(path):sha(path) for path in (old,proof,MASTER,ENTRY)}
    focus = read(old)['seeds']
    a,b,c = ROLE_NAMES
    for case, scored in cases.items():
        path = PLANS/(case+'.json')
        require(sha(path)==scored['protocol_sha256'], 'Collection protocol differs')
        plan = read(path)
        raw = ROOT/'records/evaluations'/scored['run_id']
        receipt = raw/'guarded-evaluation.json'
        require(sha(receipt)==scored['collection_sha256'],'Collection receipt changed')
        hashes.update({rel(item):sha(item) for item in (path,receipt)})
        data = {role:measured_phase(plan,raw,role,read(receipt),hashes) for role in ROLE_NAMES}
        earlier_ids = np.asarray(focus[case]['cohorts']['matched_regressions']['worlds'],dtype=int)
        first,second,candidate = (np.asarray(scored['arms'][role]['roll_pass']) for role in ROLE_NAMES)
        repeated_ids = np.flatnonzero(first & second & ~candidate)
        cohorts = {}
        for name,ids in (('archived_regressions',earlier_ids),('new_matched_regressions',repeated_ids),('all_worlds',np.arange(plan['worlds']))):
            values = {role:{key:cohort_stats(samples[ids]) for key,samples in metrics.items()} for role,metrics in data.items()}
            comparison = {role:{key:cohort_stats((samples-data[role][key])[ids]) for key,samples in data[c].items()} for role in (a,b)}
            cohorts[name] = dict(worlds=ids.tolist(),measurements=values,candidate_minus_reference=comparison)
        results[case] = dict(run_id=raw.name, original_gate_passes={role:int(np.asarray(scored['arms'][role]['roll_pass']).sum())
            for role in ROLE_NAMES}, archived_regressions_repeated=np.intersect1d(earlier_ids,repeated_ids).tolist(), cohorts=cohorts)
    result = REPORT/'physical-load-summary.json'
    put(result,dict(cases=results,sources_sha256=hashes,control_period_s=.02,physical_period_s=.004,
        measurement_control_range=[FIRST+1,LAST], phase_degrees=[180,270],reserved_seed_used=False,
        added_physics_for_instrumentation=0,learning_updates=0,hardware_io=False))
    state.update(load_result=rel(result),load_result_sha256=sha(result))
    project['staged_training_request'].update(current_protocol=rel(MASTER),current_evidence=rel(result),
        current_protocol_execution='roll-head-recontact-load-reviewed',
        next_gate='Use the measured contact-load and torque differences to select a bounded roll recovery intervention.')
    write(P,project)
    print('ROLL_HEAD_RECONTACT_LOAD_VERIFIED',sha(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score','summarize'))
    parser.add_argument('case',nargs='?',choices=CASES)
    args = parser.parse_args()
    if args.action in ('freeze','summarize'):
        return {'freeze':freeze,'summarize':summarize}[args.action]()
    with scope():
        if args.action=='batch':
            return endpoint.previous.batch()
        if args.case is None:
            parser.error('Select the original development seed')
        {'collect':endpoint.previous.sample,'score':endpoint.previous.score}[args.action](args.case)


if __name__ == '__main__':
    main()
