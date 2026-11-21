"""Check bounded neck and leg torque changes on measured early support states."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

import numpy as np

import review_nominal_low_friction_early_transfer_authority as authority
import review_nominal_low_friction_forward_support as measured
from roll_contact_clock_capture import reduce_contacts


experiment = measured.experiment
ROOT, PROJECT, read, rel, sha, write = experiment.ROOT, experiment.PROJECT, measured.read, measured.rel, measured.sha, measured.write
KEY, STAGE = 'nominal_low_friction_forward_support_torque', 'nominal-low-friction-forward-support-torque'
PLAN = experiment.PLAN.with_stem(KEY)
REPORT = experiment.REPORT/'short-torque-replay'
PROFILES = {
    'neck-unload': {'neck_pitch':.06},
    'leg-transfer': {'left_hip_pitch':.025,'left_knee':-.025,'right_hip_pitch':-.025,'right_knee':.025},
}


def freeze():
    project = read(PROJECT)
    task, stages = project['staged_training_request'], project['goal_execution']
    assert experiment.parent.allowed(project,experiment.STAGE) and not task['simulation_running']
    assert KEY not in stages and not (PLAN.exists() or REPORT.exists())
    capture, old = stages[experiment.KEY], stages[authority.KEY]
    assert capture['status']==old['status']=='reviewed'
    evidence = read(ROOT/capture['result'],capture['result_sha256'])
    assert evidence['seed']==1257 and evidence['foot_sliding_friction']==.8 and evidence['learning_updates']==0
    load = stages['nominal_low_friction_support_effort_limit_readback']
    original = read(ROOT/load['result'],load['result_sha256'])
    pairs = original['release_support']['matches']
    admitted = sorted({row['admitted']['world'] for row in pairs})
    assert len(admitted)==13
    sources = {item[key]:item[key+'_sha256'] for item,keys in
               ((capture,('protocol','collection','result')),(old,('protocol',)),(load,('result',))) for key in keys}
    for src in (__file__,authority.__file__,measured.__file__):
        path = Path(src).resolve()
        sources[rel(path)] = sha(path)
    for source,checksum in sources.items():
        assert sha(ROOT/source)==checksum
    spec = dict(status='frozen-short-diagnostic',stage=STAGE,seed=1257,friction=.8,release_control=100,
        capture_protocol=capture['protocol'],capture_result=capture['result'],capture_collection=capture['collection'],
        archived_authority_protocol=old['protocol'],sources=sources,focus=[439,779],protected_archived_admitted=admitted,
        profiles_nm=PROFILES,start_offset_before_support_ms=40,physics_dt=.004,steps_per_rollout=50,
        delay_substeps=1,commanded_pulse_substeps=20,maximum_sum_torque_change_nm=.10,
        measure_after_support_ms=80,maximum_reference_tilt_error_deg=5.,maximum_reference_support_delay_ms=20,
        minimum_target_tilt_gain_deg=3.,maximum_protected_tilt_loss_deg=2.,
        allowed_head_load_increase_body_weights=.025,maximum_lost_protected_support_ms=8,
        minimum_target_success_each_world=1,minimum_target_success_all=3,minimum_protected_each_parent=12,
        heldout_seed=2246,learning_updates=0,hardware_io=False)
    REPORT.mkdir()
    write(PLAN,spec)
    stages[KEY] = dict(status='frozen',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',PLAN,
        'Verify the actual delivered torque and the reference trajectory before comparing each local intervention.')
    write(PROJECT,project)
    print('FORWARD_SUPPORT_TORQUE_FROZEN',sha(PLAN),len(admitted),flush=True)


def rollout(engine, spec, state, effort, meta, offsets):
    model,data,mj = engine.model, engine.data, engine.mj
    for name, values in engine.fields.items():
        getattr(model,name)[:] = values[state['world']]
    mj.mj_resetData(model,data)
    data.qpos[:],data.qvel[:] = state['qpos'],state['qvel']
    duty,supply = effort['voltage_duty'],effort['supply_v']
    kt,r,cap = (meta[k] for k in ('torque_constant_nm_per_a','resistance_ohm','pwm_limit'))
    captured = []
    for index in range(spec['steps_per_rollout']):
        back = kt*data.qvel[engine.dofs]
        gain = kt/r
        raw = (duty[index]*supply[index]-back)*gain
        baseline = np.clip(raw,engine.lower,engine.upper)
        # The affected motors receive a bounded request after the original delay.
        ctrl = raw.copy()
        if spec['delay_substeps'] <= index < spec['commanded_pulse_substeps']:
            vmax = supply[index]*cap
            lo,hi = (np.clip((v-back)*gain,engine.lower,engine.upper) for v in (-vmax,vmax))
            use = offsets!=0
            ctrl[use] = np.clip(baseline[use]+offsets[use],lo[use],hi[use])
        data.ctrl[engine.acts] = ctrl
        if index==0:
            np.testing.assert_allclose(raw,effort['requested_torque_nm'][0],atol=5e-6,rtol=0)
        mj.mj_step(model,data)
        actual = data.actuator_force[engine.acts].copy()
        np.testing.assert_allclose(actual,np.clip(ctrl,engine.lower,engine.upper),atol=5e-8,rtol=0)
        pairs,normals = [],[]
        for j in range(data.ncon):
            contact = data.contact[j]
            if contact.efc_address>=0 and engine.floor in contact.geom:
                mj.mj_contactForce(model,data,j,engine.force)
                pairs.append(contact.geom.copy())
                normals.append(engine.force[0])
        _, force = reduce_contacts(np.asarray(pairs,dtype=np.int32).reshape(-1,2),np.zeros(len(pairs),dtype=np.int32),
            np.asarray(normals),engine.floor,engine.contact_kinds,1)
        captured.append((data.qpos.copy(), force[0], actual-baseline))
    assert abs(data.time-len(captured)*spec['physics_dt'])<1e-12
    position,force,delta = (np.stack(channel) for channel in zip(*captured))
    assert np.isfinite(position).all() and np.isfinite(force).all()
    assert float(np.abs(delta).sum(axis=1).max()) <= spec['maximum_sum_torque_change_nm']+1e-7
    return position,force,delta


def metrics(position, force, weight, scope, measure):
    tilt = authority.tilt(position)
    threshold = weight*scope['nominal_load_fraction']
    feet = (force[:,2:]>threshold).all(axis=1) & (force[:,2:].sum(axis=1)>=weight*scope['both_feet_fraction'])
    first = measured.first_streak(feet,scope['minimum_load_solves'])
    part = slice(measure-19,measure+1)
    return dict(confirm_ms=None if first is None else (first+1)*4, tilt_deg=float(tilt[measure]),
        mean_head_weight_fraction=float(force[part,1].mean()/weight), feet_loaded_sample_count=int(feet[part].sum()))


def compare_case(data, trace, score, world, spec, scope, meta, engine):
    i = scope['selected_worlds'].index(world)
    confirm = score['first_bilateral_support_ms']
    assert confirm is not None
    step = (confirm-spec['start_offset_before_support_ms'])//20
    start = 5*step
    end = start+spec['steps_per_rollout']
    assert step > 0 and end<=len(data['substep'])
    qpos,qvel = (trace[key][step-1,world] for key in ('qpos','qvel'))
    np.testing.assert_array_equal(qpos[3:7],data['orientation_before'][start,i])
    state = dict(world=world,qpos=qpos,qvel=qvel)
    archive = {name:data[name][start:end,i] for name in ('voltage_duty','supply_v','requested_torque_nm')}
    names = meta['joint_order']
    offsets = {name:np.array([joint.get(j,0.) for j in names]) for name,joint in spec['profiles_nm'].items()}
    zero = np.zeros(len(names))
    baseline,replicate = (rollout(engine,spec,state,archive,meta,zero) for _ in range(2))
    for a,b in zip(baseline,replicate):
        np.testing.assert_array_equal(a,b)
    endpoint = (confirm+spec['measure_after_support_ms'])//4-start-1
    assert 19<=endpoint<spec['steps_per_rollout']
    site = metrics(data['orientation_after'][start:end,i],data['normal_force_n'][start:end,i],engine.weight,scope,endpoint)
    ref = metrics(baseline[0][:,3:7],baseline[1],engine.weight,scope,endpoint)
    aligned = (ref['confirm_ms'] is not None and abs(ref['confirm_ms']-(confirm-step*20))<=spec['maximum_reference_support_delay_ms']
        and abs(ref['tilt_deg']-site['tilt_deg'])<=spec['maximum_reference_tilt_error_deg'])
    result = dict(world=world,start_control=experiment.START+step,archive_first_support_ms=confirm,observed=site,
        repeated_cpu_reference=ref,aligned=bool(aligned),profiles={})
    for name,vector in offsets.items():
        pose,force,delta = rollout(engine,spec,state,archive,meta,vector)
        for old,new in zip(baseline,(pose,force,delta)):
            np.testing.assert_array_equal(old[:1],new[:1])
        row = metrics(pose[:,3:7],force,engine.weight,scope,endpoint)
        delivered = float(np.abs(delta).sum(axis=1).max())
        tilt_gain = ref['tilt_deg']-row['tilt_deg']
        head_gain = ref['mean_head_weight_fraction']-row['mean_head_weight_fraction']
        feet_loss_ms = (ref['feet_loaded_sample_count']-row['feet_loaded_sample_count'])*4
        protected = (aligned and tilt_gain>=-spec['maximum_protected_tilt_loss_deg']
            and head_gain>=-spec['allowed_head_load_increase_body_weights'] and feet_loss_ms<=spec['maximum_lost_protected_support_ms'])
        improved = protected and tilt_gain>=spec['minimum_target_tilt_gain_deg'] and delivered>=.6*np.abs(vector).sum()
        result['profiles'][name] = dict(**row,delivered_peak_l1_nm=delivered,tilt_gain_deg=tilt_gain,
            head_weight_fraction_improvement=head_gain,feet_support_loss_ms=feet_loss_ms,
            protected=bool(protected),improved=bool(improved))
    return result


def score():
    project = read(PROJECT)
    node,task = project['goal_execution'][KEY],project['staged_training_request']
    assert experiment.parent.allowed(project,STAGE) and not task['simulation_running'] and node['status']=='frozen'
    spec = read(PLAN,node['protocol_sha256'])
    for key,value in spec['sources'].items():
        assert sha(ROOT/key)==value
    old,plan,batch,evidence = (read(ROOT/spec[key]) for key in
        ('archived_authority_protocol','capture_protocol','capture_collection','capture_result'))
    for path,checksum in old['sources'].items():
        assert sha(ROOT/path)==checksum
    source,meta = (read(ROOT/old[key]) for key in ('prior_experiment','load_metadata'))
    engine = authority.FixedStateEngine(old,source,meta)
    scope = plan['forward_support']
    folder = ROOT/plan['record']
    roles,cases = measured.ROLES,{}
    for role in roles:
        arm = evidence['roles'][role]
        met = read(ROOT/arm['metadata'],arm['metadata_sha256'])
        assert met['joint_order']==meta['joint_order'] and met['nominal_weight_n']==meta['nominal_weight_n']
        path = ROOT/arm['source']
        assert sha(path)==arm['source_sha256']
        with np.load(path,allow_pickle=False) as raw:
            signals = dict(raw)
        trace = measured.read_trace(folder/role,batch['branches'][role]['chunks'],experiment.START+1,experiment.END,('qpos','qvel'),sha)
        worlds = spec['focus']+spec['protected_archived_admitted'] if role in roles[:2] else spec['focus']
        cases[role] = [compare_case(signals,trace,arm['selected_world_readback'][str(w)],w,spec,scope,met,engine)
                       for w in worlds]
    stats = {}
    for name in spec['profiles_nm']:
        focus = [row for group in cases.values() for row in group if row['world'] in spec['focus']]
        improved = {str(w):sum(row['profiles'][name]['improved'] for row in focus if row['world']==w) for w in spec['focus']}
        retained = {role:sum(row['profiles'][name]['protected'] for row in cases[role]
                             if row['world'] in spec['protected_archived_admitted']) for role in roles[:2]}
        eligible = (sum(improved.values())>=spec['minimum_target_success_all']
            and min(improved.values())>=spec['minimum_target_success_each_world']
            and min(retained.values())>=spec['minimum_protected_each_parent'])
        stats[name] = dict(improved_by_focus=improved,protected_by_parent=retained,eligible_for_observable_preflight=bool(eligible))
    count = sum(len(x) for x in cases.values())
    results = REPORT/'analysis.json'
    write(results,dict(status='reviewed-local-physics',protocol_sha256=node['protocol_sha256'],cases=cases,summary=stats,
        cases_total=count,aligned_reference_cases=sum(row['aligned'] for group in cases.values() for row in group),
        cpu_physics_steps=count*(len(spec['profiles_nm'])+2)*spec['steps_per_rollout'],
        learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed',result=rel(results),result_sha256=sha(results))
    criteria = 'Check the protected physical response and derive a controller-available trigger.' if any(
        r['eligible_for_observable_preflight'] for r in stats.values()) else (
        'Use the measured first-transfer force and tilt to qualify a different intervention. Preserve the original full-route gates.')
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-reviewed',results,criteria)
    write(PROJECT,project)
    print('FORWARD_SUPPORT_TORQUE_REPLAY',sha(results),stats,flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score'))
    action = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'score':score}[action]()


if __name__ == '__main__':
    main()
