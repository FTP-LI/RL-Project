"""Screen a bounded sagittal actuator change at archived get-up states."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_first_support_effort import KEY as EFFORT, SUPPORT_LOAD, tilt
from selective_friction_guard import RESET_FIELDS, STATIC_FIELDS, resolve_foot_geometry_ids


KEY = 'nominal_low_friction_hip_knee_authority'
STAGE = 'nominal-low-friction-hip-knee-authority'
PRIOR_STAGE = 'nominal-low-friction-first-support-effort'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-hip-knee-authority'
JOINTS = ('left_hip_pitch','left_knee','right_hip_pitch','right_knee')
TARGET_OFFSETS_MS = (-120,-40)
MAXIMUM_TRIM_NM = .06


def freeze():
    project = read(PROJECT)
    permitted(project,PRIOR_STAGE)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    effort,load = (goal[k] for k in (EFFORT,SUPPORT_LOAD))
    assert effort['status']==load['status']=='reviewed' and effort['analysis_verified_twice']
    previous = read(ROOT/effort['protocol'],effort['protocol_sha256'])
    result = read(ROOT/effort['result'],effort['result_sha256'])
    study = read(ROOT/load['protocol'],load['protocol_sha256'])
    trace_path = ROOT/load['collection']
    trace = read(trace_path,load['collection_sha256'])
    assert (study['environment']['seed'],study['composition']['foot_friction'])==(1257,.8)
    assert previous['recurrent_missed_worlds']==[152,176,297,404,439,677,779]
    chunk, = (row for row in trace['chunks'] if row['first']==101)
    source = dict(result['release_sensor_trace'])
    instrument = trace['support_effort']
    fields = study['friction_development']['startup_fields']
    paths = {
        effort['protocol']:effort['protocol_sha256'],effort['result']:effort['result_sha256'],
        load['protocol']:load['protocol_sha256'],load['result']:load['result_sha256'],
        load['collection']:load['collection_sha256'],
        rel(trace_path.parent/chunk['file']):chunk['sha256'],
        rel(trace_path.parent/instrument['path']):instrument['sha256'],
        source['path']:source['sha256'],fields:study['references'][fields],
    }
    for path,digest in paths.items():
        assert sha(ROOT/path)==digest,path
    design = dict(status='frozen-zero-step-authority',stage=STAGE,seed=1257,foot_slide=.8,
        prior_experiment=load['protocol'],sampling_collection=load['collection'],effort_result=load['result'],
        pose_trace=rel(trace_path.parent/chunk['file']),load_trace=source['path'],startup_fields=fields,
        load_metadata=rel(trace_path.parent/instrument['path']),recurrent_worlds=previous['recurrent_missed_worlds'],
        matched_pairs=13,control_endpoint_rule='Floor of (individual load-confirmation time plus offset) to 20 ms.',
        measure_at_offsets_ms=TARGET_OFFSETS_MS,joints=JOINTS,maximum_delta_nm=MAXIMUM_TRIM_NM,paired_torque_fraction=.5,
        trim='Use success minus missed actual joint torque at each event-aligned snapshot. Test the same bounded direction on both states.',
        baseline_seed_scope='One development archive; paired static model instances, no prospective route, no policy exposure to landmarks.',
        max_fixed_states=52,forward_calls_per_state=3,
        minimum_tilt_accel_gain_deg_s2=50.,maximum_head_load_increase_body_weights=.1,
        minimum_recurrent_good_pairs=5,minimum_all_good_pairs=9,minimum_recurrent_protected_pairs=6,
        minimum_all_protected_pairs=12,minimum_baseline_contact_matches=48,
        source_sha256=paths,script_sha256=sha(Path(__file__)),actual_physics_steps=0,learning_updates=0,
        hardware_io=False,reserved_seed=2246)
    REPORT.mkdir()
    write(PLAN,design)
    goal[KEY] = dict(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Read actual signed per-joint torques and compare contact-resolved CPU angular acceleration at fixed saved states.')
    write(PROJECT,project)
    print('HIP_KNEE_AUTHORITY_FROZEN',sha(PLAN),design['max_fixed_states'],flush=True)


def load_data(spec):
    with np.load(ROOT/spec['pose_trace'],allow_pickle=False) as f:
        trace = {name:f[name] for name in ('control','qpos','qvel')}
    with np.load(ROOT/spec['load_trace'],allow_pickle=False) as f:
        motor = {name:f[name] for name in ('substep','orientation_before','normal_force_n','requested_torque_nm',
                                          'applied_torque_nm','voltage_duty','supply_v')}
    np.testing.assert_array_equal(trace['control'],np.arange(101,201))
    np.testing.assert_array_equal(motor['substep'],np.arange(501,876))
    np.testing.assert_array_equal(trace['qpos'][:74,:,3:7],motor['orientation_before'][5::5],
        err_msg='4 ms start versus the immediately preceding 20 ms endpoint')
    return trace,motor


def evaluate(spec):
    import mujoco
    from mjlab.scene import Scene
    import torch

    from roll_contact_clock_capture import contact_layout, reduce_contacts
    from run_nominal_low_friction_support_effort import config

    assert not torch.cuda.is_initialized()
    original = read(ROOT/spec['prior_experiment'])
    cfg = config(original)
    assert cfg.sim.mujoco.timestep==.004 and cfg.decimation==5
    plant = Scene(cfg.scene,device='cpu').compile()
    cfg.sim.mujoco.apply(plant)
    assert (plant.nq,plant.nv,plant.nu)==(21,20,14)
    assert plant.opt.timestep==.004
    raw = read(ROOT/spec['sampling_collection'])
    assert raw['status']=='collected' and raw['actor_weights_unchanged'] and not raw['reserved_seed_used']
    meta = read(ROOT/spec['load_metadata'])
    matches = read(ROOT/spec['effort_result'])['release_support']['matches']
    assert len(matches)==spec['matched_pairs'] and len(meta['joint_order'])==plant.nu
    with np.load(ROOT/spec['startup_fields'],allow_pickle=False) as f:
        for name in sorted(STATIC_FIELDS):
            np.testing.assert_array_equal(getattr(plant,name),f['static_'+name],err_msg=name)
        primary = {name:f['expanded_'+name] for name in RESET_FIELDS}
    names = meta['joint_order']
    joint_ids = np.array([plant.joint('robot/'+name).id for name in names])
    acts = np.array([np.flatnonzero(plant.actuator_trnid[:,0]==joint).item() for joint in joint_ids])
    assert (plant.actuator_trntype[acts]==mujoco.mjtTrn.mjTRN_JOINT).all()
    np.testing.assert_array_equal(plant.actuator_gainprm[acts,0],np.ones(plant.nu))
    np.testing.assert_array_equal(plant.actuator_gear[acts,0],np.ones(plant.nu))
    pos,vel = plant.jnt_qposadr[joint_ids],plant.jnt_dofadr[joint_ids]
    lo,hi = plant.actuator_forcerange[acts].T
    np.testing.assert_allclose(hi,meta['solver_torque_limit_nm'],atol=5e-8,rtol=0)
    np.testing.assert_array_equal(lo,-hi)
    use = np.array([names.index(joint) for joint in spec['joints']])
    floor, kinds, weight = contact_layout(plant)
    np.testing.assert_allclose(weight,meta['nominal_weight_n'],atol=1e-9,rtol=0)
    feet = resolve_foot_geometry_ids(plant)
    np.testing.assert_array_equal(primary['geom_friction'][:,feet,0],np.full((896,2),.8,dtype=np.float32))
    trace,motor = load_data(spec)
    trace_clock = original['support_effort']
    release = original['composition']['release_control']
    assert release==100 and trace_clock['minimum_load_solves']==5
    archived_load = motor['normal_force_n']
    threshold = weight*trace_clock['nominal_load_fraction']
    feet_loaded = archived_load[...,2:]
    double = (feet_loaded>threshold).all(axis=-1)&(feet_loaded.sum(axis=-1)>=weight*trace_clock['both_feet_fraction'])
    sustained = np.lib.stride_tricks.sliding_window_view(double,5,axis=0).all(axis=-1)
    confirmation = np.where(sustained.any(axis=0),sustained.argmax(axis=0)+4,-1)
    data = mujoco.MjData(plant)
    work = np.zeros(6)
    forward_calls = 0
    reconstructed_max_error = 0.

    def solve(snapshot, torque):
        nonlocal forward_calls
        w = snapshot['world']
        for name, values in primary.items():
            getattr(plant,name)[:] = values[w]
        mujoco.mj_resetData(plant,data)
        pose,speed = snapshot['pose'],snapshot['velocity']
        data.qpos[:],data.qvel[:],data.ctrl[acts] = pose,speed,torque
        mujoco.mj_forward(plant,data)
        forward_calls += 1
        assert data.time==0. and np.isfinite(data.qacc).all()
        np.testing.assert_array_equal(data.qpos,pose)
        np.testing.assert_array_equal(data.qvel,speed)
        np.testing.assert_allclose(data.actuator_force[acts],np.clip(torque,lo,hi),atol=5e-8,rtol=0)
        pairs,normal = [],[]
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.efc_address<0 or floor not in contact.geom:
                continue
            mujoco.mj_contactForce(plant,data,index,work)
            pairs.append(contact.geom.copy())
            normal.append(work[0])
        _, loads = reduce_contacts(np.asarray(pairs,dtype=int).reshape(-1,2),np.zeros(len(pairs),int),
            np.asarray(normal),floor,kinds,1)
        return data.qacc.copy(),loads[0]

    def snapshot(row, offset):
        nonlocal reconstructed_max_error
        w,ms = row['world'],row['confirmation_elapsed_ms']
        assert confirmation[w]>=0 and ms==(confirmation[w]+1)*4
        elapsed = (ms+offset)//20*20
        k = release+elapsed//20
        n,i = (k-release)*5,k-101
        np.testing.assert_array_equal(trace['qpos'][i,w,3:7],motor['orientation_before'][n,w])
        pose,speed = trace['qpos'][i,w],trace['qvel'][i,w]
        command = motor['requested_torque_nm'][n,w].astype(np.float64)
        applied = motor['applied_torque_nm'][n,w].astype(np.float64)
        kt,r = meta['torque_constant_nm_per_a'],meta['resistance_ohm']
        vin = float(motor['supply_v'][n,w])
        duty = motor['voltage_duty'][n,w].astype(np.float64)
        dq = speed[vel].astype(np.float64)
        expected = (duty*vin-kt*dq)*kt/r
        err = float(abs(expected-command).max())
        assert err<=5e-6 and (abs(duty)<=meta['pwm_limit']+1e-6).all(),(w,k,err)
        np.testing.assert_allclose(applied,np.clip(command,lo,hi),atol=5e-6,rtol=0)
        reconstructed_max_error = max(reconstructed_max_error,err)
        vm = vin*meta['pwm_limit']
        lower = np.clip((-vm-kt*dq)*kt/r,lo,hi)
        upper = np.clip((vm-kt*dq)*kt/r,lo,hi)
        assert (lower<=upper).all()
        np.testing.assert_allclose(applied,np.clip(applied,lower,upper),atol=5e-6,rtol=0)
        return dict(world=w,control=k,time_from_confirmation_ms=elapsed-ms,pose=pose,velocity=speed,
            requested=command,applied=applied,duty=duty,lower=lower,upper=upper,
            motion_rad=trace['qpos'][i+1,w,pos]-pose[pos],contacts=archived_load[n,w])

    def measure(snap, trim):
        reference,ref_load = solve(snap,snap['requested'])
        repeat,rep_load = solve(snap,snap['requested'])
        np.testing.assert_array_equal(repeat,reference)
        np.testing.assert_array_equal(rep_load,ref_load)
        applied = snap['applied']
        alternative = snap['requested'].copy()
        alternative[use] = np.clip(applied[use]+trim,snap['lower'][use],snap['upper'][use])
        actual_change = alternative[use]-applied[use]
        assert (abs(actual_change)<=spec['maximum_delta_nm']+5e-6).all()
        modified,load = solve(snap,alternative)
        delta = modified-reference
        eps = 1e-5/max(1.,float(np.linalg.norm(delta[3:6])))
        minus = snap['pose'].astype(np.float64)
        plus = minus.copy()
        mujoco.mj_integratePos(plant,minus,delta,-eps)
        mujoco.mj_integratePos(plant,plus,delta,eps)
        tilt_accel_gain = float((tilt(minus[3:7])-tilt(plus[3:7]))/(2*eps))
        same = bool(np.array_equal((ref_load>threshold)[1:],(snap['contacts']>threshold)[1:]))
        head_delta = float((load[1]-ref_load[1])/weight)
        foot_lost = bool(((ref_load[2:]>threshold)&(load[2:]<=threshold)).any())
        supports = not foot_lost and head_delta<=spec['maximum_head_load_increase_body_weights']
        return dict(world=snap['world'],control=snap['control'],offset_ms=snap['time_from_confirmation_ms'],
            tilt_deg=float(tilt(snap['pose'][3:7])),baseline_applied_nm=applied[use].tolist(),
            measured_duty=snap['duty'][use].tolist(),measured_next_20_ms_joint_motion_rad=snap['motion_rad'][use].tolist(),
            torque_margin_negative_nm=np.maximum(0.,applied[use]-snap['lower'][use]).tolist(),
            torque_margin_positive_nm=np.maximum(0.,snap['upper'][use]-applied[use]).tolist(),
            proposed_change_nm=trim.tolist(),actual_change_nm=actual_change.tolist(),baseline_contact_match=same,
            archived_load_n=snap['contacts'].tolist(),reference_load_n=ref_load.tolist(),modified_load_n=load.tolist(),
            upright_acceleration_gain_deg_s2=tilt_accel_gain,head_load_delta_body_weights=head_delta,
            foot_support_lost=foot_lost,improved=bool(same and supports and tilt_accel_gain>=spec['minimum_tilt_accel_gain_deg_s2']),
            protected=bool(same and supports and tilt_accel_gain>=-spec['minimum_tilt_accel_gain_deg_s2']))

    cases = []
    for item in matches:
        rows = []
        for offset in spec['measure_at_offsets_ms']:
            missed,admitted = [snapshot(item[k],offset) for k in ('missed','admitted')]
            sent = np.clip((admitted['applied'][use]-missed['applied'][use])*spec['paired_torque_fraction'],
                           -spec['maximum_delta_nm'],spec['maximum_delta_nm'])
            rows.append(dict(nominal_offset_ms=offset,missed=measure(missed,sent),admitted=measure(admitted,sent)))
        cases.append(dict(missed=item['missed']['world'],admitted=item['admitted']['world'],
            recurrent=item['missed']['world'] in spec['recurrent_worlds'],samples=rows,
            improved_in_at_least_one_window=any(row['missed']['improved'] for row in rows),
            protected_in_both_windows=all(row['admitted']['protected'] for row in rows)))
    assert len(cases)*2*len(spec['measure_at_offsets_ms'])==spec['max_fixed_states']
    assert forward_calls==spec['max_fixed_states']*spec['forward_calls_per_state'] and data.time==0.
    assert not torch.cuda.is_initialized()
    matched = sum(row[role]['baseline_contact_match'] for item in cases for row in item['samples']
        for role in ('missed','admitted'))
    summaries = {}
    for label,population in (('all',cases),('recurrent',[item for item in cases if item['recurrent']])):
        summaries[label] = dict(pairs=len(population),improved_pairs=sum(x['improved_in_at_least_one_window'] for x in population),
            protected_pairs=sum(x['protected_in_both_windows'] for x in population))
    admitted = (matched>=spec['minimum_baseline_contact_matches']
        and summaries['all']['improved_pairs']>=spec['minimum_all_good_pairs']
        and summaries['recurrent']['improved_pairs']>=spec['minimum_recurrent_good_pairs']
        and summaries['all']['protected_pairs']>=spec['minimum_all_protected_pairs']
        and summaries['recurrent']['protected_pairs']>=spec['minimum_recurrent_protected_pairs'])
    return dict(status='verified-archived-same-state-actuator-authority',protocol_sha256=sha(PLAN),
        tools=dict(mujoco=mujoco.__version__,gpu_initialized=False),joints=list(spec['joints']),world_seed=spec['seed'],
        forward_calls=forward_calls,archives=spec['source_sha256'],maximum_voltage_torque_reconstruction_error_nm=reconstructed_max_error,
        baseline_contact_layout_matches=matched,cases=cases,cohorts=summaries,eligible_for_position_action_preflight=admitted,
        additional_physical_steps=0,reserved_seed_used=False,learning_updates=0,hardware_io=False)


def score(check=False):
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'zero-ready')
    if not check:
        permitted(project,STAGE)
    spec = read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,digest in spec['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    result = evaluate(spec)
    out = REPORT/'analysis.json'
    if check:
        assert result==read(out,node['result_sha256'])
    else:
        assert not out.exists()
        write(out,result)
        node.update(status='reviewed',result=rel(out),result_sha256=sha(out),position_preflight_admitted=result['eligible_for_position_action_preflight'])
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(out))
        project.update(info)
        gate = ('Reconstruct an actual joint-position action through the modeled servo delay and control bounds.'
            if result['eligible_for_position_action_preflight'] else
            'Use the per-state failure mechanisms to select a different candidate or proceed with the independent nominal perturbation phase.')
        project['staged_training_request'].update(info,next_gate=gate)
        write(PROJECT,project)
    print('HIP_KNEE_AUTHORITY_VERIFIED',sha(out),result['baseline_contact_layout_matches'],
        result['cohorts'],result['eligible_for_position_action_preflight'],flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    args = cli.parse_args()
    freeze() if args.action=='freeze' else score(args.action=='check')
