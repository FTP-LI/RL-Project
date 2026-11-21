"""Measure which bounded actuator changes can improve early get-up support."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_hip_knee_authority import load_data, tilt
from selective_friction_guard import RESET_FIELDS, STATIC_FIELDS, resolve_foot_geometry_ids


KEY = 'nominal_low_friction_early_transfer_authority'
STAGE = 'nominal-low-friction-early-transfer-authority'
PREVIOUS = 'nominal-low-friction-early-left-turn-load'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-early-transfer-authority'
OUTPUT = REPORT/'analysis.json'
JOINTS = ('left_hip_roll','left_hip_pitch','left_knee','left_ankle',
          'right_hip_roll','right_hip_pitch','right_knee','right_ankle')


def freeze():
    project = read(PROJECT)
    permitted(project,PREVIOUS)
    goal = project['goal_execution']
    early = goal['nominal_low_friction_early_left_turn_load']
    load = goal['nominal_low_friction_support_effort_limit_readback']
    assert early['status']==load['status']=='reviewed'
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    phases = read(ROOT/early['protocol'],early['protocol_sha256'])
    assert read(ROOT/early['result'],early['result_sha256'])['first_support_shift_window_ms']=='400-480'
    original = read(ROOT/load['protocol'],load['protocol_sha256'])
    collection = ROOT/load['collection']
    raw = read(collection,load['collection_sha256'])
    geometry = original['friction_development']['startup_fields']
    positions, = (part for part in raw['chunks'] if part['first']==101)
    metadata = collection.parent/raw['support_effort']['path']
    contacts, = (part for part in read(metadata,raw['support_effort']['sha256'])['windows'] if part['window']=='release')
    pose_trace = rel(collection.parent/positions['file'])
    load_trace = rel(collection.parent/contacts['file'])
    sources = {node[key]:node[key+'_sha256'] for node,keys in ((early,('protocol','result')), (load,('protocol','collection','result')))
               for key in keys}
    sources.update({geometry:original['references'][geometry],pose_trace:positions['sha256'],load_trace:contacts['sha256'],
        rel(metadata):raw['support_effort']['sha256']})
    for helper in ('review_nominal_low_friction_hip_knee_authority.py','roll_contact_clock_capture.py','run_nominal_low_friction_support_effort.py'):
        path = Path(__file__).with_name(helper)
        sources[rel(path)] = sha(path)
    for path,digest in sources.items():
        assert sha(ROOT/path)==digest,path
    design = dict(stage=STAGE,seed=1257,friction=.8,release_control=100,worlds=896,
        prior_experiment=load['protocol'],effort_result=load['result'],collection=load['collection'],
        pose_trace=pose_trace,load_trace=load_trace,load_metadata=rel(metadata),startup_fields=geometry,
        sample_controls=[120,121,122,123],focus_missed_worlds=phases['worlds'],pairs=13,joints=JOINTS,
        single_joint_trim_nm=.03,maximum_joint_trim_nm=.06,maximum_staged_trim_l1_nm=.12,
        empirical_fraction=.5,minimum_consistent_pair_signs=3,minimum_delivered_trim_nm=.005,
        empirical_groups={'right-hip-knee':['right_hip_pitch','right_knee'], 'both-sagittal':[
            'left_hip_pitch','left_knee','left_ankle','right_hip_pitch','right_knee','right_ankle'],
            'both-hip-roll':['left_hip_roll','right_hip_roll']},
        minimum_upright_gain_deg_s2=50.,maximum_protected_acceleration_loss_deg_s2=50.,
        minimum_rightward_load_shift_body_weights=.03,maximum_head_load_increase_body_weights=.1,
        minimum_matching_baseline_contact_states=100,minimum_focus_pairs_with_rotation=3,
        minimum_rotation_states_per_focus_pair=2,minimum_focus_pairs_with_rotation_and_load_change=2,
        minimum_protected_admitted_pairs=12,reference_supply_v=7.0,
        sources=sources,script_sha256=sha(Path(__file__)),additional_physical_steps=0,
        learning_updates=0,reserved_seed=2246, hardware_io=False)
    REPORT.mkdir()
    write(PLAN,design)
    goal[KEY] = dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify the 400–480 ms measured torques and stable load response at archived contact states.')
    write(PROJECT,project)
    print('EARLY_TRANSFER_AUTHORITY_FROZEN',sha(PLAN),flush=True)


def candidate_profiles(spec, snapshots, focus):
    joints = spec['joints']
    nk = len(spec['sample_controls'])
    profiles = {}
    for j,name in enumerate(joints):
        for sign in (-1,1):
            vector = np.zeros((nk,len(joints)))
            vector[:,j] = sign*spec['single_joint_trim_nm']
            profiles[f'{name}/{sign:+d}'] = vector
    measured = np.asarray([[row['admitted'][k]['applied_focus']-row['missed'][k]['applied_focus'] for k in range(nk)]
                           for row in snapshots if row['missed'][0]['world'] in focus])
    median = np.median(measured,axis=0)
    sign_count = (np.sign(measured)==np.sign(median)).sum(axis=0)
    trim = np.clip(median*spec['empirical_fraction'],-spec['maximum_joint_trim_nm'],spec['maximum_joint_trim_nm'])
    trim[(sign_count<spec['minimum_consistent_pair_signs'])|(abs(trim)<spec['minimum_delivered_trim_nm'])] = 0.
    for name,group in spec['empirical_groups'].items():
        vector = np.where(np.isin(joints,group),trim,0.)
        sums = np.abs(vector).sum(axis=1,keepdims=True)
        profiles[name] = vector*(spec['maximum_staged_trim_l1_nm']/np.maximum(sums,spec['maximum_staged_trim_l1_nm']))
    return profiles, median, sign_count


class FixedStateEngine:
    def __init__(self,spec,original,metadata):
        import mujoco
        from mjlab.scene import Scene
        from roll_contact_clock_capture import contact_layout
        from run_nominal_low_friction_support_effort import config

        self.mj = mujoco
        cfg = config(original)
        self.model = Scene(cfg.scene,device='cpu').compile()
        cfg.sim.mujoco.apply(self.model)
        model = self.model
        assert (model.nq,model.nv,model.nu,cfg.decimation)==(21,20,14,5) and model.opt.timestep==.004
        with np.load(ROOT/spec['startup_fields'],allow_pickle=False) as archived:
            for name in sorted(STATIC_FIELDS):
                np.testing.assert_array_equal(getattr(model,name),archived['static_'+name],err_msg=name)
            self.fields = {name:archived['expanded_'+name] for name in RESET_FIELDS}
        feet = resolve_foot_geometry_ids(model)
        np.testing.assert_array_equal(self.fields['geom_friction'][:,feet,0],np.full((spec['worlds'],2),spec['friction'],np.float32))
        names = metadata['joint_order']
        ids = np.array([model.joint('robot/'+name).id for name in names])
        self.acts = np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in ids])
        self.dofs = model.jnt_dofadr[ids]
        self.use = np.array([names.index(name) for name in spec['joints']])
        np.testing.assert_array_equal(model.actuator_gear[self.acts,0],np.ones(model.nu))
        np.testing.assert_array_equal(model.actuator_gainprm[self.acts,0],np.ones(model.nu))
        self.lower,self.upper = model.actuator_forcerange[self.acts].T
        np.testing.assert_allclose(self.upper,metadata['solver_torque_limit_nm'],atol=5e-8,rtol=0)
        self.floor,self.contact_kinds,self.weight = contact_layout(model)
        np.testing.assert_allclose(self.weight,metadata['nominal_weight_n'],atol=1e-9,rtol=0)
        self.threshold = self.weight*original['support_effort']['nominal_load_fraction']
        self.data = mujoco.MjData(model)
        self.force = np.empty(6)
        self.calls = 0

    def solve(self, state, torque):
        from roll_contact_clock_capture import reduce_contacts

        mj,model,data = self.mj,self.model,self.data
        for name,values in self.fields.items():
            getattr(model,name)[:] = values[state['world']]
        mj.mj_resetData(model,data)
        data.qpos[:],data.qvel[:],data.ctrl[self.acts] = state['pose'],state['velocity'],torque
        mj.mj_forward(model,data)
        self.calls += 1
        assert data.time==0 and np.isfinite(data.qacc).all()
        np.testing.assert_array_equal(data.qpos,state['pose'])
        np.testing.assert_array_equal(data.qvel,state['velocity'])
        np.testing.assert_allclose(data.actuator_force[self.acts],np.clip(torque,self.lower,self.upper),atol=5e-8,rtol=0)
        pairs,normal = [],[]
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.efc_address>=0 and self.floor in contact.geom:
                mj.mj_contactForce(model,data,i,self.force)
                pairs.append(contact.geom.copy())
                normal.append(self.force[0])
        _,load = reduce_contacts(np.asarray(pairs,dtype=int).reshape(-1,2),np.zeros(len(pairs),int),
                                 np.asarray(normal),self.floor,self.contact_kinds,1)
        return data.qacc.copy(),load[0]

    def upright_gain(self,pose,delta):
        dt = 1e-5/max(1.,float(np.linalg.norm(delta[3:6])))
        minus,plus = (pose.astype(np.float64) for _ in range(2))
        self.mj.mj_integratePos(self.model,minus,delta,-dt)
        self.mj.mj_integratePos(self.model,plus,delta,dt)
        return float((tilt(minus[3:7])-tilt(plus[3:7]))/(2*dt))


def snapshot(engine,spec,metadata,trace,motor,world,control):
    i,n = control-101,(control-spec['release_control'])*5
    q,dq = trace['qpos'][i,world],trace['qvel'][i,world]
    np.testing.assert_array_equal(q[3:7],motor['orientation_before'][n,world])
    duty = motor['voltage_duty'][n,world].astype(np.float64)
    requested = motor['requested_torque_nm'][n,world].astype(np.float64)
    applied = motor['applied_torque_nm'][n,world].astype(np.float64)
    supply = float(motor['supply_v'][n,world])
    kt,r,cap = (metadata[k] for k in ('torque_constant_nm_per_a','resistance_ohm','pwm_limit'))
    back_emf = kt*dq[engine.dofs].astype(np.float64)
    reconstructed = (duty*supply-back_emf)*kt/r
    error = float(abs(reconstructed-requested).max())
    assert error<=5e-6 and (np.abs(duty)<=cap+1e-6).all()
    np.testing.assert_allclose(applied,np.clip(requested,engine.lower,engine.upper),atol=5e-6,rtol=0)

    def bounds(voltage):
        vmax = voltage*cap
        return tuple(np.clip((value-back_emf)*kt/r, engine.lower,engine.upper) for value in (-vmax,vmax))

    low,high = bounds(supply)
    np.testing.assert_allclose(applied,np.clip(applied,low,high),atol=5e-6,rtol=0)
    rail = spec['reference_supply_v']
    low_rail,high_rail = bounds(rail)
    baseline_rail = np.clip((duty*rail-back_emf)*kt/r,low_rail,high_rail)
    archived = motor['normal_force_n'][n,world].astype(np.float64)
    s = dict(world=world,control=control,pose=q,velocity=dq,duty=duty,applied_focus=applied[engine.use],
        requested=requested,applied=applied,voltage=supply,low=low,high=high,low_rail=low_rail,high_rail=high_rail,
        baseline_rail=baseline_rail,archived_load=archived,voltage_error=error)
    baseline,load = engine.solve(s,requested)
    duplicate,duplicate_load = engine.solve(s,requested)
    np.testing.assert_array_equal(baseline,duplicate)
    np.testing.assert_array_equal(load,duplicate_load)
    matched = np.array_equal(archived[1:]>engine.threshold,load[1:]>engine.threshold)
    s.update(acceleration=baseline,load=load,matched=bool(matched))
    return s


def evaluate_trim(engine,spec,s,trim):
    use = engine.use
    requested = s['requested'].copy()
    desired = s['applied_focus']+trim
    altered = np.clip(desired,s['low'][use],s['high'][use])
    requested[use] = np.where(trim!=0,altered,requested[use])
    actual = np.where(trim!=0,altered-s['applied_focus'],0.)
    rail = s['baseline_rail'][use]
    available_at_rail = np.clip(rail+trim,s['low_rail'][use],s['high_rail'][use])-rail
    assert np.max(np.abs(actual))<=spec['maximum_joint_trim_nm']+5e-6
    signed = np.sign(trim)
    delivered = bool(np.any(trim!=0) and (signed*actual>=spec['minimum_delivered_trim_nm']*(trim!=0)).all()
        and (signed*available_at_rail>=spec['minimum_delivered_trim_nm']*(trim!=0)).all())
    accel,load = engine.solve(s,requested)
    gain = engine.upright_gain(s['pose'],accel-s['acceleration'])
    delta = (load-s['load'])/engine.weight
    foot_lost = bool(((s['load'][2:]>engine.threshold)&(load[2:]<=engine.threshold)).any())
    supports = s['matched'] and not foot_lost and delta[1]<=spec['maximum_head_load_increase_body_weights']
    protection = supports and gain>=-spec['maximum_protected_acceleration_loss_deg_s2']
    rotates = supports and delivered and gain>=spec['minimum_upright_gain_deg_s2']
    shift = float(delta[3]-delta[2])
    return dict(actual_nm=actual.tolist(),available_at_minimum_supply_nm=available_at_rail.tolist(),
        upright_acceleration_gain_deg_s2=gain,rightward_load_shift_body_weights=shift,foot_support_lost=foot_lost,
        head_load_delta_body_weights=float(delta[1]),protected=bool(protection),rotates=bool(rotates),
        rotates_and_shifts=bool(rotates and shift>=spec['minimum_rightward_load_shift_body_weights']))


def evaluate(spec):
    import torch

    assert not torch.cuda.is_initialized()
    original,metadata = (read(ROOT/spec[k]) for k in ('prior_experiment','load_metadata'))
    raw = read(ROOT/spec['collection'])
    assert raw['status']=='collected' and raw['actor_weights_unchanged'] and not raw['reserved_seed_used']
    assert (original['environment']['seed'],original['composition']['foot_friction'])==(spec['seed'],spec['friction'])
    engine = FixedStateEngine(spec,original,metadata)
    trace,motor = load_data(spec)
    matches = read(ROOT/spec['effort_result'])['release_support']['matches']
    assert len(matches)==spec['pairs'] and len({row['admitted']['world'] for row in matches})==spec['pairs']
    focus = set(spec['focus_missed_worlds'])
    assert len(focus)==4 and focus.issubset({row['missed']['world'] for row in matches})
    cases = [{label:[snapshot(engine,spec,metadata,trace,motor,row[label]['world'],k)
                     for k in spec['sample_controls']] for label in ('missed','admitted')} for row in matches]
    states = [s for pair in cases for group in pair.values() for s in group]
    contact_matches = sum(s['matched'] for s in states)
    profiles,median,sign_counts = candidate_profiles(spec,cases,focus)
    results,eligible = {},[]
    for name,targets in profiles.items():
        effects = [{role:[evaluate_trim(engine,spec,s,trim) for s,trim in zip(group,targets,strict=True)]
                    for role,group in pair.items()} for pair in cases]
        focus_effects = [e['missed'] for pair,e in zip(cases,effects,strict=True) if pair['missed'][0]['world'] in focus]
        rotations = sum(sum(s['rotates'] for s in rows)>=spec['minimum_rotation_states_per_focus_pair'] for rows in focus_effects)
        transfers = sum(any(s['rotates_and_shifts'] for s in rows) for rows in focus_effects)
        protected = sum(all(s['protected'] for s in e['admitted']) for e in effects)
        ok = (contact_matches>=spec['minimum_matching_baseline_contact_states']
            and rotations>=spec['minimum_focus_pairs_with_rotation'] and transfers>=spec['minimum_focus_pairs_with_rotation_and_load_change']
            and protected>=spec['minimum_protected_admitted_pairs'])
        magnitude = float(np.max(np.abs(targets).sum(axis=1)))
        results[name] = dict(eligible=bool(ok),target_nm=targets.tolist(),focus_rotation_pairs=rotations,
            focus_rotation_and_transfer_pairs=transfers,retained_pairs=protected,maximum_l1_torque_nm=magnitude,effects=effects)
        if ok:
            eligible.append((-transfers,-rotations,-protected,magnitude,name))
    preferred = sorted(eligible)[0][-1] if eligible else None
    assert engine.calls==(2+len(profiles))*len(states) and not torch.cuda.is_initialized()
    rows = []
    for state in states:
        use = engine.use
        rows.append(dict(world=state['world'],control=state['control'],supply_v=state['voltage'],
            applied_nm=state['applied_focus'].tolist(),pwm_fraction=state['duty'][use].tolist(),
            positive_torque_margin_nm=(state['high'][use]-state['applied_focus']).tolist(),
            negative_torque_margin_nm=(state['applied_focus']-state['low'][use]).tolist(),baseline_contact_match=state['matched'],
            archived_load_n=state['archived_load'].tolist(),reconstructed_load_n=state['load'].tolist()))
    return dict(status='verified-fixed-state-early-transfer-authority',protocol_sha256=sha(PLAN),
        joint_order=spec['joints'],controls=spec['sample_controls'],pairs=[{role:group[0]['world'] for role,group in case.items()} for case in cases],
        states=rows,contact_matches=contact_matches,maximum_torque_reconstruction_error_nm=max(s['voltage_error'] for s in states),
        selected_profile=preferred,measured_median_applied_torque_delta_nm=median.tolist(),consistent_pair_signs=sign_counts.tolist(),
        candidates=results,forward_calls=engine.calls,additional_physical_steps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False)


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for name,digest in spec['sources'].items():
        assert sha(ROOT/name)==digest,name
    result=evaluate(spec)
    preferred=result['selected_profile']
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),selected_profile=preferred,
            report=rel(REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Verify the selected torque change through official joint position inputs and actuator delay, then test paired roll admission.'
            if preferred else 'Use the archived early support response to select a feedback target that preserves original successful episodes.'))
        write(PROJECT,project)
    print('EARLY_TRANSFER_AUTHORITY_VERIFIED',sha(OUTPUT),result['contact_matches'],result['forward_calls'],preferred,
        {name:(row['focus_rotation_pairs'],row['focus_rotation_and_transfer_pairs'],row['retained_pairs'])
         for name,row in result['candidates'].items()},flush=True)


if __name__=='__main__':
    arguments=ArgumentParser(description=__doc__)
    arguments.add_argument('action',choices=('freeze','score','check'))
    option=arguments.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
