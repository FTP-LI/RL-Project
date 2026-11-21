"""Screen an earlier joint combination for upright rotation and right-foot placement."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import review_nominal_low_friction_forward_geometry as geometry
import review_nominal_low_friction_right_foot_landing_authority as landing


measured,previous,authority = geometry.measured,geometry.previous,geometry.authority
experiment = measured.experiment
ROOT, PROJECT, read, rel, sha, write = geometry.ROOT, geometry.PROJECT, geometry.read, geometry.rel, geometry.sha, geometry.write
KEY, STAGE = 'nominal_low_friction_forward_momentum', 'nominal-low-friction-forward-momentum'
PLAN = geometry.PLAN.with_stem(KEY)
REPORT = geometry.REPORT/'momentum-replay'
JOINTS = ('left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch',
    'right_hip_roll','right_hip_pitch','right_knee','right_ankle')


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert experiment.parent.allowed(project,geometry.STAGE) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    node = goal[geometry.KEY]
    config = read(ROOT/node['protocol'],node['protocol_sha256'])
    evidence = read(ROOT/node['result'],node['result_sha256'])
    assert node['status']=='reviewed' and evidence['reconstructed_states']==5304 and not evidence['heldout_used']
    sources = {**config['sources'],node['protocol']:node['protocol_sha256'],node['result']:node['result_sha256']}
    for src in (__file__,previous.__file__,landing.__file__):
        path=Path(src).resolve()
        sources[rel(path)]=sha(path)
    for path,digest in sources.items():
        assert sha(ROOT/path)==digest,path
    spec = dict(stage=STAGE,geometry_protocol=node['protocol'],geometry_result=node['result'],sources=sources,
        start_controls=[118,121],support_reference_compared_to_cuda_first=True,fit_role=measured.ROLES[0],
        evaluate_roles=list(measured.ROLES),joints=list(JOINTS),finite_difference_nm=.02,maximum_joint_trim_nm=.06,
        maximum_sum_torque_change_nm=.16,lowest_vertex_band_m=.00025,minimum_free_foot_gap_mm=2.,
        acceleration_scale_deg_s2=50.,foot_downward_scale_m_s2=.15,maximum_protected_head_load_increase=.10,
        maximum_protected_left_load_loss=.05,minimum_model_left_load=.02,
        delay_substeps=1,commanded_pulse_substeps=20,steps_per_rollout=40,physics_dt=.004,compare_ms=120,
        maximum_aligned_tilt_error_deg=5.,maximum_aligned_right_foot_time_ms=20,minimum_target_tilt_gain_deg=3.,
        maximum_protected_tilt_loss_deg=2.,maximum_protected_head_load_increase_fraction=.025,
        maximum_bilateral_support_loss_ms=8,right_support_dwell_substeps=3,
        minimum_qualifying_target_replays=3,minimum_protect_per_repeat=12,
        heldout_seed=2246,hardware_io=False,learning_updates=0)
    REPORT.mkdir()
    write(PLAN,spec)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',PLAN,
        'Fit a protected multi-joint direction at two pre-support phases, then test its delivered short-horizon response.')
    write(PROJECT,project)
    print('FORWARD_MOMENTUM_FROZEN',sha(PLAN),spec['start_controls'],flush=True)


def prepare(engine, trace, raw, worlds, meta, world, control):
    offset = control-experiment.START
    q,v = [trace[k][offset-1,world] for k in ('qpos','qvel')]
    i = worlds.index(world)
    slot = offset*5
    assert np.array_equal(q[3:7], raw['orientation_before'][slot,i])
    lo_torque,hi_torque = engine.lower,engine.upper
    voltage=raw['supply_v'][slot,i]
    cap,kt,r = [meta[k] for k in ('pwm_limit','torque_constant_nm_per_a','resistance_ohm')]
    emf=kt*v[engine.dofs]
    low,high = [np.clip((sign*cap*voltage-emf)*(kt/r),lo_torque,hi_torque) for sign in (-1,1)]
    requested = raw['requested_torque_nm'][slot,i].astype(np.float64)
    reconstructed=(float(voltage)*raw['voltage_duty'][slot,i]-emf)*(kt/r)
    np.testing.assert_allclose(requested,reconstructed,atol=5e-6,rtol=0)
    return dict(world=world,pose=q,velocity=v,slot=slot,channel=i,requested=requested,applied=np.clip(requested,low,high),
        low=low,high=high)


def fit_profile(engine, spec, scope, meta, trace, raw, pairs, control):
    cols=[meta['joint_order'].index(name) for name in spec['joints']]
    eps=spec['finite_difference_nm']
    axes=np.eye(len(meta['joint_order']))[cols]
    failures,successes=[],[]
    for failure,success in pairs:
        for group,w in ((failures,failure),(successes,success)):
            state=prepare(engine,trace,raw,scope['selected_worlds'],meta,w,control)
            q=state['pose']
            nominal,force=engine.solve(state,state['requested'])
            gap,jac=landing.mesh_geometry(engine,spec)

            def response(offset):
                request=state['requested'].copy()
                moved=offset!=0
                request[moved]=np.clip(state['applied'][moved]+offset[moved],state['low'][moved],state['high'][moved])
                accel,loads=engine.solve(state,request)
                delta=accel-nominal
                return np.array([engine.upright_gain(q,delta),-(jac@delta).max(),
                                 (loads[1]-force[1])/engine.weight,(loads[2]-force[2])/engine.weight])

            slopes=np.stack([(response(eps*axis)-response(-eps*axis))/(2*eps) for axis in axes],axis=1)
            group.append(dict(world=w,gap_mm=gap*1000,left_load=float(force[2]/engine.weight),slopes=slopes))
    free=[row for row in failures if row['gap_mm']>=spec['minimum_free_foot_gap_mm']]
    assert free
    rotation=np.median([row['slopes'][0] for row in failures],axis=0)/spec['acceleration_scale_deg_s2']
    lowering=np.median([row['slopes'][1] for row in free],axis=0)/spec['foot_downward_scale_m_s2']
    # u+,u- represent one signed, common torque vector; z bounds both responses.
    def signed(a,goal=0.):
        return np.r_[a,-a,goal]
    constraints=[signed(-rotation,1.),signed(-lowering,1.),
                 np.r_[np.ones(2*len(cols)),0.]]
    limits=[0.,0.,spec['maximum_sum_torque_change_nm']]
    for row in successes:
        slopes=row['slopes']
        constraints.extend((signed(-slopes[0]),signed(slopes[2])))
        limits.extend((spec['acceleration_scale_deg_s2'],spec['maximum_protected_head_load_increase']))
        if row['left_load']>=spec['minimum_model_left_load']:
            constraints.append(signed(-slopes[3]))
            limits.append(spec['maximum_protected_left_load_loss'])
    cost=np.r_[np.full(2*len(cols),1e-3),-1.]
    result=linprog(cost,A_ub=np.stack(constraints),b_ub=np.array(limits),
        bounds=[(0,spec['maximum_joint_trim_nm'])]*(2*len(cols))+[(0,None)],method='highs')
    assert result.success,result.message
    values=result.x[:len(cols)]-result.x[len(cols):2*len(cols)]
    signed_clear=np.where(abs(values)>1e-8,values,0.)
    vector=np.zeros(len(meta['joint_order']))
    vector[cols]=signed_clear
    assert abs(vector).sum()<=spec['maximum_sum_torque_change_nm']+1e-10
    return vector,dict(control=control,discovery_worlds=len(failures),free_right_foot_worlds=len(free),
        fitted_vector_nm=dict(zip(spec['joints'],signed_clear.tolist(),strict=True)),predicted_rotation_deg_s2=float(
            rotation@signed_clear*spec['acceleration_scale_deg_s2']),
        predicted_right_foot_down_accel_m_s2=float(lowering@signed_clear*spec['foot_downward_scale_m_s2']))


def pulse_check(engine,spec,scope,meta,trace,raw,world,control,vector):
    case=prepare(engine,trace,raw,scope['selected_worlds'],meta,world,control)
    start,i=case['slot'],case['channel']
    last=start+spec['steps_per_rollout']
    measure=spec['compare_ms']//4-1
    effort={field:raw[field][start:last,i] for field in ('voltage_duty','supply_v','requested_torque_nm')}
    state=dict(world=world,qpos=case['pose'],qvel=case['velocity'])
    same,repeat,branch=(previous.rollout(engine,spec,state,effort,meta,extra)
                        for extra in (np.zeros_like(vector),np.zeros_like(vector),vector))
    for a,b,c in zip(same,repeat,branch,strict=True):
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[0],c[0])
    reference=previous.metrics(same[0][:,3:7],same[1],engine.weight,scope,measure)
    result=previous.metrics(branch[0][:,3:7],branch[1],engine.weight,scope,measure)
    archive=previous.metrics(raw['orientation_after'][start:last,i],raw['normal_force_n'][start:last,i],engine.weight,scope,measure)
    def right_elapsed(forces):
        found=measured.first_streak(forces[:measure+1,3]>engine.threshold,spec['right_support_dwell_substeps'])
        return None if found is None else (found+1)*4
    a,b,c=(right_elapsed(force) for force in (raw['normal_force_n'][start:last,i],same[1],branch[1]))
    right_aligned=a==b or (a is not None and b is not None and abs(a-b)<=spec['maximum_aligned_right_foot_time_ms'])
    aligned=right_aligned and abs(reference['tilt_deg']-archive['tilt_deg'])<=spec['maximum_aligned_tilt_error_deg']
    gain=reference['tilt_deg']-result['tilt_deg']
    protection=(aligned and gain>=-spec['maximum_protected_tilt_loss_deg']
        and result['mean_head_weight_fraction']-reference['mean_head_weight_fraction']<=spec['maximum_protected_head_load_increase_fraction']
        and 4*(reference['feet_loaded_sample_count']-result['feet_loaded_sample_count'])<=spec['maximum_bilateral_support_loss_ms'])
    qualifies=(protection and gain>=spec['minimum_target_tilt_gain_deg'] and c is not None and (b is None or c<=b+4))
    return dict(world=world,reference_aligned=bool(aligned),tilt_gain_deg=gain,protected=bool(protection),qualifies=bool(qualifies),
                first_right_support_ms=[a,b,c],archive=archive,baseline=reference,adjusted=result,
                delivered_peak_l1_nm=float(abs(branch[2]).sum(axis=1).max()))


def evaluate():
    project=read(PROJECT)
    node=project['goal_execution'][KEY]
    assert experiment.parent.allowed(project,STAGE) and not project['staged_training_request']['simulation_running']
    assert node['status']=='archive-ready'
    spec=read(PLAN,node['protocol_sha256'])
    for path,digest in spec['sources'].items():
        assert sha(ROOT/path)==digest,path
    geom=read(ROOT/spec['geometry_protocol'])
    old,plan,batch,evidence=(read(ROOT/geom[k]) for k in ('geometry_authority_protocol','earlier_capture_protocol',
        'earlier_capture_collection','earlier_capture_result'))
    for path,digest in old['sources'].items():
        assert sha(ROOT/path)==digest,path
    source,meta=(read(ROOT/old[k]) for k in ('prior_experiment','load_metadata'))
    engine=authority.FixedStateEngine(old,source,meta)
    pairs=[(int(w),v) for w,v in geom['archive_pairs'].items()]
    focus=geom['focus']
    folder=ROOT/plan['record']
    roles=spec['evaluate_roles']
    settings=plan['forward_support']
    responses={str(k):{} for k in spec['start_controls']}
    fitted={}
    for role in roles:
        trace=measured.read_trace(folder/role,batch['branches'][role]['chunks'],experiment.START+1,experiment.END,
            ('qpos','qvel'),sha)
        row=evidence['roles'][role]
        meta=read(ROOT/row['metadata'],row['metadata_sha256'])
        assert sha(ROOT/row['source'])==row['source_sha256']
        with np.load(ROOT/row['source'],allow_pickle=False) as raw:
            signals=dict(raw)
        for control in spec['start_controls']:
            if role==spec['fit_role']:
                fitted[control]=fit_profile(engine,spec,settings,meta,trace,signals,pairs,control)
            v=fitted[control][0]
            worlds=focus+[b for _,b in pairs] if role in roles[:2] else focus
            responses[str(control)][role]=[pulse_check(engine,spec,settings,meta,trace,signals,w,control,v) for w in worlds]
    summary={}
    for k,group in responses.items():
        targeted=[row for items in group.values() for row in items if row['world'] in focus]
        pass_w={w:sum(r['qualifies'] for r in targeted if r['world']==w) for w in focus}
        protect={role:sum(r['protected'] for r in group[role] if r['world'] not in focus) for role in roles[:2]}
        qualifies=(sum(pass_w.values())>=spec['minimum_qualifying_target_replays'] and all(pass_w.values())
                    and min(protect.values())>=spec['minimum_protect_per_repeat'])
        summary[k]=dict(verified_dynamic_gain_by_target=pass_w,protected_by_reference=protect,max_target_tilt_gain_deg=float(
            max(r['tilt_gain_deg'] for r in targeted)),target_references_aligned=sum(r['reference_aligned'] for r in targeted),
            eligible_for_observed_trigger=bool(qualifies))
    dest=REPORT/'analysis.json'
    count=sum(len(rows) for each in responses.values() for rows in each.values())
    write(dest,dict(status='verified-bounded-pre-support-replay',protocol_sha256=node['protocol_sha256'],
        fit={str(k):v[1] for k,v in fitted.items()},summary=summary,arms=responses,fixed_state_solves=engine.calls,
        cpu_physics_steps=count*3*spec['steps_per_rollout'],learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed',result=rel(dest),result_sha256=sha(dest))
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-reviewed',dest,
        'Use the protected early-motion response to decide between an observed trigger and a trajectory-based training candidate.')
    write(PROJECT,project)
    print('FORWARD_MOMENTUM_REPLAY_VERIFIED',sha(dest),summary,flush=True)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','evaluate'))
    mode=cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'evaluate':evaluate}[mode]()


if __name__=='__main__':
    main()
