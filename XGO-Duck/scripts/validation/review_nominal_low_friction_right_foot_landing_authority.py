"""Check unloaded right-foot motion while preserving the original left support."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import review_nominal_low_friction_early_transfer_authority as prior
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY = 'nominal_low_friction_right_foot_landing_authority'
STAGE = 'nominal-low-friction-right-foot-landing-authority'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
OUTPUT = prior.REPORT/'right-foot-landing.json'


def freeze():
    project = read(PROJECT)
    permitted(project,prior.STAGE)
    goal = project['goal_execution']
    previous = goal[prior.KEY]
    assert previous['status']=='reviewed' and previous['selected_profile'] is None and KEY not in goal
    assert not PLAN.exists() and not OUTPUT.exists()
    meta = read(ROOT/previous['protocol'],previous['protocol_sha256'])
    evidence = read(ROOT/previous['result'],previous['result_sha256'])
    qualifying_rotation = [name for name,row in evidence['candidates'].items()
        if row['focus_rotation_pairs']>=meta['minimum_focus_pairs_with_rotation']
        and row['retained_pairs']>=meta['minimum_protected_admitted_pairs']]
    assert qualifying_rotation
    write(PLAN,dict(stage=STAGE,authority_protocol=previous['protocol'],authority_protocol_sha256=previous['protocol_sha256'],
        authority_result=previous['result'],authority_result_sha256=previous['result_sha256'],
        authority_script_sha256=meta['script_sha256'],script_sha256=sha(Path(__file__)),
        profiles=qualifying_rotation,lowest_vertex_band_m=.00025,minimum_clearance_m=.00025,
        minimum_downward_acceleration_gain_m_s2=.15,minimum_qualifying_states_per_focus_pair=2,
        minimum_focus_pairs=3,minimum_protected_admitted_pairs=12,right_load_readback_consecutive=3,
        additional_physical_steps=0,learning_updates=0,reserved_seed=2246,hardware_io=False))
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    goal[KEY] = dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    project.update(info)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Measure added downward acceleration at the free right foot with the left foot supported.')
    write(PROJECT,project)
    print('RIGHT_FOOT_LANDING_AUTHORITY_FROZEN',sha(PLAN),qualifying_rotation,flush=True)


def mesh_geometry(engine,spec):
    model,data = engine.model,engine.data
    geom = model.geom('robot/right_foot_collision').id
    mesh = int(model.geom_dataid[geom])
    assert int(model.geom_type[geom])==int(engine.mj.mjtGeom.mjGEOM_MESH) and mesh>=0
    offset,size = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
    mesh_points = model.mesh_vert[offset:offset+size].astype(np.float64)
    plane = data.geom_xpos[engine.floor]
    normal = data.geom_xmat[engine.floor].reshape(3,3)[:,2]
    np.testing.assert_allclose(normal,[0.,0.,1.],rtol=0,atol=1e-10)
    points = mesh_points@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    gap = (points-plane)@normal
    lowest = float(gap.min())
    near = points[gap<=lowest+spec['lowest_vertex_band_m']]
    linear = np.empty((3,model.nv))
    plane_jacobian = []
    for point in near:
        engine.mj.mj_jac(model,data,linear,None,point,int(model.geom_bodyid[geom]))
        plane_jacobian.append(normal@linear)
    return lowest,np.asarray(plane_jacobian)


def evaluate(spec):
    import torch

    old = read(ROOT/spec['authority_protocol'],spec['authority_protocol_sha256'])
    results = read(ROOT/spec['authority_result'],spec['authority_result_sha256'])
    assert sha(Path(prior.__file__))==spec['authority_script_sha256']==old['script_sha256']
    for path,digest in old['sources'].items():
        assert sha(ROOT/path)==digest,path
    original,metadata = (read(ROOT/old[k]) for k in ('prior_experiment','load_metadata'))
    engine = prior.FixedStateEngine(old,original,metadata)
    trace,motor = prior.load_data(old)
    start = 5*(old['sample_controls'][0]-old['release_control'])
    window = slice(start,start+5*len(old['sample_controls']))
    signals = {}
    for world in old['focus_missed_worlds']:
        force=motor['normal_force_n'][:,world]
        left = (force[window,2]>engine.threshold)
        right = (force[:,3]>engine.threshold)
        n = spec['right_load_readback_consecutive']
        found = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(right[start:],n).all(axis=1))
        assert found.size and not right[window].any()
        signals[str(world)] = dict(left_loaded_reads=int(left.sum()),right_loaded_reads=int(right[window].sum()),
            window_reads=len(left),right_sustained_load_confirmed_ms=int(4*(start+found[0]+n)))
    evaluated = {name:{} for name in spec['profiles']}
    for i,pair in enumerate(results['pairs']):
        w=pair['missed']
        if w not in old['focus_missed_worlds']:
            continue
        for j,k in enumerate(old['sample_controls']):
            state = prior.snapshot(engine,old,metadata,trace,motor,w,k)
            assert state['matched'] and state['load'][2]>engine.threshold and state['load'][3]<=engine.threshold
            clearance,jacobian = mesh_geometry(engine,spec)
            baseline_normal_velocity = jacobian@state['velocity']
            for name in spec['profiles']:
                original_response = results['candidates'][name]['effects'][i]['missed'][j]
                delta = np.asarray(original_response['actual_nm'])
                tuned = state['requested'].copy()
                use = engine.use[delta!=0.]
                tuned[use] = state['applied'][use]+delta[delta!=0.]
                acc,force = engine.solve(state,tuned)
                vertical_change=jacobian@(acc-state['acceleration'])
                gain = -float(vertical_change.max())
                well_supported = force[2]>engine.threshold and not original_response['foot_support_lost']
                ok = (original_response['rotates'] and well_supported and clearance>=spec['minimum_clearance_m']
                    and gain>=spec['minimum_downward_acceleration_gain_m_s2'])
                evaluated[name].setdefault(str(w),[]).append(dict(control=k,mesh_min_clearance_mm=1000*clearance,
                    tested_lowest_vertices=len(jacobian),baseline_worst_vertex_vertical_velocity_m_s=float(baseline_normal_velocity.max()),
                    minimum_added_downward_acceleration_m_s2=gain,left_loaded_after=bool(well_supported),
                    right_loaded_after=bool(force[3]>engine.threshold),qualified=bool(ok)))
    scores = {}
    ordering = []
    for name,rows in evaluated.items():
        qualified = sum(sum(row['qualified'] for row in states)>=spec['minimum_qualifying_states_per_focus_pair'] for states in rows.values())
        gains = [point['minimum_added_downward_acceleration_m_s2'] for states in rows.values() for point in states if point['qualified']]
        profile = results['candidates'][name]
        protected=profile['retained_pairs']
        median=float(np.median(gains)) if gains else None
        eligible = qualified>=spec['minimum_focus_pairs'] and protected>=spec['minimum_protected_admitted_pairs']
        scores[name]=dict(qualified_focus_pairs=qualified,protected_admitted_pairs=protected,qualifying_endpoint_gain_median_m_s2=median,
            eligible=bool(eligible),worlds=rows)
        if eligible:
            ordering.append((-qualified,profile['maximum_l1_torque_nm'],-median,name))
    assert not torch.cuda.is_initialized() and engine.data.time==0
    preferred=min(ordering)[-1] if ordering else None
    return dict(status='verified-right-foot-landing-authority',protocol_sha256=sha(PLAN),authoritative_prior=spec['authority_result_sha256'],
        selected_profile=preferred,observed_foot_loading=signals,profiles=scores,forward_calls=engine.calls,
        additional_physical_steps=0,learning_updates=0,holdout_used=False,hardware_io=False)


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    result=evaluate(spec)
    preferred=result['selected_profile']
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),selected_profile=preferred,
            report=rel(prior.REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Convert and verify the supported right-foot landing change through the production servo position action.' if preferred else
            'Locate the missed right-leg geometry in the official action and compare causal landing timing without a static trim.'))
        write(PROJECT,project)
    print('RIGHT_FOOT_LANDING_AUTHORITY_VERIFIED',sha(OUTPUT),preferred,
        {n:(s['qualified_focus_pairs'],s['qualifying_endpoint_gain_median_m_s2']) for n,s in result['profiles'].items()},flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    option=cli.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
