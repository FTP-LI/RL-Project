"""Advance matched archived states using the same measured duty sequence."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import review_nominal_low_friction_early_transfer_authority as authority
import review_nominal_low_friction_observed_foot_height as observed
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY='nominal_low_friction_early_hip_roll_replay'
STAGE='nominal-low-friction-early-hip-roll-replay'
PLAN=ROOT/'configs/experiments'/f'{KEY}.json'
OUTPUT=authority.REPORT/'early-hip-roll-dynamic.json'


def freeze():
    project=read(PROJECT)
    permitted(project,observed.STAGE)
    goal=project['goal_execution']
    geom=goal['nominal_low_friction_right_foot_landing_authority']
    signal=goal[observed.KEY]
    reference=goal[authority.KEY]
    assert geom['status']==signal['status']==reference['status']=='reviewed'
    assert read(ROOT/geom['result'],geom['result_sha256'])['selected_profile']=='left_hip_roll/-1'
    assert read(ROOT/signal['result'],signal['result_sha256'])['qualified_rules']==0
    sources={node[name]:node[name+'_sha256'] for node in (geom,signal,reference) for name in ('protocol','result')}
    assert KEY not in goal and not PLAN.exists() and not OUTPUT.exists()
    write(PLAN,dict(stage=STAGE,sources=sources,script_sha256=sha(Path(__file__)),authority_protocol=reference['protocol'],
        start_control=120,elapsed_from_release_ms=400,substep_ms=4,substeps_per_arm=80,pulse_substeps=20,
        initial_delay_substeps=1,candidate_joint='left_hip_roll',requested_torque_offset_nm=-.03,
        minimum_completed_torque_offset_nm=.012,minimum_doses_per_world=15,minimum_target_gain_ms=8,
        maximum_reference_archived_time_error_ms=20,minimum_reference_aligned_focus=3,minimum_improved_focus=2,
        right_load_consecutive=3,maximum_protection_delay_ms=8,maximum_protection_tilt_degradation_deg=3.,
        maximum_left_support_lost_pulse_reads=1,minimum_protected_admitted=12,holdout_seed=2246,
        scope='Short CPU mechanism diagnostic with archived voltage duty and back-EMF at current joint speed. No policy recomputation.',
        training_updates=0,hardware_io=False))
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Compare measured right-foot loading after the delayed hip torque change at the same archived initial state.')
    write(PROJECT,project)
    print('EARLY_HIP_ROLL_REPLAY_FROZEN',sha(PLAN),flush=True)


def confirmed(values,threshold,spec):
    n=spec['right_load_consecutive']
    indices=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(values>threshold,n).all(axis=-1))
    return None if not indices.size else int(spec['elapsed_from_release_ms']+(indices[0]+n)*spec['substep_ms'])


def rollout(engine,spec,metadata,state,duty,supply,change):
    from roll_contact_clock_capture import reduce_contacts

    m,d=engine.model,engine.data
    for name,values in engine.fields.items():
        getattr(m,name)[:]=values[state['world']]
    engine.mj.mj_resetData(m,d)
    d.qpos[:],d.qvel[:]=state['pose'],state['velocity']
    kt,r,cap=(metadata[k] for k in ('torque_constant_nm_per_a','resistance_ohm','pwm_limit'))
    joint=metadata['joint_order'].index(spec['candidate_joint'])
    count=spec['substeps_per_arm']
    inputs,outputs,poses,velocities=[],[],[],[]
    for i in range(count):
        back_emf=kt*d.qvel[engine.dofs]
        vmax=supply[i]*cap
        requested=(duty[i]*supply[i]-back_emf)*kt/r
        baseline=np.clip(requested,engine.lower,engine.upper)
        low,high=(np.clip((voltage-back_emf)*kt/r,engine.lower,engine.upper) for voltage in (-vmax,vmax))
        result=requested.copy()
        if change and spec['initial_delay_substeps']<=i<spec['pulse_substeps']:
            result[joint]=np.clip(baseline[joint]+spec['requested_torque_offset_nm'],low[joint],high[joint])
        reads=float(np.clip(result[joint],engine.lower[joint],engine.upper[joint])-baseline[joint])
        assert -abs(spec['requested_torque_offset_nm'])-1e-7<=reads<=1e-7
        d.ctrl[engine.acts]=result
        engine.mj.mj_step(m,d)
        np.testing.assert_allclose(d.actuator_force[engine.acts],np.clip(result,engine.lower,engine.upper),atol=5e-8,rtol=0)
        pairs,normals=[],[]
        for j in range(d.ncon):
            contact=d.contact[j]
            if contact.efc_address>=0 and engine.floor in contact.geom:
                engine.mj.mj_contactForce(m,d,j,engine.force)
                pairs.append(contact.geom.copy())
                normals.append(engine.force[0])
        _,forces=reduce_contacts(np.asarray(pairs,dtype=int).reshape(-1,2),np.zeros(len(pairs),int),np.asarray(normals),
                                 engine.floor,engine.contact_kinds,1)
        inputs.append(reads)
        outputs.append(forces[0])
        poses.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
    assert abs(d.time-count*spec['substep_ms']/1000.)<1e-12
    pose,speed,force=np.asarray(poses),np.asarray(velocities),np.asarray(outputs)
    assert np.isfinite(pose).all() and np.isfinite(speed).all() and np.isfinite(force).all()
    return dict(qpos=pose,qvel=speed,force=force,tilt=authority.tilt(pose[:,3:7]),added_nm=np.asarray(inputs))


def evaluate(spec):
    import torch

    path=spec['authority_protocol']
    old=read(ROOT/path,spec['sources'][path])
    assert sha(Path(authority.__file__))==old['script_sha256']
    for path,checksum in old['sources'].items():
        assert sha(ROOT/path)==checksum,path
    original,metadata=(read(ROOT/old[k]) for k in ('prior_experiment','load_metadata'))
    engine=authority.FixedStateEngine(old,original,metadata)
    trace,motor=authority.load_data(old)
    matches=read(ROOT/old['effort_result'])['release_support']['matches']
    assert len(matches)==13 and not torch.cuda.is_initialized()
    focus=set(old['focus_missed_worlds'])
    start=(spec['start_control']-old['release_control'])*5
    part=slice(start,start+spec['substeps_per_arm'])
    results={}
    for match in matches:
        for label in ('missed','admitted'):
            w=match[label]['world']
            assert str(w) not in results
            snapshot=authority.snapshot(engine,old,metadata,trace,motor,w,spec['start_control'])
            duty=motor['voltage_duty'][part,w].astype(np.float64)
            supply=motor['supply_v'][part,w].astype(np.float64)
            original_right=motor['normal_force_n'][part,w,3]
            archived=confirmed(original_right,engine.threshold,spec)
            a,b,c=(rollout(engine,spec,metadata,snapshot,duty,supply,branch) for branch in (False,False,True))
            for key in a:
                np.testing.assert_array_equal(a[key],b[key],err_msg=f'World {w}: {key}')
                np.testing.assert_array_equal(a[key][0],c[key][0],err_msg=f'World {w} first delayed tick: {key}')
            ref=confirmed(a['force'][:,3],engine.threshold,spec)
            alt=confirmed(c['force'][:,3],engine.threshold,spec)
            advance=None if ref is None or alt is None else ref-alt
            aligned=archived is not None and ref is not None and abs(ref-archived)<=spec['maximum_reference_archived_time_error_ms']
            dose=int((c['added_nm']<=-spec['minimum_completed_torque_offset_nm']).sum())
            window=spec['pulse_substeps']
            left_lost=int(((a['force'][:window,2]>engine.threshold)&(c['force'][:window,2]<=engine.threshold)).sum())
            tilt_change=float(c['tilt'][-1]-a['tilt'][-1])
            protected=((ref is None or (alt is not None and advance>=-spec['maximum_protection_delay_ms']))
                       and left_lost<=spec['maximum_left_support_lost_pulse_reads']
                       and tilt_change<=spec['maximum_protection_tilt_degradation_deg'])
            improved=(aligned and advance is not None and advance>=spec['minimum_target_gain_ms']
                      and dose>=spec['minimum_doses_per_world'] and left_lost<=spec['maximum_left_support_lost_pulse_reads'])
            results[str(w)]=dict(world=w,pair_label=label,is_focus=w in focus,archived_right_confirmation_ms=archived,
                reference_right_confirmation_ms=ref,candidate_right_confirmation_ms=alt,advance_ms=advance,reference_alignment=bool(aligned),
                qualifying_torque_reads=dose,candidate_left_support_lost_reads=left_lost,candidate_minus_reference_tilt_deg=tilt_change,
                improved=bool(improved),protected=bool(protected),reference_right_force_n=a['force'][:,3].tolist(),
                candidate_right_force_n=c['force'][:,3].tolist(),candidate_added_torque_nm=c['added_nm'].tolist())
    local=[r for r in results.values() if r['is_focus']]
    aligns=sum(r['reference_alignment'] for r in local)
    improvements=sum(r['improved'] for r in local)
    protections=sum(r['protected'] for r in results.values() if r['pair_label']=='admitted')
    steps=spec['substeps_per_arm']*3*len(results)
    qualified=(aligns>=spec['minimum_reference_aligned_focus'] and improvements>=spec['minimum_improved_focus']
               and protections>=spec['minimum_protected_admitted'])
    return dict(status='verified-short-dynamic-mechanism-replay',protocol_sha256=sha(PLAN),scope=spec['scope'],
        results=results,focus_reference_aligned=aligns,focus_improved=improvements,admitted_protected=protections,
        eligible_for_development_followup=bool(qualified),physics_steps=steps,same_first_delayed_tick=True,
        training_updates=0,holdout_used=False,hardware_io=False)


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,checksum in spec['sources'].items():
        assert sha(ROOT/path)==checksum,path
    result=evaluate(spec)
    qualified=result['eligible_for_development_followup']
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),mechanism_qualified=qualified,
            report=rel(authority.REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Use the verified timing and original policy target to design an observable same-state CUDA position comparison.' if qualified else
            'Retire the local single-hip pulse direction and isolate right-foot recovery through a targeted state-training comparison.'))
        write(PROJECT,project)
    print('EARLY_HIP_ROLL_DYNAMIC_VERIFIED',sha(OUTPUT),result['physics_steps'],result['focus_reference_aligned'],
        result['focus_improved'],result['admitted_protected'],qualified,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    option=cli.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
