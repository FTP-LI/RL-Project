"""Measure sagittal actuator effort around the first sustained bilateral load."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY = 'nominal_low_friction_first_support_effort'
STAGE = 'nominal-low-friction-first-support-effort'
PREVIOUS = 'nominal-low-friction-release-integrator'
SUPPORT_LOAD = 'nominal_low_friction_support_effort_limit_readback'
RELEASE_RESULT = 'nominal_low_friction_release_integrator'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-first-support-effort'
JOINTS = {
    'neck-and-head-pitch':('neck_pitch','head_pitch'),
    'bilateral-hip-pitch':('left_hip_pitch','right_hip_pitch'),
    'bilateral-knee':('left_knee','right_knee'),
    'bilateral-ankle':('left_ankle','right_ankle'),
}
WINDOWS = ((-160,-80),(-80,0),(0,80))


def freeze():
    project=read(PROJECT)
    permitted(project,PREVIOUS)
    goal,task=project['goal_execution'],project['staged_training_request']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    load,release=(goal[key] for key in (SUPPORT_LOAD,RELEASE_RESULT))
    assert load['status']==release['status']=='reviewed'
    assert release['preferred'] is None
    evidence=read(ROOT/release['stage_failure_readback'],release['stage_failure_readback_sha256'])
    upstream=read(ROOT/load['result'],load['result_sha256'])
    matches=upstream['release_support']['matches']
    recurrent=evidence['prior_recurring_supine_all_three_still_no_roll']
    same_load={item['missed']['world'] for item in matches}
    assert (upstream['seed'],upstream['foot_sliding_friction'],len(matches),len(recurrent))==(1257,.8,13,7)
    sources={row[field]:row[field+'_sha256'] for row,fields in (
        (load,('protocol','collection','result')),(release,('stage_failure_readback','verification'))) for field in fields}
    for path,digest in sources.items():
        assert sha(ROOT/path)==digest
    scope=dict(status='frozen-archive-readback',stage=STAGE,seed=1257,foot_slide=.8,
        matched_pair_source=load['result'],sampling_collection=load['collection'],qualified_original_matches=13,
        recurrent_missed_worlds=sorted(same_load.intersection(recurrent)),original_missed_worlds=sorted(same_load),
        joint_groups={k:list(v) for k,v in JOINTS.items()},offset_windows_ms=[list(x) for x in WINDOWS],
        source_sha256=sources,source_script_sha256=sha(Path(__file__)),additional_physics_steps=0,
        learning_updates=0,heldout_datasets_opened=[])
    assert len(scope['recurrent_missed_worlds'])>=4
    REPORT.mkdir()
    write(PLAN,scope)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Quantify upright motion, head contact and sagittal motor effort at fixed windows relative to initial bilateral load.')
    project.update(info)
    write(PROJECT,project)
    print('FIRST_SUPPORT_EFFORT_FROZEN',sha(PLAN),scope['recurrent_missed_worlds'],flush=True)


def tilt(quat):
    q=quat.astype(np.float64)
    q /= np.linalg.norm(q,axis=-1,keepdims=True)
    return np.rad2deg(np.arccos(np.clip(1.-2.*(q[...,1]**2+q[...,2]**2),-1.,1.)))


def summary(rows):
    x=np.array(rows)
    return dict(pairs=int(len(x)),median=float(np.median(x)),
                quartiles=[float(v) for v in np.quantile(x,(.25,.75))],positive_pairs=int((x>0).sum()))


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    plan=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==plan['source_script_sha256']
    for path,digest in plan['source_sha256'].items():
        assert sha(ROOT/path)==digest
    record_path=ROOT/plan['sampling_collection']
    raw=read(record_path)
    load=project['goal_execution'][SUPPORT_LOAD]
    sampled_plan=read(ROOT/load['protocol'],raw['protocol_sha256'])
    assert raw['actor_weights_unchanged'] and not raw['reserved_seed_used']
    assert sampled_plan['composition']['foot_friction']==.8
    measured=raw['support_effort']
    metadata=read(record_path.parent/measured['path'],measured['sha256'])
    window,=(row for row in metadata['windows'] if row['window']=='release')
    source=record_path.parent/window['file']
    assert sha(source)==window['sha256']
    with np.load(source,allow_pickle=False) as saved:
        step,normal,tau,pwm,angles,after=(saved[k] for k in ('substep','normal_force_n','applied_torque_nm',
            'voltage_duty','orientation_before','orientation_after'))
    assert metadata['contact_channels']==['ground','head','left_foot','right_foot']
    np.testing.assert_array_equal(step,np.arange(501,876))
    np.testing.assert_array_equal(angles[1:],after[:-1])
    report=read(ROOT/plan['matched_pair_source'])
    weight=metadata['nominal_weight_n']
    feet=normal[...,2:]
    supported=(feet>.02*weight).all(axis=-1)&(feet.sum(axis=-1)>=.5*weight)
    sustained=np.lib.stride_tricks.sliding_window_view(supported,5,axis=0).all(axis=-1)
    first=np.where(sustained.any(axis=0),sustained.argmax(axis=0)+4,-1)
    assert tau.shape==pwm.shape==(375,896,14)
    assert np.isfinite(tau).all() and np.isfinite(pwm).all()
    assert (abs(pwm)<=metadata['pwm_limit']+1e-6).all()
    indices={name:[metadata['joint_order'].index(j) for j in names] for name,names in JOINTS.items()}

    def event(row):
        world=row['world']
        k=first[world]
        assert k>=0 and (k+1)*4==row['confirmation_elapsed_ms']
        result={}
        t=tilt(np.concatenate((angles[:1,world],after[:,world])))
        for lo,hi in WINDOWS:
            a,b=k+1+lo//4,k+1+hi//4
            assert 0<=a<b<len(t)
            counts=normal[a:b,world]
            torque=tau[a:b,world]
            duty=pwm[a:b,world]
            result[f'{lo}:{hi}']=dict(tilt_drop_deg=float(t[a]-t[b]),
                mean_head_fraction=float(counts[:,1].mean()/weight),
                peak_duty_fraction=float(abs(duty).max()/metadata['pwm_limit']),
                absolute_joint_torque_integral_nms={group:float(abs(torque[:,ids]).sum(dtype=np.float64)*.004)
                    for group,ids in indices.items()})
        return result

    entries=[dict(missed=row['missed']['world'],admitted=row['admitted']['world'],
         missed_readout=event(row['missed']),admitted_readout=event(row['admitted']))
         for row in report['release_support']['matches']]
    all_missed=sorted(row['missed'] for row in entries)
    assert all_missed==plan['original_missed_worlds']
    cohorts={'all':entries,'matches_from_recurrent_release_failures':[x for x in entries
        if x['missed'] in plan['recurrent_missed_worlds']]}
    results={}
    for name,items in cohorts.items():
        values={}
        for low,high in WINDOWS:
            window_key=f'{low}:{high}'
            paired=[(pair['missed_readout'][window_key],pair['admitted_readout'][window_key]) for pair in items]
            results_row={metric:summary([a[metric]-b[metric] for a,b in paired])
                for metric in ('tilt_drop_deg','mean_head_fraction','peak_duty_fraction')}
            results_row['absolute_joint_torque_integral_nms']={group:summary([
                a['absolute_joint_torque_integral_nms'][group]-b['absolute_joint_torque_integral_nms'][group]
                for a,b in paired]) for group in indices}
            values[window_key]=results_row
        results[name]=values
    dest=REPORT/'analysis.json'
    outcome=dict(status='verified-archived-sagittal-effort',protocol_sha256=sha(PLAN),
        release_sensor_trace=dict(path=rel(source),sha256=window['sha256']),paired_delta_definition='missed minus admitted',
        paired_windows=results,pair_readout=entries,additional_physics_steps=0,learning_updates=0,reserved_seed_used=False)
    if check:
        assert read(dest,node['result_sha256'])==outcome
    else:
        assert not dest.exists()
        write(dest,outcome)
        node.update(status='reviewed',result=rel(dest),result_sha256=sha(dest))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(dest))
        project['staged_training_request'].update(info,next_gate='Decide whether to test a targeted early sagittal action using the matched effort and rotation readback.')
        project.update(info)
        write(PROJECT,project)
    print('FIRST_SUPPORT_EFFORT_VERIFIED',sha(dest),len(entries),len(cohorts['matches_from_recurrent_release_failures']),flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    args=cli.parse_args()
    freeze() if args.action=='freeze' else score(args.action=='check')
