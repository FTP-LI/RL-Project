"""Separate recurring recovery losses from roll losses using the archived scene bank."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'project.json'
KEY = 'nominal_low_friction_failure_recurrence'
STAGE = 'nominal-low-friction-failure-recurrence'
PREVIOUS = 'nominal-low-friction-support-ppo-action-comparison'
PROTOCOL = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-failure-recurrence'
OUTPUT = REPORT/'analysis.json'
DEVELOPMENT = 'nominal_timestep_low_friction_development'
WINDOW = 'nominal_low_friction_roll_window'
SOURCES = (
    (DEVELOPMENT, ('lower-friction',)),
    ('nominal_supine_reward_friction_trial', ('parent-policy',)),
    (WINDOW, (None,)),
    ('nominal_low_friction_upright_recovery', ('original-first','original-repeat')),
    ('nominal_low_friction_upright_early_return', ('original-first','original-repeat')),
    ('nominal_low_friction_support_effort_limit_readback', (None,)),
    ('nominal_low_friction_support_ppo_action', ('parent-policy',)),
)
FIELDS = ('quaternion','qpos','roll_phase','head_contact','foot_contact')


def sha(path):
    with path.open('rb') as file:
        return file_digest(file,'sha256').hexdigest()


def read(path, checksum=None):
    if checksum is not None:
        assert sha(path)==checksum, path
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    temp.replace(path)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def permitted(project, stage):
    task, goal = project['staged_training_request'], project['goal_execution']
    assert goal['status']=='active' and not goal['pause_requested_by_user'] and task['current_stage']==stage
    assert not any(task[k] for k in ('simulation_running','learning_running','pause_requested','deployment_authorized_now'))


def freeze():
    project=read(PROJECT)
    permitted(project,PREVIOUS)
    goal,task=project['goal_execution'],project['staged_training_request']
    assert KEY not in goal and not (REPORT.exists() or PROTOCOL.exists())
    origins=[]
    for key,roles in SOURCES:
        node=goal[key]
        assert node['status']=='reviewed'
        paths={name:dict(path=node[name],sha256=node[name+'_sha256']) for name in ('protocol','collection','result')}
        for item in paths.values():
            assert sha(ROOT/item['path'])==item['sha256']
        origins.append(dict(key=key,branches=list(roles),**paths))
    old=goal[DEVELOPMENT+'_entry_review']
    latest=goal[WINDOW]
    endpoints=dict(two_ms=dict(path=old['evidence'],sha256=old['evidence_sha256']),
                   first_window_one_ms=dict(path=latest['entry_chronology'],sha256=latest['entry_chronology_sha256']))
    for item in endpoints.values():
        assert sha(ROOT/item['path'])==item['sha256']
    design=dict(status='frozen-before-archive-readback',stage=STAGE,sources=origins,seed=1257,worlds=896,
        foot_slide=.8,release_control=100,standing_readback_controls=600,late_samples=25,
        release_neighborhood_controls=[100,175],held_out=[2256,2246],repeat_rule='First original branch per acquisition.',
        recurring_failure_rule='Actual supine and failed at least four of seven acquisitions.',
        upright_clock_rule='First three 20 ms samples of two-foot contact, clear head and tilt at most 15 degrees.',
        entry_endpoints=endpoints,source_script_sha256=sha(Path(__file__)),physics_steps=0,learning_updates=0)
    REPORT.mkdir()
    write(PROTOCOL,design)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PROTOCOL),protocol_sha256=sha(PROTOCOL),hardware_io=False)
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PROTOCOL))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PROTOCOL),current_protocol_frozen=True,
                next_gate='Trace matched scene IDs across seven parent acquisitions and separate original command admission failure stages.')
    write(PROJECT,project)
    print('LOW_FRICTION_RECURRENCE_FROZEN',sha(PROTOCOL),len(origins),flush=True)


def trace_subset(folder, raw, branch, wanted):
    if 'prefix' in raw:
        parts=[(folder/'shared',raw['prefix']['chunks'])]
        if branch is not None:
            parts.append((folder/branch,raw['branches'][branch]['chunks']))
    else:
        assert branch is None
        parts=[(folder,raw['chunks'])]
    saved={}
    sources={}
    for location,chunks in parts:
        for item in chunks:
            subset=wanted[(wanted>=item['first'])&(wanted<=item['last'])]
            if not len(subset):
                continue
            path=location/item['file']
            assert sha(path)==item['sha256']
            sources[rel(path)]=item['sha256']
            with np.load(path,allow_pickle=False) as data:
                start=item['first']
                np.testing.assert_array_equal(data['control'],np.arange(start,item['last']+1))
                selected={field:data[field][subset-start] for field in FIELDS}
            for position,k in enumerate(subset):
                assert int(k) not in saved
                saved[int(k)]={field:values[position] for field,values in selected.items()}
    assert sorted(saved)==wanted.tolist()
    result={field:np.stack([saved[int(k)][field] for k in wanted]) for field in FIELDS}
    return result,sources


def summarize(traces, controls, poses, groups, prior_score, case, branch):
    q=traces['quaternion'].astype(np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    tilt=np.degrees(np.arccos(np.clip(1.-2.*(q[...,1]**2+q[...,2]**2),-1.,1.)))
    feet=traces['foot_contact'].all(axis=-1)
    head=traces['head_contact']
    entry=traces['roll_phase'][-1]==1  # ROLL in the original phase enum.
    if branch is not None and 'roles' in prior_score:
        expected=prior_score['roles'][branch]['roll_entry']
    elif branch is not None and 'results' in prior_score:
        expected=prior_score['results'][branch]['roll_entry']
    elif case==WINDOW:
        expected=prior_score['results']['early-1ms-candidate']['roll_entry']
    else:
        expected=prior_score['roll_entry']
    assert int(entry.sum())==expected
    supine=poses=='supine'
    failed=supine&~entry
    late=controls>=576
    early=(controls>100)&(controls<=175)
    stable=feet[early]&~head[early]&(tilt[early]<=15.)
    dwell=stable[:-2]&stable[1:-1]&stable[2:]
    time=np.where(dwell.any(axis=0),controls[early][2:][dwell.argmax(axis=0)],-1)
    timing={}
    for label,group in (('admitted',supine&entry),('not_admitted',failed)):
        ids=np.flatnonzero(group)
        timing[label]=dict(scenes=len(ids),qualified_by_175=int((time[ids]>=0).sum()),
            median_tilt_150_deg=float(np.median(tilt[controls==150][:,ids])),
            median_tilt_600_deg=float(np.median(tilt[-1,ids])))
    results=dict(roll_entry=int(entry.sum()),actual_supine=int(supine.sum()),supine_failed=int(failed.sum()),
        assigned_left_roll_entry=int(entry[groups['turn-left']].sum()),
        assigned_forward_roll_entry=int(entry[groups['forward']].sum()),
        late_supine_head_two_feet_and_tilt_fail=int((failed&(head[late]&feet[late]&(tilt[late]>15.)).all(axis=0)).sum()),
        failed_supine_ids=np.flatnonzero(failed).tolist(),early_upright_readback=timing)
    return results,failed,q[0]


def score(check=False):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PROTOCOL,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['source_script_sha256'] and tuple(spec['held_out'])==(2256,2246)
    controls=np.r_[100:176,576:601]
    summary={}
    source_hash={}
    parent_fail,counts,poses,orientations={},[],[],[]
    signature=None
    for source in spec['sources']:
        key=source['key']
        plan,raw,previous=(read(ROOT/source[name]['path'],source[name]['sha256']) for name in ('protocol','collection','result'))
        folder=(ROOT/source['collection']['path']).parent
        assert raw['protocol_sha256']==source['protocol']['sha256']
        assert (plan['environment']['seed'],plan['composition']['worlds'],plan['composition']['foot_friction'])==(1257,896,.8)
        assert plan['composition']['roll_start_control']==600 and plan['composition']['release_control']==100
        assert raw['status']=='collected' and raw['actor_weights_unchanged'] and raw['resets']==1
        assert not any(raw[k] for k in ('hardware_io','learning_updates','reserved_seed_used'))
        assert sha(folder/'initial.npz')==raw['initial_sha256']
        assert sha(folder/'release-poses.npz')==raw['release_sha256']
        with np.load(folder/'release-poses.npz',allow_pickle=False) as file:
            release=file['poses']
        groups=plan['seven_command']['groups']
        models={k:row for k,row in plan['models'].items() if k not in ('getup-supine-control','getup-supine-candidate')}
        current=(raw['initial_sha256'],models,groups,plan['controller'] if 'controller' in plan else plan['sequence'])
        if signature is None:
            signature=current
        else:
            assert current==signature,key
        outcomes={}
        for ordinal,branch in enumerate(source['branches']):
            trace, hashes=trace_subset(folder,raw,branch,controls)
            source_hash.update(hashes)
            result,failed,q0=summarize(trace,controls,release,groups,previous,key,branch)
            outcomes[branch or 'original-before-roll']=result
            if not ordinal:
                parent_fail[key]=np.flatnonzero(failed).tolist()
                counts.append(failed)
                poses.append(release)
                orientations.append(q0)
        summary[key]=outcomes
    failures=np.asarray(counts)
    actual=np.asarray(poses)
    same=np.asarray(orientations)
    assert len(failures)==7
    totals=failures.sum(axis=0)
    common={}
    for world in np.flatnonzero(totals>=4):
        ids=next(name for name,group in groups.items() if int(world) in group)
        angle=2*np.degrees(np.arccos(np.clip(np.abs(same[:,world]@same[-1,world]),0.,1.)))
        common[str(world)]=dict(command=ids,actual_supine_count=int((actual[:,world]=='supine').sum()),
            failed_count=int(totals[world]),acquisitions=[key for key,values in parent_fail.items() if world in values],
            maximum_release_angle_difference_from_latest_deg=float(angle.max()))
    endpoints=spec['entry_endpoints']
    older=read(ROOT/endpoints['two_ms']['path'],endpoints['two_ms']['sha256'])
    recent=read(ROOT/endpoints['first_window_one_ms']['path'],endpoints['first_window_one_ms']['sha256'])
    prior_left=older['commands']['turn-left']
    details={k:dict(world=int(k),stage=v['stage'],supported_peak_deg=v['supported_peak_deg'])
             for k,v in recent['left_candidate_missing'].items()}
    assert prior_left['admitted']==121 and 128-len(details)==120
    results=dict(status='verified-same-scene-bank-readback',protocol_sha256=sha(PROTOCOL),initial_digest=signature[0],
        acquisitions=summary,recurring_supine_failures=common,supine_failed_repeat_histogram={
            str(n):int((totals==n).sum()) for n in range(1,8)},
        left_entries=dict(two_ms=dict(admitted=121,failures=prior_left['scenes'],by_stage=prior_left['blocked_by_stage']),
            first_window_one_ms=dict(admitted=120,failures=list(details.values()),by_stage=recent['left_reasons'])),
        control_groups=groups,source_trace_sha256=source_hash,additional_physical_steps=0,learning_updates=0,
        safety_gate_changed=False,reserved_seed_used=False)
    if check:
        assert read(OUTPUT,node['result_sha256'])==results
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,results)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate='Use the recurring stage-specific losses and frozen motor clock to select the next matched intervention.')
        write(PROJECT,project)
    print('LOW_FRICTION_RECURRENCE_VERIFIED',sha(OUTPUT),
          {k:(v['command'],v['failed_count']) for k,v in common.items()},flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    action=cli.parse_args().action
    freeze() if action=='freeze' else score(check=action=='check')
