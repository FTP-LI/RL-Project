"""Align the archived right-hip/knee comparison to real foot and head contact."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from standup_metrics import projected_gravity


SOURCE = 'nominal_low_friction_right_support_trial'
KEY = 'nominal_low_friction_right_support_contact_timing'
STAGE = 'nominal-low-friction-right-support-contact-timing'
PRIOR = 'nominal-low-friction-right-support-trial'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-right-support-trial/support-timing'
FIELDS = ('quaternion','qpos','foot_positions','head_contact','foot_contact','angular_velocity','height')
ARMS = ('original-first','original-repeat','right-hip-knee-candidate')


def freeze():
    project = read(PROJECT)
    permitted(project,PRIOR)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    node = goal[SOURCE]
    assert node['status']=='reviewed' and not node['full_route_admitted']
    fields = ('protocol','result','collection','verification','motion_readback')
    sources = {node[k]:node[k+'_sha256'] for k in fields}
    for path,expected in sources.items():
        assert sha(ROOT/path)==expected
    motion = read(ROOT/node['motion_readback'],node['motion_readback_sha256'])
    assert motion['newly_admitted_in_both_comparisons']==[218] and len(motion['still_missed_in_all_arms'])==8
    spec = dict(status='frozen-archive-contact-timing',stage=STAGE,seed=1257,friction=.8,
        collection=node['collection'],shared_control=136,release_control=100,last_control=180,
        same_state_worlds=motion['newly_admitted_in_both_comparisons']+motion['still_missed_in_all_arms'],
        grouping={'gained':motion['newly_admitted_in_both_comparisons'],'persistently_missed':motion['still_missed_in_all_arms']},
        horizon_examples=[130,136,140,150,160,175],confirmation_reads=3,step_s=.02,
        event='Timestamp of first confirmed three consecutive 20 ms endpoint reads after release.',
        planar_foot_coordinates='Absolute right-minus-left along the initial horizontal heading at the intervention state.',
        source_sha256=sources,script_sha256=sha(Path(__file__)),physical_steps=0,hardware_io=False,learning_updates=0,
        low_friction_reserved_seed=2246)
    REPORT.mkdir()
    write(PLAN,spec)
    goal[KEY] = dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Read back bilateral support, head touchdown and upright dwell in the regained and persistent scenes.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_CONTACT_TIMING_FROZEN',sha(PLAN),flush=True)


def trace_rows(folder,chunks,index,first,last):
    result = {}
    for part in chunks:
        a,b = max(first,part['first']),min(last,part['last'])
        if a>b:
            continue
        path = folder/part['file']
        assert sha(path)==part['sha256']
        with np.load(path,allow_pickle=False) as rows:
            offset = slice(a-part['first'],b-part['first']+1)
            controls = rows['control'][offset]
            np.testing.assert_array_equal(controls,np.arange(a,b+1))
            values = {field:rows[field][offset][:,index] for field in FIELDS}
        for j,k in enumerate(controls):
            assert int(k) not in result
            result[int(k)] = {field:stack[j] for field,stack in values.items()}
    assert sorted(result)==list(range(first,last+1))
    return result


def first_dwell(controls,flags,n,after):
    valid = np.asarray(flags)&(controls>=after)
    windows = np.lib.stride_tricks.sliding_window_view(valid,n).all(axis=-1)
    index = np.flatnonzero(windows)
    return int(controls[index[0]+n-1]) if index.size else None


def summarize(rows,index,spec):
    k0,k1 = spec['release_control'],spec['last_control']
    t = np.arange(k0,k1+1)
    q = np.stack([rows[int(k)]['quaternion'][index] for k in t])
    g = projected_gravity(q)
    tilt = np.rad2deg(np.arccos(np.clip(-g[:,2],-1.,1.)))
    forward_pitch = np.rad2deg(np.arctan2(g[:,0],-g[:,2]))
    feet = np.stack([rows[int(k)]['foot_contact'][index] for k in t])
    head = np.array([rows[int(k)]['head_contact'][index] for k in t])
    n,t0,at = spec['confirmation_reads'],k0,spec['shared_control']
    base = rows[at]['qpos'][index][3:7].astype(np.float64)
    w,x,y,z = base
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    forward = np.array([np.cos(yaw),np.sin(yaw)])
    slots = {}
    for k in spec['horizon_examples']:
        row,j = rows[k],k-t0
        xy = row['foot_positions'][index,:,:2]
        sagittal = float((xy[1]-xy[0])@forward)
        slots[str(k)] = dict(tilt_deg=float(tilt[j]),forward_pitch_deg=float(forward_pitch[j]),
            left_contact=bool(feet[j,0]),right_contact=bool(feet[j,1]),head_contact=bool(head[j]),
            right_minus_left_sagittal_mm=sagittal*1000.,
            abs_sagittal_gap_mm=abs(sagittal)*1000.,height_mm=float(row['height'][index]*1000.),
            angular_speed_rad_s=float(np.linalg.norm(row['angular_velocity'][index])))
    return dict(first_left_contact_confirmed=first_dwell(t,feet[:,0],n,t0+1),
        first_right_contact_confirmed=first_dwell(t,feet[:,1],n,t0+1),
        first_bilateral_confirmed=first_dwell(t,feet.all(axis=1),n,t0+1),
        first_post_intervention_three_support_confirmed=first_dwell(t,feet.all(axis=1)&head,n,at+1),
        first_post_intervention_upright_and_head_clear_confirmed=first_dwell(t,(tilt<=15.)&feet.all(axis=1)&~head,n,at+1),
        checkpoints=slots)


def score(check=False):
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    if not check:
        permitted(project,STAGE)
    spec = read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for file,digest in spec['source_sha256'].items():
        assert sha(ROOT/file)==digest
    collection = ROOT/spec['collection']
    raw = read(collection)
    assert raw['status']=='collected' and raw['actor_weights_unchanged']
    assert not any(raw[k] for k in ('hardware_io','learning_updates','reserved_seed_used'))
    subset = np.array(spec['same_state_worlds'])
    at,end = spec['shared_control'],spec['last_control']
    assert raw['prefix']['controls']==at and tuple(raw['branches'])==ARMS
    shared = trace_rows(collection.parent/'shared',raw['prefix']['chunks'],subset,spec['release_control'],at)
    full = {arm:shared | trace_rows(collection.parent/arm,raw['branches'][arm]['chunks'],subset,at+1,end)
            for arm in ARMS}
    measurements = {str(w):{arm:summarize(rows,i,spec) for arm,rows in full.items()} for i,w in enumerate(subset)}
    groups = {name:{arm:{} for arm in ARMS} for name in spec['grouping']}
    keys = ('first_bilateral_confirmed','first_post_intervention_three_support_confirmed',
            'first_post_intervention_upright_and_head_clear_confirmed')
    for name,worlds in spec['grouping'].items():
        for arm in ARMS:
            for label in keys:
                values = [measurements[str(i)][arm][label] for i in worlds]
                valid = [x for x in values if x is not None]
                groups[name][arm][label] = dict(detected=len(valid),median_control=float(np.median(valid)) if valid else None,
                                              first_control=min(valid) if valid else None,last_control=max(valid) if valid else None)
    dest = REPORT/'analysis.json'
    result = dict(status='verified-contact-and-pitch-timing',protocol_sha256=sha(PLAN),worlds=len(subset),
        seed=1257,friction=.8,step_s=.02,groups=groups,measured_by_world=measurements,
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    if check:
        assert result==read(dest,node['result_sha256'])
    else:
        assert not dest.exists()
        write(dest,result)
        node.update(status='reviewed',result=rel(dest),result_sha256=sha(dest))
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(dest))
        project.update(info)
        project['staged_training_request'].update(info,next_gate='Use the measured margin before bilateral support to choose a recovery test. Evaluate nominal supply and delay independently.')
        write(PROJECT,project)
    print('LOW_FRICTION_CONTACT_TIMING_VERIFIED',sha(dest),len(subset),flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    args = cli.parse_args()
    freeze() if args.action=='freeze' else score(args.action=='check')
