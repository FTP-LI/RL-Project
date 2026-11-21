"""Compare the frozen early signal against original roll admission on each full trace."""

from argparse import ArgumentParser
import fcntl
from math import ceil
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import review_nominal_low_friction_pre_support_progress as partial
from roll_stand_walk_sequence import ROLL
from standup_metrics import projected_gravity


geometry,experiment = partial.geometry,partial.experiment
authority = geometry.authority
ROOT, PROJECT, read, rel, sha, write = partial.ROOT, partial.PROJECT, partial.read, partial.rel, partial.sha, partial.write
KEY, STAGE = 'nominal_low_friction_full_trace_progress', 'nominal-low-friction-full-trace-progress'
PLAN, REPORT = partial.PLAN.with_stem(KEY), partial.REPORT/'full-trace'


def freeze():
    project=read(PROJECT)
    goal=project['goal_execution']
    assert experiment.parent.allowed(project,partial.STAGE) and not project['staged_training_request']['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    component=goal[partial.KEY]
    assert component['status']=='reviewed' and not component['eligible_for_real_sampler']
    scope=read(ROOT/component['protocol'],component['protocol_sha256'])
    geometry_plan=goal[geometry.KEY]
    original=read(ROOT/geometry_plan['protocol'],geometry_plan['protocol_sha256'])
    old=goal[authority.KEY]
    archives = []
    for key,roles in (('nominal_low_friction_release_integrator',('reference-4ms-first','reference-4ms-repeat')),
                      ('nominal_low_friction_loaded_upright_ppo_action',geometry.measured.ROLES[:1]+geometry.measured.ROLES[2:])):
        record=goal[key]
        assert record['status']=='reviewed'
        archives.append(dict(roles=list(roles),**{tag:dict(path=record[tag],sha256=record[tag+'_sha256'])
             for tag in ('protocol','collection','result')}))
    inputs={row[k]:row[k+'_sha256'] for row,keys in ((component,('protocol','result')),
                (geometry_plan,('protocol',)),(old,('protocol',))) for k in keys}
    inputs.update({v['path']:v['sha256'] for x in archives for k,v in x.items() if k!='roles'})
    for src in (__file__,partial.__file__,geometry.__file__,authority.__file__):
        path=Path(src).resolve()
        inputs[rel(path)]=sha(path)
    for path,digest in inputs.items():
        assert sha(ROOT/path)==digest,path
    protocol=dict(stage=STAGE,source_sha256=inputs,archives=archives,metric_definition=component['protocol'],
        geometry_authority_protocol=original['geometry_authority_protocol'],focus=scope['focus'],
        matching_features='Three release projected gravity coordinates and their first 20ms change; Euclidean cost; one-to-one optimal pairing within each branch.',
        release_control=100,score_control=600,minimum_positive_fraction=12/13,minimum_focus_advantage=.10,
        eligible_heldout_seed=2246,learning_updates=0,new_physical_steps=0,hardware_io=False)
    REPORT.mkdir()
    write(PLAN,protocol)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',PLAN,
        'Pair early measurements only with the original admission outcomes of their own complete archived trajectory.')
    write(PROJECT,project)
    print('FULL_TRACE_PROGRESS_FROZEN',sha(PLAN),flush=True)


def controls(folder,chunks,needed,field):
    row, = [x for x in chunks if x['first']<=needed[0]<=needed[-1]<=x['last']]
    path=folder/row['file']
    assert sha(path)==row['sha256']
    indices=np.array(needed)-row['first']
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['control'][indices],needed)
        return data[field][indices]


def right_gap(geom, qs, w):
    geom.select(w)
    m,d,mj=geom.model,geom.data,geom.mj
    foot=geom.feet[1]
    heights=[]
    for q in qs:
        d.qpos[:]=q
        mj.mj_kinematics(m,d)
        row=d.geom_xmat[foot].reshape(3,3)[2]
        lowest=(geom.vertices[1]@row).min()+d.geom_xpos[foot,2]-d.geom_xpos[geom.engine.floor,2]
        heights.append(lowest*1000.)
    assert d.time==0.
    return np.asarray(heights)


def read_branch(geom,scope,protocol,role,folder,batch,common,poses):
    start,end=(protocol[k] for k in ('release_control','score_control'))
    d=batch['branches'][role]
    leaf=folder/role
    assert batch['prefix']['controls']==start
    restored=read(leaf/'restored-fork.json',d['restore_sha256'])
    fork=read(folder/'fork.json',batch['fork_sha256'])
    assert restored=={k:fork[k] for k in ('physical','controller_sha256')}
    before=projected_gravity(common[:,3:7])
    initial=controls(leaf,d['chunks'],[start+1],'qpos')[0,:,3:7]
    features=np.column_stack((before,projected_gravity(initial)-before))
    entered=controls(leaf,d['chunks'],[end],'roll_phase')[0]==ROLL
    fail,pass_=(np.flatnonzero((poses=='supine')&(entered==match)) for match in (False,True))
    assert len(fail)>0 and len(pass_)>=len(fail)
    cost=np.linalg.norm(features[fail,None]-features[None,pass_],axis=-1)
    a,b=linear_sum_assignment(cost)
    np.testing.assert_array_equal(a,np.arange(len(fail)))
    chosen=np.r_[fail,pass_[b]]
    first,last,dt=(scope[k] for k in ('first_ms','last_ms','control_period_ms'))
    times=list(range(start+first//dt,start+last//dt))
    raw=controls(leaf,d['chunks'],times,'qpos')[:,chosen]
    angle=authority.tilt(raw[...,3:7])
    feet=np.stack([right_gap(geom,raw[:,i],int(w)) for i,w in enumerate(chosen)],axis=1)
    body,right,joint=(s.mean(axis=0) for s in partial.shape(angle,feet,scope))
    count=len(fail)
    pairs=[]
    for n,w in enumerate(fail):
        m=count+n
        pairs.append(dict(missed=int(w),admitted=int(chosen[m]),release_feature_distance=float(cost[a[n],b[n]]),
            joint_admitted_minus_missed=float(joint[m]-joint[n]),body_admitted_minus_missed=float(body[m]-body[n]),
            foot_admitted_minus_missed=float(right[m]-right[n])))
    positive=sum(r['joint_admitted_minus_missed']>0 for r in pairs)
    need=ceil(protocol['minimum_positive_fraction']*count)
    focused=[r for r in pairs if r['missed'] in protocol['focus']]
    return dict(role=role,supine_worlds=count+len(pass_),failed=count,roll_entry_all=int(entered.sum()),
        matched_positive=positive,required_positive=need,focus_missed=len(focused),
        focus_positive=sum(r['joint_admitted_minus_missed']>=protocol['minimum_focus_advantage'] for r in focused),
        eligible=positive>=need and len(focused)>0 and all(r['joint_admitted_minus_missed']>=protocol['minimum_focus_advantage'] for r in focused),
        pairs=pairs)


def evaluate():
    project=read(PROJECT)
    node=project['goal_execution'][KEY]
    assert experiment.parent.allowed(project,STAGE) and not project['staged_training_request']['simulation_running']
    assert node['status']=='archive-ready'
    spec=read(PLAN,node['protocol_sha256'])
    for path,digest in spec['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    scope=read(ROOT/spec['metric_definition'])
    old=read(ROOT/spec['geometry_authority_protocol'])
    for path,digest in old['sources'].items():
        assert sha(ROOT/path)==digest,path
    source,meta=(read(ROOT/old[k]) for k in ('prior_experiment','load_metadata'))
    geom=geometry.Geometry(authority.FixedStateEngine(old,source,meta))
    results={}
    for x in spec['archives']:
        source,batch=(read(ROOT/x[k]['path'],x[k]['sha256']) for k in ('protocol','collection'))
        comp=source['composition']
        assert (source['environment']['seed'],comp['foot_friction'],comp['release_control'],comp['roll_start_control'])==(1257,.8,100,600)
        assert source['roll_route']['standing_dwell_controls']==25 and comp['existing_low_friction_reserved_seed']==2246
        assert batch['actor_weights_unchanged'] and not batch['reserved_seed_used']
        folder=(ROOT/x['collection']['path']).parent
        assert sha(folder/'release-poses.npz')==batch['release_sha256']
        with np.load(folder/'release-poses.npz',allow_pickle=False) as file:
            poses=file['poses']
        common=controls(folder/'shared',batch['prefix']['chunks'],[spec['release_control']],'qpos')[0]
        for role in x['roles']:
            results[role]=read_branch(geom,scope,spec,role,folder,batch,common,poses)
    passed=all(row['eligible'] for row in results.values())
    output=REPORT/'analysis.json'
    write(output,dict(status='verified-full-trajectory-alignment',protocol_sha256=node['protocol_sha256'],branches=results,
        eligible_for_real_sampler=passed,new_physical_steps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),eligible_for_real_sampler=passed)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-reviewed',output,
        'Use the full-trajectory paired proof for the next training input and preserve the original route gates.')
    write(PROJECT,project)
    print('FULL_TRACE_PROGRESS_VERIFIED',sha(output),{k:(v['matched_positive'],v['failed'],v['required_positive'],
        v['focus_positive'],v['focus_missed']) for k,v in results.items()},passed,flush=True)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','evaluate'))
    mode=cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'evaluate':evaluate}[mode]()


if __name__=='__main__':
    main()
