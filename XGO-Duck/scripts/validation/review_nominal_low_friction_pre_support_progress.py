"""Compare a fixed early rotation and foot-clearance signal on saved releases."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

import numpy as np

import review_nominal_low_friction_forward_geometry as geometry
import replay_nominal_low_friction_forward_momentum as previous


experiment = geometry.experiment
ROOT, PROJECT, read, rel, sha, write = geometry.ROOT, geometry.PROJECT, geometry.read, geometry.rel, geometry.sha, geometry.write
KEY, STAGE = 'nominal_low_friction_pre_support_progress', 'nominal-low-friction-pre-support-progress'
PLAN = geometry.PLAN.with_stem(KEY)
REPORT = geometry.REPORT/'progress-qualification'


def freeze():
    project = read(PROJECT)
    goal = project['goal_execution']
    assert experiment.parent.allowed(project,previous.STAGE) and not project['staged_training_request']['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    geom,motor,archive = (goal[k] for k in (geometry.KEY,previous.KEY,experiment.KEY))
    assert all(x['status']=='reviewed' for x in (geom,motor,archive))
    design = read(ROOT/geom['protocol'],geom['protocol_sha256'])
    result = read(ROOT/geom['result'],geom['result_sha256'])
    sources = {row[k]:row[k+'_sha256'] for row,keys in ((geom,('protocol','result')),(motor,('result',)),
        (archive,('result',))) for k in keys}
    sources.update({r['path']:r['sha256'] for r in result['arrays'].values()})
    script = Path(__file__).resolve()
    sources[rel(script)] = sha(script)
    for path,digest in sources.items():
        assert sha(ROOT/path)==digest,path
    scope = dict(stage=STAGE,geometry_analysis=geom['result'],archive_effort=archive['result'],
        pairs=design['archive_pairs'],focus=design['focus'],roles=design['roles'],source_sha256=sources,
        seed=1257,friction=.8,release_control=100,control_period_ms=20,
        first_ms=400,midpoint_ms=520,last_ms=640,
        angle_start_deg=75.,angle_target_deg=15.,foot_range_mm=60.,
        minimum_positive_pairs_each_role=12,minimum_focus_advantage=.10,
        heldout_seed=2246,new_physical_steps=0,learning_updates=0,hardware_io=False)
    REPORT.mkdir()
    write(PLAN,scope)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',PLAN,
        'Verify the fixed early posture and foot-clearance signal on all archived pairs and controls.')
    write(PROJECT,project)
    print('PRE_SUPPORT_PROGRESS_FROZEN',sha(PLAN),flush=True)


def shape(tilt, foot_clear_mm, spec):
    body = np.clip((spec['angle_start_deg']-tilt)/(spec['angle_start_deg']-spec['angle_target_deg']),0.,1.)
    foot = np.clip(1.-foot_clear_mm/spec['foot_range_mm'],0.,1.)
    return body,foot,body*foot


def evaluate():
    project=read(PROJECT)
    node=project['goal_execution'][KEY]
    assert experiment.parent.allowed(project,STAGE) and not project['staged_training_request']['simulation_running']
    assert node['status']=='archive-ready'
    spec=read(PLAN,node['protocol_sha256'])
    for name,digest in spec['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    geom=read(ROOT/spec['geometry_analysis'])
    effort=read(ROOT/spec['archive_effort'])
    pairs=[(int(a),b) for a,b in spec['pairs'].items()]
    results={}
    for role in spec['roles']:
        with np.load(ROOT/geom['arrays'][role]['path'],allow_pickle=False) as a:
            ms=(a['controls']-spec['release_control'])*spec['control_period_ms']
            worlds=a['worlds'].tolist()
            fields=a['metrics'].tolist()
            angles,gaps=(a['values'][...,fields.index(k)] for k in ('tilt_deg','right_clearance_mm'))
        first,mid,last=(spec[name] for name in ('first_ms','midpoint_ms','last_ms'))
        mean={}
        for window in ((first,last),(first,mid),(mid,last)):
            rows=(ms>=window[0])&(ms<window[1])
            scales=np.stack(shape(angles[rows],gaps[rows],spec),axis=-1)
            mean[window]=scales.mean(axis=0)
        full=mean[first,last]
        receipt=effort['roles'][role]['selected_world_readback']
        def stood(world):
            r=receipt[str(world)]
            return r['terminal_tilt_deg']<=15. and not r['late_sustained_head_and_feet']
        rows=[]
        for a,b in pairs:
            i,j=worlds.index(a),worlds.index(b)
            delta=full[j]-full[i]
            rows.append(dict(original_missed_world=a,original_admitted_world=b,
                admitted_minus_missed_body=float(delta[0]),admitted_minus_missed_foot=float(delta[1]),
                admitted_minus_missed_joint_signal=float(delta[2]),
                early_half_joint_signal_delta=float(mean[first,mid][j,2]-mean[first,mid][i,2]),
                late_half_joint_signal_delta=float(mean[mid,last][j,2]-mean[mid,last][i,2]),
                current_early_probe_preserves_pair_phenotype=(not stood(a) and stood(b))))
        positive=sum(r['admitted_minus_missed_joint_signal']>0. for r in rows)
        focus_ok=all(r['admitted_minus_missed_joint_signal']>=spec['minimum_focus_advantage'] for r in rows
                     if r['original_missed_world'] in spec['focus'])
        results[role]=dict(positive_joint_signal_pairs=positive,focus_passes=focus_ok,
            median_joint_signal_difference=float(np.median([r['admitted_minus_missed_joint_signal'] for r in rows])),
            paired_readback=rows,eligible=positive>=spec['minimum_positive_pairs_each_role'] and focus_ok)
    passing=all(v['eligible'] for v in results.values())
    out=REPORT/'analysis.json'
    write(out,dict(status='verified-offline-pre-support-signal',protocol_sha256=node['protocol_sha256'],
        scope='Previously selected archive pairs, four routes on the same development seed. Half-open release-aligned window.',
        comparisons=results,eligible_for_real_sampler=passing,new_physical_steps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed',result=rel(out),result_sha256=sha(out),eligible_for_real_sampler=passing)
    next_step=('Implement the early control-rate geometry readback; independently verify the GPU calculation before training.'
               if passing else 'Review the disagreeing early trajectory pairs before freezing a new training input.')
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-reviewed',out,next_step)
    write(PROJECT,project)
    print('PRE_SUPPORT_PROGRESS_VERIFIED',sha(out),{k:(v['positive_joint_signal_pairs'],v['focus_passes']) for k,v in results.items()},passing,flush=True)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','evaluate'))
    mode=cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'evaluate':evaluate}[mode]()


if __name__=='__main__':
    main()
