"""Recompute all saved common-walk scores and preserve the incomplete old run.

No new episodes, policy fitting, threshold changes or heldout reuse. Describe
the failed heldout component; do not choose a controller from heldout traces.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from planar_block_metrics import block_statistic
OUT=ROOT/'records/reports/common-walk-confirmation'


def full(seed,directory):
    analysis=read(directory/f's{seed}/analysis.json');folder=ROOT/analysis['evaluator']
    p=read(folder/'protocol.json');report=read(folder/'evaluation.json');runtime=read(folder/'runtime-contract.json')
    assert report['status']=='completed' and runtime['control_steps']==1000
    c=p['criteria'];result=[];failures=[]
    for name,ids in p['groups'].items():
        path=folder/f'{name}-trace.npz';item=next(r for r in report['conditions'] if r['condition']==name)
        assert sha(path)==item['trace_sha256']
        with np.load(path) as f:t={k:f[k].copy() for k in f.files}
        active=t['active'];counts=active.sum(0);assert active.shape==(1000,32) and active[0].all()
        mean=lambda value:np.where(active,value,0).sum(0)/counts
        command=np.asarray(p['commands'][name]);axis=int(np.argmax(abs(command)));sign=float(np.sign(command[axis]))
        signed=mean(t['yaw_rate'] if axis==2 else t['velocity'][...,axis])*sign
        yaw=mean(abs(t['yaw_rate']-command[2]))
        block=block_statistic(t['velocity'],active,command)['mean_error_per_episode']
        after=active.copy();after[:50]=False;valid=after.sum(0)
        hips=[p['joint_names'].index(s+'_hip_roll') for s in ('left','right')]
        angles=np.rad2deg(t['joint_positions'][...,hips]);inward=(angles[...,0]<-20)&(angles[...,1]>20)
        fraction=lambda values:np.divide((values*after).sum(0),valid,out=np.ones(32),where=valid>0)
        contacts=t['contact']*active[...,None];nc=contacts.sum((0,2))
        slip=np.divide(np.where(contacts,t['foot_speed'],0).sum((0,2)),nc,out=np.full(32,np.nan),where=nc>0)
        w,x,y,z=np.moveaxis(t['root_quaternion'],-1,0)
        local_y=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
        sep=((t['foot_positions'][...,0,:]-t['foot_positions'][...,1,:])*local_y).sum(-1)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))
        pose=c['posture'];posture=fraction((abs(t['height']-pose['nominal_height_m'])>pose['height_tolerance_m'])|(tilt>pose['tilt_max_deg']))
        close=fraction(sep<pose['minimum_signed_site_separation_m'])
        limits=np.asarray(p['joint_limits']);tol=np.deg2rad(c['joint_tolerance_deg'])
        bounded=((t['joint_positions']>=limits[:,0]-tol)&(t['joint_positions']<=limits[:,1]+tol)).all(-1)
        tests=dict(survived=np.asarray([r['outcome']=='horizon_complete' for r in item['episodes']]),
            block_error=block<=c['linear_error'][name],yaw_error=yaw<=c['yaw_error_max'],
            signed_speed=signed>=c['signed_speed'][name],inward=fraction(inward)<=c['inward_fraction_max'],
            contact_slip=slip<=c['contact_slip_max'],physical_joints=(bounded|~active).all(0),
            posture=posture<=pose['maximum_bad_sample_fraction'],feet_separation=close<=pose['maximum_bad_sample_fraction'])
        if name=='stand':tests['standing_drift']=np.asarray([r['final_displacement_m'] for r in item['episodes']])<=c['stand_drift_median_max']
        passed=np.logical_and.reduce(list(tests.values()));old=next(r for r in analysis['rows'] if r['condition']==name)
        assert int(passed.sum())==old['successes']
        np.testing.assert_allclose([block.mean(),yaw.mean(),signed.mean(),np.nanmean(slip),posture.mean(),close.mean()],
            [old[k] for k in ('block_error','yaw_error','signed_speed','contact_slip','posture_bad_fraction','close_feet_fraction')],atol=1e-6,rtol=1e-5)
        cases=[]
        for i in np.flatnonzero(~passed):
            row=dict(seed=seed,condition=name,world=ids[int(i)],failed_components=[k for k,v in tests.items() if not v[i]],
                block_error=float(block[i]),yaw_error=float(yaw[i]),signed_speed=float(signed[i]),slip=float(slip[i]),
                posture_bad_fraction=float(posture[i]),close_feet_fraction=float(close[i]))
            cases.append(row);failures.append(row)
        result.append(dict(condition=name,successes=int(passed.sum()),episodes=32,
            functional_pass=old['functional_development_pass'],failures=cases))
    return dict(seed=seed,rows=result,total_successes=sum(x['successes'] for x in result),episodes=224,
        all_commands_passed=analysis['functional_development_passed'],control_transitions=runtime['control_transitions'],
        actual_reset_calls=runtime['resets_after_initialization'],failures=failures,
        analysis_sha256=sha(directory/f's{seed}/analysis.json'),run_id=p['run_id'])


def partial():
    master=read(ROOT/'configs/experiments/common_walk_confirmation.json');p=read(ROOT/master['protocols']['1041'])
    folder=ROOT/'records/evaluations'/p['run_id'];state=read(folder/'evaluation.json');movies=[]
    assert state['status']=='failed' and state['actual_control_steps']==79
    for row in state['conditions']:
        name=row['condition'];path=folder/f'{name}-trace.npz';assert sha(path)==row['trace_sha256']
        with np.load(path) as f:qpos=f['qpos'].copy()
        v=row['video'];video=folder/v['path'];assert sha(video)==v['sha256'] and not v['complete_clip']
        assert v['frame_times_seconds']==np.arange(0,79,2).astype(float).astype(float).__mul__(.02).tolist()
        steps=np.rint(np.asarray(v['frame_times_seconds'])/.02).astype(int)
        with np.load(folder/'physical-initial.npz') as f:initial=f['qpos'][v['environment_index']].copy()
        hashes=[hashlib.sha256((initial if step==0 else qpos[step-1,0]).tobytes()).hexdigest() for step in steps]
        assert hashes==v['qpos_sha256_by_frame']
        # Full decode is also independently performed by the media atlas.
        movies.append(dict(path=str(video.relative_to(ROOT)),frames=len(steps),physical_range=[0,1.56],partial=True,sha256=sha(video)))
    return dict(seed=1041,controls=79,transitions=17696,terminated_world=98,
        failure_retained=True,full_motion_acceptance=False,movies=movies)


def main():
    recovery=OUT/'reset-recovery';suite=read(recovery/'suite-progress.json')
    assert suite['status']=='closed-functional-failed' and suite['completed_seeds']==[1042,1043,2040]
    assert suite['failed_seed']==2040 and read(recovery/'heldout-consumption.json')['seed']==2040
    rows=[full(1040,OUT)]+[full(seed,recovery) for seed in (1042,1043,2040)]
    interrupted=partial();new=sum(r['control_transitions'] for r in rows[1:]);assert new==672000
    frozen={}
    for name in ('common_walk_confirmation.json','common_walk_reset_recovery.json','getup_teacher_anchor_pair.json'):
        path=ROOT/'configs/experiments'/name;p=read(path)
        for source,digest in p['source_sha256'].items():assert sha(ROOT/source)==digest,source
        frozen[name]=dict(sha256=sha(path),sources=len(p['source_sha256']))
    proof=dict(status='passed-integrity-functional-failed',checked_at=datetime.now(timezone.utc).isoformat(),
        full_evaluations=rows,partial=interrupted,new_transitions=new,original_transitions=241696,
        total_transitions=913696,new_successes=sum(r['total_successes'] for r in rows[1:]),new_episodes=672,
        heldout_seed_consumed=2040,heldout_not_for_retuning=True,learning_updates=0,physics_steps_added_by_review=0,
        actual_resets_exercised=False,reset_helper_cpu_only=True,frozen=frozen,
        source_sha256=sha(Path(__file__)),all_nine_motion_goal_complete=False)
    write(OUT/'closure-review.json',proof)
    print('COMMON_WALK_CLOSURE',[(r['seed'],r['total_successes'],r['all_commands_passed']) for r in rows],
        'heldout_failures',rows[-1]['failures'],'913696transitions','0learning',flush=True)


if __name__=='__main__':main()
