"""CPU diagnosis of the closed common-plant screen; never advance physics.

Recompute episode failures from immutable traces with the original criteria.
Separate a missing requested release stratum from a failed recovery hold, but
retain both in the original pass/fail result. Repeated environment indices are
diagnostic associations, not evidence that a physical parameter caused failure.
Validation: compare all recomputed outcomes to the independently checked rows,
verify input hashes and write a traceable, separate postprocessing artifact.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from planar_block_metrics import block_statistic
from standup_metrics import projected_gravity, summarize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/common-plant-screen'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def longest(mask):
    run=best=0
    for value in mask:
        run=run+1 if value else 0
        best=max(best,run)
    return best


def walking():
    proof=read(OUT/'walk/common-verification.json');assert proof['status']=='passed'
    folder=ROOT/proof['record'];report=read(folder/'evaluation.json');plan=read(folder/'protocol.json')
    rows=[];criteria=plan['criteria'];pose=criteria['posture'];names=plan['joint_names']
    hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    for item in report['conditions']:
        name=item['condition'];path=folder/f'{name}-trace.npz';assert sha(path)==item['trace_sha256']
        t=load(path);active=t['active'];command=np.array(plan['commands'][name]);n=active.shape[1]
        mean=lambda x:np.where(active,x,0).sum(axis=0)/active.sum(axis=0)
        block=block_statistic(t['velocity'],active,command)['mean_error_per_episode']
        yaw=mean(abs(t['yaw_rate']-command[2]));axis=int(np.argmax(abs(command)))
        signed=mean(t['yaw_rate'] if axis==2 else t['velocity'][:,:,axis])*np.sign(command[axis])
        contacts=t['contact']&active[:,:,None];count=contacts.sum(axis=(0,2))
        slip=np.divide(np.where(contacts,t['foot_speed'],0).sum(axis=(0,2)),count,out=np.full(n,np.nan),where=count>0)
        after=active.copy();after[:50]=False;den=after.sum(axis=0)
        angles=np.rad2deg(t['joint_positions'][:,:,hips]);inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
        inward_fraction=(inward&after).sum(axis=0)/den
        w,x,y,z=np.moveaxis(t['root_quaternion'],-1,0)
        local_y=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
        separation=((t['foot_positions'][:,:,0]-t['foot_positions'][:,:,1])*local_y).sum(axis=-1)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))
        bad_pose=(abs(t['height']-pose['nominal_height_m'])>pose['height_tolerance_m'])|(tilt>pose['tilt_max_deg'])
        masks=dict(survival=np.array([e['outcome']=='horizon_complete' for e in item['episodes']]),
            linear_block_error=block<=criteria['linear_error'][name],yaw_error=yaw<=criteria['yaw_error_max'],
            signed_speed=signed>=criteria['signed_speed'][name],inward=inward_fraction<=criteria['inward_fraction_max'],
            slip=slip<=criteria['contact_slip_max'],joint_limits=np.all(t['joint_envelope_ok']|~active,axis=0),
            posture=(bad_pose&after).sum(axis=0)/den<=pose['maximum_bad_sample_fraction'],
            foot_separation=((separation<pose['minimum_signed_site_separation_m'])&after).sum(axis=0)/den<=pose['maximum_bad_sample_fraction'])
        if name=='stand':masks['standing_drift']=np.array([e['final_displacement_m'] for e in item['episodes']])<=criteria['stand_drift_median_max']
        passed=np.logical_and.reduce(list(masks.values()));expected=next(r for r in read(OUT/'walk/s1009/analysis.json')['rows'] if r['condition']==name)
        assert int(passed.sum())==expected['successes']
        failed=[]
        for i in np.flatnonzero(~passed):
            failed.append(dict(group_index=int(i),environment_index=item['environment_ids'][i],
                failed_checks=[k for k,v in masks.items() if not v[i]],
                block_error_m_s=float(block[i]),block_error_limit=criteria['linear_error'][name],
                yaw_error_rad_s=float(yaw[i]),signed_speed=float(signed[i]),contact_slip_m_s=float(slip[i])))
        rows.append(dict(condition=name,successes=int(passed.sum()),episodes=n,failures=failed,trace_sha256=sha(path)))
    return rows


def recovery():
    proof=read(OUT/'getup/common-verification.json');assert proof['status']=='passed'
    folder=ROOT/proof['record'];plan=read(folder/'protocol.json');report=read(folder/'evaluation.json');criteria=plan['criteria']
    start=round(criteria['freeze_seconds']/criteria['step_dt']);hold=round(criteria['hold_seconds']/criteria['step_dt'])
    latest=round(criteria['deadline_seconds']/criteria['step_dt']);rows=[];fail_counter=Counter();initial=[]
    for item in report['conditions']:
        pose=item['condition'];path=folder/f'{pose}-trace.npz';assert sha(path)==item['trace_sha256']
        t=load(path);result=summarize(t,criteria);assert result['episodes']==item['episodes']
        gravity=projected_gravity(t['quaternion']);tilt=np.rad2deg(np.arccos(np.clip(-gravity[...,2],-1,1)))
        masks=dict(height=abs(t['height']-criteria['nominal_height_m'])<=criteria['height_tolerance_m'],
            tilt=tilt<=criteria['tilt_max_deg'],angular_speed=np.linalg.norm(t['angular_velocity'],axis=-1)<=criteria['angular_speed_max_rad_s'],
            feet=t['foot_contact'].all(axis=-1),joint_limits=t['joint_envelope_ok'],active=t['active'])
        stable=np.logical_and.reduce(list(masks.values()));missed=[];failures=[]
        for e in result['episodes']:
            i=e['env_index'];has_hold=e['hold_start_seconds'] is not None
            if not e['release_pose_valid']:missed.append(dict(env_index=i,actual=e['release_pose'],has_timely_hold=has_hold))
            if not has_hold:
                fail_counter[i]+=1
                lengths={k:longest(v[start:,i])*criteria['step_dt'] for k,v in masks.items()}
                # Fixed three-second window with the fewest unstable samples,
                # restricted to on-time starts. This is explanation, not a gate.
                starts=range(start,min(latest+1,len(stable)-hold+1))
                best=min(starts,key=lambda a:np.count_nonzero(~stable[a:a+hold,i]))
                failures.append(dict(env_index=i,release_pose=e['release_pose'],requested_release_valid=e['release_pose_valid'],
                    longest_stable_seconds=longest(stable[start:,i])*criteria['step_dt'],
                    longest_individual_checks_seconds=lengths,best_on_time_window_start_seconds=best*criteria['step_dt'],
                    bad_samples_in_best_window={k:int(np.count_nonzero(~v[best:best+hold,i])) for k,v in masks.items()},
                    final_height_m=float(t['height'][-1,i]),final_tilt_deg=float(tilt[-1,i]),
                    final_feet=t['foot_contact'][-1,i].tolist(),joint_violation_samples=int((~t['joint_envelope_ok'][start:,i]).sum()),
                    final_actor_gravity_median=np.median(t['observations'][-100:,i,3:6],axis=0).tolist(),
                    final_joint_positions_radians=t['joint_positions'][-1,i].tolist(),
                    initial_actor_joint_estimates_radians=(t['observations'][0,i,6:20]+np.array(report['policy_home'])).tolist()))
        # Actor joint estimates contain observation noise. Use saved raw qpos
        # for the physical reset comparison instead of mislabelling those data.
        initial.append(t['initial_qpos'][:,7:])
        rows.append(dict(condition=pose,original_successes=result['successes'],episodes=result['attempts'],
            original_gate_passed=result['successes']/result['attempts']>=plan['condition_success_rate_min'],
            wrong_release_count=len(missed),wrong_release_with_timely_hold=sum(e['has_timely_hold'] for e in missed),
            actual_requested_pose_count=sum(e['release_pose_valid'] for e in result['episodes']),
            total_timely_holds=sum(e['hold_start_seconds'] is not None for e in result['episodes']),
            missed_strata=missed,no_timely_hold=failures,trace_sha256=sha(path)))
    identical=all(np.array_equal(initial[0],v) for v in initial[1:])
    spread=np.ptp(initial[0],axis=0)
    return dict(conditions=rows,repeated_no_hold_environment_indices=dict(fail_counter),
        initial_physical_joint_qpos_identical_across_four_strata=identical,
        initial_physical_joint_qpos_spread_radians=spread.tolist(),
        caution='Post-hoc release coverage and hold counts do not replace the original four-stratum gates; pose/seed associations are not causal isolation')


def main():
    assert read(OUT/'progress.json')['status']=='completed-review-required'
    analysis=read(OUT/'analysis.json');assert analysis['actual_control_transitions']==308800 and analysis['learning_updates']==0
    result=dict(status='passed',walking=walking(),getup=recovery(),learning_updates=0,physics_steps=0,
        original_analysis_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        decision='Keep original results closed. Diagnose recovery interface, contact and release coverage before any new bounded physical protocol; no learning extension')
    (OUT/'failure-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],walking_failures=[e for r in result['walking'] for e in r['failures']],
        getup=[{k:v for k,v in r.items() if k not in ('missed_strata','no_timely_hold')} for r in result['getup']['conditions']],
        getup_no_holds={r['condition']:[{k:v for k,v in e.items() if k not in ('initial_actor_joint_estimates_radians','final_joint_positions_radians')} for e in r['no_timely_hold']] for r in result['getup']['conditions']},
        initial_physical_joint_qpos_spread_radians=result['getup']['initial_physical_joint_qpos_spread_radians'],physics_steps=0),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
