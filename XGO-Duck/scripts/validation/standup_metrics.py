"""Phase-aware recovery criteria, independent of the simulator and policy.

A fallen initial pose is expected. Success requires the requested pose at
release, recovery by the deadline, then an uninterrupted stable hold. A timeout
cannot reset away a failure or truncate the hold. Frames are not trials.
"""
import numpy as np


def projected_gravity(quaternion):
    q=np.asarray(quaternion,dtype=float)
    np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1.,atol=2e-4,rtol=0)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((-2*(x*z-w*y),-2*(y*z+w*x),-(1-2*(x*x+y*y))),axis=-1)


def classify_release(gravity,minimum_alignment=.75):
    gravity=np.asarray(gravity)
    result=[]
    for g in gravity:
        axis=int(np.argmax(np.abs(g)))
        if abs(g[axis])<minimum_alignment or axis==2:result.append('other')
        elif axis==0:result.append('prone' if g[0]>0 else 'supine')
        else:result.append('left-side' if g[1]>0 else 'right-side')
    return result


def summarize(trace,plan):
    dt=float(plan['step_dt']);times=np.asarray(trace['time'])
    n=trace['height'].shape[1];horizon=len(times)
    np.testing.assert_allclose(times,np.arange(1,horizon+1)*dt,atol=1e-7,rtol=0)
    assert horizon==round(plan['max_seconds']/dt)
    release=int(round(plan['freeze_seconds']/dt))-1
    pose=classify_release(projected_gravity(trace['quaternion'][release]),plan['release_alignment_min'])
    tilt=np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1,1))
    upright=(np.abs(trace['height']-plan['nominal_height_m'])<=plan['height_tolerance_m'])
    upright&=tilt<=np.deg2rad(plan['tilt_max_deg'])
    upright&=np.linalg.norm(trace['angular_velocity'],axis=-1)<=plan['angular_speed_max_rad_s']
    upright&=np.all(trace['foot_contact'],axis=-1)
    upright&=trace['joint_envelope_ok'] & trace['active']
    hold_steps=round(plan['hold_seconds']/dt)
    rows=[]
    for env in range(n):
        requested=trace['requested_pose'][env]
        found=None
        for start in range(release,horizon-hold_steps+1):
            # Samples describe the interval ending at each time. A complete
            # three-second hold needs 150 successive samples, not 149 gaps.
            onset=times[start]-dt
            if onset<plan['freeze_seconds']-1e-8:continue
            if onset>plan['deadline_seconds']+1e-8:break
            if upright[start:start+hold_steps,env].all():found=start;break
        valid_pose=pose[env]==requested
        success=valid_pose and found is not None
        reason='success' if success else ('wrong-release-pose' if not valid_pose else 'no-timely-stable-hold')
        rows.append(dict(env_index=env,requested_pose=requested,release_pose=pose[env],release_pose_valid=valid_pose,
            success=success,outcome=reason,hold_start_seconds=None if found is None else float(times[found]-dt),
            hold_end_seconds=None if found is None else float(times[found+hold_steps-1]),
            upright_fraction_after_release=float(upright[release+1:,env].mean()),
            joint_violation_samples=int((~trace['joint_envelope_ok'][release+1:,env]).sum())))
    return dict(episodes=rows,successes=sum(x['success'] for x in rows),attempts=n,
                wrong_release_pose=sum(not x['release_pose_valid'] for x in rows),
                scope='Recovery and stable hold only; no walking transition or hardware claim')
