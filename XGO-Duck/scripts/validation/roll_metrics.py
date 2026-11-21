"""Phase-aware forward-roll assessment from saved physical trajectories.

The score cannot be earned by standing, a backward/side roll, an unsupported
flip, repeated revolutions, or a reset. Forward angle is geometric and yaw/sign
invariant; supported progress follows the source task's contact/flatness rule.
Completion is separate from the subsequent two-second stable landing hold.
"""
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1., atol=2e-4, rtol=0)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3,3))


def progress(trace):
    r = rotation(trace['quaternion'])
    lateral_horizontal = np.maximum(np.linalg.norm(r[...,:2,1], axis=-1), 1e-6)
    horizontal_forward = (r[...,0,0]*r[...,1,1]-r[...,1,0]*r[...,0,1])/lateral_horizontal
    pitch = np.arctan2(-r[...,2,0], horizontal_forward)
    delta = np.zeros_like(pitch)
    delta[1:] = (np.diff(pitch, axis=0)+np.pi)%(2*np.pi)-np.pi
    flat = np.clip((.866-np.abs(r[...,2,1]))/(.866-.5), 0, 1)
    flat = flat*flat*(3-2*flat)
    geometric = np.cumsum(delta, axis=0)
    supported = np.cumsum(np.where(delta<0, delta, delta*flat*trace['ground_contact']), axis=0)
    return geometric, supported, r


def summarize(trace, plan):
    dt=plan['step_dt']; times=np.asarray(trace['time']); h=len(times)
    np.testing.assert_allclose(times, np.arange(h)*dt, atol=1e-7, rtol=0)
    assert h==round(plan['max_seconds']/dt)+1
    active=np.asarray(trace['active']);n=active.shape[1]
    assert not np.any(np.diff(active.astype(int),axis=0)>0), 'Resets cannot erase failed first episodes'
    geom,supported,r=progress(trace)
    head_top_z=rotation(trace['head_quaternion'])[...,2,1]  # XgoDuck jaw_soft local +Y.
    tilt=np.arccos(np.clip(r[...,2,2],-1,1))
    stable=(np.abs(trace['height']-plan['nominal_height_m'])<=plan['height_tolerance_m'])
    stable&=tilt<=np.deg2rad(plan['tilt_max_deg'])
    stable&=np.linalg.norm(trace['angular_velocity'],axis=-1)<=plan['angular_speed_max_rad_s']
    stable&=np.all(trace['foot_contact'],axis=-1)&trace['joint_envelope_ok']&active
    lower=np.deg2rad(360-plan['completion_tolerance_deg']); upper=np.deg2rad(360+plan['completion_tolerance_deg'])
    hold=round(plan['hold_seconds']/dt); rows=[]
    for i in range(n):
        head=None; first_feet=None; completion=None; hold_start=None
        for k in range(h):
            if not active[k,i]:break
            if first_feet is None and np.all(trace['foot_contact'][k,i]):first_feet=k
            if (head is None and first_feet is not None and first_feet<k and
                np.deg2rad(20)<geom[k,i]<np.deg2rad(170) and
                trace['head_contact'][k,i] and head_top_z[k,i]<-.3):head=k
            if head is not None and min(geom[k,i],supported[k,i])>=lower:
                completion=k;break
        if completion is not None and times[completion]<=plan['deadline_seconds']+1e-8:
            for start in range(completion+1,h-hold+1):
                onset=times[start]-dt
                if onset>plan['deadline_seconds']+1e-8:break
                end=start+hold
                if stable[start:end,i].all() and np.all((geom[start:end,i]>=lower)&(geom[start:end,i]<=upper)):
                    hold_start=start;break
        last=h if hold_start is None else hold_start+hold
        path=active[:last,i]
        no_repeated_roll=bool(np.all(geom[:last,i][path]<=upper))
        no_backward_roll=bool(np.all(geom[:last,i][path]>=-np.deg2rad(plan['backward_tolerance_deg'])))
        joints_ok=bool(trace['joint_envelope_ok'][:last,i][path].all())
        head_depth=float(np.max(trace['head_penetration_m'][:last,i][path],initial=0))
        contact_ok=head_depth<=plan['head_penetration_max_m']
        initial_ok=(tilt[0,i]<=np.deg2rad(plan['tilt_max_deg']) and
                    abs(trace['height'][0,i]-plan['nominal_height_m'])<=plan['height_tolerance_m'])
        success=bool(hold_start is not None and no_repeated_roll and no_backward_roll and joints_ok and contact_ok and initial_ok)
        reasons=[]
        if not initial_ok:reasons.append('invalid-standing-start')
        if head is None:reasons.append('no-feet-headtop-sequence')
        if completion is None:reasons.append('no-supported-forward-revolution')
        if hold_start is None:reasons.append('no-timely-two-second-landing-hold')
        if not no_repeated_roll:reasons.append('excess-revolution')
        if not no_backward_roll:reasons.append('backward-rotation')
        if not joints_ok:reasons.append('actual-joint-limit-exceeded')
        if not contact_ok:reasons.append('head-penetration-exceeded')
        rows.append(dict(env_index=i,success=success,reasons=reasons,
            head_contact_seconds=None if head is None else float(times[head]),
            completion_seconds=None if completion is None else float(times[completion]),
            hold_start_seconds=None if hold_start is None else float(times[hold_start]-dt),
            hold_end_seconds=None if hold_start is None else float(times[hold_start+hold-1]),
            geometric_max_deg=float(np.rad2deg(geom[:,i][active[:,i]]).max(initial=0)),
            supported_max_deg=float(np.rad2deg(supported[:,i][active[:,i]]).max(initial=0)),
            head_penetration_max_m=head_depth,
            joint_violation_samples=int((~trace['joint_envelope_ok'][:,i]&active[:,i]).sum())))
    return dict(attempts=n,successes=sum(e['success'] for e in rows),episodes=rows)
