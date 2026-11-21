"""Additional prospective final-window hold; retain original recovery scores."""
import numpy as np
from standup_metrics import projected_gravity
from getup_coverage_metrics import group_rows


def evaluate_end_hold(trace,p,original_rows):
    c=p['criteria'];dt=c['step_dt'];end=c['max_seconds'];start=end-c['hold_seconds']
    indices=np.flatnonzero(np.asarray(trace['time'])>start+1e-8)
    assert len(indices)==round(c['hold_seconds']/dt) and abs(start-c['deadline_seconds'])<1e-7
    tilt=np.arccos(np.clip(-projected_gravity(trace['quaternion'][indices])[...,2],-1,1))
    stable=abs(trace['height'][indices]-c['nominal_height_m'])<=c['height_tolerance_m']
    stable &= tilt<=np.deg2rad(c['tilt_max_deg'])
    stable &= np.linalg.norm(trace['angular_velocity'][indices],axis=-1)<=c['angular_speed_max_rad_s']
    stable &= trace['foot_contact'][indices].all(-1)&trace['joint_envelope_ok'][indices]&trace['active'][indices]
    flags=stable.all(0)
    rows=[dict(r,hold_start_seconds=float(start) if flags[i] else None,
               success=bool(flags[i] and r['release_pose_valid'])) for i,r in enumerate(original_rows)]
    result=group_rows(rows,p['success_rate_min'],p['actual_release_minimum'])
    result.update(start_seconds=float(start),end_seconds=float(end),stable_by_environment=flags.tolist(),
                  original_recovery_score_unchanged=True,scope='Additional final3s standing gate, frozen before new dynamics')
    return result
