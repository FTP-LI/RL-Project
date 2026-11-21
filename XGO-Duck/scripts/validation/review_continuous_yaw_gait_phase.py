"""Separate repeated gait yaw from accumulated heading on saved forward tracks."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'records/reports/continuous-roll-projected-entry'
PROOF = OUT/'analysis.json'
DEST = OUT/'forward-yaw-gait-phase.json'


def sha(path):
    import hashlib

    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def landed_edges(contact):
    # Require three saved samples before and after the edge to skip chatter.
    return np.array([k for k in range(3,len(contact)-2) if not contact[k-3:k].any()
                     and contact[k:k+3].all()])


def summarize(track,world,begin,expected):
    end = begin+1001
    s = slice(begin+1,end)
    assert track['active'][s,world].all()
    yaw = track['angular_velocity'][s,world,2]
    np.testing.assert_allclose(abs(yaw).mean(),expected['yaw_error_rad_s'],atol=1e-7,rtol=0)
    w,x,y,z = track['quaternion'][s,world].astype(np.float64).T
    heading = np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    left = landed_edges(track['foot_contact'][s,world,0])
    gait = []
    for a,b in zip(left[:-1],left[1:]):
        if 15<=b-a<=65:
            time = np.linspace(a,b,32,endpoint=False)
            gait.append((int(b-a),np.interp(time,np.arange(a,b+1),yaw[a:b+1])))
    frequency,phase_r2 = None,None
    if len(gait)>=6:
        intervals,values = zip(*gait,strict=True)
        frequency = float(50./np.median(intervals))
        windows = np.asarray(values)
        early,late = np.array_split(windows,2)
        # Fit only in the first half, test only in the second. A constant from
        # the same fitting data is the reference for the descriptive R².
        reference = (late-early.mean())**2
        error = (late-early.mean(axis=0))**2
        phase_r2 = float(1-error.mean()/reference.mean())
    return dict(world=world,failed_checks=expected['failed_checks'],mean_abs_body_yaw_rad_s=float(abs(yaw).mean()),
        mean_signed_body_yaw_rad_s=float(yaw.mean()),net_heading_deg=float(np.rad2deg(heading[-1]-heading[0])),
        heading_range_deg=float(np.rad2deg(np.ptp(heading))),detected_cycles=len(gait),
        cadence_hz=frequency,held_half_phase_r2=phase_r2)


def main():
    proof = json.loads(PROOF.read_text(encoding='utf-8'))
    trace = ROOT/proof['record']/'trace.npz'
    assert not DEST.exists() and proof['status']=='passed-independent-verification' and not proof['heldout_used']
    assert sha(trace)==proof['source_sha256'][trace.relative_to(ROOT).as_posix()]
    rows = []
    with np.load(trace) as data:
        fields = {key:data[key] for key in ('active','quaternion','angular_velocity','foot_contact')}
        for world,expected in enumerate(proof['forward']):
            if expected['entered_forward'] and 'survived' not in expected['failed_checks']:
                rows.append(summarize(fields,world,expected['forward_start'],expected))
    groups = {}
    keys = ('mean_abs_body_yaw_rad_s','mean_signed_body_yaw_rad_s','net_heading_deg','heading_range_deg',
            'detected_cycles','cadence_hz','held_half_phase_r2')
    for label,failures in (('passes',[]),('yaw_only',['yaw_error'])):
        selected = [row for row in rows if row['failed_checks']==failures]
        groups[label] = dict(worlds=len(selected))
        for key in keys:
            values = [abs(row[key]) if key in ('net_heading_deg','mean_signed_body_yaw_rad_s') else row[key]
                      for row in selected if row[key] is not None]
            groups[label][key] = dict(count=len(values),quantiles=np.quantile(values,[.1,.5,.9]).tolist() if values else None)
    assert groups['passes']['worlds']==111 and groups['yaw_only']['worlds']==15
    report = dict(status='verified-saved-development-phase-review',groups=groups,rows=rows,heldout_used=False,
        zero_new_dynamics=True,headings_are_world_frame=True,angular_yaw_is_body_frame=True,
        absolute_value_in_group_summary=['net_heading_deg','mean_signed_body_yaw_rad_s'],
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (Path(__file__),PROOF,trace)})
    DEST.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(groups,indent=2),flush=True)


if __name__=='__main__':
    main()
