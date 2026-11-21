"""Verify the proposed lateral push and timing against a saved nominal sequence."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'records/reports/continuous-forward-yaw-pair'
DRAFT = ROOT/'configs/experiments/continuous_forward_lateral_push_draft.json'
DEST = OUT/'lateral-push-geometry-preflight.json'


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lateral_world(quaternion,magnitude):
    w,x,y,z = np.asarray(quaternion,dtype=np.float64).T
    np.testing.assert_allclose(w*w+x*x+y*y+z*z,1.,rtol=0,atol=2e-4)
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    delta = magnitude*np.column_stack((-np.sin(yaw),np.cos(yaw),np.zeros_like(yaw)))
    return yaw,delta


def recovery_window(trace,control,plan):
    rules = plan['walking']['criteria']
    length = rules['block_steps']
    assert length==50 and control+2*length<len(trace['active'])
    full = slice(control+1,control+2*length+1)
    measured = slice(control+length+1,control+2*length+1)
    speed = trace['velocity'][measured].mean(axis=0)
    error = np.linalg.norm(speed-np.asarray(plan['walking']['commands']['forward'])[:2],axis=1)
    return (trace['active'][full].all(axis=0)&trace['joint_envelope_ok'][full].all(axis=0)
            &(speed[:,0]>=rules['signed_speed']['forward'])&(error<=rules['linear_error']['forward']))


def main():
    assert not DEST.exists()
    draft,proof = load(DRAFT),load(OUT/'analysis.json')
    assert draft['status']=='draft-not-executable' and not draft['heldout_used']
    assert proof['passed_development_gate'] and proof['status']=='passed-independent-verification'
    folder = ROOT/proof['record']
    protocol = folder/'protocol.json'
    assert sha(protocol)==proof['protocol_sha256']
    p = load(protocol)
    fork = draft['proposed_shared_fork_control']
    assert fork==sum(p[key] for key in ('initial_stand_controls','roll_controls','landing_stand_controls'))
    assert draft['proposed_maximum_unique_controls']==fork+3*(p['maximum_controls']-fork)
    assert draft['proposed_worlds']==p['worlds'] and not proof['heldout_used']
    stock = ROOT/draft['reference_source']
    source = stock.read_text(encoding='utf-8')
    assert 'VELOCITY_PUSH_RANGE = (-0.3, 0.3)' in source
    assert 'VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)' in source
    assert 'func=mdp.push_by_setting_velocity' in source
    wide = proof['branches']['wider-input']
    start = np.asarray(wide['forward_at'])
    eligible = start>=0
    successful = np.array([row['success'] for row in wide['forward']])
    assert (start[eligible]<=fork).all() and (fork+100<=start[eligible]+p['forward_controls']).all()
    archive = folder/'clips/wider-input/trace.npz'
    assert sha(archive)==proof['source_sha256'][archive.relative_to(ROOT).as_posix()]
    local = fork-proof['shared_controls']
    with np.load(archive) as data:
        trace = {name:data[name] for name in ('active','velocity','joint_envelope_ok')}
        original = data['qvel'][local,eligible].copy()
        quaternion = data['quaternion'][local,eligible]
        raw_quaternion = data['qpos'][local,eligible,3:7]
        np.testing.assert_allclose(quaternion,raw_quaternion,atol=2e-7,rtol=0)
        quaternion_difference = float(np.abs(quaternion-raw_quaternion).max())
        assert trace['active'][local,eligible].all()
        recovery = recovery_window(trace,local,p)
    role_values = draft['proposed_roles_body_horizontal_lateral_m_s']
    assert len(role_values)==3 and role_values['left-push']==-role_values['right-push']==.08
    results = {}
    for role,value in role_values.items():
        yaw,delta = lateral_world(quaternion,value)
        forward = np.column_stack((np.cos(yaw),np.sin(yaw),np.zeros_like(yaw)))
        np.testing.assert_allclose((forward*delta).sum(axis=1),0.,atol=1e-12,rtol=0)
        candidate = original.copy()
        candidate[:,:3] += delta.astype(candidate.dtype)
        np.testing.assert_array_equal(candidate[:,2:],original[:,2:])
        measured = (candidate-original)[:,:2]
        np.testing.assert_allclose(np.linalg.norm(measured,axis=1),abs(value),atol=5e-8,rtol=0)
        results[role] = dict(target_lateral_m_s=value,first_order_horizontal_velocity_max_m_s=float(
            np.linalg.norm(measured,axis=1).max()),joint_angular_vertical_velocity_unchanged=True)
    count = int((successful&recovery).sum())
    verdict = count>=p['minimum_continuous_successes']
    data = dict(status='passed-saved-geometry-preflight',fork_control=fork,eligible_worlds=int(eligible.sum()),
        nominal_full_and_recovery_successes=count,nominal_screen_passed=verdict,roles=results,
        maximum_link_vs_free_joint_quaternion_rounding=quaternion_difference,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in
                       (Path(__file__),DRAFT,OUT/'analysis.json',protocol,archive,stock)},
        scope='Saved nominal trajectory and arithmetic only. Actual CUDA writer, injected delta and physical response remain to be verified.',
        physical_substeps=0,heldout_used=False)
    DEST.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('CONTINUOUS_LATERAL_PUSH_GEOMETRY',data['eligible_worlds'],count,verdict,results,flush=True)


if __name__=='__main__':
    main()
