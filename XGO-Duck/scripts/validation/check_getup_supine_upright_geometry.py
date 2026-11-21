"""Check whether a small upright correction can preserve both supporting feet."""
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_getup_contact_entry import PAIRS
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_getup_nominal_supine import ROLES
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/'configs/experiments/getup_supine_upright_geometry_draft.json'
OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'
FIELDS = ('time','qpos','foot_contact','native_head_leg_contact','quaternion')


def checked_worlds(name):
    proof_path = ROOT/'records/reports'/name/'analysis.json'
    proof = read(proof_path)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    folder = ROOT/proof['record']
    assert sha(folder/'protocol.json')==proof['protocol_sha256']
    prefix = folder/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['prefix_sha256']
    worlds = set.intersection(*({row['env_index'] for row in proof['arms'][role]['timing']['episodes']
                                if row['requested_pose']=='supine' and not row['success']} for role in ROLES))
    assert len(worlds)==1
    world, = worlds
    with np.load(prefix) as values:
        shared = {key:values[key] if key=='time' else values[key][:,world] for key in FIELDS}
    for role in ROLES:
        branch = folder/f'trace-{role}.npz'
        assert sha(branch)==proof['arms'][role]['trace_sha256']
        with np.load(branch) as values:
            trace = {key:np.concatenate((shared[key],values[key] if key=='time' else values[key][:,world])) for key in FIELDS}
        gravity = projected_gravity(trace['quaternion'])
        qualifies = ((trace['time']>2.)&(gravity[:,2]>-np.cos(np.deg2rad(15.)))&trace['native_head_leg_contact']
                     &trace['foot_contact'].all(axis=-1))
        confirmed = np.flatnonzero(np.convolve(qualifies.astype(int),np.ones(5,dtype=int),mode='valid')==5)
        assert confirmed.size
        k = int(confirmed[0])+4
        yield dict(role=role,world=world,seconds=float(trace['time'][k]),source_sha256=sha(branch),
                   qpos=trace['qpos'][k],scene=folder/'scene-static-fields.npz')


def targets(model, row, spec, with_neck):
    q = row['qpos'].astype(float)
    names = [f'{side}_{joint}' for side in ('left','right')
             for joint in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]
    if with_neck:
        names.append('neck_pitch')
    joints = [model.joint('robot/'+name) for name in names]
    coords = np.asarray([x.qposadr.item() for x in joints])
    limits = np.asarray([x.range for x in joints])
    sites = [model.site('robot/'+side+'_foot').id for side in ('left','right')]
    head = model.geom('robot/head_collision').id
    legs = [model.geom('robot/'+part).id for part in PAIRS]
    data = mujoco.MjData(model)

    def geometry(pose):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model,data)
        closest = min(mujoco.mj_geomDistance(model,data,head,leg,.2,None) for leg in legs)
        return data.site_xpos[sites].copy(),data.site_xmat[sites].reshape(2,3,3).copy(),float(closest)

    feet, original_frames, baseline_gap = geometry(q)
    trunk = Rotation.from_quat(np.roll(q[3:7],-1))
    up = trunk.as_matrix()[:,2]
    axis = np.cross(up,[0.,0.,1.])
    assert np.linalg.norm(axis)>.1
    increment = np.deg2rad(spec['trunk_rotation_degrees'])
    oriented = q.copy()
    oriented[3:7] = np.roll((Rotation.from_rotvec(axis/np.linalg.norm(axis)*increment)*trunk).as_quat(),1)
    cap, shift = spec['maximum_joint_delta_rad'],spec['maximum_translation_m']
    lower = np.concatenate((np.full(3,-shift),np.maximum(-cap,limits[:,0]-q[coords])))
    upper = np.concatenate((np.full(3,shift),np.minimum(cap,limits[:,1]-q[coords])))
    if not (lower<upper).all():
        return dict(feasible=False,reason='joint-bound-no-feasible-interval')

    def pose(values):
        proposed = oriented.copy()
        proposed[:3] += values[:3]
        proposed[coords] += values[3:]
        return proposed

    target_gap = baseline_gap+spec['minimum_head_leg_gap_gain_m']

    def residual(values):
        xyz,frames,gap = geometry(pose(values))
        relative = np.swapaxes(original_frames,1,2)@frames
        error = Rotation.from_matrix(relative).as_rotvec()
        return np.concatenate(((xyz-feet).ravel(),.01*error.ravel(),[2.*max(0.,target_gap-gap)],.0008*values[3:]))

    fit = least_squares(residual,np.clip(np.zeros(len(lower)),lower,upper),bounds=(lower,upper),
                        max_nfev=160,gtol=1e-10,ftol=1e-10)
    candidate = pose(fit.x)
    xyz,frames,gap = geometry(candidate)
    foot_error = np.linalg.norm(xyz-feet,axis=1)
    sole_angle = np.rad2deg(Rotation.from_matrix(np.swapaxes(original_frames,1,2)@frames).magnitude())
    assert data.time==0. and abs(fit.x[3:]).max()<=cap+1e-10
    assert (candidate[coords]>=limits[:,0]-1e-10).all() and (candidate[coords]<=limits[:,1]+1e-10).all()
    initial_tilt = np.rad2deg(np.arccos(np.clip(up[2],-1.,1.)))
    updated_tilt = np.rad2deg(np.arccos(np.clip(Rotation.from_quat(np.roll(candidate[3:7],-1)).as_matrix()[2,2],-1.,1.)))
    assert abs(initial_tilt-updated_tilt-spec['trunk_rotation_degrees'])<1e-7
    passed = (fit.success and foot_error.max()<=spec['maximum_sole_position_error_m']
              and sole_angle.max()<=spec['maximum_sole_frame_error_degrees'] and gap>=target_gap-1e-7)
    return dict(feasible=bool(passed),solver_converged=bool(fit.success),joint_names=names,
        joint_delta_rad=fit.x[3:].tolist(),translation_m=fit.x[:3].tolist(),sole_error_mm=(foot_error*1000).tolist(),
        sole_angle_error_deg=sole_angle.tolist(),initial_tilt_deg=float(initial_tilt),target_tilt_deg=float(updated_tilt),
        initial_head_leg_gap_mm=1000*baseline_gap,target_head_leg_gap_mm=1000*gap)


def main():
    from mjlab.scene import Scene
    from getup_knee_support_supply_recovery import shared

    spec = read(PLAN)
    assert spec['status']=='draft-cpu-geometry-only' and not spec['heldout_used']
    source = ROOT/spec['source']
    assert read(source)['status']=='passed-paired-saved-review'
    frozen_plant = ROOT/'configs/experiments/getup_knee_support_supply_recovery.json'
    cfg, _, _ = shared.prepare(read(frozen_plant)['environment'])
    model = Scene(cfg.env.scene,device='cpu').compile()
    compiled = model_arrays(model)
    result = {}
    for name in spec['cohorts']:
        rows = []
        for row in checked_worlds(name):
            with np.load(row['scene']) as data:
                compare_arrays(compiled,{field:data[field] for field in data.files})
            row.update(legs=targets(model,row,spec,False),legs_and_neck=targets(model,row,spec,True))
            del row['qpos']
            row['scene'] = row['scene'].relative_to(ROOT).as_posix()
            rows.append(row)
        assert len(rows)==3
        result[name] = rows
    assert not torch.cuda.is_initialized()
    out = OUT/'nominal-supine-upright-geometry.json'
    assert not out.exists()
    write(out,dict(status='passed-saved-geometry-review',results=result,physical_substeps=0,
                   learning_updates=0,heldout_used=False,scope='A geometric projection on saved poses; no force or motion outcome.',
                   source_sha256={path.relative_to(ROOT).as_posix():sha(path)
                                  for path in (source,PLAN,frozen_plant,Path(__file__))}))
    print('GETUP_UPRIGHT_GEOMETRY',[(name,[(x['legs']['feasible'],x['legs_and_neck']['feasible']) for x in rows])
                                  for name,rows in result.items()],flush=True)


if __name__=='__main__':
    main()
