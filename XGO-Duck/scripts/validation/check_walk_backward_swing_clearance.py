"""Check exact swing-sole clearance on saved stalled states without stepping."""
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
import torch

from check_walk_lateral_kinematics import addresses
import collect_walk_backward_onset as onset
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_roll_foot_transfer import SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_swing_clearance.json'
OUT = ROOT/'records/reports/common-walk-backward-swing-clearance'


def swing(model, geometry, sites, q, joints, side, spec):
    names,coords,dofs,limits = joints
    mask = np.array([name.startswith(side+'_') for name in names])
    assert int(mask.sum())==5
    coords,dofs,limits = coords[mask],dofs[mask],limits[mask]
    data = geometry.data
    selected = spec['swing_sides'].index(side)
    stance = 1-selected
    geom,vertices,_ = geometry.feet[side]

    def clearance(pose):
        # These archives have qpos and contact but no qvel. Only position
        # kinematics is needed to compare the sole mesh with the fixed plane.
        data.qpos[:] = pose
        mujoco.mj_kinematics(model,data)
        np.testing.assert_array_equal(data.qpos,pose)
        assert data.time==0.
        world = vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        normal = data.geom_xmat[geometry.floor].reshape(3,3)[:,2]
        return float(1000*np.min((world-data.geom_xpos[geometry.floor])@normal))

    before = clearance(q)
    feet = data.site_xpos[sites].copy()
    frames = data.site_xmat[sites].reshape(2,3,3).copy()
    normal = data.geom_xmat[geometry.floor].reshape(3,3)[:,2].copy()
    np.testing.assert_allclose(normal,[0.,0.,1.],atol=1e-9,rtol=0)
    pos,rotation = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    # The position-stage COM transform initializes cdof for the site Jacobian.
    mujoco.mj_comPos(model,data)
    mujoco.mj_jacSite(model,data,pos,rotation,int(sites[selected]))
    analytic = pos[:,dofs]
    epsilon = 1e-6
    numeric = np.empty_like(analytic)
    for j,coord in enumerate(coords):
        pose = q.astype(float).copy()
        pose[coord] += epsilon
        clearance(pose)
        above = data.site_xpos[sites[selected]].copy()
        pose[coord] -= 2*epsilon
        clearance(pose)
        numeric[:,j] = (above-data.site_xpos[sites[selected]])/(2*epsilon)
    assert np.linalg.norm(analytic)>1e-6
    np.testing.assert_allclose(analytic,numeric,atol=2e-8,rtol=1e-6)
    cap = spec['maximum_joint_offset_rad']
    lower,upper = np.maximum(-cap,limits[:,0]-q[coords]),np.minimum(cap,limits[:,1]-q[coords])
    if not (lower<upper).all():
        return dict(feasible=False,reason='no-joint-feasible-interval')
    orient = spec['orientation_penalty_m_per_rad']
    system = np.vstack((analytic,orient*rotation[:,dofs],.002*np.eye(len(dofs))))
    goal = np.r_[normal*spec['site_lift_target_m'],np.zeros(3+len(dofs))]
    fit = lsq_linear(system,goal,bounds=(lower,upper),method='bvls',tol=1e-12)
    assert fit.success and np.isfinite(fit.x).all() and abs(fit.x).max()<=cap+1e-10
    envelope = []
    for amount in (.25,.5,.75,1.):
        target = q.astype(float).copy()
        target[coords] += amount*fit.x
        assert (target[coords]>=limits[:,0]-1e-10).all() and (target[coords]<=limits[:,1]+1e-10).all()
        current = clearance(target)
        motion = data.site_xpos[sites]-feet
        final_frame = data.site_xmat[sites[selected]].reshape(3,3)
        angle = np.rad2deg(np.arccos(np.clip((np.trace(final_frame@frames[selected].T)-1.)/2.,-1.,1.)))
        envelope.append(dict(fraction=amount,sole_clearance_mm=current,
            stance_site_motion_m=float(np.linalg.norm(motion[stance])),
            swing_site_horizontal_motion_m=float(np.linalg.norm(motion[selected,:2])),swing_rotation_deg=float(angle)))
    last = envelope[-1]
    rise = last['sole_clearance_mm']-before
    feasible = (rise>=1000*spec['minimum_sole_mesh_rise_m'] and
        last['sole_clearance_mm']>=1000*spec['minimum_final_plane_clearance_m'] and
        all(row['stance_site_motion_m']<=spec['maximum_stance_site_motion_m'] and
            row['swing_site_horizontal_motion_m']<=spec['maximum_swing_horizontal_site_motion_m'] and
            row['swing_rotation_deg']<=spec['maximum_swing_foot_rotation_deg'] for row in envelope))
    return dict(feasible=bool(feasible),swing_joints=np.asarray(names)[mask].tolist(),
                delta_rad=fit.x.tolist(),sole_rise_mm=float(rise),initial_clearance_mm=before,
                jacobian_max_error_m_per_rad=float(np.max(abs(analytic-numeric))),
                samples=envelope)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-kinematics-only'
    assert not any(spec[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==spec['stage'] and ROOT/stage['current_protocol']==PLAN
    for file,expected in spec['sources'].items():
        assert sha(ROOT/file)==expected,file
    reference = ROOT/spec['parent_protocol']
    parent = read(reference)
    report_path = ROOT/spec['parent_report']
    prior = read(report_path)
    assert prior['status']=='closed-mechanism-gate-not-met' and not prior['heldout_used']
    assert prior['protocol_sha256']==sha(reference)
    p = onset.environment(parent,spec['seeds'][0])
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(ROOT/spec['scene']) as saved:
        compare_arrays(model_arrays(model),{name:saved[name] for name in saved.files})
    joints = addresses(model,p)
    geometry = SoleGeometry(model)
    sites = np.asarray([model.site(f'robot/{side}_foot').id for side in spec['swing_sides']])
    cohorts = []
    for seed in spec['seeds']:
        proof = report_path.with_name(f'seed{seed}-analysis.json')
        assert sha(proof)==prior['sources'][proof.name]
        checked = read(proof)
        assert checked['status']=='passed-integrity' and checked['protocol_sha256']==sha(reference)
        assert not checked['heldout_used'] and not checked['learning_updates']
        record = ROOT/checked['record']
        prefix = record/'trace-shared-prefix.npz'
        assert sha(prefix)==checked['shared_prefix_sha256'] and sha(record/'protocol.json')==sha(reference)
        with np.load(prefix) as saved:
            state,feet = (saved[name][-1] for name in ('qpos','contact'))
        worlds = checked['common_near_static']
        assert len(worlds)>0 and len(set(worlds))==len(worlds)
        rows = []
        for world in worlds:
            both = bool(feet[world].all())
            trials = {side:swing(model,geometry,sites,state[world],joints,side,spec)
                      if both else dict(feasible=False,reason='saved-state-not-double-support') for side in spec['swing_sides']}
            rows.append(dict(world=world,native_bilateral_contact=both,sides=trials))
        counts = {side:sum(row['sides'][side]['feasible'] for row in rows) for side in spec['swing_sides']}
        cohorts.append(dict(seed=seed,prefix_sha256=sha(prefix),stalls=len(worlds),feasible=counts,rows=rows))
    admitted = {side:all(row['feasible'][side]/row['stalls']>=spec['minimum_feasible_fraction_each_seed']
                         for row in cohorts) for side in spec['swing_sides']}
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-clearance-review',protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)),cohorts=cohorts,geometry_admission=admitted,
        source_readback_previous_attempt=dict(exit_code=1,error="KeyError: qvel is not a file in the archive",
            previous_script_sha256='118769f4ed1ff222755dd09a0607e49a57270295805f1eab57f9f72494a4fb0f',
            resolved_with='Pure position kinematics from archived qpos and original contact labels'),
        dynamic_response_verified=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_SWING_CLEARANCE',[(row['seed'],row['stalls'],row['feasible']) for row in cohorts],admitted,flush=True)


if __name__=='__main__':
    main()
