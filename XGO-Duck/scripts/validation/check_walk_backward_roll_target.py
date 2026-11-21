"""Verify a stance-preserving roll target on saved backward stalls."""
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from check_walk_lateral_kinematics import addresses
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import onset, SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_roll_target.json'
OUT = ROOT/'records/reports/common-walk-backward-roll-target'
SIDES = ('left','right')


def sole_measure(geometry, sites, pose):
    model,data = geometry.model,geometry.data
    data.qpos[:] = pose
    mujoco.mj_kinematics(model,data)
    assert data.time==0.
    ground = data.geom_xpos[geometry.floor]
    normal = data.geom_xmat[geometry.floor].reshape(3,3)[:,2]
    gaps = []
    for side in SIDES:
        geom,vertices,_ = geometry.feet[side]
        world = vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        gaps.append(np.min((world-ground)@normal))
    return data.site_xpos[sites].copy(),data.site_xmat[sites].reshape(2,3,3).copy(),np.asarray(gaps),normal.copy()


def target(geometry,sites,pose,coords,limits,stance,d):
    cap = d['maximum_joint_offset_rad']
    shift = np.asarray(d['maximum_root_translation_m'])
    if not ((pose[coords]>=limits[:,0])&(pose[coords]<=limits[:,1])).all():
        return dict(feasible=False,reason='initial-joints-outside-target-limits')
    lower = np.r_[-shift,np.maximum(-cap,limits[:,0]-pose[coords])]
    upper = np.r_[shift,np.minimum(cap,limits[:,1]-pose[coords])]
    if not (lower<upper).all():
        return dict(feasible=False,reason='no-bounded-target-interval')
    swing = 1-stance
    xyz,frames,gap,normal = sole_measure(geometry,sites,pose)
    np.testing.assert_allclose(normal,[0.,0.,1.],rtol=0,atol=1e-9)
    original = Rotation.from_quat(np.roll(pose[3:7],-1))
    angle = np.deg2rad(d['roll_toward_stance_deg'])*(1 if stance==1 else -1)
    goal = xyz.copy()
    goal[swing] += d['swing_site_lift_target_m']*normal

    def at(values,fraction=1.):
        state = pose.astype(float).copy()
        state[:3] += fraction*values[:3]
        state[3:7] = np.roll((original*Rotation.from_rotvec([angle*fraction,0.,0.])).as_quat(),1)
        state[coords] += fraction*values[3:]
        return state

    def angles(current):
        return Rotation.from_matrix(np.einsum('nij,nkj->nik',current,frames)).as_rotvec()

    def residual(values):
        current,orient,clear,_ = sole_measure(geometry,sites,at(values))
        difference = current-goal
        difference[stance] *= d['stance_position_weight']
        gap_goal = d['minimum_swing_plane_clearance_m']+d['target_gap_margin_m']
        return np.r_[difference.ravel(),d['orientation_weight_m_per_rad']*angles(orient).ravel(),
                     d['stance_gap_weight']*(clear[stance]-gap[stance]),
                     d['swing_gap_weight']*max(0.,gap_goal-clear[swing]),
                     d['root_delta_penalty']*values[:3],d['joint_delta_penalty_m_per_rad']*values[3:]]

    fit = least_squares(residual,np.zeros(len(lower)),bounds=(lower,upper),max_nfev=d['maximum_solver_evaluations'],
                        ftol=1e-9,xtol=1e-9,gtol=1e-9)
    assert np.isfinite(fit.x).all() and abs(fit.x[3:]).max()<=cap+1e-10
    stages = []
    for fraction in (.25,.5,.75,1.):
        q = at(fit.x,fraction)
        assert ((q[coords]>=limits[:,0]-1e-10)&(q[coords]<=limits[:,1]+1e-10)).all()
        current,orient,clear,_ = sole_measure(geometry,sites,q)
        move = current-xyz
        stages.append(dict(fraction=fraction,stance_site_motion_m=float(np.linalg.norm(move[stance])),
            stance_plane_gap_change_m=float(clear[stance]-gap[stance]),
            swing_horizontal_motion_m=float(np.linalg.norm(move[swing,:2])),
            maximum_foot_rotation_deg=float(np.rad2deg(np.linalg.norm(angles(orient),axis=-1)).max()),
            swing_plane_clearance_m=float(clear[swing]),swing_sole_rise_m=float(clear[swing]-gap[swing])))
    final = stages[-1]
    feasible = (fit.success and final['swing_plane_clearance_m']>=d['minimum_swing_plane_clearance_m'] and
        final['swing_sole_rise_m']>=d['minimum_swing_sole_rise_m'] and
        all(row['stance_site_motion_m']<=d['maximum_stance_site_motion_m'] and
            abs(row['stance_plane_gap_change_m'])<=d['maximum_stance_plane_gap_change_m'] and
            row['swing_horizontal_motion_m']<=d['maximum_swing_horizontal_motion_m'] and
            row['maximum_foot_rotation_deg']<=d['maximum_foot_rotation_deg'] for row in stages))
    return dict(feasible=bool(feasible),solver_converged=bool(fit.success),root_translation_m=fit.x[:3].tolist(),
                joint_delta_rad=fit.x[3:].tolist(),stages=stages)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-kinematics-only'
    assert not any(spec[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==spec['stage'] and ROOT/stage['current_protocol']==PLAN
    files = {}
    for key in ('axis_evidence','swing_evidence','parent_protocol'):
        path,expected = spec[key]
        assert sha(ROOT/path)==expected
        files[key] = ROOT/path
    axis = read(files['axis_evidence'])
    swing = read(files['swing_evidence'])
    assert axis['status']=='passed-saved-attitude-review' and axis['roll_target_geometry_admitted']
    assert swing['status']=='passed-saved-clearance-review' and not any(swing['geometry_admission'].values())
    assert not axis['heldout_used'] and not swing['heldout_used']
    parent = read(files['parent_protocol'])
    p = onset.environment(parent,spec['seeds'][0])
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    proof_dir = ROOT/'records/reports/common-walk-backward-actuator-support'
    scene = proof_dir/'cpu-scene-fields.npz'
    assert sha(scene)==spec['scene_sha256']
    with np.load(scene) as saved:
        compare_arrays(model_arrays(model),{name:saved[name] for name in saved.files})
    names,coords,_,limits = addresses(model,p)
    geometry = SoleGeometry(model)
    sites = np.array([model.site(f'robot/{side}_foot').id for side in SIDES])
    cohorts = []
    for seed,old in zip(spec['seeds'],swing['cohorts'],strict=True):
        assert seed==old['seed']
        proof = read(proof_dir/f'seed{seed}-analysis.json')
        assert proof['status']=='passed-integrity' and not proof['heldout_used'] and not proof['learning_updates']
        assert proof['protocol_sha256']==sha(files['parent_protocol'])
        record = ROOT/proof['record']
        path = record/'trace-shared-prefix.npz'
        assert sha(path)==old['prefix_sha256']==proof['shared_prefix_sha256']
        with np.load(path) as saved:
            poses,feet = saved['qpos'][-1],saved['contact'][-1]
        worlds = [row['world'] for row in old['rows']]
        assert worlds==proof['common_near_static'] and feet[worlds].all()
        rows = [dict(world=i,stance={side:target(geometry,sites,poses[i],coords,limits,k,spec['design'])
                    for k,side in enumerate(SIDES)}) for i in worlds]
        counts = {side:sum(row['stance'][side]['feasible'] for row in rows) for side in SIDES}
        cohorts.append(dict(seed=seed,worlds=len(rows),counts=counts,rows=rows))
    admitted = {side:all(row['counts'][side]/row['worlds']>=spec['design']['minimum_feasible_fraction_each_seed']
                        for row in cohorts) for side in SIDES}
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-roll-target-geometry',protocol_sha256=sha(PLAN),
         script_sha256=sha(Path(__file__)),joint_names=names,cohorts=cohorts,geometry_admission=admitted,
         dynamic_execution_authorized=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_ROLL_TARGET',[(row['seed'],row['worlds'],row['counts']) for row in cohorts],admitted,flush=True)


if __name__=='__main__':
    main()
