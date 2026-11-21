"""Solve the saved side shift with actuator bounds enforced inside the fit."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

import collect_walk_backward_onset as onset
from check_walk_lateral_kinematics import (MAX_FOOT_DRIFT_M,MAX_TARGET_RAD,RIDGE_M_PER_RAD,
                                         TARGET_M,addresses)
from collect_walk_backward_lateral_probe import OUT,PLAN,SEEDS
from common_scene_contract import compare_arrays,model_arrays
from record_io import write
from s1_reference_inputs import ROOT,read,sha


def target(model,q,sites,coords,vel,limits,direction):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model,data)
    feet = data.site_xpos[sites].copy()
    frames = data.site_xmat[sites].reshape(2,3,3).copy()
    jac = []
    for site in sites:
        positional = np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,positional,None,int(site))
        jac.append(positional[:,vel])
    w,x,y,z = q[3:7]
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    lateral = np.array([-np.sin(yaw),np.cos(yaw),0.])
    shift = direction*TARGET_M*lateral
    objective = np.vstack((*jac,RIDGE_M_PER_RAD*np.eye(len(vel))))
    rhs = np.concatenate((-np.tile(shift,2),np.zeros(len(vel))))
    lower = np.maximum(-MAX_TARGET_RAD,limits[:,0]-q[coords])
    upper = np.minimum(MAX_TARGET_RAD,limits[:,1]-q[coords])
    assert (lower<upper).all()
    result = lsq_linear(objective,rhs,bounds=(lower,upper),method='bvls',tol=1e-12)
    assert result.success and np.isfinite(result.x).all()
    proposed = q.astype(float).copy()
    proposed[:3] += shift
    proposed[coords] += result.x
    data.qpos[:] = proposed
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,proposed)
    error = np.linalg.norm(data.site_xpos[sites]-feet,axis=1)
    current = data.site_xmat[sites].reshape(2,3,3)
    rotations = np.einsum('nij,nkj->nik',current,frames)
    degrees = np.rad2deg(np.arccos(np.clip((np.trace(rotations,axis1=1,axis2=2)-1)/2,-1.,1.)))
    assert abs(result.x).max()<=MAX_TARGET_RAD+1e-10
    assert ((proposed[coords]>=limits[:,0]-1e-10)&(proposed[coords]<=limits[:,1]+1e-10)).all()
    actual = ((proposed[:3]-data.site_xpos[sites].mean(axis=0))-(q[:3]-feet.mean(axis=0)))@lateral
    return dict(target_rad=result.x.tolist(),maximum_joint_delta_rad=float(abs(result.x).max()),
        exact_foot_drift_mm=(1000*error).tolist(),foot_frame_rotation_degrees=degrees.tolist(),
        exact_body_relative_to_midfoot_mm=float(1000*actual),feasible=bool(error.max()<=MAX_FOOT_DRIFT_M))


def main():
    from mjlab.scene import Scene

    path = OUT/'bounded-kinematics-preflight.json'
    source = OUT/'kinematics-preflight.json'
    prior = read(source)
    assert not path.exists() and prior['status']=='passed-saved-kinematics' and not prior['heldout_used']
    spec = read(PLAN)
    cfg,_ = onset.config(onset.environment(spec,SEEDS[0]))
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(OUT/'cpu-scene-fields.npz') as proof:
        compare_arrays(model_arrays(model),{key:proof[key] for key in proof.files})
    names,coords,vel,limits = addresses(model,spec['environment'])
    assert names==prior['joints']
    sites = np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'robot/'+side+'_foot')
                        for side in ('left','right')])
    assert (sites>=0).all()
    cohorts = []
    for c in prior['cohorts']:
        raw = read(OUT/f'seed{c["seed"]}-analysis.json')
        prefix = ROOT/raw['record']/'trace-shared-prefix.npz'
        assert sha(prefix)==c['source_prefix_sha256']==raw['shared_prefix_sha256']
        entries = []
        with np.load(prefix) as archive:
            for old in c['rows']:
                i = old['world']
                assert archive['contact'][-1,i].all()
                q = archive['qpos'][-1,i]
                entries.append(dict(world=i,left=target(model,q,sites,coords,vel,limits,1.),
                    right=target(model,q,sites,coords,vel,limits,-1.)))
        cohorts.append(dict(seed=c['seed'],worlds=len(entries),rows=entries,
            unconstrained_passed=c['feasible_both'],
            bounded_passed=sum(row['left']['feasible'] and row['right']['feasible'] for row in entries)))
    write(path,dict(status='passed-saved-bounded-kinematics',source_sha256=sha(source),script_sha256=sha(Path(__file__)),
        maximum_joint_delta_rad=MAX_TARGET_RAD,maximum_foot_drift_mm=MAX_FOOT_DRIFT_M*1000,
        body_shift_target_mm=TARGET_M*1000,cohorts=cohorts,physical_substeps=0,learning_updates=0,
        heldout_used=False,dynamic_response_verified=False))
    print('WALK_BOUNDED_KINEMATICS',[(row['seed'],row['worlds'],row['unconstrained_passed'],row['bounded_passed'])
           for row in cohorts],flush=True)


if __name__=='__main__':
    main()
