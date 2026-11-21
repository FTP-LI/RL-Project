"""Check bounded joint targets for a sustained, feet-fixed lateral shift."""
from pathlib import Path

import mujoco
import numpy as np

import collect_walk_backward_onset as onset
from collect_walk_backward_lateral_probe import OUT, PLAN, SEEDS
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


TARGET_M = .002
MAX_TARGET_RAD = .03
MAX_FOOT_DRIFT_M = .0005
RIDGE_M_PER_RAD = .002


def addresses(model,p):
    legs = [name for name in p['joint_names'] if name.startswith(('left_','right_'))]
    assert len(legs)==10
    joints = [model.joint('robot/'+name).id for name in legs]
    assert all(model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE for i in joints)
    return (legs,model.jnt_qposadr[joints],model.jnt_dofadr[joints],
            np.asarray([p['joint_limits'][p['joint_names'].index(name)] for name in legs]))


def local_target(model,positions,site_ids,coords,vel,limits,sign):
    data = mujoco.MjData(model)
    data.qpos[:] = positions
    mujoco.mj_forward(model,data)
    feet = data.site_xpos[site_ids].copy()
    jac = []
    for site in site_ids:
        position_jacobian = np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,position_jacobian,None,int(site))
        jac.append(position_jacobian[:,vel])
    jac = np.concatenate(jac,axis=0)
    w,x,y,z = positions[3:7]
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    side = np.array([-np.sin(yaw),np.cos(yaw),0.])
    target = sign*TARGET_M*side
    objective = np.concatenate((jac,RIDGE_M_PER_RAD*np.eye(len(vel))),axis=0)
    goal = np.concatenate((-np.tile(target,2),np.zeros(len(vel))))
    delta,*_ = np.linalg.lstsq(objective,goal,rcond=None)
    q = positions.astype(float).copy()
    q[:3] += target
    q[coords] += delta
    data.qpos[:] = q
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,q)
    actual_foot_drift = np.linalg.norm(data.site_xpos[site_ids]-feet,axis=1)
    predicted_foot_drift = np.linalg.norm((jac@delta+np.tile(target,2)).reshape(2,3),axis=1)
    lateral_shift = ((q[:3]-data.site_xpos[site_ids].mean(axis=0))-(positions[:3]-feet.mean(axis=0)))@side
    bounds = bool(((q[coords]>=limits[:,0])&(q[coords]<=limits[:,1])).all())
    return dict(target_rad=delta.tolist(),max_joint_change_rad=float(abs(delta).max()),
        predicted_foot_drift_mm=(1000*predicted_foot_drift).tolist(),
        exact_foot_drift_mm=(1000*actual_foot_drift).tolist(),
        exact_body_relative_to_midfoot_mm=float(1000*lateral_shift),joint_limits_kept=bounds,
        feasible=bool(bounds and abs(delta).max()<=MAX_TARGET_RAD and actual_foot_drift.max()<=MAX_FOOT_DRIFT_M))


def main():
    from mjlab.scene import Scene

    receipt = OUT/'kinematics-preflight.json'
    assert not receipt.exists()
    spec = read(PLAN)
    proof = read(OUT/'analysis.json')
    assert proof['protocol_sha256']==sha(PLAN) and not proof['heldout_used']
    p = onset.environment(spec,SEEDS[0])
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(OUT/'cpu-scene-fields.npz') as reference:
        compare_arrays(model_arrays(model),{name:reference[name] for name in reference.files})
    names,coords,vel,limits = addresses(model,p)
    site_ids = np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'robot/'+side+'_foot')
                           for side in ('left','right')])
    assert (site_ids>=0).all()
    cohorts = []
    for seed in SEEDS:
        verified = OUT/f'seed{seed}-analysis.json'
        assert sha(verified)==proof['sources'][verified.name]
        row = read(verified)
        prefix = ROOT/row['record']/'trace-shared-prefix.npz'
        assert sha(prefix)==row['shared_prefix_sha256']
        archive = []
        with np.load(prefix) as data:
            for world in row['common_near_static']:
                if not data['contact'][-1,world].all():
                    continue
                qpos = data['qpos'][-1,world]
                archive.append(dict(world=world,left=local_target(model,qpos,site_ids,coords,vel,limits,1.),
                    right=local_target(model,qpos,site_ids,coords,vel,limits,-1.)))
        cohorts.append(dict(seed=seed,common_stalls=len(row['common_near_static']),double_support=len(archive),
            feasible_both=sum(row['left']['feasible'] and row['right']['feasible'] for row in archive),
            rows=archive,source_prefix_sha256=sha(prefix)))
    write(receipt,dict(status='passed-saved-kinematics',target_mm=TARGET_M*1000,
        maximum_target_rad=MAX_TARGET_RAD,maximum_foot_drift_mm=MAX_FOOT_DRIFT_M*1000,
        joints=names,cohorts=cohorts,source_sha256=sha(Path(__file__)),physical_substeps=0,
        learning_updates=0,heldout_used=False,dynamic_response_verified=False))
    print('WALK_LATERAL_KINEMATICS',[(row['seed'],row['double_support'],row['feasible_both']) for row in cohorts],flush=True)


if __name__=='__main__':
    main()
