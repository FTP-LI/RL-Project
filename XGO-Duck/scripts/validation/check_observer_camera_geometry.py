"""Check declared camera geometry against a saved initial robot state on CPU.

Why: the native observer screenshot crops the robot, although saved videos
show the whole body. Reconstruct the nominal robot at the actual saved reset
qpos, project its displayed mesh vertices using MuJoCo's camera calculation,
and compare the observer parameters with the saved-video parameters. This
does not access a native window or the active worker. No renderer/context,
environment, physics integration, policy inference or learning is created.
Nominal masses/geometry and a 4:3 reference viewport are used, so this cannot
prove the active viewer's actual camera state or full-body framing.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args=parser.parse_args()
    record=ROOT/'records/experiments'/args.run_id
    assert record.parent==ROOT/'records/experiments'
    request=json.loads((record/'training-request.json').read_text())
    packet=record/'first-action-inputs.npz'
    assert sha(packet)==json.loads((record/'first-action.json').read_text())['packet_sha256']
    observer=record/'sources/live_observer.txt'
    assert sha(observer)==request['source_sha256']['live_observer']
    setup=next(n for n in ast.walk(ast.parse(observer.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='setup')
    settings={}
    for node in ast.walk(setup):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=ast.unparse(node.targets[0])
            if target.startswith('self.handle.cam.'):
                name=target.rsplit('.',1)[1]
                if name in ('distance','azimuth','elevation'):settings[name]=float(ast.literal_eval(node.value))
                elif name=='type':assert ast.unparse(node.value)=='mujoco.mjtCamera.mjCAMERA_TRACKING'
                elif name=='trackbodyid':assert ast.unparse(node.value)=='self.root_body'
    assert set(settings)=={'distance','azimuth','elevation'}
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    cfg=load_env_cfg('Mjlab-Velocity-Flat-XgoDuck',play=False)
    model=cfg.scene.entities['robot'].spec_fn().compile()
    data=mujoco.MjData(model)
    with np.load(packet,allow_pickle=False) as saved:
        assert saved['physical_qpos'].shape==(4096,model.nq)
        assert saved['physical_qvel'].shape==(4096,model.nv)
        data.qpos[:]=saved['physical_qpos'][0]
        data.qvel[:]=saved['physical_qvel'][0]
    before=data.qpos.copy()
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,before)
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
    assert body>0
    opt=mujoco.MjvOption();vertices=[];mesh_names=[]
    for geom in range(model.ngeom):
        if not opt.geomgroup[model.geom_group[geom]] or model.geom_rgba[geom,3]==0:continue
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=int(model.geom_dataid[geom]);begin=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        points=model.mesh_vert[begin:begin+count].astype(float)
        vertices.append(points@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
        mesh_names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom))
    points=np.concatenate(vertices);assert len(mesh_names)>=10
    scene=mujoco.MjvScene(model,maxgeom=100)
    location=json.loads((record/'live-log-location.json').read_text())
    log=Path(location['log_dir'])
    if not log.is_absolute():log=Path.home()/'XGO-Duck/workspace/xgoduck_rl'/log
    config_file=log/'params/env.yaml';video=yaml.load(config_file.read_text(),Loader=yaml.BaseLoader)['viewer']
    choices={'observer_declared':settings,'video_declared':{name:float(video[name]) for name in settings}}
    result=[]
    for name,values in choices.items():
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid=body
        for field,value in values.items():setattr(camera,field,value)
        mujoco.mjv_updateCamera(model,data,camera,scene)
        eye=np.mean([c.pos for c in scene.camera],axis=0)
        forward=np.asarray(scene.camera[0].forward,dtype=float);up=np.asarray(scene.camera[0].up,dtype=float)
        right=np.cross(forward,up);delta=points-eye;z=delta@forward
        near=float(scene.camera[0].frustum_near)
        half=float(scene.camera[0].frustum_top)/near
        projected=np.column_stack(((delta@right)/(z*half*4/3),(delta@up)/(z*half)))
        bbox=np.array([projected.min(axis=0),projected.max(axis=0)])
        result.append(dict(camera=name,declared=values,lookat=camera.lookat.tolist(),eye=eye.tolist(),
            model_fovy_deg=float(model.vis.global_.fovy),reference_aspect=4/3,
            mesh_count=len(mesh_names),vertices=len(points),minimum_depth_m=float(z.min()),
            near_clip_m=near,normalized_visible_box=[[-1,-1],[1,1]],projected_mesh_box=bbox.tolist(),
            nominal_meshes_inside_view=bool((z>near).all() and (np.abs(projected)<=1).all())))
    np.testing.assert_array_equal(data.qpos,before)
    source_model=Path.home()/'XGO-Duck/workspace/xgoduck_rl/src/mjlab_microduck/robot/xgoduck/robot_walk.xml'
    out=record/'camera-geometry';out.mkdir(exist_ok=True)
    report=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        mujoco_release=mujoco.__version__,input_sha256={str(p):sha(p) for p in (packet,observer,config_file,source_model)},
        results=result,native_active_camera_observed=False,randomized_model_parameters_reconstructed=False,
        geometry_scope='Displayed nominal mesh vertices at actual saved reset qpos; excludes primitive collision geometries and native viewport/UI state',
        physics_integration_steps=0,environment_created=False,graphics_context_created=False,
        simulation_state_modified=False,active_worker_modified=False,training_started=False)
    (out/'check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
