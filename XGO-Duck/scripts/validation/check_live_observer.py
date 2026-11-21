"""Check observer isolation on a real CPU model, optionally probe WSLg display.

Synthetic sensor values are used only for software checks, never gait evidence.
No GPU environment, policy inference, optimizer update or physics step is run.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from live_observer import LiveObserver, open_passive, close_passive


class FakeHandle:
    def __init__(self): self.syncs=0; self.closed=False
    def is_running(self): return not self.closed
    def lock(self): return nullcontext()
    def set_texts(self,text): self.text=text
    def sync(self): self.syncs+=1
    def close(self): self.closed=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui-probe',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=ROOT/'official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck/scene_walk.xml'
    model=mujoco.MjModel.from_xml_path(str(source))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i)
           for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
    assert len(names)==14
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot')]
    assert min(feet)>=0
    copied={name:torch.from_numpy(getattr(data,name).copy()[None])
            for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','xfrc_applied')}
    head=[names.index(n) for n in ('neck_pitch','head_pitch','head_yaw','head_roll')]
    homes=torch.zeros((1,14),dtype=torch.float64)
    homes[0,head]=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    velocities=torch.zeros((1,14),dtype=torch.float64)
    velocities[0,head]=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    commands={'twist':torch.tensor([[.15,0.,0.]],dtype=torch.float64),
              'head_pose':torch.tensor([[.05,-.1,.15,-.2]],dtype=torch.float64)}
    robot=SimpleNamespace(joint_names=names,
        find_sites=lambda requested,preserve_order:(feet,['left_foot','right_foot']),
        data=SimpleNamespace(site_pos_w=torch.from_numpy(data.site_xpos.copy()[None]),
                             default_joint_pos=homes,joint_vel=velocities,
                             joint_pos=torch.from_numpy(data.qpos[model.jnt_qposadr[joints]].copy()[None])))
    env=SimpleNamespace(common_step_counter=0,step_dt=0.02,
        sim=SimpleNamespace(mj_model=model,expanded_fields=set(),data=SimpleNamespace(**copied)),
        scene={'robot':robot,'feet_ground_contact':SimpleNamespace(data=SimpleNamespace(found=torch.zeros((1,2,1),dtype=torch.bool))),
               'self_collision':SimpleNamespace(data=SimpleNamespace(found=torch.zeros((1,1),dtype=torch.bool)))},
        command_manager=SimpleNamespace(get_command=lambda name:commands[name]))
    env.unwrapped=env
    original={name:value.clone() for name,value in copied.items()}
    sensor_original={name:value.clone() for name,value in
                     {'homes':homes,'velocities':velocities,**commands}.items()}
    model_before=hashlib.sha256(model.qpos0.tobytes()+model.body_pos.tobytes()+model.geom_pos.tobytes()).hexdigest()
    rng_before=torch.get_rng_state().clone()
    output=ROOT/'records/reports/live-view-entry-check'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output,prefix='observer-check-') as temporary:
        observer=LiveObserver(env,temporary)
        observer.handle=FakeHandle()
        observer.observe()
        rows=[json.loads(line) for line in (Path(temporary)/'live-view.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['common_step_counter']==0
        assert len(rows[0]['joint_positions_rad'])==14
        np.testing.assert_allclose(rows[0]['head_pose_target_rad'],[.15,.1,.45,.2],atol=1e-12)
        np.testing.assert_allclose(rows[0]['head_joint_positions_rad'],[0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(rows[0]['head_pose_error_actuator_rad'],[-.15,-.1,-.45,-.2],atol=1e-12)
        np.testing.assert_allclose(rows[0]['head_joint_velocities_rad_s'],[1,2,3,4],atol=1e-12)
        assert 'Head target deg' in observer.handle.text[2] and 'Head actual deg' in observer.handle.text[2]
        expected=float(np.linalg.norm(data.site_xpos[feet[0],:2]-data.site_xpos[feet[1],:2]))
        assert abs(rows[0]['foot_site_distance_xy_m']-expected)<1e-12
        assert observer.handle.syncs==1
        observer.observe()  # Rate limit prevents an extra render/log row.
        assert observer.handle.syncs==1
        observer.handle.close();observer.next_frame=0;env.common_step_counter=10
        observer.observe()
        observer.close()
        status=json.loads((Path(temporary)/'live-view-status.json').read_text())
        assert status['status']=='observer-finished' and status['display_closed_common_step']==10
        assert status['display_closed_at'] and status['telemetry_stops_when_display_closes']
        assert len((Path(temporary)/'live-view.jsonl').read_text().splitlines())==1
    assert all(torch.equal(copied[name],before) for name,before in original.items())
    assert all(torch.equal(value,sensor_original[name]) for name,value in
               {'homes':homes,'velocities':velocities,**commands}.items())
    assert torch.equal(torch.get_rng_state(),rng_before)
    assert hashlib.sha256(model.qpos0.tobytes()+model.body_pos.tobytes()+model.geom_pos.tobytes()).hexdigest()==model_before
    assert not torch.cuda.is_initialized()
    gui=False
    if args.gui_probe:
        viewer, threads = open_passive(model,data)
        try:
            viewer.cam.distance=.8
            viewer.cam.elevation=-15
            viewer.cam.lookat[:]=[0,0,.13]
            deadline=time.perf_counter()+6
            print('VIEWER_PROBE_OPEN: STATIC MODEL, NO TRAINING OR PHYSICS STEPS',flush=True)
            while time.perf_counter()<deadline and viewer.is_running():
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,'VIEWER CONNECTION TEST','STATIC MODEL - no training'))
                viewer.sync()
                gui=True
                time.sleep(.1)
        finally:
            close_passive(viewer,threads)
        assert gui
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],observer_sha256=hashlib.sha256((ROOT/'scripts/training/live_observer.py').read_bytes()).hexdigest(),
        cpu_model_source=str(source.relative_to(ROOT)),joint_count=14,
        isolated_model_and_source_tensors_unchanged=True,torch_rng_unchanged=True,
        foot_gap_recomputed=True,rate_limit_checked=True,cuda_initialized=False,
        head_command_target_actual_and_velocity_checked=True,
        display_close_event_survives_final_cleanup=True,
        gui_requested=args.gui_probe,gui_open_and_sync_verified=gui,
        simulation_steps=0,training_started=False,synthetic_sensor_values=True,
        stable_gait_verified=False)
    (output/('gui-check.json' if args.gui_probe else 'check.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
