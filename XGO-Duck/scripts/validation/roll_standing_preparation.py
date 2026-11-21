"""Actual standing-policy preparation in the existing roll physical world.

Do not reset joints, root, actuators or physical clock at the handoff. The
roll policy starts a fresh raw-output history, as a newly selected skill.
Keep every noisy initial proposal; preparation failures remain failures.
"""
from pathlib import Path
import time
import numpy as np
from s1_reference_inputs import ROOT,sha
from record_io import write


def prepare_standing(plan,env,obs,robot,ids,sim_home,limits,tolerance,cpu,folder):
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageDraw
    steps=round(plan['prepare_seconds']/.02);n=plan['episodes']
    assert steps in (0,50) and env.step_dt==.02
    model=ROOT/plan['preparation_model'];assert sha(model)==plan['preparation_model_sha256']
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    meta=session.get_modelmeta().custom_metadata_map
    assert meta['joint_names'].split(',')==plan['joint_names'] and float(meta['action_scale'])==1.
    home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    before=int(env.common_step_counter);wall=time.perf_counter()
    previous=np.zeros((n,14),np.float32);alive=np.ones(n,bool);last=sim_home.copy()
    curves={k:[] for k in ('time','active','qpos','qvel','height','quaternion','angular_velocity','foot_contact',
                          'observations','raw_actions','targets','joint_envelope_ok','standing_ok','terminated','truncated')}
    frames=[];times=[]
    def capture(k,actor,raw,targets,term,trunc):
        pos=cpu(robot.data.joint_pos[:,ids]);bounded=((pos>=limits[:,0]-tolerance)&(pos<=limits[:,1]+tolerance)).all(axis=1)
        q=cpu(robot.data.root_link_quat_w);height=cpu(robot.data.root_link_pos_w[:,2])
        omega=cpu(robot.data.root_link_ang_vel_b)
        feet=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2)
        tilt=np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))
        c=plan['criteria']
        standing=(abs(height-c['nominal_height_m'])<=c['height_tolerance_m'])&(tilt<=np.deg2rad(c['tilt_max_deg']))
        standing&=(np.linalg.norm(omega,axis=1)<=c['angular_speed_max_rad_s'])&feet.all(axis=1)&bounded
        alive[:] &= bounded&~term&~trunc
        if steps:alive[:] &= (height>=.08)&(tilt<=np.deg2rad(45.))
        values=dict(time=-steps*.02+k*.02,active=alive.copy(),qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),
            height=height,quaternion=q,angular_velocity=omega,foot_contact=feet,standing_ok=standing,
            observations=actor.copy(),raw_actions=raw.copy(),targets=targets.copy(),joint_envelope_ok=bounded,
            terminated=term.copy(),truncated=trunc.copy())
        assert all(np.isfinite(v).all() for key,v in values.items() if key not in ('time','active','joint_envelope_ok','terminated','truncated'))
        for key,value in values.items():curves[key].append(value)
    initial=cpu(obs['actor']);initial[:,6:20]+=sim_home-home;initial[:,34:48]=0
    capture(0,initial,previous,last,np.zeros(n,bool),np.zeros(n,bool))
    with torch.no_grad():
        for k in range(steps):
            if k%2==0:
                counter=int(env.common_step_counter);im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,56),fill=(16,25,34))
                draw.text((12,8),f'STAND PREPARATION | env 0 | t={-steps*.02+k*.02:.2f}s | seed {plan["seed"]}',fill='white')
                draw.text((12,30),'Same physical episode | official walking policy, zero command | no joint reset',fill='white')
                frames.append(np.asarray(im));times.append(-steps*.02+k*.02)
                assert counter==int(env.common_step_counter)
            actor=cpu(obs['actor']);np.testing.assert_allclose(actor[:,48:],0.,atol=1e-6,rtol=0)
            actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
            raw=np.zeros((n,14),np.float32)
            for i in np.flatnonzero(alive):raw[i]=session.run(None,{'obs':actor[i:i+1]})[0][0]
            last=home+raw
            obs,_,term,trunc,_=env.step(torch.as_tensor(last-sim_home,device=env.device,dtype=torch.float32))
            capture(k+1,actor,raw,last,cpu(term),cpu(trunc));previous=raw.copy()
    assert int(env.common_step_counter)-before==steps
    if steps:alive &= np.asarray(curves['standing_ok'])[-10:].all(axis=0)
    file=folder/'standing-preparation.npz';np.savez_compressed(file,**{k:np.asarray(v) for k,v in curves.items()})
    result=dict(status='completed',seconds=plan['prepare_seconds'],control_steps=steps,
        start_counter=before,end_counter=int(env.common_step_counter),alive=alive.tolist(),
        all_proposals_retained=True,physical_reset_at_handoff=False,preparation_trace_sha256=sha(file),
        trigger_stable_hold_seconds=.2 if steps else 0.,maximum_preparation_tilt_deg=45.,minimum_preparation_height_m=.08,
        wall_seconds=time.perf_counter()-wall,prefix_frames=len(frames),prefix_times_seconds=times,
        model=str(model.relative_to(ROOT)),model_sha256=sha(model),last_targets=last.tolist(),
        roll_history_at_handoff='zero for newly selected roll model; actuator state and physical qpos/qvel retained')
    write(folder/'standing-preparation.json',result)
    return obs,alive,frames,times,last
