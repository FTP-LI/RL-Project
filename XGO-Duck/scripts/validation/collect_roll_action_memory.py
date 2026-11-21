"""New roll collector derived from the frozen official-reference evaluator.

Keep physical config, model HOME/raw-history adapter, full contact geometry
and old outcome rule. Both arms calculate the same .15 old-value EMA; only
candidate applies it. Save both raw/filtered actions and initial parameters.
The frozen original remains unchanged; this is not full runtime emulation.
"""
from evaluate_roll_reference import ROOT,config,summarize,asdict,time,np,sha

def evaluate(plan,folder,state):
    from roll_action_memory import ActionMemory
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider'])
    inp,out=session.get_inputs()[0],session.get_outputs()[0]
    assert inp.shape==[1,61] and out.shape==[1,14]
    meta=session.get_modelmeta().custom_metadata_map
    names=meta['joint_names'].split(',');home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    assert float(meta['action_scale'])==1.
    cfg=config(plan);dump_yaml(folder/'env.yaml',asdict(cfg))
    n=plan['episodes'];h=round(plan['criteria']['max_seconds']/.02)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    try:
        robot=env.scene['robot'];ids,actual_names=robot.find_joints(names,preserve_order=True);assert actual_names==names
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=np.broadcast_to(cpu(action._offset),(n,14)).copy()
        assert np.max(abs(np.asarray(cpu(action._scale) if torch.is_tensor(action._scale) else action._scale)-1))<1e-7
        m=env.sim.mj_model
        jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
        assert min(jids)>=0 and np.all(m.jnt_limited[jids])
        limits=m.jnt_range[jids].copy();tol=np.deg2rad(plan['actual_joint_limit_tolerance_deg'])
        head_local=robot.find_bodies('jaw_soft')[0][0]
        head_global=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/jaw_soft');assert head_global>=0
        head_geoms=np.flatnonzero(m.geom_bodyid==head_global)
        floor_geoms=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
        assert len(head_geoms) and len(floor_geoms)
        env.seed(plan['seed']);obs,_=env.reset()
        initial_qvel=cpu(env.sim.data.qvel)
        expanded={key:cpu(getattr(env.sim.model,key)) for key in sorted(env.sim.expanded_fields)}
        np.savez_compressed(folder/'expanded-model-initial.npz',**expanded)
        state['expanded_model_sha256']=sha(folder/'expanded-model-initial.npz')
        initial_qpos=cpu(env.sim.data.qpos);previous=np.zeros((n,14),np.float32);alive=np.ones(n,bool);memory=ActionMemory(n);filtered=previous.copy();applied=previous.copy()
        state.update(step_dt=env.step_dt,physics_dt=env.physics_dt,policy_home=home.tolist(),
            simulation_home=sim_home[0].tolist(),joint_names=names,joint_limits=limits.tolist(),
            head_geom_ids=head_geoms.tolist(),floor_geom_ids=floor_geoms.tolist(),
            observation_adapter='relative joint position and raw action history use model HOME',
            action_adapter='physical target = model HOME + '+('raw output' if plan['arm']=='control' else '.15 old + .85 new output memory')+'; no freeze',
            formal_acceptance_eligible=False)
        trace={name:[] for name in ('time','active','height','quaternion','head_quaternion','angular_velocity',
            'foot_contact','head_contact','ground_contact','head_penetration_m','joint_positions','joint_envelope_ok',
            'observations','raw_policy_actions','filtered_policy_actions','applied_policy_actions','applied_targets','terminated','truncated')}
        def sample(step,actor,raw,targets,valid,term,timeout):
            # Pooled contacts carry world IDs. Select only actual valid contact
            # slots; never interpret uninitialized capacity as penetration.
            wd=env.sim.wp_data;c=wd.contact;count=int(wd.nacon.numpy()[0])
            distances=c.dist.numpy();assert 0<=count<=len(distances)
            pairs=c.geom.numpy()[:count];world=c.worldid.numpy()[:count]
            selected=(np.isin(pairs[:,0],head_geoms)&np.isin(pairs[:,1],floor_geoms))|\
                     (np.isin(pairs[:,1],head_geoms)&np.isin(pairs[:,0],floor_geoms))
            depth=np.zeros(n,np.float32)
            np.maximum.at(depth,world[selected],np.maximum(-distances[:count][selected],0))
            pos=cpu(robot.data.joint_pos[:,ids])
            contact=lambda name:cpu(env.scene[name].data.found).reshape(n,-1).any(axis=1)
            values=dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                quaternion=cpu(robot.data.root_link_quat_w),head_quaternion=cpu(robot.data.body_link_quat_w[:,head_local]),
                angular_velocity=cpu(robot.data.root_link_ang_vel_b),joint_positions=pos,
                joint_envelope_ok=((pos>=limits[:,0]-tol)&(pos<=limits[:,1]+tol)).all(axis=1),
                foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2),
                head_contact=contact('head_ground_contact'),ground_contact=contact('robot_ground_contact'),
                head_penetration_m=depth,observations=actor.copy(),raw_policy_actions=raw.copy(),filtered_policy_actions=filtered.copy(),applied_policy_actions=applied.copy(),
                applied_targets=targets.copy(),terminated=term.copy(),truncated=timeout.copy())
            assert all(np.isfinite(v[valid]).all() for key,v in values.items() if key!='time')
            for key,value in values.items():trace[key].append(value)
        initial_actor=cpu(obs['actor']);initial_actor[:,6:20]+=sim_home-home;initial_actor[:,34:48]=0
        sample(0,initial_actor,previous,sim_home,alive.copy(),np.zeros(n,bool),np.zeros(n,bool))
        video=folder/'videos/forward-roll_env0.mp4';times=[];start=time.monotonic()
        with media.VideoWriter(str(video),shape=(720,960),fps=25,qp=22) as writer:
            def frame(step):
                im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,56),fill=(16,25,34))
                draw.text((12,8),f'ROLL ACTION MEMORY {plan["arm"]} | env 0 | t={step*.02:.2f}s | seed {plan["seed"]}',fill='white')
                draw.text((12,30),'Fixed policy | full contact | maneuver by 7 s | stable landing 2 s',fill='white')
                writer.add_image(np.asarray(im));times.append(step*.02)
                if step in (0,50,100,150,200,350,450):im.save(folder/'videos'/f'roll_t{step*.02:g}.jpg')
            frame(0)
            with torch.no_grad():
                for step in range(1,h+1):
                    actor=cpu(obs['actor']);assert actor.shape==(n,61)
                    np.testing.assert_allclose(actor[:,48:],0,rtol=0,atol=1e-6)
                    actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                    raw=np.zeros((n,14),np.float32)
                    for i in np.flatnonzero(alive):raw[i]=session.run([out.name],{inp.name:actor[i:i+1]})[0][0]
                    assert np.isfinite(raw).all()
                    filtered=memory.update(raw)
                    applied=raw.copy() if plan['arm']=='control' else filtered.copy()
                    targets=home+applied
                    obs,_,term,timeout,_=env.step(torch.tensor(targets-sim_home,device=env.device,dtype=torch.float32))
                    term,timeout=cpu(term),cpu(timeout)
                    valid=alive&~term&(~timeout|(step==h))
                    sample(step,actor,raw,targets,valid,term,timeout)
                    if alive[0] and (step%2==0 or term[0] or timeout[0]):frame(step)
                    alive&=~(term|timeout);previous=raw.copy()
                    if step%100==0 or step==h:print(f'forward-roll {step}/{h}; active {alive.sum()}/{n}; wall {time.monotonic()-start:.1f}s',flush=True)
                    if step<h:
                        reset_ids=torch.as_tensor(np.flatnonzero(term|timeout),device=env.device)
                        if len(reset_ids):obs,_=env.reset(env_ids=reset_ids);previous[reset_ids.cpu().numpy()]=0;memory.reset(reset_ids.cpu().numpy())
        arrays={k:np.asarray(v) for k,v in trace.items()};arrays['initial_qpos']=initial_qpos;arrays['initial_qvel']=initial_qvel
        path=folder/'forward-roll-trace.npz';np.savez_compressed(path,**arrays)
        result=summarize(arrays,plan['criteria'])
        result.update(condition='forward-roll',numeric_gate_passed=result['successes']/n>=plan['success_fraction_min'],
            trace_sha256=sha(path),wall_seconds=time.monotonic()-start,
            video=dict(path=str(video.relative_to(folder)),sha256=sha(video),frame_times_seconds=times,fps=25,environment_index=0))
        state['conditions']=[result];state['all_conditions_passed']=result['numeric_gate_passed']
        print(f'ROLL_REFERENCE_RESULT: {result["successes"]}/{n}',flush=True)
    finally:env.close()



