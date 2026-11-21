"""Reusable complete-episode collector extracted for new fixed getup studies.

Based on the frozen release-coverage collector, kept byte-for-byte in its old
path. New callers provide the explicit protocol, output and config factory.
Add per-world native contact and processed-neck correction evidence; preserve
raw actor history, all first episodes, full11s videos and zero learning.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,time,shutil,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_coverage_metrics import POSES,evaluate
from getup_release_coverage import render_index

def collect(role,master,plan_path,out,prepare,actor_model):
    from contextlib import ExitStack
    import mediapy
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from getup_value_bootstrap import td
    from recover_getup_value_startup import startup
    from training_watch import Snapshot
    from common_scene_contract import compare_arrays
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original single visible VS Code terminal'
    p=master['arms'][role]
    progress=out/f'{role}-progress.json';assert not progress.exists(),'No repeat or extension'
    record=ROOT/'records/evaluations'/p['run_id'];record.mkdir();(record/'videos').mkdir()
    write(record/'protocol.json',p)
    for name in master['source_sha256']:shutil.copy2(ROOT/name,record/Path(name).name)
    state=dict(status='running',run_id=p['run_id'],path=str(record),pid=os.getpid(),conditions=[],
        model=str(ROOT/p['model']),model_sha256=p['model_sha256'],learning_updates=0,scope=master['scope'],
        started_at=datetime.now(timezone.utc).isoformat(),control_steps=0,protocol_sha256=sha(plan_path))
    env=snapshot=None;trace=[];peak=0;frame_times=[];video_hashes={pose:[] for pose in POSES}
    def publish():
        if env is not None:state['control_steps']=int(env.common_step_counter)
        write(record/'evaluation.json',state);write(progress,state)
    publish();write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(record),run_id=p['run_id'],pid=os.getpid()))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,meta=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            actor=actor_model(p,env.device);before={k:v.cpu().clone() for k,v in actor.state_dict().items()}
            startup(env,obs,actor,p,record)
            cpu=lambda x:x.detach().cpu().numpy().copy()
            initial=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),actor_obs=cpu(obs['actor']),
                critic_obs=cpu(obs['critic']),imu=cpu(env._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),
                cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
            np.savez_compressed(record/'paired-start.npz',**initial)
            if role=='candidate':
                original=Path(read(out/'control-progress.json')['path'])
                with np.load(original/'paired-start.npz') as f:compare_arrays(initial,{k:f[k] for k in f.files})
                write(record/'matched-start.json',dict(status='passed',physics_steps=0,physical_observations_imu_rng_exact=True))
            robot=env.scene['robot'];ids,names=robot.find_joints(meta['joint_names'].split(','),preserve_order=True)
            import mujoco
            jids=[mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in names]
            ranges=env.sim.mj_model.jnt_range[jids].copy();tolerance=np.deg2rad(2.)
            snapshot=Snapshot(env,record);slots=int(env.sim.wp_data.contact.dist.shape[0]);start=time.monotonic()
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe());parity=[]
            with ExitStack() as stack:
                writers={pose:stack.enter_context(mediapy.VideoWriter(str(record/'videos'/f'{pose}_env{i}.mp4'),shape=(480,640),fps=25,qp=22)) for pose,i in p['video_worlds'].items()}
                def frame(step):
                    for pose,index in p['video_worlds'].items():
                        im=Image.fromarray(render_index(snapshot,index));draw=ImageDraw.Draw(im)
                        draw.rectangle((0,0,640,56),fill='#10202d')
                        draw.text((8,6),f'GETUP NECK RELIEF {role} | initial {pose} | env {index} | t={step*.02:.2f}s',fill='white')
                        draw.text((8,28),'Actual sampled state; freeze2s, recover by8s, hold3s; no extra physics',fill='white')
                        writers[pose].add_image(np.asarray(im));video_hashes[pose].append(__import__('hashlib').sha256(cpu(env.sim.data.qpos[index]).tobytes()).hexdigest())
                        if step in (0,100,150,250,400,550):im.save(record/'videos'/f'{pose}_t{step*.02:g}.jpg')
                    frame_times.append(step*.02)
                frame(0)
                with torch.no_grad():
                    for step in range(1,551):
                        raw=torch.zeros((320,14),device=env.device) if step<=100 else actor(obs)
                        sent=raw.clone();observed=cpu(obs['actor'])
                        if step in (1,100,101,150,250,400,550):
                            parity.append(dict(step=step,obs=observed[:32],actions=cpu(actor(td('actor',obs['actor'][:32])))))
                        action_term=env.action_manager.get_term('joint_pos')
                        action_term.control_observation=obs['actor']
                        obs,reward,term,timeout,_=env.step(raw)
                        torch.testing.assert_close(raw,sent,atol=0,rtol=0)
                        torch.testing.assert_close(obs['actor'][:,34:48],sent,atol=0,rtol=0)
                        positions=cpu(robot.data.joint_pos[:,ids]);qpos=cpu(env.sim.data.qpos);qvel=cpu(env.sim.data.qvel)
                        valid=cpu(~term & (~timeout | (step==550)))
                        bounded=((positions>=ranges[:,0]-tolerance)&(positions<=ranges[:,1]+tolerance)).all(1)
                        trace.append(dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                            quaternion=cpu(robot.data.root_link_quat_w),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(320,2,-1).any(2),
                            joint_envelope_ok=bounded,joint_positions=positions,observations=observed,raw_policy_actions=cpu(sent),
                            applied_targets=cpu(env.action_manager.get_term('joint_pos')._processed_actions),qpos=qpos,qvel=qvel,
                            episode_ages=cpu(env.episode_length_buf),terminated=cpu(term),truncated=cpu(timeout),
                            neck_correction=cpu(action_term.relief_applied),neck_shadow_correction=cpu(action_term.relief_correction),
                            neck_latched=cpu(action_term.relief_latched),native_head_leg_contact=cpu(env._getup_contact['contact']),
                            native_contact_eligible=cpu(env._getup_contact['eligible'])))
                        assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and torch.isfinite(reward).all()
                        assert not term.any() and (not timeout.any() if step<550 else bool(timeout.all())),'Unexpected episode boundary; no reset'
                        assert bool((env.episode_length_buf==step).all()),'Unexpected reset'
                        count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                        if step%2==0:frame(step)
                        if step%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                        if step%100==0 or step==550:
                            publish();print('GETUP_NECK_RELIEF',role,step,'/550; 320 first episodes; wall',round(time.monotonic()-start,1),flush=True)
            arrays={key:np.asarray([r[key] for r in trace]) for key in trace[0]}
            arrays.update(requested_pose=np.array([POSES[i] for i in p['split']['pose_codes']]),initial_qpos=initial['qpos'],initial_qvel=initial['qvel'])
            result=evaluate(arrays,p);rows=result['episodes']
            np.savez_compressed(record/'coverage-trace.npz',**arrays)
            np.savez_compressed(record/'inference-parity.npz',observations=np.concatenate([v['obs'] for v in parity]),actions=np.concatenate([v['actions'] for v in parity]))
            assert all(torch.equal(v.cpu(),before[k]) for k,v in actor.state_dict().items())
            write(record/'actor-integrity.json',dict(status='passed',actor_and_normalizer_unchanged=True,optimizer_steps=0))
            for pose,index in p['video_worlds'].items():
                group=[r for r in rows if r['requested_pose']==pose];score=result['requested'][pose]
                state['conditions'].append(dict(condition=pose,episodes=group,successes=score['timely_holds'],
                    numeric_gate_passed=score['passed'],strict_requested_successes=sum(r['success'] for r in group),
                    video=dict(path=f'videos/{pose}_env{index}.mp4',sha256=sha(record/'videos'/f'{pose}_env{index}.mp4'),
                        frame_times_seconds=frame_times,environment_index=index,fps=25,qpos_sha256_by_frame=video_hashes[pose],
                        complete_clip=True,source='actual-sampled-GPU-state-render',extra_physics_steps=0)))
            write(record/'coverage.json',result)
            state.update(status='completed',all_conditions_passed=result['functional_coverage_gate'],elapsed_seconds=time.monotonic()-start,
                trace_sha256=sha(record/'coverage-trace.npz'))
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));traceback.print_exc();raise
        finally:
            if trace and state['status']!='completed':np.savez_compressed(record/'partial-trace.npz',**{k:np.asarray([v[k] for v in trace]) for k in trace[0]})
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),control_transitions=int(env.common_step_counter)*320,
                    physics_substeps=int(env.common_step_counter)*5,contact_peak=peak,contact_slots=slots if 'slots' in locals() else None,
                    resets_after_initialization=0,optimizer_steps=0,extra_render_physics_steps=0))
                if hasattr(env,'_getup_contact_audits'):write(record/'contact-audits.json',env._getup_contact_audits)
                env.close()
            state['ended_at']=datetime.now(timezone.utc).isoformat();publish();sys.stdout,sys.stderr=stdout,stderr

