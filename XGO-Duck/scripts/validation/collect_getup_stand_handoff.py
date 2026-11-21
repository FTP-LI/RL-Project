"""Visible complete-episode collector for the new continuous handoff study.

One environment and one sensed stream survive the switch. Both frozen actors
are evaluated in both arms; only the candidate selects the standing output.
Save enough raw inputs/state to independently reconstruct every target/history.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import hashlib, os, shutil, sys, time, traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read, sha
from record_io import write
from getup_stand_handoff_pair import PLAN, OUT, prepare, actor_model
from getup_stand_handoff import HandoffState
from getup_coverage_metrics import POSES, evaluate
from getup_release_coverage import render_index


def startup(env, obs, actors, p, record):
    """Exercise the actual CUDA processors without integrating or resampling."""
    import onnxruntime as ort
    from getup_value_bootstrap import td
    from recover_getup_value_startup import startup as source_startup
    from common_scene_contract import compare_arrays
    cpu = lambda x:x.detach().cpu().numpy().copy()
    term = env.action_manager.get_term('joint_pos')
    extra = ('previous_target','previous_valid','standing_mask','requested_target','projected_destination')
    saved_extra = {k:getattr(term,k).clone() for k in extra}
    source_startup(env,obs,actors['getup'],p,record)
    # The inherited check restores its own fields; restore the new target
    # processor memory too, so zero-step probing cannot seed a fake target.
    for k,v in saved_extra.items():
        setattr(term,k,v.clone())
    saved = {k:getattr(term,k).clone() for k in (*extra,'_raw_actions','_processed_actions')}
    histories = {k:getattr(env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')}
    age = env.episode_length_buf.clone()
    physical = dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(env.device).clone()
    home = np.asarray(p['model_home'],np.float32)
    stand_home = np.asarray(p['standing_home'],np.float32)
    memory = HandoffState(home,stand_home,env.num_envs,enabled=p['handoff_enabled'],
                          device=env.device,settings=p['handoff_settings'])
    _, destination = memory.inputs(obs['actor'],age,env.step_dt)
    assert not memory.standing.any() and not memory.shadow_latched.any()
    np.testing.assert_allclose(cpu(destination[:,6:20])+stand_home,
                               cpu(obs['actor'][:,6:20])+home,atol=1e-7,rtol=0)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['standing_model']),sess_options=options,
                                   providers=['CPUExecutionProvider'])
    selected = cpu(destination[:32])
    expected = np.concatenate([session.run(None,{'obs':x[None]})[0] for x in selected])
    with torch.no_grad():
        actual = cpu(actors['standing'](td('actor',destination[:32])))
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    probe = torch.full_like(histories['_action'],.01)
    env.episode_length_buf.fill_(249)
    env.action_manager.process_action(probe)
    old = cpu(term._processed_actions)
    np.testing.assert_allclose(old,np.broadcast_to(home+.01,old.shape),atol=1e-7,rtol=0)
    env.episode_length_buf.fill_(250)
    mask = torch.arange(env.num_envs,device=env.device)%2==0
    term.standing_mask = mask
    env.action_manager.process_action(probe)
    requested = np.broadcast_to(home+.01,old.shape).copy()
    requested[cpu(mask)] = stand_home+.01
    limits = np.asarray(p['joint_limits'],np.float32)
    legal = np.clip(requested,limits[:,0],limits[:,1])
    target = old+np.clip(legal-old,-.04,.04)
    np.testing.assert_allclose(cpu(term.requested_target),requested,atol=1e-7,rtol=0)
    np.testing.assert_allclose(cpu(term._processed_actions),target,atol=1e-7,rtol=0)
    torch.testing.assert_close(probe,torch.full_like(probe,.01),atol=0,rtol=0)
    torch.testing.assert_close(env.action_manager._action,probe,atol=0,rtol=0)
    for k,v in saved.items():
        setattr(term,k,v.clone())
    for k,v in histories.items():
        getattr(env.action_manager,k).copy_(v)
    env.episode_length_buf.copy_(age)
    compare_arrays(physical,dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel)))
    assert env.common_step_counter==0
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(env.device))
    for k,v in saved_extra.items():
        torch.testing.assert_close(getattr(term,k),v,atol=0,rtol=0)
    np.savez_compressed(record/'standing-startup-inputs.npz',observations=selected,
                         torch_actions=actual,onnx_actions=expected)
    proof = read(record/'startup.json')
    proof.update(standing_import_error=float(abs(actual-expected).max()),standing_inputs=32,
        cuda_model_home_switch_and_slew_checked=True,separate_model_histories_zero=True,
        new_processor_memory_restored_exact=True,physical_steps=0)
    write(record/'startup.json',proof)


def collect(role, master):
    import mediapy, imageio_ffmpeg, mujoco
    from PIL import Image, ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from getup_value_bootstrap import td
    from training_watch import Snapshot
    from common_scene_contract import compare_arrays
    from evaluate_standup_reference import Tee
    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    p = master['arms'][role]
    progress = OUT/f'{role}-progress.json'
    assert not progress.exists(), 'No repeated attempt or extension'
    record = ROOT/'records/evaluations'/p['run_id']
    record.mkdir(); (record/'videos').mkdir()
    write(record/'protocol.json',p)
    for name in master['source_sha256']:
        shutil.copy2(ROOT/name,record/Path(name).name)
    state = dict(status='running',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        conditions=[],learning_updates=0,scope=master['scope'],control_steps=0,
        model_sha256=p['model_sha256'],standing_model_sha256=p['standing_model_sha256'],
        protocol_sha256=sha(PLAN),started_at=datetime.now(timezone.utc).isoformat())
    env = snapshot = None
    trace, frame_times, parity = [], [], []
    hashes = {pose:[] for pose in POSES}
    peak = 0
    def publish():
        if env is not None:
            state['control_steps'] = int(env.common_step_counter)
        write(record/'evaluation.json',state); write(progress,state)
    publish()
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(record),run_id=p['run_id'],pid=os.getpid()))
    write(record/'command.json',dict(argv=sys.argv,pid=os.getpid(),foreground_tty=os.isatty(1),
        started_at=state['started_at'],scope=master['scope']))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout,log), Tee(stderr,log)
        try:
            configure_torch_backends(allow_tf32=False); torch.set_num_threads(4)
            cfg, display, meta = prepare(p); write(record/'display.json',display)
            env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
            env.seed(p['seed']); obs,_ = env.reset()
            standing_p = dict(p,model=p['standing_model'],model_sha256=p['standing_model_sha256'])
            actors = dict(getup=actor_model(p,env.device),standing=actor_model(standing_p,env.device))
            before = {role:{k:v.cpu().clone() for k,v in a.state_dict().items()} for role,a in actors.items()}
            startup(env,obs,actors,p,record)
            cpu = lambda x:x.detach().cpu().numpy().copy()
            initial = dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),actor_obs=cpu(obs['actor']),
                critic_obs=cpu(obs['critic']),imu=cpu(env._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),
                cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
            np.savez_compressed(record/'paired-start.npz',**initial)
            if role=='candidate':
                original = Path(read(OUT/'control-progress.json')['path'])
                with np.load(original/'paired-start.npz') as f:
                    compare_arrays(initial,{k:f[k] for k in f.files})
                write(record/'matched-start.json',dict(status='passed',physics_steps=0,
                                                       physical_observations_imu_rng_exact=True))
            memory = HandoffState(p['model_home'],p['standing_home'],p['num_envs'],enabled=p['handoff_enabled'],
                                  device=env.device,settings=p['handoff_settings'])
            robot = env.scene['robot']
            ids, names = robot.find_joints(meta['joint_names'].split(','),preserve_order=True)
            jids = [mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in names]
            ranges = env.sim.mj_model.jnt_range[jids].copy()
            np.testing.assert_allclose(ranges,p['joint_limits'],atol=1e-7,rtol=0)
            tolerance = np.deg2rad(2.)
            term = env.action_manager.get_term('joint_pos')
            snapshot = Snapshot(env,record)
            slots = int(env.sim.wp_data.contact.dist.shape[0]); start = time.monotonic()
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
            with ExitStack() as stack:
                writers = {pose:stack.enter_context(mediapy.VideoWriter(str(record/'videos'/f'{pose}_env{i}.mp4'),
                    shape=(480,640),fps=25,qp=22)) for pose,i in p['video_worlds'].items()}
                def frame(step):
                    for pose,index in p['video_worlds'].items():
                        image = Image.fromarray(render_index(snapshot,index)); draw = ImageDraw.Draw(image)
                        phase = 'STAND' if bool(memory.standing[index]) else 'GETUP'
                        draw.rectangle((0,0,640,56),fill='#10202d')
                        draw.text((8,6),f'GETUP -> STAND {role} | {pose} | env{index} | t={step*.02:.2f}s',fill='white')
                        draw.text((8,28),f'Last action: {phase}; continuous state; freeze2s, recover by8s, hold3s',fill='white')
                        writers[pose].add_image(np.asarray(image))
                        hashes[pose].append(hashlib.sha256(cpu(env.sim.data.qpos[index]).tobytes()).hexdigest())
                        if step in (0,100,250,300,400,550):
                            image.save(record/'videos'/f'{pose}_t{step*.02:g}.jpg')
                    frame_times.append(step*.02)
                frame(0)
                with torch.no_grad():
                    for step in range(1,551):
                        sensed = obs['actor']
                        source_obs, destination_obs = memory.inputs(sensed,env.episode_length_buf,env.step_dt)
                        getup = actors['getup'](td('actor',source_obs))
                        standing = actors['standing'](td('actor',destination_obs))
                        raw = memory.select(getup,standing,env.episode_length_buf,env.step_dt)
                        observed, sent = cpu(sensed), raw.clone()
                        if step in (1,100,101,250,251,400,550):
                            parity.append(dict(getup_obs=cpu(source_obs[:32]),getup_raw=cpu(getup[:32]),
                                stand_obs=cpu(destination_obs[:32]),stand_raw=cpu(standing[:32])))
                        term.standing_mask = memory.standing.clone()
                        previous = cpu(term.previous_target)
                        obs,reward,terminated,truncated,_ = env.step(raw)
                        torch.testing.assert_close(raw,sent,atol=0,rtol=0)
                        torch.testing.assert_close(obs['actor'][:,34:48],sent,atol=0,rtol=0)
                        target = cpu(term._processed_actions)
                        if step>=251:
                            assert (abs(target-previous)<=.040002).all(), 'Target slew exceeded'
                        positions = cpu(robot.data.joint_pos[:,ids])
                        bounded = ((positions>=ranges[:,0]-tolerance)&(positions<=ranges[:,1]+tolerance)).all(1)
                        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
                        valid = cpu(~terminated & (~truncated | (step==550)))
                        trace.append(dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                            quaternion=cpu(robot.data.root_link_quat_w),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(p['num_envs'],2,-1).any(2),
                            joint_envelope_ok=bounded,joint_positions=positions,observations=observed,
                            getup_actions=cpu(getup),standing_actions=cpu(standing),raw_policy_actions=cpu(sent),
                            requested_targets=cpu(term.requested_target),applied_targets=target,qpos=qpos,qvel=qvel,
                            phase_standing=cpu(memory.standing),shadow_handoff=cpu(memory.shadow_latched),
                            episode_ages=cpu(env.episode_length_buf),terminated=cpu(terminated),truncated=cpu(truncated),
                            native_head_leg_contact=cpu(env._getup_contact['contact']),
                            native_contact_eligible=cpu(env._getup_contact['eligible'])))
                        assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and torch.isfinite(reward).all()
                        assert not terminated.any() and (not truncated.any() if step<550 else bool(truncated.all()))
                        assert bool((env.episode_length_buf==step).all()), 'Physical reset at switch is forbidden'
                        count = int(env.sim.wp_data.nacon.numpy()[0]); peak = max(peak,count)
                        assert 0<=count<slots
                        if step%2==0:
                            frame(step)
                        if step%25==0:
                            assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                        if step%100==0 or step==550:
                            state['standing_worlds']=int(memory.standing.sum()); publish()
                            print('GETUP_STAND_HANDOFF',role,step,'/550; standing',state['standing_worlds'],
                                  '; wall',round(time.monotonic()-start,1),flush=True)
            arrays = {k:np.asarray([row[k] for row in trace]) for k in trace[0]}
            arrays.update(requested_pose=np.array([POSES[i] for i in p['split']['pose_codes']]),
                          initial_qpos=initial['qpos'],initial_qvel=initial['qvel'])
            result = evaluate(arrays,p)
            np.savez_compressed(record/'coverage-trace.npz',**arrays)
            np.savez_compressed(record/'inference-parity.npz',**{k:np.concatenate([v[k] for v in parity]) for k in parity[0]})
            for label,actor in actors.items():
                assert all(torch.equal(v.cpu(),before[label][k]) for k,v in actor.state_dict().items())
            write(record/'actor-integrity.json',dict(status='passed',both_actors_and_normalizers_unchanged=True,optimizer_steps=0))
            for pose,index in p['video_worlds'].items():
                score = result['requested'][pose]
                state['conditions'].append(dict(condition=pose,successes=score['timely_holds'],numeric_gate_passed=score['passed'],
                    video=dict(path=f'videos/{pose}_env{index}.mp4',sha256=sha(record/'videos'/f'{pose}_env{index}.mp4'),
                        frame_times_seconds=frame_times,environment_index=index,fps=25,
                        qpos_sha256_by_frame=hashes[pose],complete_clip=True,
                        source='actual-sampled-GPU-state-render',extra_physics_steps=0)))
            write(record/'coverage.json',result)
            state.update(status='completed',all_conditions_passed=result['functional_coverage_gate'],
                         elapsed_seconds=time.monotonic()-start,trace_sha256=sha(record/'coverage-trace.npz'))
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
            traceback.print_exc(); raise
        finally:
            if trace and state['status']!='completed':
                np.savez_compressed(record/'partial-trace.npz',**{k:np.asarray([v[k] for v in trace]) for k in trace[0]})
            if snapshot is not None:
                snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),
                    control_transitions=int(env.common_step_counter)*p['num_envs'],
                    physics_substeps=int(env.common_step_counter)*5,contact_peak=peak,
                    contact_slots=slots if 'slots' in locals() else None,resets_after_initialization=0,
                    optimizer_steps=0,extra_render_physics_steps=0))
                if hasattr(env,'_getup_contact_audits'):
                    write(record/'contact-audits.json',env._getup_contact_audits)
                env.close()
            state['ended_at']=datetime.now(timezone.utc).isoformat(); publish()
            sys.stdout, sys.stderr = stdout, stderr
