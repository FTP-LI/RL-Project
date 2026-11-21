"""Freeze and run a fixed-policy, continuous development sequence."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from action_video import ActionVideo
from collect_roll_confirmation import model_input, prepare
from command_input_mapping import transform
from common_scene_contract import compare_arrays, install_guard, model_arrays
from common_walk_startup import make_actor, td
from getup_stand_handoff import projected_slew
from record_io import write
from roll_endpoint_readback_capture import check_geom_layout
from roll_stand_walk_sequence import FAILED, FORWARD, ROLL, STAND_LANDING, Sequence
from training_watch import Snapshot
from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target


OUT = ROOT/'records/reports/continuous-roll-stand-walk'
PAGE = ROOT/'records/roll-stand-walk-live.html'
PLAN = ROOT/'configs/experiments/continuous_roll_stand_walk.json'
DRAFT = PLAN.with_name('continuous_roll_stand_walk_draft.json')
STARTUP = ROOT/'records/reports/common-plant-screen/startup'
PROJECT = ROOT/'project.json'
TZ = ZoneInfo('Asia/Shanghai')


def cpu(value):
    return value.detach().cpu().numpy().copy()


def configuration(spec):
    cfg = prepare(spec['roll'])
    cfg.episode_length_s = (spec['maximum_controls'] + 1)*.02
    assert not cfg.auto_reset and cfg.scene.num_envs == spec['worlds']
    return cfg


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists()
    p = read(DRAFT)
    assert p['status'] == 'draft-not-executable' and not p['hardware_io']
    boundary = ROOT/p['boundary_preflight']
    proof = read(boundary)
    assert proof['status'] == 'passed-offline-boundary-preflight' and not proof['heldout_used']
    for file, digest in proof['sources'].items():
        assert sha(ROOT/file) == digest, file
    assert proof['paired_boundaries'] == 117 and proof['switch_rate_rad_s'] == p['bridge_slew_rad_s']
    roll_plan, walk_plan = ROOT/p['nominal_roll'], ROOT/p['walking_interface']
    roll = dict(read(roll_plan)['evaluations']['development-b'], seed=p['seed'], episodes=p['worlds'])
    walk = read(walk_plan)
    confirm = ROOT/'records/reports/roll-endpoint-confirmation/analysis.json'
    yaw = ROOT/'records/reports/common-walk-actuator-yaw-feedback-heldout/README.md'
    assert all(not x['heldout_used'] for x in (roll, walk))
    screen = ROOT/walk['feedback_screen']
    assert read(screen)['nominated_cap_rad_s'] == walk['feedback']['maximum_yaw_input_rad_s']
    forward = walk['environment']['commands']['forward']
    assert forward == [.15, 0, 0] and p['forward_controls'] == 1000
    assert not any(f'_s{p["seed"]}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    controls = sum(p[name] for name in ('initial_stand_controls','roll_controls','landing_stand_controls','forward_controls'))
    assert p['maximum_control_transitions'] == controls*p['worlds']
    assert p['maximum_physics_substeps'] == controls*5
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    p.update(status='frozen-before-dynamics', run_id=f'duck_switch_roll-stand-walk_development_s{p["seed"]}_{stamp}',
             roll=roll, walking=walk['environment'], feedback=walk['feedback'], feedback_screen=walk['feedback_screen'],
             maximum_controls=controls)
    files = [DRAFT, boundary, roll_plan, walk_plan, confirm, yaw, screen, ROOT/roll['model'],
             ROOT/walk['environment']['model'], STARTUP/'roll/scene-static-fields.npz',
             STARTUP/'roll/nominal-primary-fields.npz', Path(__file__)]
    files += [ROOT/'scripts/validation'/name for name in ('roll_stand_walk_sequence.py','roll_landing_latch.py',
        'verify_continuous_roll_stand_walk.py','roll_metrics.py','common_scene_contract.py','collect_roll_confirmation.py',
        'roll_endpoint_readback_capture.py','common_walk_startup.py','walking_actuator_yaw_feedback.py',
        'planar_block_metrics.py','command_input_mapping.py','common_skill_plant.py')]
    files += [ROOT/'scripts/training'/name for name in ('action_video.py','training_live_page.py',
        'training_watch.py','getup_stand_handoff.py')]
    files.append(ROOT/'configs/environments/action_recording.json')
    p['frozen_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    p['training_source_sha256'] = read(roll_plan)['training_source_sha256']
    p['library_source_sha256'] = roll['library_source_sha256']
    cfg = configuration(p)
    model = Scene(cfg.scene, device='cpu').compile()
    check_geom_layout(model)
    with np.load(STARTUP/'roll/scene-static-fields.npz') as data:
        compare_arrays(model_arrays(model), {key: data[key] for key in data.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False)
    write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), physical_substeps=0,
          shared_compiled_scene=True, frozen_files=len(files), seed=p['seed'], observation=PAGE.relative_to(ROOT).as_posix()))
    print('ROLL_STAND_WALK_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    from s1_reference_inputs import WORK

    p = read(PLAN)
    assert p['status'] == 'frozen-before-dynamics' and not p['hardware_io'] and not p['heldout_used']
    assert not p['learning_updates'] and not p['automatic_retry'] and not p['automatic_extension']
    for root, key in ((ROOT,'frozen_sha256'), (WORK,'training_source_sha256'), (Path('/'),'library_source_sha256')):
        for file, digest in p[key].items():
            assert sha(root/file) == digest, file
    return p


def onnx_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def actors(env, obs, p, folder):
    term = env.action_manager.get_term('joint_pos')
    simulation_home = torch.as_tensor(term._offset, device=env.device).expand(p['worlds'],-1)
    scale = torch.as_tensor(term._scale, device=env.device)
    assert (scale == 1).all()
    order = list(term._target_names)
    limits = torch.tensor(p['walking']['joint_limits'], device=env.device, dtype=torch.float32)
    np.testing.assert_array_equal(np.asarray(p['walking']['joint_limits']),
                                  np.array([env.sim.mj_model.joint('robot/'+name).range for name in order]))
    home, networks, maximum = {}, {}, {}
    for role, meta in (('roll',p['roll']),('walk',p['walking'])):
        net = make_actor(meta,env.device)
        session = onnx_session(ROOT/meta['model'])
        tags = session.get_modelmeta().custom_metadata_map
        assert order == tags['joint_names'].split(',') and float(tags['action_scale']) == 1.
        home[role] = torch.tensor([float(x) for x in tags['default_joint_pos'].split(',')],
                                  device=env.device, dtype=torch.float32)
        observed = model_input(obs['actor'],simulation_home,home[role],torch.zeros_like(simulation_home))
        actual = cpu(net(td(observed)))
        expected = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in cpu(observed)])
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        networks[role] = net
        maximum[role] = float(abs(actual-expected).max())
    assert env.common_step_counter == env._sim_step_counter == 0
    np.savez_compressed(folder/'startup.npz',qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),obs=cpu(obs['actor']),
                        simulation_home=cpu(simulation_home),roll_home=cpu(home['roll']),walk_home=cpu(home['walk']))
    write(folder/'startup.json',dict(status='passed',physical_substeps=0,actual_onnx_parity=maximum,
          startup_sha256=sha(folder/'startup.npz'), simulation_home=cpu(simulation_home[0]).tolist(), joints=order))
    return networks,home,simulation_home,limits


def run(zero):
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from roll_endpoint_readback_capture import EndpointReadback

    p = checked()
    task = read(PROJECT)['staged_training_request']
    assert task['current_stage'] == p['stage'] and task['current_protocol_frozen'] and not task['pause_requested']
    assert ROOT/task['current_protocol'] == PLAN
    assert read(OUT/'windows-entry.json')['protocol_sha256'] == sha(PLAN)
    configure_torch_backends(allow_tf32=False)
    folder = OUT/'zero' if zero else ROOT/'records/evaluations'/p['run_id']
    folder.mkdir(exist_ok=False)
    write(folder/'protocol.json', p)
    cfg = configuration(p)
    dump_yaml(folder/'env.yaml',asdict(cfg))
    cls = install_guard(p['roll'],folder,STARTUP,'rollout')
    env = cls(cfg=cfg,device='cuda:0',render_mode=None)
    try:
        env.seed(p['seed'])
        obs,_ = env.reset()
        networks,homes,sim_home,limits = actors(env,obs,p,folder)
        endpoint = EndpointReadback(env,folder)
        endpoint.capture()
        np.testing.assert_array_equal(endpoint.rows[-1]['fresh'],endpoint.rows[-1]['constraint_points']>0)
        if zero:
            endpoint.save(1)
            print('ROLL_STAND_WALK_ZERO',p['seed'],flush=True)
            return
        with np.load(OUT/'zero/startup.npz') as before, np.load(folder/'startup.npz') as current:
            compare_arrays({key:current[key] for key in current.files},{key:before[key] for key in before.files})
        with np.load(OUT/'zero/endpoint-contact.npz') as before:
            for key,value in endpoint.rows[0].items():
                np.testing.assert_array_equal(value,before[key][0],err_msg=key)
        endpoint.paired_start_verified = True
        collect(env,obs,p,folder,endpoint,networks,homes,sim_home,limits)
    finally:
        env.close()


class PolicyControl:
    def __init__(self,p,networks,home,sim_home,limits):
        self.p, self.networks, self.home, self.sim_home, self.limits = p,networks,home,sim_home,limits
        self.previous = sim_home.clone()
        self.history = torch.zeros_like(sim_home)
        self.hip = [20+p['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left','right')]
        self.previous_features = None
        self.feedback = ActuatorYawFeedback(read(ROOT/p['feedback_screen']),p['feedback']['maximum_yaw_input_rad_s'])

    def next(self,step,obs,sequence):
        phase = sequence.phase.copy()
        new_model = np.isin(phase,(ROLL,STAND_LANDING)) & (sequence.phase_changed == step-1)
        self.history[new_model] = 0.
        shape = self.history.shape
        actor_input = torch.zeros((shape[0],61),device=obs.device)
        raw = torch.zeros_like(self.history)
        requested = self.previous.clone()
        desired = np.zeros((shape[0],3),dtype=np.float32)
        forward = phase == FORWARD
        desired[forward] = self.p['walking']['commands']['forward']
        observed = cpu(obs)
        features = np.column_stack((observed[:,2],observed[:,self.hip].sum(axis=1)))
        changes = features-self.previous_features if self.previous_features is not None else np.zeros_like(features)
        self.previous_features = features
        correction = np.zeros(shape[0],dtype=np.float32)
        if forward.any():
            correction[forward] = self.feedback.commands(np.column_stack((features[forward],changes[forward])))
        for role,mask in (('roll',phase==ROLL),('walk',(phase!=ROLL)&(phase!=FAILED))):
            if not mask.any():
                continue
            adapted = model_input(obs[mask],self.sim_home[mask],self.home[role],self.history[mask])
            if role == 'walk':
                adapted[:,48:51] = torch.from_numpy(transform(desired[mask])).to(obs.device)
            actor_input[mask] = adapted
            raw[mask] = self.networks[role](td(adapted))
            requested[mask] = self.home[role]+raw[mask]
        assert torch.isfinite(raw).all()
        destination = torch.clamp(requested,min=self.limits[:,0],max=self.limits[:,1])
        alternate = torch.zeros_like(raw)
        if forward.any():
            adapted = actor_input[forward].clone()
            adapted[:,50] = torch.from_numpy(correction[forward]).to(obs.device)
            alternate[forward] = self.networks['walk'](td(adapted))
            destination[forward],_ = projected_feedback_target(destination[forward],alternate[forward],
                self.home['walk'],self.limits,self.p['feedback']['maximum_joint_correction_rad'])
        bridge = (np.isin(phase,(ROLL,STAND_LANDING))
                  & (step - sequence.phase_changed <= self.p['cross_model_bridge_controls']))
        applied = destination.clone()
        if bridge.any():
            applied[bridge],_ = projected_slew(self.previous[bridge],destination[bridge],self.limits,
                                                .02,self.p['bridge_slew_rad_s'])
        applied[phase==FAILED] = self.previous[phase==FAILED]
        self.history = raw.clone()
        self.previous = applied.clone()
        return applied,dict(actor_input=cpu(actor_input),raw_policy_actions=cpu(raw),requested_targets=cpu(requested),
            applied_targets=cpu(applied),alternate_action=cpu(alternate),feedback_yaw=correction,
            phase_used=phase,bridge=bridge,history_reset=new_model,external_command=desired)


def capture(env,endpoint,alive,ids,head,feet,limits,tolerance):
    robot = env.scene['robot']
    end = endpoint.rows[-1]
    raw = env.sim.wp_data
    count = int(raw.nacon.numpy()[0])
    pairs, worlds, distance = (getattr(raw.contact,name).numpy()[:count] for name in ('geom','worldid','dist'))
    first, second = pairs.T
    head_floor = ((first==endpoint.floor)&(endpoint.kinds[second]==1))|((second==endpoint.floor)&(endpoint.kinds[first]==1))
    depth = np.zeros(env.num_envs,dtype=np.float32)
    np.maximum.at(depth,worlds[head_floor],np.maximum(-distance[head_floor],0.))
    joints = cpu(robot.data.joint_pos[:,ids])
    return dict(active=alive.copy(),quaternion=cpu(robot.data.root_link_quat_w),
        head_quaternion=cpu(robot.data.body_link_quat_w[:,head]),height=cpu(robot.data.root_link_pos_w[:,2]),
        angular_velocity=cpu(robot.data.root_link_ang_vel_b),ground_contact=end['fresh'][:,0],
        head_contact=end['fresh'][:,1],foot_contact=end['fresh'][:,2:],head_penetration_m=depth,
        joint_envelope_ok=((joints>=limits[:,0]-tolerance)&(joints<=limits[:,1]+tolerance)).all(axis=1),
        joint_positions=joints,qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),
        velocity=cpu(robot.data.root_link_lin_vel_b[:,:2]),foot_positions=cpu(robot.data.site_pos_w[:,feet]),
        foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:,feet,:2]),axis=-1))


def collect(env,obs,p,folder,endpoint,networks,homes,sim_home,limits):
    from roll_stand_walk_sequence import score_forward

    robot = env.scene['robot']
    joints, order = robot.find_joints(p['walking']['joint_names'],preserve_order=True)
    assert order == p['walking']['joint_names']
    feet, sites = robot.find_sites(('left_foot','right_foot'),preserve_order=True)
    assert list(sites) == ['left_foot','right_foot']
    head = robot.find_bodies('jaw_soft')[0][0]
    tolerance = np.deg2rad(p['roll']['actual_joint_limit_tolerance_deg'])
    limit_np = cpu(limits)
    alive = np.ones(env.num_envs,dtype=bool)
    sequence = Sequence(p)
    policy = PolicyControl(p,networks,homes,sim_home,limits)
    states, controls = [], []
    snapshot = Snapshot(env,folder)
    video = ActionVideo(snapshot,'switch',folder,p['maximum_controls']*.02)
    log = folder/'execution.log'
    started = monotonic()
    view = dict(status='running',run_id=p['run_id'],purpose='站立、前滚翻、站立、前进 · 连续仿真',
                controls=0,maximum_controls=p['maximum_controls'],transitions=0,learning_updates=0,updates=[])

    def publish(k,status='running',frame=None):
        view.update(status=status,updated_at=datetime.now(TZ).isoformat(),controls=k,transitions=k*p['worlds'])
        if frame:
            view['frame'] = frame
        if video.clips and video.clips[0]['complete_clip']:
            view['latest_clip'] = video.clips[0]['file']
        with patch.object(live,'PAGE',PAGE):
            live.publish(folder,view)

    def take(k):
        frame = capture(env,endpoint,alive,joints,head,feet,limit_np,tolerance)
        assert all(np.isfinite(value).all() for value in frame.values())
        sequence.update(k,frame)
        states.append(dict(frame,phase_next=sequence.phase.copy()))

    (folder/'README.md').write_text('# 站立、前滚翻、站立、前进\n\n'
        '同一物理环境完整采集。切换不重置，保留模型原始动作历史与实际执行器目标。\n\n'
        '[方案](protocol.json) · [执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    k = 0
    try:
        take(0)
        video.sample()
        publish(0,frame=video.image['file'])
        for k in range(1,p['maximum_controls']+1):
            base = obs['actor'].clone()
            target, saved = policy.next(k,base,sequence)
            obs,_,term,timeout,_ = env.step(target-sim_home)
            received = env.action_manager.get_term('joint_pos')._processed_actions
            torch.testing.assert_close(received,target,rtol=0,atol=1e-6)
            alive &= ~cpu(term|timeout)
            controls.append(dict(saved,observed=cpu(base),terminated=cpu(term),truncated=cpu(timeout),
                                 actual_motor_target=cpu(received)))
            endpoint.capture()
            assert not endpoint.first_mismatches, endpoint.first_mismatches
            take(k)
            if video.writer is not None:
                if alive[0]:
                    video.sample()
                else:
                    video._sample_terminal()
            if k%25 == 0:
                assert monotonic()-started < p['maximum_sampling_wall_seconds']
            if k%50 == 0:
                elapsed = monotonic()-started
                view['updates'].append(dict(control=k,transitions_per_second=k*p['worlds']/elapsed))
                row = f'ROLL_STAND_WALK {k}/{p["maximum_controls"]} active {alive.sum()} roll {sum(sequence.stand_at>=0)} forward {sum(sequence.forward_at>=0)}'
                print(row,flush=True)
                with log.open('a',encoding='utf-8') as file:
                    file.write(row+'\n')
                publish(k,frame=snapshot.capture()['file'])
        assert len(env.guard_resets)==1 and env.guard_steps==env.common_step_counter==p['maximum_controls']
        arrays = {key:np.asarray([item[key] for item in states]) for key in states[0]}
        score = score_forward(arrays,sequence.forward_at,p)
        write(folder/'raw-score.json',dict(roll_switch_at=sequence.stand_at.tolist(),forward_at=sequence.forward_at.tolist(),
                                          forward=score))
        status = 'collected-review-required'
    except BaseException:
        publish(len(controls),status='stopped-review-required')
        raise
    finally:
        video_close_error = None
        try:
            try:
                video.close()
            except BaseException as exc:
                video_close_error = repr(exc)
                write(folder/'video-close-error.json',dict(status='captured-and-continued',error=video_close_error,
                    controls=len(controls),learning_updates=0,hardware_io=False))
            for name,rows in (('trace',states),('commands',controls)):
                if rows:
                    np.savez_compressed(folder/f'{name}.npz',**{key:np.asarray([x[key] for x in rows]) for key in rows[0]})
            endpoint.save(p['maximum_controls']+1)
        finally:
            snapshot.close()
    write(folder/'evaluation.json',dict(status=status,record=folder.relative_to(ROOT).as_posix(),
        protocol_sha256=sha(PLAN),actual_control_transitions=len(controls)*p['worlds'],
        actual_physics_substeps=env._sim_step_counter,physical_resets=1,learning_updates=0,heldout_used=False,
        trace_sha256=sha(folder/'trace.npz'),command_sha256=sha(folder/'commands.npz'),video=video.clips))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','zero','sample'))
    args = parser.parse_args()
    if args.command=='freeze':
        freeze()
    else:
        with torch.no_grad():
            run(args.command=='zero')


if __name__=='__main__':
    main()
