"""Freeze, admit and collect the nominal continuous XGO-Duck route."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
from math import ceil
import os
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from action_video import ActionVideo
from collect_routed_turn_feedback_pair import save_trace
from getup_anchor_runtime import RecordingSnapshot
from getup_coverage_metrics import POSES
from nominal_skill_composition_controller import COMMAND, FAILED, ROLLING, ROLL_STANDING, Composition
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback, check_geom_layout
from roll_stand_walk_sequence import ROLL, STAND_LANDING
from routed_getup_sequence import STAND
from routed_seven_command_controller import balanced_groups
from run_continuous_roll_stand_walk import capture, cpu, onnx_session
from run_local_getup_ppo import MemoryWatch
from run_routed_turn_feedback_pair import base, original, route
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


PROJECT = ROOT/'project.json'
DESIGN = ROOT/'configs/experiments/nominal_skill_composition_design.json'
PLAN = DESIGN.with_name('nominal_skill_composition.json')
REPORT = ROOT/'records/reports/nominal-skill-composition'
KEY, STAGE = 'nominal_skill_composition', 'nominal-skill-composition'
PREPARATION = STAGE+'-controller-preflight'
ENTRY = Path(__file__).resolve()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def allowed(project, stage):
    s,g = project['staged_training_request'],project['goal_execution']
    return (g['status']=='active' and not g['pause_requested_by_user'] and not s['pause_requested']
            and not s['deployment_authorized_now'] and not s['learning_running'] and s['current_stage']==stage)


def config(plan):
    cfg,_,_ = original.prepare(plan['environment'])
    cfg.env.episode_length_s = (plan['composition']['maximum_controls']+1)*.02
    cfg.env.viewer.width, cfg.env.viewer.height = 640,480
    assert not cfg.env.auto_reset and (cfg.env.sim.mujoco.timestep,cfg.env.decimation)==(.004,5)
    motor=cfg.env.scene.entities['robot'].articulation.actuators[0]
    assert motor.vin_range==(7.7,7.7) and (motor.delay_min_lag,motor.delay_max_lag)==(1,1)
    assert cfg.env.events['foot_friction'].params['ranges']==(1.,1.)
    return cfg.env


def freeze():
    from mjlab.scene import Scene

    project = read(PROJECT)
    state,task = project['goal_execution'][KEY],project['staged_training_request']
    assert allowed(project,PREPARATION) and not task['simulation_running']
    assert state['status']=='controller-preflight-design-registered' and sha(DESIGN)==state['design_sha256']
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    d = read(DESIGN)
    seed,n = d['development_seed'],d['worlds']
    assert seed not in (d['reserved_seed'],d['existing_low_friction_reserved_seed'])
    for parent in ('records/evaluations','records/experiments'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/parent).iterdir()), seed
    source = ROOT/d['source_review']
    assert sha(source)==d['source_review_sha256']==state['source_review_sha256']
    audited = read(source)
    assert audited['original_getup_standing_bridge_controls']==d['getup_to_stand_bridge_controls']==25
    assert audited['original_roll_standing_bridge_controls']==d['stand_to_roll_and_roll_to_official_stand_bridge_controls']==0
    assert d['command_entry_control']+d['command_controls']==d['maximum_controls']==2125
    paths = {name:ROOT/name for name in audited['source_sha256']}
    assert all(sha(path)==audited['source_sha256'][name] for name,path in paths.items())
    nominal, routed = (read(paths[name]) for name in ('records/reports/official-route-independent-s1235/analysis.json',
                                                    'artifacts/delivery/routed-seven-command-simulation.json'))
    route_plan = ROOT/nominal['protocol']
    original_route = ROOT/routed['protocol']['path']
    assert sha(route_plan)==nominal['protocol_sha256'] and sha(original_route)==routed['protocol']['sha256']
    spec, roll = read(original_route),read(route_plan)
    plant,seven = spec['environment'],spec['seven_command']
    assert spec['sequence']['boundary_controls']==[100,550,600] and spec['sequence']['controller']['standing_dwell_controls']==25
    assert d['requested_pose_counts'].keys()==dict.fromkeys(POSES).keys()
    codes = np.random.default_rng(seed).permutation(np.repeat(range(4),[d['requested_pose_counts'][name] for name in POSES]))
    assert len(codes)==n and plant['success_rate_min']==d['qualification']['minimum_getup_success_fraction']
    assign = balanced_groups(codes,seven['commands'], seed+70000)
    assert len(assign)==7 and all(len(ids)==128 for ids in assign.values())
    minimum = d['qualification']['minimum_complete_fraction_per_command']
    assert all(ceil(minimum*len(seven['groups'][name]))==need for name,need in seven['minimum_by_command'].items())
    assert ceil(d['qualification']['minimum_complete_fraction_overall']*routed['worlds'])==seven['minimum_overall']
    seven.update(groups=assign,minimum_by_command={name:ceil(minimum*len(ids)) for name,ids in assign.items()},
                 minimum_overall=ceil(d['qualification']['minimum_complete_fraction_overall']*n))
    start,roll_end,after = (d[name] for name in ('roll_start_control','roll_deadline_control','roll_to_base_stand_control'))
    assert roll_end-start==roll['roll_controls'] and after-roll_end==roll['landing_stand_controls']
    assert roll['minimum_roll_completion']/roll['worlds']==d['qualification']['minimum_roll_fraction']
    assert d['command_entry_control']-after==d['official_stand_to_base_stand_bridge_controls']==25
    roll.update(worlds=n,minimum_roll_completion=ceil(d['qualification']['minimum_roll_fraction']*n))
    model_specs = {role:dict(model=row['path'],model_sha256=row['sha256'],seed=seed) for role,row in audited['policies'].items()}
    assert all(sha(ROOT/row['model'])==row['model_sha256'] for row in model_specs.values())
    run = f'duck_nominal_skill_composition_s{seed}_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    folder = ROOT/'records/evaluations'/run
    plant.update(num_envs=n,seed=seed,stage=STAGE,run_id=run,split=dict(pose_codes=codes.tolist(),
                 groups={name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(POSES)}))
    assert audited['joint_names']==spec['walking']['joint_names']==roll['walking']['joint_names']
    assert plant['model_home']==audited['policies']['getup-supine']['policy_home']
    spec.update(status='frozen-before-dynamics',stage=STAGE,record=rel(folder),composition=d,roll_route=roll,
        models=model_specs,policy_home={role:row['policy_home'] for role,row in audited['policies'].items()},
        maximum_unique_controls=d['maximum_controls'],maximum_physics_substeps=d['maximum_controls']*5,
        maximum_control_transitions=d['maximum_controls']*n,data_scope='fresh-development',
        reserved_holdout=False,heldout_used=False,hardware_io=False,learning_updates=0)
    spec['source_sha256'].update({rel(path):sha(path) for path in (ENTRY,ENTRY.with_name('nominal_skill_composition_controller.py'),
        ENTRY.with_name('verify_nominal_skill_composition.py'),ENTRY.with_name('run_continuous_roll_stand_walk.py'),
        ENTRY.with_name('roll_stand_walk_sequence.py'),ENTRY.with_name('roll_landing_latch.py'),
        ENTRY.with_name('roll_endpoint_readback_capture.py'),ENTRY.with_name('routed_seven_command_score.py'),
        ENTRY.with_name('verify_forward_support_nominal_recovery.py'),ROOT/'scripts/training/action_video.py')})
    spec['references'].update({rel(path):sha(path) for path in (DESIGN,source,route_plan,original_route,*paths.values())})
    base.shared.assert_sources(spec)
    envcfg = config(spec)
    actual = Scene(envcfg.scene,device='cpu').compile()
    base.exact_scene(actual)
    check_geom_layout(actual)
    actors = load_actors(spec,'cpu',lambda:100)
    proof = boundary_check(spec,actors,torch.zeros((n,61),dtype=torch.float32))
    assert not torch.cuda.is_initialized()
    REPORT.mkdir()
    write(PLAN,spec)
    write(REPORT/'preflight.json',dict(protocol_sha256=sha(PLAN),cpu_scene_exact=True,physical_substeps=0,
        composition_boundaries=proof,observer=preflight(),seed=seed))
    state.update(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(REPORT/'preflight.json'))
    task.update(current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                current_protocol_execution='nominal-skill-composition-cpu-preflight-verified',
                current_evidence=rel(REPORT/'preflight.json'))
    write(PROJECT,project)
    print('NOMINAL_SKILL_COMPOSITION_FROZEN',sha(PLAN),flush=True)


def checked(status):
    project,plan = read(PROJECT),read(PLAN)
    state,task = project['goal_execution'][KEY],project['staged_training_request']
    assert allowed(project,STAGE) and state['status']==status and not task['simulation_running']
    assert task['current_protocol']==rel(PLAN) and task['current_protocol_frozen'] and sha(PLAN)==state['protocol_sha256']
    base.shared.assert_sources(plan)
    return project,plan


def load_actors(plan, device, counter):
    actors = {name:route.guarded_actor(item,device) for name,item in plan['models'].items()}
    supine = actors['getup-supine']
    actors['getup'] = route.routes({'reference-a':supine,'reference-b':supine,'candidate':actors['getup-right-side']},
                                  actors['getup-fallback'],counter,plan)['candidate']
    return actors


def boundary_check(plan, actors, sensed):
    ctl = Composition(plan,actors,sensed.device)
    previous = ctl.roll.sim_home.clone()
    upright = sensed.clone()
    upright[:,:6] = torch.as_tensor([0.,0.,0.,0.,0.,-1.],device=sensed.device)
    d = plan['composition']
    start,end,command = (d[name] for name in ('roll_start_control','roll_to_base_stand_control','command_entry_control'))
    ctl.roll_route.phase.fill(ROLL)
    ctl.roll_route.phase[1:3] = [STAND_LANDING,FAILED]
    ctl.roll_route.phase_changed.fill(start-d['getup_to_stand_control'])
    ctl.roll.history.fill_(.125)
    rolling = ctl.action(start,upright,previous)
    assert rolling.mode[:3].tolist()==[ROLLING,ROLL_STANDING,FAILED] and not np.any(rolling.bridge)
    assert not rolling.actor_input[:2,34:48].any() and not rolling.actor_input[:2,48:].any()
    np.testing.assert_array_equal(cpu(rolling.target[2]),cpu(previous[2]))
    for i,role in enumerate(('roll','walk')):
        np.testing.assert_allclose(rolling.actor_input[i,6:20]+cpu(ctl.roll.home[role]),
                                   cpu(upright[i,6:20]+previous[i]),atol=2e-7,rtol=0)
    ctl.roll_ready.fill(True)
    ctl.commands.history[STAND].fill_(.125)
    stance = ctl.action(end,upright,previous)
    assert stance.bridge.all() and not stance.actor_input[:,34:48].any()
    term = SimpleNamespace(**{key:torch.zeros_like(previous) for key in ('requested_target','projected_destination','_processed_actions','previous_target')})
    ctl.apply(term,previous,end,stance)
    assert float((term._processed_actions-previous).abs().max())<=.0400002
    ctl.stand_reads.fill(ctl.commands.dwell_steps)
    ctl.commands.dwell.fill_(ctl.commands.dwell_steps)
    ctl.commands.history[STAND].fill_(.125)
    terminal = ctl.action(command,upright,previous)
    assert (terminal.mode==COMMAND).all() and terminal.bridge.all()
    carried = ctl.commands.route_masks['base']
    assert (terminal.actor_input[carried,34:48]==.125).all() and not terminal.actor_input[~carried,34:48].any()
    assert torch.isfinite(terminal.raw).all()
    return dict(roll_and_official_stand_home_verified=True,roll_switch_bridge_controls=0,roll_switch_histories_zero=True,
        base_stand_history_zero=True,base_stand_maximum_target_step_rad=float((term._processed_actions-previous).abs().max()),
        seven_command_roles=len(plan['seven_command']['groups']),base_history_carried=int(carried.sum()),
        new_policy_history_zero=int((~carried).sum()),input_source='observations with upright fixture values for switch selection')


def qualify(env, obs, plan, actors, folder):
    folder.mkdir()
    term=env.action_manager.get_term('joint_pos')
    joints=plan['walking']['joint_names']
    assert list(term._target_names)==joints
    previous={name:cpu(getattr(env.sim.data,name)) for name in ('qpos','qvel','ctrl')}
    previous['actor']=cpu(obs['actor'])
    rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(env.device).clone())
    x = obs['actor']
    assert tuple(x.shape)==(plan['environment']['num_envs'],61)
    slots = np.unique(np.linspace(0,len(x)-1,32,dtype=int))
    home = torch.as_tensor(plan['environment']['model_home'],device=env.device,dtype=torch.float32)
    torch.testing.assert_close(torch.as_tensor(term._offset,device=env.device).expand(len(x),-1)[0],home,atol=0,rtol=0)
    errors={}
    for role,item in plan['models'].items():
        pose = torch.as_tensor(plan['policy_home'][role],device=env.device,dtype=torch.float32)
        adapted=x[slots].clone()
        adapted[:,6:20]+=home-pose
        adapted[:,34:]=0.
        reference=onnx_session(ROOT/item['model'])
        expected=np.concatenate([reference.run(None,{'obs':row[None]})[0] for row in cpu(adapted)])
        actual=cpu(actors[role](base.shared.td('actor',adapted)))
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        errors[role]=float(abs(actual-expected).max())
    original_getup=original.boundary_fixture(plan,{'stand':actors['stand-and-base'],'forward':actors['forward-official']},env.device,term)
    fixture=boundary_check(plan,actors,x)
    assert env.common_step_counter==env._sim_step_counter==0
    for name in ('qpos','qvel','ctrl'):
        np.testing.assert_array_equal(previous[name],cpu(getattr(env.sim.data,name)))
    np.testing.assert_array_equal(previous['actor'],cpu(env.obs_buf['actor']))
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state(env.device))
    np.savez_compressed(folder/'reset.npz',**previous)
    receipt=dict(status='passed-actual-cuda-zero',protocol_sha256=sha(PLAN),physical_substeps=0,physical_resets=1,
                 actual_reset_observations=len(slots),max_abs_model_errors=errors,original_getup_boundary=original_getup,
                 composition_boundary=fixture,reset_sha256=sha(folder/'reset.npz'))
    write(folder/'proof.json',receipt)
    return receipt


def cuda_env(plan):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    env=ManagerBasedRlEnv(cfg=config(plan),device='cuda:0',render_mode=None)
    env.guard_slots=int(env.sim.wp_data.contact.dist.shape[0])
    return env


def zero():
    project,plan=checked('zero-ready')
    folder=REPORT/'zero'
    assert not folder.exists()
    env=cuda_env(plan)
    try:
        obs,_=env.reset(seed=plan['environment']['seed'])
        actors=load_actors(plan,env.device,lambda:env.common_step_counter)
        receipt=qualify(env,obs,plan,actors,folder)
        endpoint=EndpointReadback(env,folder)
        endpoint.capture()
        assert not endpoint.first_mismatches
        endpoint.save(1)
    finally:
        env.close()
    project['goal_execution'][KEY].update(status='sample-ready',zero=rel(folder/'proof.json'),zero_sha256=sha(folder/'proof.json'))
    project['staged_training_request'].update(current_protocol_execution='nominal-skill-composition-actual-cuda-zero-verified',
                                              current_evidence=rel(folder/'proof.json'))
    write(PROJECT,project)
    print('NOMINAL_SKILL_COMPOSITION_ZERO',sha(folder/'proof.json'),max(receipt['max_abs_model_errors'].values()),flush=True)


def render_status(observer, camera, folder, run, count, horizon, worlds, progress, mode):
    now=monotonic()
    before,stamp,updates=progress
    if count>before:
        updates.append(dict(control=count,transitions_per_second=(count-before)*worlds/max(now-stamp,.01)))
    data=dict(status='running',run_id=run,purpose=mode,controls=count,maximum_controls=horizon,
              transitions=count*worlds,learning_updates=0,updates=updates,
              frame=camera.capture()['file'],updated_at=datetime.now(timezone.utc).isoformat())
    observer.publish(folder,data)
    return count,now,updates


def collect(env, plan, actors, folder, observer, watch, ledger):
    d,plant = plan['composition'],plan['environment']
    o, n, end = d['observer'],d['worlds'],d['maximum_controls']
    ctl=Composition(plan,actors,env.device)
    robot=env.scene['robot']
    joints, order=robot.find_joints(plan['walking']['joint_names'],preserve_order=True)
    feet, names=robot.find_sites(('left_foot','right_foot'),preserve_order=True)
    assert order==plan['walking']['joint_names'] and list(names)==['left_foot','right_foot']
    addresses=[int(env.sim.mj_model.joint('robot/'+name).qposadr[0]) for name in order]
    term=env.action_manager.get_term('joint_pos')
    endpoint=EndpointReadback(env,folder)
    alive=np.ones(n,bool)
    bounds=np.asarray(plant['joint_limits'])
    tol=np.deg2rad(plan['roll_route']['roll']['actual_joint_limit_tolerance_deg'])
    head=robot.find_bodies('jaw_soft')[0][0]
    process, physical, step = term.process_actions,env.sim.step,env.step
    rows,chunks,boundaries=[],[],[]
    keys = ('release_control','getup_to_stand_control','roll_start_control','roll_to_base_stand_control','command_entry_control')
    audit_controls={d[key] for key in keys}
    initial=cpu(env.sim.data.qpos)
    np.savez_compressed(folder/'initial.npz',qpos=initial,requested_pose=np.asarray(POSES)[plant['split']['pose_codes']])
    camera=RecordingSnapshot(env,folder)
    movie=ActionVideo(camera,'switch',folder,end*.02)
    start=monotonic()
    progress=(0,start,[])
    latest_action, previous, motor = None,None,[]

    def overwrite(raw):
        process(raw)
        ctl.apply(term,previous,env.common_step_counter,latest_action)

    def physical_step():
        assert ledger['physics_substeps'] < end*5
        if ledger['controls'] in audit_controls:
            motor.append(cpu(env.sim.data.ctrl))
        physical()
        ledger['physics_substeps']+=1

    def observe(control):
        endpoint.capture()
        assert not endpoint.first_mismatches, endpoint.first_mismatches
        frame=capture(env,endpoint,alive,joints,head,feet,bounds,tol)
        assert np.isfinite(frame['qpos']).all() and np.isfinite(frame['qvel']).all()
        ctl.observe_roll(control,frame)
        ctl.observe_base_stand(control,frame)
        return frame

    def flush():
        if rows:
            chunks.append(save_trace(folder,rows,o['trace_compression_level']))
            rows.clear()

    def as_array(value):
        return cpu(value) if isinstance(value,torch.Tensor) else value

    ledger.update(chunks=chunks,boundaries=boundaries,qpos_joint_addresses=addresses,
                  initial_sha256=sha(folder/'initial.npz'),camera_world=0,
                  camera_command=next(key for key,ids in plan['seven_command']['groups'].items() if 0 in ids))
    observer.begin_collect()
    try:
        with patch.object(term,'process_actions',overwrite), patch.object(env.sim,'step',physical_step), \
             patch.object(env,'reset',side_effect=AssertionError('Only the initial reset is allowed')):
            movie.sample()
            observe(0)
            progress=render_status(observer,camera,folder,plant['run_id'],0,end,n,progress,'标称连续动作｜跌倒准备')
            for control in range(end):
                watch.check()
                assert observer.check(), observer.refusal
                assert monotonic()-start<o['sampling_wall_seconds']
                assert env.common_step_counter==control and env._sim_step_counter==5*control
                sensed=env.obs_buf['actor'].clone()
                previous=term.previous_target.clone()
                latest_action=ctl.action(control,sensed,previous)
                if control==d['command_entry_control']:
                    seven=plan['seven_command']
                    reached=latest_action.mode==COMMAND
                    admission={name:int(reached[ids].sum()) for name,ids in seven['groups'].items()}
                    ledger['command_admission']=admission
                    if (int(reached.sum())<seven['minimum_overall'] or
                            any(admission[name]<seven['minimum_by_command'][name] for name in admission)):
                        ledger['reason']='complete-command-gate-unreachable-at-entry'
                        break
                _,_,terminated,truncated,_=step(latest_action.raw)
                ledger['controls']=t=control+1
                assert env.common_step_counter==t and env._sim_step_counter==ledger['physics_substeps']==5*t
                assert bool((env.episode_length_buf==t).all()) and not truncated.any()
                alive &= ~cpu(terminated)
                frame=observe(t)
                original_feet=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=-1)
                assert torch.isfinite(term._processed_actions).all()
                if control in audit_controls:
                    assert len(motor)==5
                    boundary=folder/f'first-target-{control:04d}.npz'
                    np.savez_compressed(boundary,observed=cpu(sensed),actor_input=as_array(latest_action.actor_input),
                        raw=cpu(latest_action.raw),mode=latest_action.mode,bridge=as_array(latest_action.bridge),
                        previous_target=cpu(previous),actual_target=cpu(term._processed_actions),
                        expected_articulation_target=cpu(robot.data.joint_pos_target[:,joints]),motor=np.stack(motor))
                    boundaries.append(dict(control=control,file=boundary.name,sha256=sha(boundary)))
                    motor.clear()
                rows.append(dict(control=t,mode=latest_action.mode,roll_phase=ctl.roll_route.phase.copy(),
                    roll_ready=ctl.roll_ready.copy(),base_stand_reads=ctl.stand_reads.copy(),original_feet=original_feet,
                    terminated=cpu(terminated),**frame))
                movie.sample()
                if t==d['release_control']:
                    poses=np.asarray(classify_release(projected_gravity(frame['quaternion']),plant['criteria']['release_alignment_min']))
                    np.savez_compressed(folder/'release-poses.npz',poses=poses)
                    counts={name:int((poses==name).sum()) for name in (*POSES,'other')}
                    ledger.update(actual_release_counts=counts,release_sha256=sha(folder/'release-poses.npz'))
                    q=d['qualification']
                    if (counts['right-side']<q['minimum_actual_right_side_releases'] or counts['other'] or
                            min(counts[name] for name in POSES)<q['minimum_each_other_actual_release']):
                        ledger['reason']='actual-release-coverage'
                        break
                if t%o['trace_controls_per_chunk']==0:
                    flush()
                if t%o['status_and_frame_every_controls']==0 or t==end:
                    mode=('起身' if t<=550 else '站立与滚翻' if t<=1100 else '站立与指令')
                    progress=render_status(observer,camera,folder,plant['run_id'],t,end,n,progress,'标称连续动作｜'+mode)
                    print('NOMINAL_SKILL_COMPOSITION',t,end,'roll',int((ctl.roll_route.stand_at>=0).sum()),flush=True)
                if t==d['roll_deadline_control'] and (ctl.roll_route.stand_at>=0).sum()<plan['roll_route']['minimum_roll_completion']:
                    ledger['reason']='original-roll-gate-unreachable'
                    break
            if ledger['controls'] and movie.writer is not None:
                movie._sample_terminal()
    finally:
        flush()
        endpoint.save(ledger['controls']+1)
        movie.close()
        camera.close()
        if movie.clips:
            ledger['video']=dict(movie.clips[-1],sha256=sha(folder/movie.clips[-1]['file']))


def sample():
    project,plan=checked('sample-ready')
    state,task=project['goal_execution'][KEY],project['staged_training_request']
    assert sha(REPORT/'zero/proof.json')==state['zero_sha256']
    folder=ROOT/plan['record']
    folder.mkdir()
    write(folder/'protocol.json',plan)
    owner=os.getpid()
    state.update(status='sampling',worker_pid=owner)
    task.update(simulation_running=True,current_protocol_execution='nominal-skill-composition-running')
    write(PROJECT,project)

    def permitted():
        current=read(PROJECT)
        return (allowed(current,STAGE) and current['goal_execution'][KEY].get('worker_pid')==owner
                and current['staged_training_request']['simulation_running'] and sha(PLAN)==state['protocol_sha256'])

    run=plan['environment']['run_id']
    observer=RecordedObserver(run,REPORT,permitted)
    observer.state.update(purpose='标称连续动作｜准备',maximum_controls=plan['composition']['maximum_controls'])
    ledger=dict(status='review-required',run_id=run,protocol_sha256=sha(PLAN),controls=0,physics_substeps=0,resets=0,
                hardware_io=False,learning_updates=0,reserved_seed_used=False)
    watch=MemoryWatch(plan['composition']['observer']['gpu_memory_fraction_limit'])
    env=None
    try:
        observer.start()
        with watch.active():
            env=cuda_env(plan)
            obs,_=env.reset(seed=plan['environment']['seed'])
            ledger['resets']=1
            actors=load_actors(plan,env.device,lambda:env.common_step_counter)
            receipt=qualify(env,obs,plan,actors,folder/'zero')
            prior=read(REPORT/'zero/proof.json')
            assert sha(REPORT/'zero/reset.npz')==prior['reset_sha256']
            with np.load(REPORT/'zero/reset.npz',allow_pickle=False) as expected, np.load(folder/'zero/reset.npz',allow_pickle=False) as actual:
                for field in expected.files:
                    np.testing.assert_array_equal(actual[field],expected[field],err_msg=field)
            assert receipt['composition_boundary']==prior['composition_boundary']
            ledger['process_local_zero_sha256']=sha(folder/'zero/proof.json')
            collect(env,plan,actors,folder,observer,watch,ledger)
            ledger['status']='collected' if ledger['controls']==plan['composition']['maximum_controls'] else 'stopped-at-frozen-gate'
    except BaseException as error:
        ledger['reason']=repr(error)
        raise
    finally:
        try:
            if env is not None:
                env.close()
            ledger['gpu_samples']=watch.rows
            ledger['transitions']=ledger['controls']*plan['composition']['worlds']
            write(folder/'collection.json',ledger)
        finally:
            try:
                observer.close(ledger)
            finally:
                current=read(PROJECT)
                key=current['goal_execution'][KEY]
                if key.get('worker_pid')==owner:
                    del key['worker_pid']
                    key.update(status='score-pending' if ledger['status']!='review-required' else 'review-required',
                               collection=rel(folder/'collection.json'),collection_sha256=sha(folder/'collection.json'))
                    current['staged_training_request'].update(simulation_running=False,current_evidence=rel(folder/'collection.json'))
                    write(PROJECT,current)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','zero','sample','score'))
    action=parser.parse_args().action
    if action=='score':
        from verify_nominal_skill_composition import score
        return score()
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'zero':zero,'sample':sample}[action]()


if __name__=='__main__':
    main()
