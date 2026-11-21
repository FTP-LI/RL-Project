"""Compare three roll targets after restoring one actual standing boundary."""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from action_video import ActionVideo
from collect_roll_confirmation import model_input
from common_scene_contract import compare_arrays, install_guard
from common_walk_startup import td
from getup_stand_handoff import projected_slew
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from roll_entry_impossibility import maximum_possible
from roll_entry_snapshot import RollEntrySnapshot
from roll_landing_latch import RollLandingLatch
from roll_stand_walk_sequence import ROLL, Sequence
from run_continuous_roll_stand_walk import PLAN as PARENT
from run_continuous_roll_stand_walk import PROJECT, STARTUP, TZ, PolicyControl, actors, capture, configuration, cpu
from run_continuous_roll_stand_walk import checked as checked_parent
from training_watch import Snapshot


OUT = ROOT/'records/reports/continuous-roll-entry-isolation'
PRIOR = ROOT/'records/reports/continuous-roll-stand-walk'
PAGE = ROOT/'records/roll-entry-isolation-live.html'
PLAN = ROOT/'configs/experiments/continuous_roll_entry_isolation.json'
DRAFT = PLAN.with_name('continuous_roll_entry_isolation_draft.json')
ROLES = ('original-roll-target','projection-only','projection-and-slew')
LABELS = ('原始目标','目标投影','目标投影与限速')


def checked():
    plan = read(PLAN)
    assert plan['status']=='frozen-before-dynamics' and plan['roles']==list(ROLES)
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    checked_parent()
    return plan


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft, parent = read(DRAFT), checked_parent()
    assert draft['status']=='draft-not-executable' and tuple(draft['branches'])==ROLES
    assert all(draft[key] is False for key in ('hardware_io','heldout_used','automatic_retry'))
    assert draft['learning_updates']==0
    cpu_path = PRIOR/'roll-entry-fork-cpu-preflight.json'
    prior = read(cpu_path)
    assert prior['status']=='passed-cpu-zero-step-fork' and prior['verified']['counters']==[0,0]
    assert prior['verified']['fields']==294
    for name,digest in prior['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    stop_path = PRIOR/'roll-entry-impossibility-replay.json'
    assert read(stop_path)['first_stop']['global_control']==72
    assert draft['shared_stand_controls']==parent['initial_stand_controls']==50
    assert draft['maximum_roll_controls_per_branch']==parent['roll_controls']==450
    assert draft['worlds']==parent['worlds']==128 and parent['minimum_roll_completion']==116
    assert parent['roll']['criteria']['head_penetration_max_m']==.005
    seed = draft['development_seed']
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    p = dict(parent,seed=seed,roll=dict(parent['roll'],seed=seed),
             maximum_controls=draft['shared_stand_controls']+draft['maximum_roll_controls_per_branch'])
    paths = [PARENT,DRAFT,cpu_path,stop_path,ROOT/draft['source'],Path(__file__)]
    paths += [ROOT/'scripts/validation'/name for name in ('roll_entry_snapshot.py','roll_entry_impossibility.py',
        'getup_release_scene_state.py','getup_release_fork_state.py','verify_roll_entry_isolation.py')]
    run_id = f'duck_switch_roll-entry-isolation_development_s{seed}_{datetime.now(TZ):%Y%m%dT%H%M%S}'
    plan = dict(draft,status='frozen-before-dynamics',run_id=run_id,roles=list(ROLES),plant=p,
        minimum_roll_completion=parent['minimum_roll_completion'],maximum_warp_snapshot_bytes=268435456,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    budget = draft['shared_stand_controls']+len(ROLES)*draft['maximum_roll_controls_per_branch']
    assert budget==plan['maximum_controls_all_branches'] and budget*5==plan['maximum_physics_substeps']
    assert budget*draft['worlds']==plan['maximum_control_transitions']
    OUT.mkdir()
    write(PLAN,plan)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),observation=PAGE.name,
        cpu_snapshot_sha256=sha(cpu_path),automatic_stop_replay_sha256=sha(stop_path),physical_substeps=0))
    print('ROLL_ENTRY_ISOLATION_FROZEN',run_id,sha(PLAN),flush=True)


def target_for(role,local_step,requested,previous,limits,p):
    if role==ROLES[0]:
        return requested
    bounded = torch.clamp(requested,min=limits[:,0],max=limits[:,1])
    if role==ROLES[2] and local_step<=p['cross_model_bridge_controls']:
        return projected_slew(previous,bounded,limits,.02,p['bridge_slew_rad_s'])[0]
    return bounded


def save_rows(path, rows):
    if rows:
        np.savez_compressed(path,**{key:np.stack([row[key] for row in rows]) for key in rows[0]})


def collect(env,obs,plan,folder,endpoint,networks,homes,sim_home,limits):
    p = plan['plant']
    worlds = plan['worlds']
    warm, horizon = plan['shared_stand_controls'],plan['maximum_roll_controls_per_branch']
    robot = env.scene['robot']
    joints, order = robot.find_joints(p['walking']['joint_names'],preserve_order=True)
    feet, sites = robot.find_sites(('left_foot','right_foot'),preserve_order=True)
    assert order==p['walking']['joint_names'] and list(sites)==['left_foot','right_foot']
    head = robot.find_bodies('jaw_soft')[0][0]
    tolerance = np.deg2rad(p['roll']['actual_joint_limit_tolerance_deg'])
    bound = cpu(limits)
    term = env.action_manager.get_term('joint_pos')
    sequence = Sequence(p)
    policy = PolicyControl(p,networks,homes,sim_home,limits)
    alive = np.ones(worlds,dtype=bool)
    prefix, stand_commands, branches = [], [], {}
    started = monotonic()
    view = dict(run_id=plan['run_id'],maximum_controls=plan['maximum_controls_all_branches'],
                learning_updates=0,updates=[])
    frame_path = 'frame-step-0.jpg'
    role_name = '共同站立'
    video_root = folder/'clips'
    video_root.mkdir()
    snapshots = Snapshot(env,folder)
    cameras = {}
    original_step = env.sim.step

    def publish(status='running', sample=None):
        count = env.guard_steps
        if sample is not None:
            view['updates'].append(dict(control=count,transitions_per_second=count*worlds/max(.01,monotonic()-started)))
        view.update(status=status,purpose=f'滚翻入口同初态 · {role_name}',updated_at=datetime.now(TZ).isoformat(),
                    controls=count,transitions=count*worlds,frame=frame_path)
        with patch.object(live,'PAGE',PAGE):
            live.publish(folder,view)

    def observe():
        state = capture(env,endpoint,alive,joints,head,feet,bound,tolerance)
        assert all(np.isfinite(value).all() for value in state.values())
        return state

    def step(applied):
        nonlocal alive
        obs,_,terminated,truncated,_ = env.step(applied-sim_home)
        alive &= ~cpu(terminated|truncated)
        torch.testing.assert_close(term._processed_actions,applied,rtol=0,atol=1e-6)
        endpoint.capture()
        assert not endpoint.first_mismatches,endpoint.first_mismatches
        return obs['actor'],cpu(terminated),cpu(truncated)

    def log(value):
        print(value,flush=True)
        with (folder/'execution.log').open('a',encoding='utf-8') as f:
            f.write(value+'\n')

    (folder/'README.md').write_text('# 滚翻入口同初态对照\n\n'
        '三组共用 50 步站立轨迹和同一物理快照。原片从共同站立开始。若已不可能达到门槛，保留到停止点。\n\n'
        '[冻结方案](protocol.json) · [采样日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    try:
        for role in ROLES:
            path = video_root/role
            path.mkdir()
            cameras[role] = ActionVideo(snapshots,'switch',path,(warm+horizon)*.02)
            cameras[role].sample()
        snapshots.capture()
        publish()
        prefix.append(observe())
        sequence.update(0,prefix[0])
        actor_obs = obs['actor']
        for k in range(1,warm+1):
            raw_obs = cpu(actor_obs)
            applied,saved = policy.next(k,actor_obs,sequence)
            actor_obs,terminated,truncated = step(applied)
            stand_commands.append(dict(saved,observed=raw_obs,terminated=terminated,truncated=truncated))
            prefix.append(observe())
            sequence.update(k,prefix[-1])
            for camera in cameras.values():
                camera.sample()
            if k%10==0:
                frame_path = snapshots.capture()['file']
                publish(sample=k)
        eligible = sequence.phase==ROLL
        assert len(env.guard_resets)==1 and eligible.any()
        save_rows(folder/'standing-trace.npz',prefix)
        save_rows(folder/'standing-commands.npz',stand_commands)
        frozen_endpoints = list(endpoint.rows)
        fork = RollEntrySnapshot(env,policy,plan['maximum_warp_snapshot_bytes'])
        receipt = fork.verify()
        assert receipt['counters']==[warm,warm*5]
        write(folder/'fork.json',dict(captured=receipt,fields=sorted(fork.expected),eligible=int(eligible.sum())))
        first_obs, first_raw, first_motor = None,None,None
        for role,label in zip(ROLES,LABELS,strict=True):
            role_name = label
            replay = fork.verify() if role==ROLES[0] else fork.restore()
            assert replay==receipt
            write(video_root/role/'restored-fork.json',replay)
            endpoint.rows = list(frozen_endpoints)
            endpoint.first_mismatches = {}
            endpoint.folder = video_root/role
            endpoint.paired_start_verified = True
            alive = prefix[-1]['active'].copy()
            start_state = observe()
            for name,value in start_state.items():
                np.testing.assert_array_equal(value,prefix[-1][name],err_msg=name)
            states,commands,motor = [start_state],[],[]
            latch = RollLandingLatch(p['roll']['criteria'],worlds)
            latch.advance(dict(start_state,active=start_state['active']&eligible))
            timeline = [maximum_possible(latch,horizon)]
            camera = cameras[role]
            local = 0
            try:
                while local<horizon and timeline[-1][0]>=plan['minimum_roll_completion']:
                    local += 1
                    base = env.obs_buf['actor'].clone()
                    history = torch.zeros_like(policy.history) if local==1 else policy.history
                    inputs = model_input(base,sim_home,homes['roll'],history)
                    raw = networks['roll'](td(inputs))
                    assert torch.isfinite(raw).all()
                    request = homes['roll']+raw
                    applied = target_for(role,local,request,policy.previous,limits,p)
                    applied = torch.where(torch.from_numpy(eligible&alive).to(env.device)[:,None],applied,policy.previous)
                    if local==1:
                        if first_obs is None:
                            first_obs,first_raw = cpu(inputs),cpu(raw)
                        np.testing.assert_array_equal(cpu(inputs),first_obs)
                        np.testing.assert_array_equal(cpu(raw),first_raw)

                        def delayed_step(*args,**kwargs):
                            motor.append(cpu(env.sim.data.ctrl))
                            return original_step(*args,**kwargs)

                        env.sim.step = delayed_step
                    _,terminated,truncated = step(applied)
                    env.sim.step = original_step
                    if local==1:
                        assert len(motor)==5
                        if first_motor is None:
                            first_motor = motor[0]
                        np.testing.assert_array_equal(motor[0],first_motor)
                        np.savez_compressed(video_root/role/'first-motor.npz',substeps=np.stack(motor))
                    commands.append(dict(observed=cpu(base),actor_input=cpu(inputs),raw_policy_actions=cpu(raw),
                        requested_targets=cpu(request),applied_targets=cpu(applied),
                        actual_motor_target=cpu(term._processed_actions),terminated=terminated,truncated=truncated))
                    policy.history,policy.previous = raw,applied
                    state = observe()
                    states.append(state)
                    latch.advance(dict(state,active=state['active']&eligible))
                    timeline.append(maximum_possible(latch,horizon))
                    if camera.writer is not None:
                        camera.sample() if alive[0] else camera._sample_terminal()
                    if local%10==0:
                        assert monotonic()-started<p['maximum_sampling_wall_seconds']
                        frame = snapshots.capture()
                        frame_path = f'frame-{role}-{warm+local}.jpg'
                        (folder/frame['file']).replace(folder/frame_path)
                        publish(sample=local)
                reason = 'full-roll-window' if local==horizon else 'original-threshold-impossible'
                if camera.writer is not None:
                    camera._sample_terminal()
                clip, = camera.clips
                branches[role] = dict(controls=local,reason=reason,certified=timeline[-1][1],
                    possible=timeline[-1][0],timeline=[list(row) for row in timeline],
                    ready_steps=latch.first_ready.tolist(),video=dict(clip,path=f'clips/{role}/{clip["file"]}',
                        sha256=sha(video_root/role/clip['file'])))
                log(f'ROLL_ENTRY_ISOLATION {role} controls={local} possible={timeline[-1][0]} certified={timeline[-1][1]} {reason}')
            finally:
                env.sim.step = original_step
                save_rows(video_root/role/'trace.npz',states)
                save_rows(video_root/role/'commands.npz',commands)
                endpoint.save(warm+horizon+1)
        assert env.guard_steps==warm+sum(row['controls'] for row in branches.values())
        assert env.guard_steps<=plan['maximum_controls_all_branches'] and len(env.guard_resets)==1
        write(folder/'evaluation.json',dict(status='collected-review-required',record=folder.relative_to(ROOT).as_posix(),
            protocol_sha256=sha(PLAN),controls=env.guard_steps,substeps=env.guard_steps*5,transitions=env.guard_steps*worlds,
            branches=branches,physical_resets=1,learning_updates=0,heldout_used=False))
        role_name = '三组采样结束，原片见本轮文档'
        with (folder/'README.md').open('a',encoding='utf-8') as f:
            f.write(' · '.join(f'[{label}原片]({branches[role]["video"]["path"]})'
                              for role,label in zip(ROLES,LABELS,strict=True))+'\n')
        publish()
    except BaseException:
        publish('stopped-review-required')
        raise
    finally:
        for camera in reversed(list(cameras.values())):
            camera.close()
        snapshots.close()


def run(zero):
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    plan = checked()
    p = plan['plant']
    task = read(PROJECT)['staged_training_request']
    assert task['current_stage']==plan['stage'] and task['current_protocol_frozen'] and not task['pause_requested']
    assert ROOT/task['current_protocol']==PLAN
    assert read(OUT/'windows-entry.json')['protocol_sha256']==sha(PLAN)
    folder = OUT/'zero' if zero else ROOT/'records/evaluations'/plan['run_id']
    folder.mkdir(exist_ok=False)
    write(folder/'protocol.json',plan)
    configure_torch_backends(allow_tf32=False)
    cfg = configuration(p)
    dump_yaml(folder/'env.yaml',asdict(cfg))
    cls = install_guard(p['roll'],folder,STARTUP,'rollout')
    env = cls(cfg=cfg,device='cuda:0',render_mode=None)
    try:
        env.seed(plan['development_seed'])
        obs,_ = env.reset()
        networks,homes,sim_home,limits = actors(env,obs,p,folder)
        endpoint = EndpointReadback(env,folder)
        endpoint.capture()
        assert not endpoint.first_mismatches
        if zero:
            endpoint.save(1)
            print('ROLL_ENTRY_ISOLATION_ZERO',plan['development_seed'],flush=True)
            return
        for file in ('startup.npz',):
            with np.load(OUT/'zero'/file) as before,np.load(folder/file) as current:
                compare_arrays({key:current[key] for key in current.files},{key:before[key] for key in before.files})
        with np.load(OUT/'zero/endpoint-contact.npz') as before:
            for key,value in endpoint.rows[0].items():
                np.testing.assert_array_equal(value,before[key][0],err_msg=key)
        collect(env,obs,plan,folder,endpoint,networks,homes,sim_home,limits)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','zero','sample'))
    args = parser.parse_args()
    if args.command=='freeze':
        freeze()
    else:
        with torch.no_grad():
            run(args.command=='zero')
