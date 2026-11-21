"""Collect a single standing prefix and three restored continuous routes."""

from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from check_nominal_pre_roll_standing_snapshot import StandingFork
from getup_release_fork_state import signature
from local_getup_scale import gpu_sample
from run_nominal_skill_composition import (COMMAND, POSES, STAND, ActionVideo, Composition, EndpointReadback,
    capture, classify_release, cpu, projected_gravity, save_trace, sha, write)
from training_watch import Snapshot


REFERENCES = ('first-common-standing-policy-repeat','second-common-standing-policy-repeat')
CANDIDATE = 'official-pre-roll-standing'
ROLES = (*REFERENCES,CANDIDATE)
LABELS = dict(zip(('shared',*ROLES),('共同轨迹','原站立对照一','原站立对照二','官方站立候选'),strict=True))


def collect(env, plan, actors, folder, observer, watch, ledger):
    d,plant = plan['composition'],plan['environment']
    pair,seven = plan['standing_pair'],plan['seven_command']
    o,n,end = d['observer'],d['worlds'],d['maximum_controls']
    shared = pair['common_snapshot_control']
    total = shared+len(ROLES)*(end-shared)
    ctl = Composition(plan,actors,env.device)
    robot = env.scene['robot']
    joints,order = robot.find_joints(plan['walking']['joint_names'],preserve_order=True)
    feet,names = robot.find_sites(('left_foot','right_foot'),preserve_order=True)
    assert order==plan['walking']['joint_names'] and list(names)==['left_foot','right_foot']
    addresses = [int(env.sim.mj_model.joint('robot/'+name).qposadr[0]) for name in order]
    head = robot.find_bodies('jaw_soft')[0][0]
    bounds = np.asarray(plant['joint_limits'])
    tol = np.deg2rad(plan['roll_route']['roll']['actual_joint_limit_tolerance_deg'])
    term = env.action_manager.get_term('joint_pos')
    process,physical,step = term.process_actions,env.sim.step,env.step
    audit_controls = {shared,*(d[key] for key in ('release_control','getup_to_stand_control','roll_start_control',
                                                 'roll_to_base_stand_control','command_entry_control'))}
    alive = np.ones(n,bool)
    initial = cpu(env.sim.data.qpos)
    np.savez_compressed(folder/'initial.npz',qpos=initial,requested_pose=np.asarray(POSES)[plant['split']['pose_codes']])
    ledger.update(initial_sha256=sha(folder/'initial.npz'),qpos_joint_addresses=addresses,camera_world=0,
                  camera_command=next(key for key,ids in seven['groups'].items() if 0 in ids),branches={})
    branch_dir = folder/'shared'
    branch_dir.mkdir()
    (folder/'frames').mkdir()
    endpoint = EndpointReadback(env,branch_dir)
    role = 'shared'
    branches = ledger['branches']
    motor, updates = [],[]
    latest_action, previous = None,None
    started = last_time = monotonic()
    last_count = 0
    state = dict(run_id=plant['run_id'],maximum_controls=total,learning_updates=0,updates=updates)
    frame_hashes = {name:{} for name in LABELS}
    start_weight = signature({key:actor.state_dict() for key,actor in actors.items()})

    def overwrite(raw):
        process(raw)
        ctl.apply(term,previous,env.common_step_counter,latest_action)

    def integrate():
        assert ledger['physics_substeps'] < total*5
        if env.common_step_counter in audit_controls:
            motor.append(cpu(env.sim.data.ctrl))
        physical()
        ledger['physics_substeps']+=1

    def forbid_reset(*args,**kwargs):
        raise AssertionError('Only the initial reset is allowed')

    def guard():
        watch.check()
        assert observer.check(), observer.refusal
        assert monotonic()-started<o['sampling_wall_seconds']

    def observe(control):
        endpoint.capture()
        assert not endpoint.first_mismatches, endpoint.first_mismatches
        frame=capture(env,endpoint,alive,joints,head,feet,bounds,tol)
        assert np.isfinite(frame['qpos']).all() and np.isfinite(frame['qvel']).all()
        ctl.observe_roll(control,frame)
        ctl.observe_base_stand(control,frame)
        return frame

    def publish():
        nonlocal last_count,last_time
        count,now = ledger['controls'],monotonic()
        if count>last_count:
            updates.append(dict(control=count,transitions_per_second=(count-last_count)*n/max(now-last_time,.01)))
            last_count,last_time = count,now
        k = env.common_step_counter
        path = f'frames/{role}-{k:04d}.jpg'
        if not (folder/path).exists():
            Image.fromarray(camera.render()).save(folder/path,quality=88)
        state.update(status='running',purpose=f'滚翻前站立对照｜{LABELS[role]}',controls=count,transitions=count*n,
                     frame=path,phase=role,sequence_control=k,updated_at=datetime.now(timezone.utc).isoformat())
        observer.publish(folder,state)
        print('PRE_ROLL_STAND',role,k,count,total,'roll',int((ctl.roll_route.stand_at>=0).sum()),flush=True)

    def as_array(value):
        return cpu(value) if isinstance(value,torch.Tensor) else value

    def collect_segment(first,last,record,writers):
        nonlocal latest_action,previous
        rows,standing = [],[]
        line = first
        base_history = None
        record.update(controls=first,chunks=[],boundaries=[])
        try:
            for control in range(first,last):
                guard()
                assert env.common_step_counter==control and env._sim_step_counter==5*control
                if role==CANDIDATE and control==shared:
                    base_history = ctl.commands.history[STAND].clone()
                    ctl.commands.history[STAND].zero_()
                    ctl.command_actors[STAND] = actors['forward-official']
                if role==CANDIDATE and control==d['roll_start_control']:
                    ctl.command_actors[STAND] = actors['stand-and-base']
                    ctl.commands.history[STAND].copy_(base_history)
                sensed = env.obs_buf['actor'].clone()
                previous = term.previous_target.clone()
                latest_action = ctl.action(control,sensed,previous)
                if control==d['command_entry_control']:
                    reached = latest_action.mode==COMMAND
                    admission = {name:int(reached[ids].sum()) for name,ids in seven['groups'].items()}
                    record.update(command_admission=admission,command_entry_worlds=np.flatnonzero(reached).tolist())
                    publish()
                    if (int(reached.sum())<seven['minimum_overall'] or
                            any(admission[name]<seven['minimum_by_command'][name] for name in admission)):
                        record['reason']='complete-command-gate-unreachable-at-entry'
                        break
                _,_,terminated,truncated,_ = step(latest_action.raw)
                ledger['controls']+=1
                line = record['controls'] = control+1
                assert env.common_step_counter==line and env._sim_step_counter==5*line
                assert ledger['physics_substeps']==ledger['controls']*5
                assert bool((env.episode_length_buf==line).all()) and not truncated.any()
                alive[:] &= ~cpu(terminated)
                frame = observe(line)
                original_feet = cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=-1)
                assert torch.isfinite(term._processed_actions).all()
                if control in audit_controls:
                    assert len(motor)==5
                    boundary = branch_dir/f'first-target-{control:04d}.npz'
                    np.savez_compressed(boundary,observed=cpu(sensed),actor_input=as_array(latest_action.actor_input),
                        raw=cpu(latest_action.raw),mode=latest_action.mode,bridge=as_array(latest_action.bridge),
                        previous_target=cpu(previous),actual_target=cpu(term._processed_actions),
                        expected_articulation_target=cpu(robot.data.joint_pos_target[:,joints]),motor=np.stack(motor))
                    record['boundaries'].append(dict(control=control,file=boundary.name,sha256=sha(boundary)))
                    motor.clear()
                if shared<=control<d['roll_start_control']:
                    standing.append(dict(actor_input=as_array(latest_action.actor_input),mode=latest_action.mode,
                        raw=cpu(latest_action.raw),actual_target=cpu(term._processed_actions)))
                rows.append(dict(control=line,mode=latest_action.mode,roll_phase=ctl.roll_route.phase.copy(),
                    roll_ready=ctl.roll_ready.copy(),base_stand_reads=ctl.stand_reads.copy(),original_feet=original_feet,
                    terminated=cpu(terminated),**frame))
                for writer in writers:
                    writer.sample()
                if line==d['release_control']:
                    poses=np.asarray(classify_release(projected_gravity(frame['quaternion']),plant['criteria']['release_alignment_min']))
                    np.savez_compressed(folder/'release-poses.npz',poses=poses)
                    counts={name:int((poses==name).sum()) for name in (*POSES,'other')}
                    ledger.update(actual_release_counts=counts,release_sha256=sha(folder/'release-poses.npz'))
                    q=d['qualification']
                    if (counts['right-side']<q['minimum_actual_right_side_releases'] or counts['other'] or
                            min(counts[name] for name in POSES)<q['minimum_each_other_actual_release']):
                        record['reason']='actual-release-coverage'
                        break
                if len(rows)==o['trace_controls_per_chunk']:
                    record['chunks'].append(save_trace(branch_dir,rows,o['trace_compression_level']))
                    rows.clear()
                if line%o['status_and_frame_every_controls']==0 or line==last:
                    publish()
                if line==d['roll_deadline_control'] and (ctl.roll_route.stand_at>=0).sum()<plan['roll_route']['minimum_roll_completion']:
                    record['reason']='original-roll-gate-unreachable'
                    break
        finally:
            if rows:
                record['chunks'].append(save_trace(branch_dir,rows,o['trace_compression_level']))
            if standing:
                target=branch_dir/'pre-roll-standing.npz'
                np.savez_compressed(target,**{key:np.stack([row[key] for row in standing]) for key in standing[0]})
                record['standing_sha256']=sha(target)
        return line

    observer.begin_collect()
    with ExitStack() as stack:
        stack.enter_context(patch.object(term,'process_actions',overwrite))
        stack.enter_context(patch.object(env.sim,'step',integrate))
        stack.enter_context(patch.object(env,'_reset_idx',forbid_reset))
        camera = Snapshot(env,folder)
        stack.callback(camera.close)
        render,last_render,pixels = camera.render,None,None

        def render_once():
            nonlocal last_render,pixels
            key = role,env.common_step_counter
            if key!=last_render:
                qpos=cpu(env.sim.data.qpos[0])
                pixels=render()
                np.testing.assert_array_equal(camera.data.qpos.astype(qpos.dtype),qpos)
                frame_hashes[role][key[1]]=sha256(qpos.tobytes()).hexdigest()
                last_render=key
            return pixels

        camera.render=render_once
        writers={}
        for name in ROLES:
            sub = folder/name
            sub.mkdir()
            writers[name]=ActionVideo(camera,'switch',sub,end*.02)
            stack.callback(writers[name].close)
            writers[name].sample()
        observe(0)
        publish()
        prefix = ledger['prefix'] = {}
        collect_segment(0,shared,prefix,writers.values())
        if prefix['controls']!=shared:
            ledger['reason']=prefix['reason']
            endpoint.save(prefix['controls']+1)
            return

        shared_alive,shared_contacts=alive.copy(),list(endpoint.rows)
        live_gpu=gpu_sample()
        limit=o['gpu_memory_fraction_limit']
        memory=int((live_gpu['total_mib']*limit-live_gpu['used_mib'])*1024**2)
        guard()
        fork=StandingFork(env,ctl,actors['getup'],memory)
        captured=dict(physical=fork.plant.verify(),controller_sha256=fork.expected)
        assert captured['physical']['counters']==[shared,shared*5]
        required={'warp.qpos','env.obs_buf','controller.history','rng'}
        assert required <= set(fork.plant.expected)
        assert any(name.startswith('obs-delay.actor.') for name in fork.plant.expected)
        assert any(name.startswith('actuator.0.delay.') for name in fork.plant.expected)
        write(folder/'fork.json',dict(**captured,fields=sorted(fork.plant.expected),field_signatures=fork.plant.expected))
        ledger['fork_sha256']=sha(folder/'fork.json')
        print('PRE_ROLL_FORK',captured,flush=True)
        for name in ROLES:
            guard()
            role,branch_dir=name,folder/name
            ledger['active_branch']=role
            restored=fork.restore()
            assert restored==captured
            write(branch_dir/'restored-fork.json',restored)
            endpoint.rows,endpoint.folder,endpoint.first_mismatches=list(shared_contacts),branch_dir,{}
            endpoint.paired_start_verified=True
            alive[:]=shared_alive
            branch=branches[name]=dict(restore_sha256=sha(branch_dir/'restored-fork.json'))
            print('PRE_ROLL_RESTORE',role,restored,flush=True)
            collect_segment(shared,end,branch,(writers[name],))
            endpoint.save(branch['controls']+1)
            writer=writers[name]
            if writer.writer is not None:
                writer._sample_terminal()
            clip, = writer.clips
            hashes=frame_hashes['shared'] | frame_hashes[role]
            write(branch_dir/'recording-states.json',
                [dict(common_step=k,qpos_sha256=hashes[k]) for k in clip['common_steps']])
            branch.update(status='collected' if branch['controls']==end else 'stopped-at-frozen-gate',
                          contact_sha256=sha(branch_dir/'endpoint-contact.npz'),
                          video=dict(clip,sha256=sha(branch_dir/clip['file'])))
            state['latest_clip']=f'{role}/{clip["file"]}'
            publish()
        assert signature({key:actor.state_dict() for key,actor in actors.items()})==start_weight
        assert ledger['controls']==shared+sum(item['controls']-shared for item in branches.values())
        assert ledger['physical_substeps']==5*ledger['controls']
        ledger['actor_weights_unchanged']=True
