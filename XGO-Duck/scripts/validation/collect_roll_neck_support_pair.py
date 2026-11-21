"""Sample two original policies and the neck-offset route from one CUDA state."""

from time import monotonic
from unittest.mock import patch

import numpy as np

from check_continuous_forward_fork_state import controller_snapshot, restore_controller
from collect_forward_right_support_continuation import save_trace
from getup_release_fork_state import digest, signature
from roll_entry_snapshot import RollEntrySnapshot
from roll_neck_contact_window import apply_neck_pitch
from roll_screen_stop_guard import check_online, check_state
from roll_stand_walk_sequence import Sequence
from run_roll_stand_forward_diagnostic import require


LABELS = {'shared':'共同起始轨迹','reference-a':'第一次原策略参照','reference-b':'第二次原策略参照',
          'neck-pitch-positive':'滚翻支撑交接颈部俯仰候选'}


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        spec = plan['neck_support']
        plant, roles, schedule = (spec[key] for key in ('plant','roles','offset'))
        shared, horizon = (plant[key] for key in ('shared_snapshot_control','last_roll_control'))
        worlds, budget = plan['worlds'], spec['assessment']['maximum_unique_batched_controls_per_seed']
        joint_index = plan['walking']['joint_names'].index(schedule['joint'])
        joint_limits = plan['walking']['joint_limits'][joint_index]
        robot = env.scene['robot']
        joints, order = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, names = robot.find_sites(('left_foot','right_foot'), preserve_order=True)
        require(order == plan['walking']['joint_names'] and list(names) == ['left_foot','right_foot'], 'Robot geometry order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        alive = np.ones(worlds,dtype=bool)
        bounds = base.cpu(limits)
        policy = base.PolicyControl(plan,networks,homes,sim_home,limits)
        sequence = Sequence(plan)
        term = env.action_manager.get_term('joint_pos')
        began = last_time = monotonic()
        role, last_count = 'shared', 0
        branches, clips = {}, {}
        view = dict(run_id=plan['run_id'],maximum_controls=budget,learning_updates=0,updates=[])

        def guard():
            require(may_continue() and monotonic()-began<plan['maximum_sampling_wall_seconds'],
                    'Observation, pause, time or memory guard stopped the cohort')

        def snapshot():
            return base.capture(env,endpoint,alive,joints,head,feet,bounds,tolerance)

        def advance(k, motor):
            guard()
            require(env.guard_steps<budget and env.common_step_counter==k-1, 'Control counter differs before issuing a target')
            observed = env.obs_buf['actor'].clone()
            target, row = policy.next(k,observed,sequence)
            if roles.get(role,0.) and schedule['last_zero_before_active_control']<k<schedule['first_zero_after_active_control']:
                # Change only the physical neck target. The actor's action history
                # and the untouched policy targets stay available for CPU replay.
                projected = apply_neck_pitch(base.cpu(target),k,row['phase_used'],joint_index,joint_limits,schedule,roles[role])
                target = base.torch.as_tensor(projected,device=target.device)
            if motor is not None:
                physics = env.sim.step

                def record_motor(*args, **kwargs):
                    motor.append(base.cpu(env.sim.data.ctrl))
                    return physics(*args,**kwargs)

                with patch.object(env.sim,'step',record_motor):
                    _,_,terminated,truncated,_ = env.step(target-sim_home)
            else:
                _,_,terminated,truncated,_ = env.step(target-sim_home)
            base.torch.testing.assert_close(term._processed_actions,target,atol=1e-6,rtol=0)
            alive[:] &= ~base.cpu(terminated|truncated)
            endpoint.capture()
            require(not endpoint.first_mismatches,'Solver contact differs from endpoint reader')
            return dict(row,observed=base.cpu(observed),actual_motor_target=base.cpu(term._processed_actions),
                        terminated=base.cpu(terminated),truncated=base.cpu(truncated))

        def record(k, states):
            state = snapshot()
            decision = check_state(state,k,plan)
            require(not decision['stop'],f'{role} physical guard at {k}: {decision}')
            sequence.update(k,state)
            states.append(dict(state,phase_next=sequence.phase.copy(),guard_phase_valid=np.asarray(True)))
            decision = check_online(sequence,state,k,plan)
            # At the declared final control a failed roll count is the outcome
            # of this branch. The other same-snapshot arms are still required.
            require(not decision['stop'] or (k==horizon and decision['reason']=='roll-gate-impossible'),
                    f'{role} early roll guard at {k}: {decision}')
            return decision

        guard()
        zero = RollEntrySnapshot(env,policy,plan['maximum_warp_snapshot_bytes'])
        initial = zero.verify()
        require(initial['counters']==[0,0] and zero.restore()==initial,'Actual CUDA zero-step restore failed')
        np.testing.assert_array_equal(base.cpu(env.obs_buf['actor']),base.cpu(obs['actor']))
        base.write(folder/'zero-readback.json',dict(initial=initial,fields=sorted(zero.expected)))
        del zero
        camera = base.Snapshot(env,folder)
        render, cached, last_render = camera.render,None,None

        def render_once():
            nonlocal cached,last_render
            key = role,env.common_step_counter
            if key!=last_render:
                cached,last_render = render(),key
            return cached

        camera.render = render_once

        def publish(frame=False):
            nonlocal last_count,last_time
            control, count = env.common_step_counter,env.guard_steps
            now = monotonic()
            if count>last_count:
                view['updates'].append(dict(control=count,transitions_per_second=(count-last_count)*worlds/max(now-last_time,.01)))
                last_time,last_count = now,count
            if frame:
                src = camera.capture()['file']
                view['frame'] = f'frame-{role}-{control}.jpg'
                (folder/src).replace(folder/view['frame'])
            view.update(status='running',purpose=LABELS[role],controls=count,transitions=count*worlds,
                        updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder,view)
            print('ROLL_NECK_SUPPORT',plan['seed'],role,count,budget,control,
                  'certified',int((sequence.stand_at>=0).sum()),flush=True)

        def steps(first,last,states,commands,recorders,motor=None):
            for k in range(first,last+1):
                commands.append(advance(k,motor if k==first else None))
                decision = record(k,states)
                for video in recorders:
                    if video.writer is not None:
                        video.sample() if alive[0] else video._sample_terminal()
                if k%50==0 or k==last:
                    publish(frame=True)
            return decision

        common = folder/'shared'
        common.mkdir()
        current_role = None
        try:
            for name in roles:
                branch = folder/'clips'/name
                branch.mkdir(parents=True)
                clips[name] = base.ActionVideo(camera,'switch',branch,horizon*.02)
                clips[name].sample()
            prefix, prefix_commands = [],[]
            record(0,prefix)
            publish(frame=True)
            try:
                steps(1,shared,prefix,prefix_commands,clips.values())
            finally:
                prefix_hashes = save_trace(common,prefix,prefix_commands)
            contacts = list(endpoint.rows)
            memory = controller_snapshot(sequence)
            controller_sha = digest(signature(memory))
            np.savez_compressed(folder/'fork-controller.npz',**memory)
            fork = RollEntrySnapshot(env,policy,plan['maximum_warp_snapshot_bytes'])
            physical = fork.verify()
            require(physical['counters']==[shared,5*shared], 'Fork boundary differs')
            base.write(folder/'fork.json',dict(physical=physical,controller_sha256=controller_sha,fields=sorted(fork.expected),
                controller_file_sha256=base.sha(folder/'fork-controller.npz')))
            reference_motor = None
            for role in roles:
                guard()
                current_role = role
                branch = folder/'clips'/role
                restored = fork.restore()
                restore_controller(sequence,memory)
                require(restored==physical and digest(signature(controller_snapshot(sequence)))==controller_sha,
                        'Branch restore changed physical state or roll latch')
                base.write(branch/'restored-fork.json',dict(physical=restored,controller_sha256=controller_sha))
                endpoint.rows,endpoint.folder,endpoint.first_mismatches = list(contacts),branch,{}
                alive[:] = prefix[-1]['active']
                boundary = dict(snapshot(),phase_next=sequence.phase.copy(),guard_phase_valid=np.asarray(True))
                for name,value in boundary.items():
                    np.testing.assert_array_equal(value,prefix[-1][name],err_msg=name)
                states,commands,motor = [boundary],[],[]
                try:
                    steps(shared+1,shared+1,states,commands,(clips[role],),motor)
                    require(len(motor)==5, 'Initial motor control did not record every physical substep')
                    # Later forces depend on newly solved positions. Always save
                    # the measured substeps before comparing the common first one.
                    motor_path = branch/'first-motor.npz'
                    np.savez_compressed(motor_path,substeps=np.stack(motor))
                    if reference_motor is None:
                        reference_motor = motor[0]
                    np.testing.assert_array_equal(motor[0],reference_motor)
                    decision = steps(shared+2,horizon,states,commands,(clips[role],))
                finally:
                    hashes = save_trace(branch,states,commands)
                    endpoint.save(horizon+1)
                clips[role].finish(complete=False)
                clip, = clips[role].clips
                path = f'clips/{role}/{clip["file"]}'
                branches[role] = dict(controls=len(commands),gate_decision=decision,trace_sha256=hashes,
                    contact_sha256=base.sha(branch/'endpoint-contact.npz'),first_motor_sha256=base.sha(motor_path),
                    video=dict(clip,path=path,sha256=base.sha(folder/path)))
                view['latest_clip'] = path
                publish()
            require(env.guard_steps==budget and env.common_step_counter==horizon and len(env.guard_resets)==1,
                    'Physical counts differ from the fixed cohort budget')
            base.write(folder/'guarded-evaluation.json',dict(status='collected-review-required',protocol_sha256=base.sha(base.PLAN),
                actual_control_steps=budget,actual_physics_substeps=budget*5,recorded_control_transitions=budget*worlds,
                physical_resets=1,prefix_sha256=prefix_hashes,branches=branches,learning_updates=0,heldout_used=False,hardware_io=False))
        except BaseException as exc:
            base.write(folder/'paired-stop.json',dict(role=current_role or 'shared',reason=str(exc),
                physical_controls=int(env.guard_steps),episode_control=int(env.common_step_counter),branches=branches))
            raise
        finally:
            for video in reversed(list(clips.values())):
                video.close()
            camera.close()
    return collect
