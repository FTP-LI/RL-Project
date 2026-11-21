"""Independently read back the original roll latch and paired motor targets."""

import numpy as np
import torch

from check_continuous_forward_fork_state import controller_snapshot
from getup_release_fork_state import digest, signature
from review_roll_forward_input_stop import measure
from roll_metrics import progress
from roll_neck_contact_window import apply_neck_pitch
from roll_stand_walk_sequence import STATE_FIELDS, Sequence
from run_roll_stand_forward_diagnostic import read, require, sha
from verify_continuous_roll_stand_walk import arrays, check_policy
from verify_forward_right_support_continuation import checked_arrays
from verify_roll_forward_input_integration import checked_contacts, joined, video


def read_back(plan, folder, manifest):
    torch.set_num_threads(2)
    spec = plan['neck_support']
    shared,horizon = (spec['plant'][key] for key in ('shared_snapshot_control','last_roll_control'))
    budget = spec['assessment']['maximum_unique_batched_controls_per_seed']
    runtime = read(folder/'runtime-contract.json')
    require(read(folder/'protocol.json')==plan and manifest['protocol_sha256']==sha(folder/'protocol.json')
            and manifest['recorded_control_transitions']==runtime['control_transitions']==budget*plan['worlds']
            and manifest['actual_control_steps']==runtime['control_steps']==budget and runtime['physics_substeps']==budget*5
            and manifest['physical_resets']==runtime['interface_checks']==1 and runtime['status']=='closed'
            and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps']), 'Actual physical budget differs')
    require(read(folder/'zero-readback.json')['initial']['counters']==[0,0], 'Missing actual CUDA zero-step receipt')
    prefix,prefix_commands = checked_arrays(folder/'shared',manifest['prefix_sha256'])
    require(len(prefix['active'])==shared+1 and len(prefix_commands['actor_input'])==shared,'The shared prefix is incomplete')
    fork = read(folder/'fork.json')
    controller_file = folder/'fork-controller.npz'
    memory = arrays(controller_file)
    ready = Sequence(plan)
    for k in range(shared+1):
        ready.update(k,{key:prefix[key][k] for key in STATE_FIELDS})
        np.testing.assert_array_equal(ready.phase,prefix['phase_next'][k])
    require(fork['physical']['counters']==[shared,5*shared] and sha(controller_file)==fork['controller_file_sha256']
            and digest(signature(memory))==digest(signature(controller_snapshot(ready)))==fork['controller_sha256']
            and {'env.obs_buf','controller.history','controller.previous_features'}<=set(fork['fields'])
            and any('.delay.' in name for name in fork['fields']), 'Snapshot evidence is incomplete')
    startup = arrays(folder/'startup.npz')
    joint = plan['walking']['joint_names'].index(spec['offset']['joint'])
    limits = plan['walking']['joint_limits'][joint]
    first = None
    arms = {}
    for role,magnitude in spec['roles'].items():
        root = folder/'clips'/role
        meta = manifest['branches'][role]
        restore = read(root/'restored-fork.json')
        require(restore=={name:fork[name] for name in ('physical','controller_sha256')},'Read-back fork differs')
        trace,commands = checked_arrays(root,meta['trace_sha256'])
        require(meta['controls']==len(commands['actor_input'])==horizon-shared
                and len(trace['active'])==horizon-shared+1,'A branch is incomplete')
        for name,values in prefix.items():
            np.testing.assert_array_equal(values[-1],trace[name][0],err_msg=name)
        motor = root/'first-motor.npz'
        require(sha(motor)==meta['first_motor_sha256'],'First physical motor receipt changed')
        substeps = arrays(motor)['substeps']
        require(substeps.shape==(5,plan['worlds'],14) and np.isfinite(substeps).all(),'Invalid first physical motor array')
        before = {name:commands[name][0] for name in ('observed','raw_policy_actions','applied_targets','actual_motor_target')}
        before['first_physics_motor'] = substeps[0]
        if first is None:
            first = before
        for name,value in before.items():
            np.testing.assert_array_equal(value,first[name],err_msg=name)
        motion,actions = joined(prefix,trace,1),joined(prefix_commands,commands,0)
        checked_contacts(root,meta,motion,horizon)
        np.testing.assert_array_equal(actions['phase_used'],motion['phase_next'][:-1])
        np.testing.assert_array_equal(motion['active'][1:],np.logical_and.accumulate(~(actions['terminated']|actions['truncated']),axis=0))
        require(motion['guard_phase_valid'].all() and not actions['bridge'].any(),'The original rollout contract changed')
        score = measure(motion,plan)
        stop = score['stop']
        require(stop is None if not meta['gate_decision']['stop'] else stop==meta['gate_decision'], 'The original roll guard differs')
        target = apply_neck_pitch(actions['applied_targets'],np.arange(1,horizon+1)[:,None],actions['phase_used'],
                                  joint,limits,spec['offset'],magnitude)
        np.testing.assert_allclose(actions['actual_motor_target'],target,atol=1e-6,rtol=0)
        nonzero = np.abs(target[...,joint]-actions['applied_targets'][...,joint])>1e-6
        if magnitude:
            altered = np.flatnonzero(nonzero.any(axis=1))+1
            require(altered.tolist()==list(range(74,100)) and not nonzero[:shared].any(),'Unexpected target timing')
        else:
            require(not nonzero.any(),'An original-policy reference was modified')
        policy = check_policy(dict(actions,actual_motor_target=actions['applied_targets']),startup,plan,actions['phase_used'])
        invisible = np.flatnonzero(~motion['active'][:,0])
        end = int(invisible[0]) if invisible.size else horizon
        clip = video(folder,meta['video'],0,end,horizon)
        angle,_,_ = progress({name:motion[name][plan['initial_stand_controls']:] for name in ('quaternion','ground_contact')})
        pos = np.maximum(np.diff(angle,axis=0),0.)
        ground = motion['ground_contact'][plan['initial_stand_controls']+1:]
        numbers = np.arange(plan['initial_stand_controls']+1,horizon+1)
        window = (numbers>=74)&(numbers<=100)
        gaps = np.rad2deg((pos[window]*~ground[window]).sum(axis=0))
        outcomes = score['passed']
        arms[role] = dict(roll_success=int(outcomes.sum()),roll_pass=outcomes.tolist(),terminal_guard=stop,
            changed_motor_control_worlds=int(nonzero.sum()),window_contact_gap_median_deg=float(np.median(gaps)),
            window_contact_gap_degrees=gaps.tolist(),failure_categories=score['failure_counts'],
            world_diagnostics=score['worlds'],policy=policy,video=clip)
    require(not torch.cuda.is_initialized(),'Independent replay must remain on CPU')
    comparisons = {}
    for role in ('reference-a','reference-b'):
        original,altered = (np.asarray(arms[label]['roll_pass']) for label in (role,'neck-pitch-positive'))
        wins,losses = int((~original&altered).sum()),int((original&~altered).sum())
        comparisons[role] = dict(gains=wins,losses=losses,net=wins-losses,
            paired_support_gap_change_median_deg=float(np.median(np.asarray(arms['neck-pitch-positive']['window_contact_gap_degrees'])
                                                                -arms[role]['window_contact_gap_degrees'])))
    candidate = arms['neck-pitch-positive']
    minimum_gain = spec['assessment']['minimum_paired_success_gain_against_each_reference']
    passed = candidate['roll_success']>=plan['minimum_roll_completion'] and all(row['net']>=minimum_gain for row in comparisons.values())
    return dict(run_id=plan['run_id'],seed=plan['seed'],arms=arms,comparisons=comparisons,passed_development_gate=passed,
        physical_transitions=runtime['control_transitions'],heldout_used=False,learning_updates=0,hardware_io=False)
