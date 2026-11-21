"""Recompute roll acceptance, actor assignments, and exact fork receipts."""

import numpy as np
import torch

from check_continuous_forward_fork_state import controller_snapshot
from collect_roll_checkpoint_selection import CANDIDATE, REFERENCES, ROLES
from getup_release_fork_state import digest, signature
from review_roll_forward_input_stop import measure
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, STAND_START, Sequence
from run_roll_stand_forward_diagnostic import read, require, sha
from verify_continuous_roll_stand_walk import arrays, check_policy
from verify_forward_right_support_continuation import checked_arrays
from verify_roll_forward_input_integration import checked_contacts, joined, video


def read_back(plan, folder, manifest):
    torch.set_num_threads(2)
    spec = plan['checkpoint_selection']
    shared,horizon,first_roll = (spec['plant'][key] for key in ('shared_snapshot_control','last_roll_control','first_roll_actor_control'))
    budget = spec['assessment']['maximum_unique_batched_controls_per_seed']
    runtime = read(folder/'runtime-contract.json')
    require(read(folder/'protocol.json')==plan and manifest['protocol_sha256']==sha(folder/'protocol.json')
            and manifest['actual_control_steps']==runtime['control_steps']==budget
            and manifest['recorded_control_transitions']==runtime['control_transitions']==budget*plan['worlds']
            and manifest['actual_physics_substeps']==runtime['physics_substeps']==budget*5
            and manifest['physical_resets']==runtime['interface_checks']==1 and runtime['status']=='closed'
            and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps']), 'Actual physical budget differs')
    startup_check = read(folder/'checkpoint-actor-admission.json')
    require(read(folder/'zero-readback.json')['initial']['counters']==[0,0]
            and startup_check['physical_controls']==startup_check['physical_substeps']==0
            and startup_check['actual_cuda_inputs']==plan['worlds']
            and startup_check['joint_names']==plan['walking']['joint_names']
            and startup_check['models']=={'final':plan['roll']['model_sha256'],'earlier':plan['earlier_roll']['model_sha256']},
            'Missing actual initial actor equivalence')
    prefix,commands = checked_arrays(folder/'shared',manifest['prefix_sha256'])
    require(len(prefix['active'])==shared+1 and len(commands['actor_input'])==shared,'The common stance is incomplete')
    fork = read(folder/'fork.json')
    memory_path = folder/'fork-controller.npz'
    memory = arrays(memory_path)
    ready = Sequence(plan)
    for index in range(shared+1):
        ready.update(index,{key:prefix[key][index] for key in STATE_FIELDS})
        np.testing.assert_array_equal(ready.phase,prefix['phase_next'][index])
    require(fork['physical']['counters']==[shared,shared*5] and sha(memory_path)==fork['controller_file_sha256']
            and digest(signature(memory))==digest(signature(controller_snapshot(ready)))==fork['controller_sha256']
            and {'env.obs_buf','controller.history','controller.previous_features'}<=set(fork['fields'])
            and any('.delay.' in field for field in fork['fields']), 'The exact restore inventory differs')
    startup = arrays(folder/'startup.npz')
    first,arms = None,{}
    require(tuple(manifest['branches'])==ROLES,'Restored branch order differs')
    for role in ROLES:
        branch = folder/'clips'/role
        item = manifest['branches'][role]
        model = plan['earlier_roll'] if role==CANDIDATE else plan['roll']
        require(read(branch/'roll-actor.json')==dict(role=role,model_sha256=model['model_sha256'],restore_control=shared,
                    first_roll_actor_control=first_roll) and read(branch/'restored-fork.json')=={
                        name:fork[name] for name in ('physical','controller_sha256')},'The branch actor or saved state differs')
        trace,actions = checked_arrays(branch,item['trace_sha256'])
        require(len(actions['actor_input'])==item['controls']==horizon-shared and len(trace['active'])==horizon-shared+1,
                'The declared first episode is incomplete')
        for name,values in prefix.items():
            np.testing.assert_array_equal(values[-1],trace[name][0],err_msg=name)
        motor_path = branch/'first-motor.npz'
        require(sha(motor_path)==item['first_motor_sha256'],'First actual motor sample differs')
        motor = arrays(motor_path)['substeps']
        require(motor.shape==(5,plan['worlds'],14) and np.isfinite(motor).all(),'The initial motor array is invalid')
        first_command = {name:actions[name][0] for name in ('observed','actor_input','raw_policy_actions','actual_motor_target')}
        first_command['actual_first_substep_torque'] = motor[0]
        if first is None:
            first = first_command
        for name,value in first_command.items():
            np.testing.assert_array_equal(value,first[name],err_msg=name)
        motion,controls = joined(prefix,trace,1),joined(commands,actions,0)
        checked_contacts(branch,item,motion,horizon)
        phase = controls['phase_used']
        np.testing.assert_array_equal(phase,motion['phase_next'][:-1])
        np.testing.assert_array_equal(motion['active'][1:],np.logical_and.accumulate(~(controls['terminated']|controls['truncated']),axis=0))
        require((phase[shared]==STAND_START).all() and (phase[first_roll-1]==ROLL).any()
                and not (phase[:first_roll-1]==ROLL).any() and motion['guard_phase_valid'].all(), 'The initial command or actor boundary differs')
        replay = check_policy(controls,startup,dict(plan,roll=model),phase)
        roll = measure(motion,plan)
        stop = item['gate_decision'] if item['gate_decision']['stop'] else None
        require(roll['stop']==stop,'Original roll and safety guard did not replay')
        end = np.flatnonzero(~motion['active'][:,0])
        clip = video(folder,item['video'],0,int(end[0]) if end.size else horizon,horizon)
        start = plan['initial_stand_controls']
        angle,_,_ = progress({name:motion[name][start:] for name in ('quaternion','ground_contact')})
        missing = np.rad2deg((np.maximum(np.diff(angle,axis=0),0.)[73-start:100-start]
                              *~motion['ground_contact'][74:101]).sum(axis=0))
        arms[role] = dict(roll_success=int(roll['passed'].sum()),roll_pass=roll['passed'].tolist(),reference_actor_sha256=model['model_sha256'],
            failed_conditions=roll['failure_counts'],world_diagnostics=roll['worlds'],window_contact_gap_degrees=missing.tolist(),
            window_contact_gap_median_deg=float(np.median(missing)),maximum_head_depth_m=float(motion['head_penetration_m'].max()),
            policy=replay,video=clip)
    require(not torch.cuda.is_initialized(),'Independent scoring must remain on CPU')
    selected = np.asarray(arms[CANDIDATE]['roll_pass'])
    reference_vectors = [np.asarray(arms[role]['roll_pass']) for role in REFERENCES]
    comparison = {}
    for role,values in zip(REFERENCES,reference_vectors,strict=True):
        gains,losses = int((~values&selected).sum()),int((values&~selected).sum())
        support = np.asarray(arms[CANDIDATE]['window_contact_gap_degrees'])-arms[role]['window_contact_gap_degrees']
        comparison[role] = dict(gains=gains,losses=losses,net=gains-losses,paired_contact_gap_change_median_deg=float(np.median(support)))
    gate = (int(selected.sum())>=plan['minimum_roll_completion'] and all(row['net']>=spec['assessment'][
        'minimum_paired_success_gain_against_each_reference'] for row in comparison.values()))
    return dict(run_id=plan['run_id'],seed=plan['seed'],arms=arms,comparisons=comparison,passed_development_gate=gate,
        same_actor_reference_flip_count=int((reference_vectors[0]!=reference_vectors[1]).sum()),
        physical_transitions=runtime['control_transitions'],physical_substeps=runtime['physics_substeps'],
        heldout_used=False,learning_updates=0,hardware_io=False)
