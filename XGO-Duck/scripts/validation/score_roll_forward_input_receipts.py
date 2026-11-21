"""Replay complete rolling routes with first-substep or archived-stop evidence."""

from collections import Counter
import re

import numpy as np
import torch

from collect_roll_forward_input_integration import ROLES
from getup_release_fork_state import digest, signature
from run_roll_stand_forward_diagnostic import read, require, sha
from verify_forward_right_support_continuation import checked_arrays
from verify_continuous_roll_stand_walk import arrays
from verify_roll_forward_input_integration import checked_contacts, joined, replay_route, video


def trace_hashes(folder):
    return {name:sha(folder/name) for name in ('trace.npz', 'commands.npz')}


def archived_stop(folder, plan_path, command_log, collected):
    stop = read(folder/'paired-stop.json')
    plan = read(plan_path)
    shared, budget = (plan['integration'][key] for key in ('shared_control_boundary', 'maximum_unique_controls'))
    horizon = plan['maximum_controls']
    require(collected['status'] == 'review-required' and collected['run_id'] == plan['run_id']
            and stop['stopped'] and stop['detail'] is None and stop['role'] == ROLES[1]
            and stop['physical_controls'] == budget and stop['episode_control'] == horizon
            and collected['reason'] == 'AssertionError: '+stop['reason']
            and set(stop['branches']) == {ROLES[0]}, 'Unexpected archived stop')
    log = command_log.read_text(encoding='utf-8')
    reason = stop['reason']
    indices = [int(index) for index in re.findall(r'^ \[(\d+), \d+, \d+\]:', reason, re.MULTILINE)]
    mismatches = re.search(r'Mismatched elements: (\d+) / (\d+)', reason)
    maximum = re.search(r'Max absolute difference among violations: ([0-9.e+\-]+)', reason)
    require('np.testing.assert_array_equal(np.stack(first_motor), motor_reference)' in log
            and reason.strip() in log and indices and min(indices) == 1 and mismatches and maximum
            and int(mismatches[2]) == 5*plan['worlds']*14, 'The torque diagnostic does not isolate later substeps')
    branch = folder/'clips'/ROLES[1]
    require(not (branch/'first-motor.npz').exists(), 'The original stop should precede the candidate torque archive')
    clip, = read(branch/'clips.json')
    clip_path = f'clips/{ROLES[1]}/{clip["file"]}'
    branches = dict(stop['branches'], **{ROLES[1]:dict(controls=horizon-shared, stop=None,
        trace_sha256=trace_hashes(branch), contact_sha256=read(branch/'endpoint-contact.json')['sha256'],
        first_motor_sha256=None, video=dict(clip, path=clip_path, sha256=sha(folder/clip_path)))})
    return dict(protocol_sha256=sha(plan_path), prefix_sha256=trace_hashes(folder/'shared'), branches=branches,
        evidence_scope='trajectory-replay-after-archived-torque-assertion', independent_first_motor=False,
        archived_stop_sha256=sha(folder/'paired-stop.json'), command_log_sha256=sha(command_log),
        paired_motor_log=dict(first_mismatch_substep_index=indices[0], mismatched_values=int(mismatches[1]),
                              maximum_absolute_difference_n_m=float(maximum[1])))


def score(plan, folder, receipt):
    runtime = read(folder/'runtime-contract.json')
    require(read(folder/'protocol.json') == plan and folder.name == plan['run_id']
            and sha(folder/'protocol.json') == receipt['protocol_sha256'], 'Owned protocol changed')
    design = plan['integration']
    shared, horizon = design['shared_control_boundary'], plan['maximum_controls']
    budget, worlds = design['maximum_unique_controls'], plan['worlds']
    require(runtime['control_steps'] == budget and runtime['physics_substeps'] == budget*5
            and runtime['control_transitions'] == budget*worlds and runtime['interface_checks'] == 1
            and runtime['status'] == 'closed' and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps']),
            'The physical budget, reset or safety contract differs')
    prefix, prefix_commands = checked_arrays(folder/'shared', receipt['prefix_sha256'])
    require(len(prefix['active']) == shared+1 and len(prefix_commands['actor_input']) == shared, 'Incomplete shared prefix')
    fork = read(folder/'fork.json')
    memory_path = folder/'fork-controller.npz'
    memory = arrays(memory_path)
    physical = fork['physical']
    require(read(folder/'zero-readback.json')['initial']['counters'] == [0, 0]
            and physical['counters'] == [shared, shared*5] and sha(memory_path) == fork['controller_file_sha256']
            and digest(signature(memory)) == fork['controller_sha256']
            and {'env.obs_buf','controller.history','controller.previous_features'} <= set(fork['fields'])
            and any('.delay.' in name for name in fork['fields']), 'Snapshot evidence differs')
    startup = arrays(folder/'startup.npz')
    independent = receipt.get('independent_first_motor', True)
    raw_scores, arms, first_commands, motors = {}, {}, {}, {}
    for role in ROLES:
        item = receipt['branches'][role]
        branch = folder/'clips'/role
        require(read(branch/'restored-fork.json') == dict(physical=physical, controller_sha256=fork['controller_sha256'])
                and item['stop'] is None and item['controls'] == horizon-shared, 'Branch restore evidence differs')
        states, commands = checked_arrays(branch, item['trace_sha256'])
        require(len(states['active']) == horizon-shared+1 and len(commands['actor_input']) == horizon-shared,
                'Incomplete branch trace')
        for name, values in prefix.items():
            np.testing.assert_array_equal(values[-1], states[name][0], err_msg=name)
        first_commands[role] = {name:commands[name][0] for name in (
            'observed', 'actor_input', 'raw_policy_actions', 'applied_targets', 'actual_motor_target')}
        if item['first_motor_sha256'] is not None:
            motor_path = branch/'first-motor.npz'
            require(sha(motor_path) == item['first_motor_sha256'], 'First motor evidence changed')
            motor = arrays(motor_path)['substeps']
            require(motor.shape == (5,worlds,14) and np.isfinite(motor).all(), 'The first sampled motor control is incomplete')
            motors[role] = motor
        route, logical = joined(prefix, states, 1), joined(prefix_commands, commands, 0)
        checked_contacts(branch, item, route, horizon)
        common, roll_pass, sequence, scores, policy, changed = replay_route(plan, route, logical, startup, role)
        for name, value in common.items():
            np.testing.assert_array_equal(value, memory[name], err_msg=name)
        end = np.flatnonzero(~route['active'][:, 0])
        clip = video(folder, item['video'], 0, int(end[0]) if end.size else horizon, horizon)
        raw_scores[role] = dict(roll_switch_at=sequence.stand_at.tolist(), roll_world_pass=roll_pass.tolist(),
                                forward_at=sequence.forward_at.tolist(), forward=scores)
        eligible = [row for row in scores if row['entered_forward']]
        yaw = [row['yaw_error_rad_s'] for row in eligible if row['yaw_error_rad_s'] is not None]
        speed = [row['signed_speed_m_s'] for row in eligible if row['signed_speed_m_s'] is not None]
        roll, full = int(roll_pass.sum()), sum(row['success'] for row in scores)
        arms[role] = dict(roll_completed=roll, passed=full, entered_forward=len(eligible), adapted_controls=changed,
            passed_development_gate=roll>=plan['minimum_roll_completion'] and full>=plan['minimum_continuous_successes'],
            median_abs_yaw_rad_s=float(np.median(yaw)) if yaw else None,
            world_median_signed_speed_m_s=float(np.median(speed)) if speed else None,
            failure_counts=dict(Counter(reason for row in scores for reason in row['failed_checks'])), policy=policy, video=clip)
    original, candidate = (first_commands[role] for role in ROLES)
    for name in original:
        np.testing.assert_array_equal(original[name], candidate[name], err_msg=name)
    if independent:
        before, after = (motors[role] for role in ROLES)
        np.testing.assert_array_equal(before[0], after[0], err_msg='First applied motor torque')
        delta = np.abs(after-before)
        motor_proof = dict(independent_first_substep_equal=True,
                           subsequent_max_abs_difference_n_m_by_substep=delta[1:].max(axis=(1,2)).tolist())
    else:
        require(set(motors) == {ROLES[0]}, 'Archived diagnostic evidence changed')
        motor_proof = dict(independent_first_substep_equal=False, **receipt['paired_motor_log'])
    old, new = (raw_scores[role]['forward'] for role in ROLES)
    paired_yaw = [b['yaw_error_rad_s']-a['yaw_error_rad_s'] for a,b in zip(old,new,strict=True)
                  if a.get('yaw_error_rad_s') is not None and b.get('yaw_error_rad_s') is not None]
    require(not torch.cuda.is_initialized(), 'Independent replay used CUDA')
    return dict(run_id=plan['run_id'], seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'],
        exact_cuda_fork=True, exact_first_policy_and_motor_target=True, first_motor_readback=motor_proof,
        evidence_scope=receipt.get('evidence_scope','independent-first-physical-substep'),
        physical_transitions=budget*worlds, physical_substeps=budget*5, safety_violations=0,
        paired_entering_worlds=len(paired_yaw), paired_yaw_median_change_rad_s=float(np.median(paired_yaw)) if paired_yaw else None,
        arms=arms), raw_scores
