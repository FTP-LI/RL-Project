"""Independently score the same-state rolling route using the original gates."""

from collections import Counter
from unittest.mock import patch

import numpy as np

import verify_continuous_roll_stand_walk as policy_reader
from check_continuous_forward_fork_state import controller_snapshot
from collect_roll_forward_input_integration import ROLES, remap
from command_input_mapping import transform
from getup_release_fork_state import digest, signature
from roll_screen_stop_guard import check_online
from roll_stand_walk_sequence import ROLL, STAND_LANDING, Sequence, score_forward
from run_roll_stand_forward_diagnostic import put, read, rel, require, sha
from verify_continuous_forward_yaw_pair import joined
from verify_forward_right_support_continuation import checked_arrays
from verify_roll_entry_isolation import video


def replay_route(plan, trace, commands, startup, role):
    design = plan['integration']
    shared, horizon = design['shared_control_boundary'], plan['maximum_controls']
    require(len(trace['active']) == horizon+1 and len(commands['phase_used']) == horizon and trace['guard_phase_valid'].all(),
            'The first-episode route buffers do not match the original horizon')
    np.testing.assert_array_equal(commands['phase_used'], trace['phase_next'][:-1])
    np.testing.assert_array_equal(trace['active'][1:], np.logical_and.accumulate(~(
        commands['terminated'] | commands['truncated']), axis=0))
    sequence, common, rolling = Sequence(plan), None, None
    for k, expected in enumerate(trace['phase_next']):
        state = {key: values[k] for key, values in trace.items() if key not in ('phase_next','guard_phase_valid')}
        changed = sequence.update(k, state)
        decision = check_online(sequence, state, k, plan)
        require(not decision['stop'], f'The archived route violates its original stop gate: {decision}')
        np.testing.assert_array_equal(sequence.phase, expected)
        if k == shared:
            common = controller_snapshot(sequence)
        if k == sequence.roll_end:
            rolling = (sequence.stand_at >= 0).copy()
        if k < horizon:
            switching = np.isin(sequence.phase, (ROLL, STAND_LANDING))
            np.testing.assert_array_equal(commands['history_reset'][k], changed & switching)
            np.testing.assert_array_equal(commands['bridge'][k], switching & (
                k+1-sequence.phase_changed <= plan['cross_model_bridge_controls']))
    require(common is not None and rolling is not None, 'The original roll cutoff or shared boundary is absent')
    control = np.arange(horizon, dtype=np.float32)[:, None]
    entered = sequence.forward_at[None, :]

    def mapping(values):
        return remap(values, control, entered, design) if role == ROLES[1] else transform(values)

    with patch.object(policy_reader, 'transform', mapping):
        policy = policy_reader.check_policy(commands, startup, plan, commands['phase_used'])
    external = commands['external_command']
    prior = transform(external)
    mapped = mapping(external)
    change = mapped[..., 0] != prior[..., 0]
    np.testing.assert_array_equal(mapped[..., 1:], prior[..., 1:])
    require(not change[:shared+1].any(), 'The forward input changed before the authorized CUDA restore')
    if role == ROLES[1]:
        exposed = ((external[..., 0] == np.float32(design['external_forward_command_m_s']))
                   & (entered >= 0)
                   & (control > entered+design['delay_from_each_world_forward_entry_controls'])).any(axis=0)
        first = change.argmax(axis=0)
        np.testing.assert_array_equal(first[exposed], sequence.forward_at[exposed]+design['delay_from_each_world_forward_entry_controls']+1)
        require(not change[:, ~exposed].any(), 'The adapted input changed worlds outside forward motion')
    scores = score_forward(trace, sequence.forward_at, plan)
    return common, rolling, sequence, scores, policy, int(change.sum())


def checked_contacts(branch, meta, trace, horizon):
    reader = read(branch/'endpoint-contact.json')
    path = branch/'endpoint-contact.npz'
    require(reader['complete'] and reader['zero_matches_standalone'] and not reader['first_mismatches']
            and reader['cache_counters_and_rng_unchanged_at_each_boundary']
            and sha(path) == meta['contact_sha256'] == reader['sha256'], 'Independent contact evidence differs')
    c = policy_reader.arrays(path)
    contact = np.concatenate((trace['ground_contact'][..., None], trace['head_contact'][..., None], trace['foot_contact']), axis=-1)
    np.testing.assert_array_equal(contact, c['fresh'])
    np.testing.assert_array_equal(contact, c['constraint_points'] > 0)
    np.testing.assert_array_equal(trace['qpos'][..., 3:7], c['quaternion'])
    np.testing.assert_array_equal(c['control'], np.arange(horizon+1))
    np.testing.assert_array_equal(c['substep'], np.arange(horizon+1)*5)


def score(plan, folder):
    owned = read(folder/'protocol.json')
    receipt = read(folder/'guarded-evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    design = plan['integration']
    shared, horizon = design['shared_control_boundary'], plan['maximum_controls']
    budget, worlds = design['maximum_unique_controls'], plan['worlds']
    require(owned == plan and folder.name == plan['run_id'] and receipt['status'] == 'collected-review-required'
            and receipt['saved_episode_controls'] == horizon and receipt['actual_control_steps'] == runtime['control_steps'] == budget
            and receipt['actual_physics_substeps'] == runtime['physics_substeps'] == budget*5
            and receipt['recorded_control_transitions'] == runtime['control_transitions'] == budget*worlds
            and receipt['physical_resets'] == runtime['interface_checks'] == 1
            and runtime['status'] == 'closed' and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps'])
            and not (receipt['heldout_used'] or receipt['learning_updates'] or receipt['hardware_io']),
            'Physical budget, identity, reset or safety contract differs')
    prefix, prefix_commands = checked_arrays(folder/'shared', receipt['prefix_sha256'])
    require(len(prefix['active']) == shared+1 and len(prefix_commands['actor_input']) == shared, 'Shared route is incomplete')
    zero, fork = (read(folder/name) for name in ('zero-readback.json', 'fork.json'))
    memory_path = folder/'fork-controller.npz'
    memory = policy_reader.arrays(memory_path)
    physical = fork['physical']
    require(zero['initial']['counters'] == [0, 0] and physical['counters'] == [shared, shared*5]
            and sha(memory_path) == fork['controller_file_sha256']
            and digest(signature(memory)) == fork['controller_sha256']
            and {'env.obs_buf','controller.history','controller.previous_features'} <= set(fork['fields'])
            and any('.delay.' in name for name in fork['fields']), 'Exact snapshot evidence is incomplete')
    startup = policy_reader.arrays(folder/'startup.npz')
    arms, paired_scores = {}, {}
    first_reference = None
    for role in ROLES:
        branch = folder/'clips'/role
        item = receipt['branches'][role]
        require(read(branch/'restored-fork.json') == dict(physical=physical, controller_sha256=fork['controller_sha256'])
                and not item['stop'] and item['controls'] == horizon-shared, 'The archived branch differs from its contract')
        states, commands = checked_arrays(branch, item['trace_sha256'])
        require(len(states['active']) == horizon-shared+1 and len(commands['actor_input']) == horizon-shared,
                'The archived branch is missing controls')
        for key in prefix:
            np.testing.assert_array_equal(prefix[key][-1], states[key][0], err_msg=key)
        motor_path = branch/'first-motor.npz'
        require(sha(motor_path) == item['first_motor_sha256'], 'Physical motor record changed')
        first = {key:commands[key][0] for key in ('observed','actor_input','raw_policy_actions','actual_motor_target')}
        first['motor_substeps'] = policy_reader.arrays(motor_path)['substeps']
        require(first['motor_substeps'].shape[0] == 5, 'The first control lacks physical substeps')
        if first_reference is None:
            first_reference = first
        for key, value in first.items():
            np.testing.assert_array_equal(value, first_reference[key], err_msg=key)
        route, logical = joined(prefix, states, 1), joined(prefix_commands, commands, 0)
        checked_contacts(branch, item, route, horizon)
        common, roll_pass, sequence, scores, policy, changes = replay_route(plan, route, logical, startup, role)
        for key, value in common.items():
            np.testing.assert_array_equal(value, memory[key], err_msg=key)
        passes = sum(row['success'] for row in scores)
        rolls = int(roll_pass.sum())
        end = np.flatnonzero(~route['active'][:, 0])
        camera_end = int(end[0]) if end.size else horizon
        clip = video(folder, item['video'], 0, camera_end, horizon)
        raw = branch/'raw-score.json'
        put(raw, dict(roll_switch_at=sequence.stand_at.tolist(), roll_world_pass=roll_pass.tolist(),
                      forward_at=sequence.forward_at.tolist(), forward=scores))
        entered = [row for row in scores if row['entered_forward']]
        paired_scores[role] = scores
        arms[role] = dict(roll_completed=rolls, entered_forward=len(entered), passed=passes, adapted_controls=changes,
            passed_development_gate=rolls >= plan['minimum_roll_completion'] and passes >= plan['minimum_continuous_successes'],
            median_abs_yaw_rad_s=float(np.median([row['yaw_error_rad_s'] for row in entered])) if entered else None,
            world_median_signed_speed_m_s=float(np.median([row['signed_speed_m_s'] for row in entered])) if entered else None,
            failure_counts=dict(Counter(reason for row in scores for reason in row['failed_checks'])),
            policy=policy, video=clip, raw=rel(raw), raw_sha256=sha(raw))
    original, adapted = (paired_scores[role] for role in ROLES)
    common_delta = [b['yaw_error_rad_s']-a['yaw_error_rad_s'] for a,b in zip(original, adapted, strict=True)
                    if a['entered_forward'] and b['entered_forward']]
    require(not policy_reader.torch.cuda.is_initialized(), 'Independent replay must stay on CPU')
    return dict(run_id=plan['run_id'], seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'],
                exact_cuda_fork=True, exact_controller_and_first_motor=True, arms=arms,
                paired_entering_worlds=len(common_delta),
                paired_yaw_median_change_rad_s=float(np.median(common_delta)) if common_delta else None,
                physical_transitions=budget*worlds, physical_substeps=budget*5, safety_violations=0)
