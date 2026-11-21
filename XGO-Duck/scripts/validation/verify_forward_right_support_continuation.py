"""Verify the shared state, motors, original forward gates, and full videos."""

import numpy as np

from audit_forward_yaw_by_support_age import contact_age
from forward_right_support_continuation import CONTINUATION_FIELDS, RightSupportContinuation
from right_hip_pitch_duration_control import PHASE_FIELDS, SCHEDULE_FIELDS
from roll_stand_walk_sequence import score_forward
from run_forward_phase_hip_pitch import summary
from run_roll_stand_forward_diagnostic import StandForward, put, read, rel, require, sha
from verify_continuous_forward_yaw_pair import joined
from verify_continuous_roll_stand_walk import arrays, check_policy
from verify_roll_entry_isolation import video


def checked_arrays(folder, hashes):
    require(set(hashes) == {'trace.npz', 'commands.npz'}, 'Incomplete branch trace')
    for name, digest in hashes.items():
        require(sha(folder/name) == digest, 'Trace changed: '+name)
    return arrays(folder/'trace.npz'), arrays(folder/'commands.npz')


def verify_route(trace, commands, plan, startup, enabled):
    sequence = StandForward(plan)
    for k, phase in enumerate(trace['phase_next']):
        sequence.update(k, {key: value[k] for key, value in trace.items()})
        np.testing.assert_array_equal(sequence.phase, phase)
    np.testing.assert_array_equal(commands['phase_used'], trace['phase_next'][:-1])
    np.testing.assert_array_equal(sequence.forward_at, plan['initial_stand_controls'])
    require(trace['active'].all() and not (commands['terminated'] | commands['truncated']).any()
            and trace['joint_envelope_ok'].all()
            and (trace['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
            'The saved trajectory violates a physical guard')
    original = commands['applied_targets']
    policy = check_policy(dict(commands, actual_motor_target=original), startup, plan, commands['phase_used'])
    design = plan['duration_probe']
    controller = RightSupportContinuation(plan, design)
    bound = controller.onset
    for k, observed in enumerate(commands['observed']):
        item = {key: commands[key][k] for key in ('requested_targets', 'applied_targets', 'phase_used')}
        expected, detail = controller.next(k, item, observed, enabled)
        for key, value in detail.items():
            np.testing.assert_array_equal(value, commands[key][k], err_msg=key)
        actual = commands['actual_motor_target'][k]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        neutral = np.clip(item['requested_targets'], bound.lower, bound.upper)
        require(np.isfinite(actual).all() and (actual >= bound.lower-1e-6).all()
                and (actual <= bound.upper+1e-6).all() and np.abs(actual-neutral).max() <= bound.cap+1e-6,
                'An actual motor target exceeded its original envelope')

    hits = commands['right_extension_applied']
    supports, ages = contact_age(trace['foot_contact'][:-1])
    low, high = design['recognized_right_support_age_controls']
    starts = hits & ~np.vstack((np.zeros_like(hits[:1]), hits[:-1]))
    match = hits & (supports == 2)
    k, w = np.nonzero(starts & match)
    unique = len(np.unique(np.column_stack((w, k-ages[k, w]+1)), axis=0))
    count = int(hits.sum())
    require(not hits[:design['first_zero_based_control']].any() and enabled == bool(count),
            'An arm did not apply its declared correction')
    return score_forward(trace, sequence.forward_at, plan), dict(
        policy, continuation_controls=count, actual_right_support_controls=int(match.sum()),
        actual_right_support_precision=float(match.sum()/count) if count else None,
        actual_right_support_age_4_to_8_controls=int((match & (ages >= low) & (ages <= high)).sum()),
        affected_worlds=int(hits.any(axis=0).sum()), duplicate_starts_on_same_actual_support=len(k)-unique)


def score(plan, folder, bootstrap_seed):
    receipt = read(folder/'guarded-evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    design = plan['forward_pair']
    shared = design['shared_prefix_controls']
    horizon = shared+design['branch_controls']
    maximum = plan['maximum_controls']
    require(receipt['actual_control_steps'] == runtime['control_steps'] == maximum
            and receipt['recorded_control_transitions'] == runtime['control_transitions'] == maximum*plan['worlds']
            and receipt['actual_physics_substeps'] == runtime['physics_substeps'] == maximum*5
            and receipt['physical_resets'] == runtime['interface_checks'] == 1
            and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps']
                     or receipt['learning_updates'] or receipt['heldout_used'] or receipt['hardware_io']),
            'The actual run exceeded the physical or reset budget')
    init = read(folder/'zero-readback.json')
    fork = read(folder/'fork.json')
    state_fields = {f'{prefix}.{name}' for prefix, names in (
        ('onset', SCHEDULE_FIELDS), ('continuation', CONTINUATION_FIELDS), ('phase', PHASE_FIELDS)) for name in names}
    require(init['initial']['counters'] == [0, 0] and fork['physical']['counters'] == [shared, 5*shared]
            and state_fields <= set(init['fields']) & set(fork['fields'])
            and {'env.obs_buf', 'controller.history', 'controller.previous_features'} <= set(fork['fields'])
            and any('.delay.' in name for name in fork['fields']), 'The physical snapshot is incomplete')
    prefix, prefix_commands = checked_arrays(folder/'shared', receipt['prefix_sha256'])
    require(len(prefix['active']) == shared+1 and len(prefix_commands['actual_motor_target']) == shared,
            'The saved shared segment is incomplete')
    startup = arrays(folder/'startup.npz')
    arms, raw, first_motor = {}, {}, None
    for arm in design['arm_order']:
        branch = folder/'clips'/arm
        item = receipt['branches'][arm]
        require(read(branch/'restored-fork.json') == fork['physical'], 'Arms are not from the same live state')
        suffix, suffix_commands = checked_arrays(branch, item['trace_sha256'])
        require(len(suffix['active']) == design['branch_controls']+1
                and len(suffix_commands['actual_motor_target']) == item['controls'] == design['branch_controls'],
                'The saved arm is incomplete')
        for key in prefix:
            np.testing.assert_array_equal(prefix[key][-1], suffix[key][0], err_msg=key)
        first = {key: suffix_commands[key][0] for key in ('observed', 'actor_input', 'raw_policy_actions', 'actual_motor_target')}
        if first_motor is None:
            first_motor = first
        for key in first:
            np.testing.assert_array_equal(first[key], first_motor[key], err_msg=key)
        trace, commands = joined(prefix, suffix, 1), joined(prefix_commands, suffix_commands, 0)
        endpoint = read(branch/'endpoint-contact.json')
        contact_path = branch/'endpoint-contact.npz'
        require(endpoint['complete'] and endpoint['zero_matches_standalone'] and not endpoint['first_mismatches']
                and endpoint['cache_counters_and_rng_unchanged_at_each_boundary']
                and sha(contact_path) == item['contact_sha256'] == endpoint['sha256'],
                'Same-time contact readback changed')
        contact = arrays(contact_path)
        np.testing.assert_array_equal(contact['control'], np.arange(horizon+1))
        np.testing.assert_array_equal(contact['substep'], np.arange(horizon+1)*5)
        np.testing.assert_array_equal(contact['fresh'], contact['constraint_points'] > 0)
        np.testing.assert_array_equal(contact['fresh'][:, :, 2:], trace['foot_contact'])
        rows, controls = verify_route(trace, commands, plan, startup, arm == 'right-support-continuation')
        raw[arm] = rows
        metrics, _ = summary(rows)
        score_path = branch/'raw-score.json'
        put(score_path, dict(forward=rows))
        arms[arm] = dict(metrics, controls=controls, raw=rel(score_path), raw_sha256=sha(score_path),
                         video=video(folder, item['video'], 0, horizon, horizon),
                         action_gate_passed=metrics['passed'] >= plan['minimum_continuous_successes'])
    control, candidate = (raw[arm] for arm in design['arm_order'])
    change = np.array([new['yaw_error_rad_s']-old['yaw_error_rad_s'] for old, new in zip(control, candidate, strict=True)])
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(change), (4000, len(change)))
    bounds = np.quantile(np.median(change[draws], axis=1), [.025, .975]).tolist()
    improvement = sum(new['success'] and not old['success'] for old, new in zip(control, candidate, strict=True))
    regression = sum(old['success'] and not new['success'] for old, new in zip(control, candidate, strict=True))
    return dict(seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'], arms=arms,
                paired_worlds=len(change), paired_yaw_median_change_rad_s=float(np.median(change)),
                paired_yaw_median_bootstrap_95_rad_s=bounds, worlds_gaining_pass=improvement, worlds_losing_pass=regression,
                physical_transitions=runtime['control_transitions'], physical_substeps=runtime['physics_substeps'],
                physical_resets=1, exact_fork_readback=True, identical_first_motor=True, run_id=plan['run_id'])
