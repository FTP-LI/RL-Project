"""Independently score the exact same-state stride branches and repeat floor."""

from collections import Counter
from unittest.mock import patch

import numpy as np

import verify_continuous_roll_stand_walk as policy_reader
import verify_forward_right_support_continuation as previous
from audit_forward_stride_response import stride_mean, stride_windows
from collect_forward_stride_response import StrideControl, mapped_command
from run_roll_stand_forward_diagnostic import put, read, rel, require, sha
from verify_continuous_forward_yaw_pair import joined
from verify_roll_entry_isolation import video


def checked_contact(branch, summary, trace, horizon):
    stamp = read(branch/'endpoint-contact.json')
    path = branch/'endpoint-contact.npz'
    require(stamp['complete'] and stamp['zero_matches_standalone'] and not stamp['first_mismatches']
            and stamp['cache_counters_and_rng_unchanged_at_each_boundary']
            and sha(path) == stamp['sha256'] == summary['contact_sha256'], 'Physical contact evidence differs')
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['control'], np.arange(horizon+1))
        np.testing.assert_array_equal(data['substep'], np.arange(horizon+1)*5)
        np.testing.assert_array_equal(data['fresh'], data['constraint_points'] > 0)
        np.testing.assert_array_equal(data['fresh'][:, :, 2:], trace['foot_contact'])


def score(plan, folder):
    receipt = read(folder/'guarded-evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    design = plan['stride_response']
    fork, length = design['shared_controls'], design['branch_controls']
    start, horizon, maximum = design['standing_controls'], fork+length, plan['maximum_controls']
    require(receipt['actual_control_steps'] == runtime['control_steps'] == maximum
            and receipt['recorded_control_transitions'] == runtime['control_transitions'] == maximum*plan['worlds']
            and receipt['actual_physics_substeps'] == runtime['physics_substeps'] == maximum*5
            and receipt['physical_resets'] == runtime['interface_checks'] == 1
            and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps']
                     or receipt['learning_updates'] or receipt['heldout_used'] or receipt['hardware_io']),
            'The actual dynamics did not match the frozen budget')
    snap = read(folder/'fork.json')
    initial = read(folder/'zero-readback.json')
    fields = {f'{prefix}.{field}' for prefix, names in (
        ('onset', previous.SCHEDULE_FIELDS), ('continuation', previous.CONTINUATION_FIELDS),
        ('phase', previous.PHASE_FIELDS)) for field in names}
    require(initial['initial']['counters'] == [0, 0] and snap['physical']['counters'] == [fork, 5*fork]
            and fields <= set(initial['fields']) & set(snap['fields'])
            and {'env.obs_buf', 'controller.history', 'controller.previous_features'} <= set(snap['fields'])
            and any('.delay.' in field for field in snap['fields']), 'Incomplete CUDA state binding')
    prefix, command = previous.checked_arrays(folder/'shared', receipt['prefix_sha256'])
    require(len(prefix['active']) == fork+1 and len(command['actual_motor_target']) == fork, 'Incomplete shared route')
    startup = policy_reader.arrays(folder/'startup.npz')
    role_data, means, stride_data, first = {}, {}, {}, None
    checked_plan = dict(plan, forward_controls=horizon-start)
    after_ramp = fork+design['input_ramp_controls']
    for role in design['role_order']:
        branch = folder/'clips'/role
        item = receipt['branches'][role]
        require(read(branch/'restored-fork.json') == snap['physical'] and item['controls'] == length,
                'Branch is not from the exact common CUDA state')
        states, commands = previous.checked_arrays(branch, item['trace_sha256'])
        require(len(states['active']) == length+1 and len(commands['actual_motor_target']) == length, 'Missing branch controls')
        for key in prefix:
            np.testing.assert_array_equal(prefix[key][-1], states[key][0], err_msg=key)
        first_step = {key: commands[key][0] for key in ('observed','actor_input','raw_policy_actions','actual_motor_target')}
        if first is None:
            first = first_step
        for key, value in first_step.items():
            if key != 'actual_motor_target' or role != 'right-hip-reverse':
                np.testing.assert_array_equal(value, first[key], err_msg=key)
        full_trace, full_commands = joined(prefix, states, 1), joined(command, commands, 0)
        checked_contact(branch, item, full_trace, horizon)
        # Rebuild the actor input across the transition, including the raw
        # history. The evaluator still reads the original external command.
        with patch.object(previous, 'RightSupportContinuation', StrideControl), +             patch.object(policy_reader, 'transform', lambda values: mapped_command(
                 values, design, role, np.arange(len(values), dtype=np.float32)[:, None])):
            rows, controller = previous.verify_route(full_trace, full_commands, checked_plan, startup, role == 'right-hip-reverse')
        yaw = full_trace['angular_velocity'][..., 2].astype(np.float64)
        means[role] = np.abs(yaw[fork+1:]).mean(axis=0)
        after = horizon-after_ramp
        window = stride_windows(full_trace['foot_contact'], after_ramp, after, design['complete_strides_per_anchor'])
        stride_data[role] = stride_mean(yaw[after_ramp:], window)
        tail_yaw = np.abs(yaw[after_ramp+1:]).mean(axis=0)
        projected = (np.abs(yaw[start+1:fork+1]).sum(axis=0)+(1050-fork)*tail_yaw)/1000
        speed = full_trace['velocity'][after_ramp+1:, :, 0].mean(axis=0)
        unchanged_gates = np.array([not (set(row['failed_checks'])-{'yaw_error'}) for row in rows])
        projected_ok = ((projected <= design['original_yaw_limit_rad_s']) & unchanged_gates
                        & (speed >= design['original_minimum_forward_speed_m_s']))
        score_path = branch/'raw-score.json'
        put(score_path, dict(observed_forward=rows, projection_rad_s=projected.tolist(), steady_speed_m_s=speed.tolist(),
                             three_second_mean_abs_yaw_rad_s=means[role].tolist(), stride_windows_from_ramp=window.tolist()))
        role_data[role] = dict(raw=rel(score_path), raw_sha256=sha(score_path),
            three_second_median_abs_yaw_rad_s=float(np.median(means[role])),
            median_speed_m_s=float(np.median(speed)), projected_joint_gate_worlds=int(projected_ok.sum()),
            observed_non_yaw_failures=dict(Counter(check for row in rows for check in row['failed_checks'] if check != 'yaw_error')),
            both_cycle_worlds=int(np.isfinite(stride_data[role]).all(axis=1).sum()), controller=controller,
            video=video(folder, item['video'], 0, horizon, horizon))

    lo, hi = ('reference-a', 'reference-b')
    tolerance = float(np.quantile(np.abs(means[hi]-means[lo]), .95))
    reference_mean = (means[lo]+means[hi])/2
    reference_strides = (stride_data[lo]+stride_data[hi])/2
    for role, item in role_data.items():
        if role in (lo, hi):
            continue
        common = np.isfinite(reference_strides).all(axis=1) & np.isfinite(stride_data[role]).all(axis=1)
        gain = float(np.median(reference_mean-means[role]))
        cycles = np.median((reference_strides-stride_data[role])[common], axis=0).tolist() if common.any() else []
        item.update(reduction_rad_s=gain, cycle_reduction_by_start_side_rad_s=cycles, common_cycle_worlds=int(common.sum()),
                    passes_screen=bool(gain > tolerance and common.sum() >= design['original_minimum_passing_worlds']
                    and all(value > 0 for value in cycles)
                    and item['projected_joint_gate_worlds'] >= design['original_minimum_passing_worlds']))
    repeat_common = np.isfinite(stride_data[hi]).all(axis=1) & np.isfinite(stride_data[lo]).all(axis=1)
    require(repeat_common.sum() >= design['original_minimum_passing_worlds'], 'The reference pair lacks complete cycle coverage')
    cycle_noise = np.quantile(np.abs(stride_data[hi]-stride_data[lo])[repeat_common], .95, axis=0)
    return dict(run_id=plan['run_id'], seed=plan['seed'], physical_transitions=runtime['control_transitions'],
                physical_substeps=runtime['physics_substeps'], exact_snapshot=True, physical_resets=1,
                repeat_p95_rad_s=tolerance, repeat_cycle_p95_rad_s=cycle_noise.tolist(), arms=role_data)
