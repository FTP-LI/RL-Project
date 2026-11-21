"""Verify the matched start and score the actual one-step and sustained responses."""

from unittest.mock import patch

import numpy as np

import verify_forward_hip_yaw_pulse as baseline
from audit_forward_yaw_by_support_age import contact_age
from forward_phase_hip_pitch import HipPitchSchedule
from right_hip_pitch_duration_control import PHASE_FIELDS, SCHEDULE_FIELDS, schedule_state, targets
from run_roll_stand_forward_diagnostic import read, require, sha


score_shared_route, verify_original_motor = baseline.score, baseline.verify_motor
EXTRA = ('one-extra-control', 'continue-through-eight')


def replay(plan, commands, schedule, start, before_raw, role, matched=None, snapshot_at=None):
    design = plan['duration_probe']
    retained = None
    target_trace = []
    extending = matched if role in EXTRA else None
    for offset, observed in enumerate(commands['observed']):
        k = start+offset
        if k == snapshot_at:
            retained = schedule_state(schedule)
        row = {field: commands[field][offset] for field in ('requested_targets', 'applied_targets', 'phase_used')}
        expected, detail = targets(schedule, design, k, row, observed, extending)
        for key, value in detail.items():
            np.testing.assert_array_equal(value, commands[key][offset], err_msg=key)
        actual = commands['actual_motor_target'][offset]
        delta = np.zeros_like(expected)
        for side, (joint, sign) in enumerate(zip(schedule.pitch, schedule.sign, strict=True), 1):
            delta[detail['hip_pitch_pulse_side'] == side, joint] = sign*schedule.delta
        delta[detail['right_extension_applied'], schedule.pitch[1]] = design['single_control_delta_rad']
        np.testing.assert_allclose(actual-row['applied_targets'], delta, atol=1e-6, rtol=0)
        extending = detail['right_extension_applied'] if role == 'continue-through-eight' else None
        target_trace.append(expected)
    verify_original_motor(plan, commands, before_raw, np.stack(target_trace))
    return retained


def response(trace, count):
    xa, xb = (np.abs(trace[role][:count]) for role in ('zero-a', 'zero-b'))
    zero = (xa+xb)*.5
    effects = {role: (np.abs(trace[role][:count])-zero).mean(axis=0) for role in EXTRA}
    return dict(window_controls=count, zero_pair_gap_p95_rad_s=float(np.quantile(np.abs((xa-xb).mean(axis=0)), .95)),
                median_change_in_mean_abs_yaw_rad_s={role: float(np.median(data)) for role, data in effects.items()},
                sustained_minus_single_paired_median_rad_s=float(np.median(effects[EXTRA[1]]-effects[EXTRA[0]])),
                reduced_scenes={role: int((data < 0).sum()) for role, data in effects.items()})


def verify_right(folder, plan, shared, common, fork, matched_schedule):
    design = plan['duration_probe']
    roles = design['roles']
    k, width = fork['control'], plan['pulse']['response_controls']
    require(design['first_zero_based_control'] <= k < plan['pulse']['shared_controls']
            and set(fork['restored']) == set(fork['branches']) == set(roles)
            and all(value == fork['captured'] for value in fork['restored'].values())
            and fork['captured']['counters'] == [k, k*5], 'Branch snapshot readback differs')
    contact = shared['foot_contact'][:-1]
    label, age = contact_age(contact)
    tick = np.arange(len(contact))[:, None]
    prior = np.maximum.accumulate(np.where(common['hip_pitch_pulse_side'] == 2, tick, -1), axis=0)
    low, high = design['recognized_right_support_age_controls']
    match = common['right_extension_eligible'] & (label == 2) & (age >= low) & (age <= high)
    match &= (tick >= design['first_zero_based_control']) & (prior >= tick-age+1) & (prior < tick)
    selected = np.flatnonzero(match[k])
    np.testing.assert_array_equal(selected, fork['selected_worlds'])
    minimum = design['minimum_common_worlds']
    require(len(selected) >= minimum and (match[:k].sum(axis=1) < minimum).all(), 'Not the first common right support boundary')
    series, duration, motor_initial, motor_first = {}, {}, {}, {}
    for role in roles:
        path = folder/'right'/f'{role}.npz'
        require(sha(path) == fork['branches'][role], 'The branch trace changed')
        bundle = baseline.load(path)
        physical = {key[6:]: value for key, value in bundle.items() if key.startswith('state_')}
        commands = {key[8:]: value for key, value in bundle.items() if key.startswith('command_')}
        for key, values in physical.items():
            np.testing.assert_array_equal(values[0], shared[key][k], err_msg=key)
            require(values.shape[0] == width+1 and np.isfinite(values).all(), 'Invalid physical trajectory')
        inputs = ('observed', 'actor_input', 'requested_targets', 'raw_policy_actions', 'applied_targets', 'observer_label',
                  'observer_age', 'onset_latch', 'right_extension_eligible', 'hip_pitch_pulse_side')
        for field in inputs:
            np.testing.assert_array_equal(commands[field][0], common[field][k], err_msg=field)
        require(physical['active'].all() and physical['joint_envelope_ok'].all()
                and (physical['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(), 'A branch exceeded physical guards')
        schedule = HipPitchSchedule(plan, plan['phase_hip_pitch'])
        for field, value in matched_schedule.items():
            initial = bundle[f'initial_schedule_{field}']
            np.testing.assert_array_equal(initial, value, err_msg=field)
            setattr(schedule, field, initial.copy())
        replay(plan, commands, schedule, k, common['raw_policy_actions'][k-1], role, match[k])
        applied = commands['right_extension_applied']
        if role in EXTRA:
            np.testing.assert_array_equal(applied[0], match[k])
            steps = applied[:, selected].sum(axis=0)
            side = physical['foot_contact'][:-1]
            duration[role] = dict(applied_controls=int(steps.sum()), minimum=int(steps.min()), maximum=int(steps.max()),
                median=float(np.median(steps)), on_same_actual_support=int((applied & side[..., 1] & ~side[..., 0]).sum()))
        motor = bundle['motor_substeps']
        require(motor.shape[:2] == (5*width, plan['worlds']) and np.isfinite(motor).all(), 'A motor substep sequence is incomplete')
        motor_initial[role], motor_first[role] = bundle['initial_motor'], motor[:5, selected]
        series[role] = physical['angular_velocity'][1:, selected, 2].astype(np.float64)
    for value in motor_initial.values():
        np.testing.assert_array_equal(value, motor_initial['zero-a'])
    motor_delta = {role: np.abs(values-motor_first['zero-a']).max(axis=(1, 2)).tolist()
                   for role, values in motor_first.items() if role != 'zero-a'}
    return dict(fork_control=k, worlds=len(selected), all_starts_exact=True, observer_latch_exact=True,
                first_motor_difference_by_substep=motor_delta, duration=duration,
                primary=response(series, plan['pulse']['primary_response_controls']), complete=response(series, width))


def score(plan, folder):
    fields = read(folder/'pulse-zero.json')
    require(set(fields['schedule_fields']) == {f'pitch_schedule.{name}' for name in SCHEDULE_FIELDS}
            and set(fields['phase_fields']) == {f'stand_forward.{name}' for name in PHASE_FIELDS}, 'The controller snapshot inventory is incomplete')
    k = read(folder/'pulse-forks.json')['right']['control']
    retained = None

    def check_prefix(spec, commands, before_raw):
        nonlocal retained
        retained = replay(spec, commands, HipPitchSchedule(spec, spec['phase_hip_pitch']), 0, before_raw, 'zero-a', snapshot_at=k)

    def check_right(directory, spec, index, shared, common, fork):
        require(index == 0 and retained is not None, 'Failed to replay the shared controller to the branch')
        return verify_right(directory, spec, shared, common, fork, retained)

    with patch.multiple(baseline, SIDES=('right',), verify_motor=check_prefix, verify_side=check_right):
        return score_shared_route(plan, folder)
