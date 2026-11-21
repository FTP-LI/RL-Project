"""Recompute the common mask, motor intervention and short yaw response from raw traces."""

import numpy as np

from forward_support_joint_pulse import JOINTS, ROLES, SIDES, common_mask, observer_for
from run_roll_stand_forward_diagnostic import require, sha
from verify_forward_hip_yaw_pulse import load, response, verify_motor


def verify_side(folder, plan, side, shared, shared_commands, fork):
    name = SIDES[side]
    design = plan['pulse']
    first, k = design['earliest_fork_control'], fork['control']
    controls, worlds = design['response_controls'], plan['worlds']
    require(first <= k < design['shared_controls'] and fork['captured']['counters'] == [k, 5*k]
            and set(fork['restored']) == set(fork['branches']) == set(ROLES)
            and all(value == fork['captured'] for value in fork['restored'].values()),
            'The recorded starts are not the same pre-action state')

    window = slice(first, k+1)
    observer = observer_for(plan)
    mask = common_mask(plan, shared_commands['requested_targets'][window], shared_commands['applied_targets'][window],
                       shared['foot_contact'][window], observer.predict(shared_commands['observed'][window]), side)
    selected = np.flatnonzero(mask[-1])
    np.testing.assert_array_equal(selected, fork['selected_worlds'])
    require(len(selected) >= design['minimum_worlds_per_fork'] and
            (mask[:-1].sum(axis=1) < design['minimum_worlds_per_fork']).all(), 'The first common fork was not matched')

    initial, motor, yaw = {}, {}, {}
    for role, intervention in ROLES.items():
        path = folder/name/f'{role}.npz'
        require(sha(path) == fork['branches'][role], 'A raw branch changed')
        data = load(path)
        states = {key[6:]: value for key, value in data.items() if key.startswith('state_')}
        command = {key[8:]: value for key, value in data.items() if key.startswith('command_')}
        for key, samples in states.items():
            np.testing.assert_array_equal(samples[0], shared[key][k], err_msg=key)
            require(samples.shape[0] == controls+1 and np.isfinite(samples).all(), 'Invalid recorded physical trajectory')
        for key in ('observed', 'actor_input', 'requested_targets', 'raw_policy_actions', 'applied_targets'):
            np.testing.assert_array_equal(command[key][0], shared_commands[key][k], err_msg=key)
        require(states['active'].shape == (controls+1, worlds) and states['active'].all()
                and states['joint_envelope_ok'].all()
                and (states['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                'A branch exceeded the original physical guards')

        correction = np.zeros_like(command['applied_targets'])
        if intervention is not None:
            joint, sign = intervention
            index = plan['walking']['joint_names'].index(f'{name}_{joint}')
            correction[0, selected, index] = sign*design['delta_rad']
        verify_motor(plan, command, shared_commands['raw_policy_actions'][k-1], command['applied_targets']+correction)
        steps = data['motor_substeps']
        require(steps.shape[:2] == (5*controls, worlds) and np.isfinite(steps).all(), 'Motor delay trace is incomplete')
        initial[role], motor[role] = data['initial_motor'], steps[:5, selected]
        yaw[role] = states['angular_velocity'][1:, selected, 2].astype(np.float64)

    for ctrl in initial.values():
        np.testing.assert_array_equal(ctrl, initial['zero-a'])
    motor_deltas = {role: np.abs(value-motor['zero-a']).max(axis=(1, 2)).astype(float).tolist()
                    for role, value in motor.items() if role != 'zero-a'}
    scores = {}
    for joint in JOINTS:
        label = joint.replace('_', '-')
        trace = {'zero-a': yaw['zero-a'], 'zero-b': yaw['zero-b'],
                 'positive': yaw[f'{label}-positive'], 'negative': yaw[f'{label}-negative']}
        scores[joint] = dict(primary=response(trace, design['primary_response_controls']), complete=response(trace, controls))
    return dict(fork_control=k, worlds=len(selected), all_six_starts_exact=True, identical_sensor_mask_verified=True,
                joints=scores, first_control_motor_max_abs_difference_by_substep=motor_deltas)
