"""Verify matched physical targets and score the 40 ms rebound contrast."""
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from common_walk_yaw_history_probe import OUT, PLAN, ROLES
from common_walk_forward_phase_screen import check_sources, groups_and_commands
from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import verify_common_walk_forward_phase as common
import verify_common_walk_yaw_pulse as pulse


def score(arms, spec, forward):
    signs = np.asarray(spec['phase']['pulse_signs'])[forward]
    assert (signs == 1).sum() == (signs == -1).sum() == 16
    assert all(t['active'][:2, forward].all() for t in arms.values())
    a, b, c = (arms[role]['yaw_rate'][:2, forward].astype(np.float64) for role in ROLES)
    command, actuator = (b-a)*signs, (c-a)*signs
    paired = actuator-command
    gap = float(np.abs(paired[0]).mean())
    old, new = float(command[1].mean()), float(actuator[1].mean())
    improvement = float(paired[1].mean())
    fraction = improvement/abs(old) if old < 0 else None
    gate = spec['history_probe']['admission']
    rng = np.random.default_rng(spec['gate']['bootstrap_seed']+1)
    picks = np.concatenate([rng.choice(np.flatnonzero(signs == s), size=(spec['gate']['bootstrap_samples'], 16))
                            for s in (-1, 1)], axis=1)
    interval = np.quantile(paired[1, picks].mean(axis=1), [.025, .975]).tolist()
    passes = (gap <= gate['maximum_paired_immediate_yaw_difference_rad_s'] and
              old <= -spec['gate']['minimum_effect_rad_s'] and fraction is not None and
              fraction >= gate['minimum_40ms_rebound_reduction_fraction'] and interval[0] > 0)
    return dict(command_20ms_rad_s=float(command[0].mean()), actuator_20ms_rad_s=float(actuator[0].mean()),
                mean_absolute_immediate_gap_rad_s=gap, command_40ms_rad_s=old, actuator_40ms_rad_s=new,
                paired_40ms_difference_rad_s=improvement, paired_40ms_ci95_rad_s=interval,
                rebound_reduction_fraction=fraction, passes_history_isolation_gate=bool(passes))


def main():
    spec, state = read(PLAN), read(OUT/'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['protocol_sha256'] == sha(PLAN)
    assert not torch.cuda.is_initialized()
    check_sources(spec)
    record = ROOT/state['record']
    assert sha(record/'protocol.json') == sha(PLAN)
    n = spec['environment']['parallel_worlds']
    p, k = spec['environment'], spec['phase']['warmup_controls']
    controls = spec['maximum_control_transitions']//n
    assert state['actual'] == dict(attempted_controls=controls, completed_controls=controls,
                                  attempted_substeps=5*controls, completed_substeps=5*controls, initial_resets=1)
    assert state['actor_unchanged'] and state['contact_peak'] < state['contact_slots']
    startup, zero = (read(record/name) for name in ('startup.json', 'zero-snapshot.json'))
    assert startup['status'] == zero['status'] == 'passed' and startup['physical_steps'] == zero['physical_substeps'] == 0
    assert startup['cpu_cuda_rng_unchanged'] and startup['complete_processor_memory_restored']
    forks = read(record/'fork-readback.json')
    reference = forks[ROLES[0]]['physical']
    assert reference['counters'] == [k, k*5]
    assert reference['state_sha256'] == digest(read(record/'fork-state-fields.json'))
    assert all(row == forks[ROLES[0]] for row in forks.values()) and set(forks) == set(ROLES)
    assert forks[ROLES[0]]['controller_sha256'] == digest(read(record/'fork-controller-state.json'))
    initial_path = record/'physical-initial.npz'
    assert sha(initial_path) == state['initial_state_sha256']
    initial = common.arrays(initial_path)
    prefix_path = record/'trace-shared-prefix.npz'
    prefix, prefix_resets = common.arrays(prefix_path), common.arrays(record/'resets-shared-prefix.npz')
    groups, desired = groups_and_commands(p)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    arms, first, replays, videos, hashes = {}, {}, {}, {}, {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        hashes[role] = sha(path)
        post = common.arrays(path)
        first[role] = {key: post[key][0] for key in ('observations', 'actions', 'requested_joint_targets',
                                                    'applied_joint_targets', 'active')}
        arms[role] = {key: post[key] for key in ('active', 'yaw_rate')}
        trace = {key: np.concatenate((prefix[key], post[key])) for key in prefix}
        np.testing.assert_allclose(trace['qpos'][..., state['qpos_joint_addresses']], trace['joint_positions'], atol=1e-6, rtol=0)
        replays[role] = {'verified_unscored_resets': common.verify_resets(prefix_resets, trace)+common.verify_resets(
            common.arrays(record/f'resets-{role}.npz'), trace)}
        info = next(row['video'] for row in state['conditions'] if row['role'] == role)
        videos[role] = common.verify_movie(record, info, trace, initial)
        if role == 'actuator-pulse':
            # The recorded intervention target is checked against the
            # command branch below. Replay the ordinary projection elsewhere.
            trace['applied_joint_targets'][k] = first['reference']['applied_joint_targets']
        with patch.object(pulse, 'ROLES', ('reference', 'actuator-pulse', 'command-pulse')):
            replays[role].update(pulse.verify_inputs(trace, spec, desired, role, session))

    normal, command, actuator = (first[role] for role in ROLES)
    for key in ('observations', 'actions', 'requested_joint_targets', 'active'):
        np.testing.assert_array_equal(actuator[key], normal[key])
    np.testing.assert_array_equal(command['applied_joint_targets'], actuator['applied_joint_targets'])
    np.testing.assert_array_equal(command['observations'][:, :50], normal['observations'][:, :50])
    np.testing.assert_array_equal(command['observations'][:, 51:], normal['observations'][:, 51:])
    change = actuator['applied_joint_targets']-normal['applied_joint_targets']
    mask = np.asarray(spec['phase']['pulse_signs']) != 0
    assert not change[~mask].any() and change.any()
    maximum = float(abs(change).max())
    assert maximum <= spec['history_probe']['pulse']['maximum_applied_target_change_rad']
    readback = common.arrays(record/'actuator-pulse-readback.npz')
    for saved, actual in (('observations', command['observations']), ('alternate_raw', command['actions']),
                          ('enabled', mask), ('raw', normal['actions']), ('applied', actuator['applied_joint_targets'])):
        np.testing.assert_array_equal(readback[saved], actual)
    motors = common.arrays(record/'first-motor-commands.npz')
    assert set(motors) == set(ROLES)
    assert all(np.array_equal(row, motors['reference']) for row in motors.values())
    result = score(arms, spec, groups['forward'])
    write(OUT/'analysis.json', dict(status='passed-integrity', protocol_sha256=sha(PLAN), record=state['record'],
          maximum_target_change_rad=maximum, matched_applied_targets=True, first_delayed_motor_exact=True,
          actuator_raw_and_requested_unchanged=True, fork=reference, response=result, inputs=replays,
          videos=videos, trace_sha256=hashes, readback_sha256=sha(record/'actuator-pulse-readback.npz'),
          actual_control_transitions=controls*n, actual_physics_substeps=controls*5,
          learning_updates=0, heldout_used=False, motion_acceptance=False))
    print('WALK_YAW_HISTORY_VERIFIED', result, flush=True)


if __name__ == '__main__':
    main()
