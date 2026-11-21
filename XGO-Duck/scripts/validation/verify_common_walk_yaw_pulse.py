"""Audit the one-control intervention and its paired yaw impulse response."""
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from common_walk_yaw_pulse import OUT, PLAN, ROLES, checked, groups_and_commands
from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import verify_common_walk_forward_phase as common
from walking_command_controller import create_controller


def verify_inputs(trace, spec, desired, role, session):
    p, settings = spec['environment'], spec['phase']
    active, observed = trace['active'], trace['observations']
    h, n = active.shape
    k = settings['warmup_controls']
    assert (h, n) == (p['maximum_control_steps'], p['parallel_worlds'])
    assert active[0].all() and not (np.diff(active.astype(int), axis=0) > 0).any()
    assert all(np.isfinite(value[active]).all() for value in trace.values())
    np.testing.assert_allclose(observed[..., 51:], 0., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(observed[0, :, 34:48], 0.)
    np.testing.assert_array_equal(observed[1:, :, 34:48][active[1:]], trace['actions'][:-1][active[1:]])
    np.testing.assert_array_equal(trace['actions'][~active], 0.)
    expected = np.asarray(p['model_home'], dtype=np.float32)+trace['actions']
    lo, hi = np.asarray(p['joint_limits'], dtype=np.float32).T
    np.testing.assert_allclose(trace['requested_joint_targets'], expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(trace['applied_joint_targets'], np.clip(expected, lo, hi), atol=1e-6, rtol=0)
    controller = create_controller(n, .02)
    pulse = np.asarray(settings['pulse_signs'], dtype=np.float32)*np.float32(settings['peak_rad_s'])
    samples, changed, maximum = 0, 0, 0.
    for i, row in enumerate(observed):
        source = row.copy()
        source[:, 48:51] = desired
        command = controller.map(desired, source)
        if i == k and role == ROLES[-1]:
            command[active[i], 2] += pulse[active[i]]
            changed = int(np.count_nonzero(pulse[active[i]]))
        np.testing.assert_allclose(row[:, 48:51], command, atol=1e-7, rtol=0)
        if i in (0, k-1, k, k+1, k+2, k+5, h-1):
            for j in np.flatnonzero(active[i]):
                action = session.run(None, {'obs': row[j:j+1]})[0][0]
                actual = trace['actions'][i, j]
                np.testing.assert_allclose(actual, action, atol=2e-5, rtol=2e-5)
                maximum = max(maximum, float(abs(action-actual).max()))
                samples += 1
        controller.reset(np.flatnonzero(trace['terminated'][i] | trace['truncated'][i]))
    assert changed == (32 if role == ROLES[-1] else 0)
    return dict(replayed_inputs=samples, maximum_onnx_error=maximum, pulse_worlds=changed,
                survived_window=int(active[-1].sum()), joint_failed_samples=int((active & ~trace['joint_envelope_ok']).sum()))


def response(arms, spec, forward):
    gate = spec['gate']
    signs = np.asarray(spec['phase']['pulse_signs'])[forward]
    assert (signs == -1).sum() == (signs == 1).sum() == 16
    for trace in arms.values():
        assert trace['active'][:6, forward].all()
    a, b, pulse = (arms[role]['yaw_rate'][:6, forward].astype(np.float64) for role in ROLES)
    rng = np.random.default_rng(gate['bootstrap_seed'])
    picks = np.concatenate([rng.choice(np.flatnonzero(signs == s), size=(gate['bootstrap_samples'], 16))
                            for s in (-1, 1)], axis=1)
    rows = {}
    for i in range(6):
        paired = {}
        for role, ref in zip(ROLES[:2], (a, b), strict=True):
            delta = signs*(pulse[i]-ref[i])
            interval = np.quantile(delta[picks].mean(axis=1), [.025, .975])
            paired[role] = dict(mean_rad_s=float(delta.mean()), ci95_rad_s=interval.tolist(),
                                positive_worlds=int((delta > 0).sum()), signed_responses_rad_s=delta.tolist())
        placebo = signs*(b[i]-a[i])
        rows[str(20*(i+1))] = dict(paired=paired, placebo_signed_mean_rad_s=float(placebo.mean()),
                                   placebo_absolute_median_rad_s=float(np.median(abs(placebo))))

    def signal(row):
        return (abs(row['placebo_signed_mean_rad_s']) <= gate['maximum_placebo_rad_s']
                and all(v['mean_rad_s'] >= gate['minimum_effect_rad_s'] and v['ci95_rad_s'][0] > 0
                        and v['positive_worlds'] >= gate['minimum_positive_worlds'] for v in row['paired'].values()))

    immediate, delayed = rows['20'], rows['60']
    ratio = {role: (delayed['paired'][role]['mean_rad_s']/immediate['paired'][role]['mean_rad_s']
                    if immediate['paired'][role]['mean_rad_s'] > 0 else None) for role in ROLES[:2]}
    nominated = signal(immediate) and signal(delayed) and all(
        value is not None and value >= gate['minimum_60ms_fraction_of_20ms'] for value in ratio.values())
    return dict(windows=rows, immediate_effect_verified=signal(immediate), delayed_effect_verified=signal(delayed),
                delayed_to_immediate_ratio=ratio, can_test_60ms_feedback=nominated)


def main():
    spec, state = checked(), read(OUT/'coordinator.json')
    assert not torch.cuda.is_initialized() and state['status'] == 'collected-review-required'
    record = ROOT/state['record']
    assert sha(PLAN) == state['protocol_sha256'] == sha(record/'protocol.json')
    n, h = (spec['environment'][key] for key in ('parallel_worlds', 'maximum_control_steps'))
    k = spec['phase']['warmup_controls']
    controls = k+3*(h-k)
    assert state['actual'] == dict(attempted_controls=controls, completed_controls=controls,
                                  attempted_substeps=5*controls, completed_substeps=5*controls, initial_resets=1)
    assert state['actor_unchanged'] and state['contact_peak'] < state['contact_slots']
    initial_file = record/'physical-initial.npz'
    assert sha(initial_file) == state['initial_state_sha256']
    initial = common.arrays(initial_file)
    startup, zero = (read(record/name) for name in ('startup.json', 'zero-snapshot.json'))
    assert startup['status'] == zero['status'] == 'passed' and startup['physical_steps'] == zero['physical_substeps'] == 0
    assert startup['cpu_cuda_rng_unchanged'] and startup['complete_processor_memory_restored']
    forks = read(record/'fork-readback.json')
    reference = forks[ROLES[0]]['physical']
    inventory = read(record/'fork-state-fields.json')
    assert digest(inventory) == reference['state_sha256'] and len(inventory) == reference['fields']
    assert reference['counters'] == [k, 5*k]
    assert all(forks[role] == forks[ROLES[0]] for role in ROLES)
    assert forks[ROLES[0]]['controller_sha256'] == digest(read(record/'fork-controller-state.json'))
    assert reference['stateless_upright_config_sha256'] == zero['snapshot']['stateless_upright_config_sha256']
    p = spec['environment']
    groups, desired = groups_and_commands(p)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    prefix_path = record/'trace-shared-prefix.npz'
    prefix = common.arrays(prefix_path)
    assert prefix['active'].shape == (k, n)
    reset_prefix = common.arrays(record/'resets-shared-prefix.npz')
    arms, first, inputs, videos, hashes = {}, {}, {}, {}, {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        hashes[role] = sha(path)
        post = common.arrays(path)
        assert post['active'].shape == (h-k, n)
        arms[role] = {key: post[key] for key in ('yaw_rate', 'active')}
        first[role] = {key: post[key][0] for key in ('observations', 'actions', 'requested_joint_targets',
                                                    'applied_joint_targets', 'active')}
        trace = {key: np.concatenate((prefix[key], post[key])) for key in prefix}
        np.testing.assert_allclose(trace['qpos'][..., state['qpos_joint_addresses']], trace['joint_positions'], atol=1e-6, rtol=0)
        inputs[role] = verify_inputs(trace, spec, desired, role, session)
        inputs[role]['verified_unscored_resets'] = common.verify_resets(reset_prefix, trace) + common.verify_resets(
            common.arrays(record/f'resets-{role}.npz'), trace)
        info = next(item['video'] for item in state['conditions'] if item['role'] == role)
        videos[role] = common.verify_movie(record, info, trace, initial)
    with patch.object(common, 'ROLES', ROLES):
        start = common.verify_first(first, common.arrays(record/'first-motor-commands.npz'), desired[:, 0] > 0.)
    measured = response(arms, spec, groups['forward'])
    result = dict(status='passed-integrity', protocol_sha256=sha(PLAN), record=state['record'], fork=reference,
                  first_control=start, inputs=inputs, videos=videos, response=measured,
                  trace_sha256=hashes, shared_prefix_sha256=sha(prefix_path), actual_control_transitions=n*controls,
                  actual_physics_substeps=5*controls, learning_updates=0, heldout_used=False,
                  motion_acceptance=False, scope=spec['scope'])
    write(OUT/'analysis.json', result)
    print('WALK_YAW_PULSE_VERIFIED', measured['immediate_effect_verified'], measured['delayed_effect_verified'],
          measured['can_test_60ms_feedback'], flush=True)


if __name__ == '__main__':
    main()
