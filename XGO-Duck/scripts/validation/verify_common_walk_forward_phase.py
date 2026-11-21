"""Verify the shared prefix, candidate interface, original scores and footage."""
from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from common_walk_forward_phase_metrics import command_score, compare_forward
from common_walk_forward_phase_screen import OUT, PLAN, ROLES, check_sources, groups_and_commands, publish
from getup_release_fork_state import digest
from record_io import write
from walking_forward_phase import ForwardPhase


def arrays(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def verify_resets(events, trace):
    assert events['other_world_states_exact'].all()
    for i, (step, world) in enumerate(zip(events['control_step'], events['world'], strict=True)):
        assert trace['terminated'][step-1, world] or trace['truncated'][step-1, world]
        assert trace['active'][step-1, world] == events['first_episode_just_ended'][i]
        assert not trace['active'][step:, world].any()
        np.testing.assert_array_equal(trace['qpos'][step-1, world], events['before_qpos'][i])
    assert np.isfinite(events['after_qpos']).all() and np.isfinite(events['after_qvel']).all()
    return len(events['world'])


def verify_movie(record, info, trace, initial):
    file = record / info['path']
    assert sha(file) == info['sha256'] and info['environment_index'] == 0 and info['extra_physics_steps'] == 0
    reader = imageio_ffmpeg.read_frames(str(file), pix_fmt='rgb24')
    meta = next(reader)
    frames = sum(1 for _ in reader)
    assert meta['fps'] == info['fps'] == 25 and meta['size'] == (640, 480)
    times = info['frame_times_seconds']
    steps = np.rint(np.asarray(times) / .02).astype(int)
    assert frames == len(steps) and steps[0] == 0 and (np.diff(steps) > 0).all()
    stop = min(200, int(trace['active'][:, 0].sum()))
    expected = list(range(0, stop+1, 2))
    if expected[-1] != stop:
        expected.append(stop)
    assert steps.tolist() == expected and info['complete_clip'] == (stop == 200)
    np.testing.assert_allclose(times, steps * .02, atol=1e-12, rtol=0)
    hashes = [sha256((initial['qpos'][0] if k == 0 else trace['qpos'][k-1, 0]).tobytes()).hexdigest() for k in steps]
    assert hashes == info['qpos_sha256_by_frame']
    return dict(path=file.relative_to(ROOT).as_posix(), sha256=info['sha256'], frames=frames,
                environment=0, start_seconds=0., end_seconds=times[-1], complete_window=info['complete_clip'],
                physical_source_hashes_verified=True)


def verify_inputs(trace, p, spec, desired, role, session):
    h, n = trace['active'].shape
    active, observed = trace['active'], trace['observations']
    assert (h, n) == (1000, 224) and active[0].all()
    assert not (np.diff(active.astype(int), axis=0) > 0).any()
    assert all(np.isfinite(value[active]).all() for value in trace.values())
    np.testing.assert_allclose(np.linalg.norm(trace['root_quaternion'][active], axis=-1), 1., atol=2e-4, rtol=0)
    np.testing.assert_allclose(observed[..., 51:61], 0., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(observed[0, :, 34:48], 0.)
    np.testing.assert_array_equal(observed[1:, :, 34:48][active[1:]], trace['actions'][:-1][active[1:]])
    np.testing.assert_array_equal(trace['actions'][~active], 0.)
    home = np.asarray(p['model_home'], dtype=np.float32)
    limits = np.asarray(p['joint_limits'], dtype=np.float32)
    expected = home + trace['actions']
    np.testing.assert_allclose(trace['requested_joint_targets'], expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(trace['applied_joint_targets'], np.clip(expected, limits[:, 0], limits[:, 1]), atol=1e-6, rtol=0)
    controller = ForwardPhase(p, spec['phase'], desired)
    replayed, changed_count, max_error = 0, 0, 0.
    for i in range(h):
        o = observed[i].copy()
        o[:, 48:51] = desired
        mapped = controller.map(o, i+1, role == ROLES[-1], active[i])
        np.testing.assert_allclose(observed[i, :, 48:51], mapped, atol=1e-7, rtol=0)
        changed_count += int(np.count_nonzero(mapped[controller.forward, 2]))
        if i in (0, 49, 50, 99, 199, 499, 749, 999):
            for j in np.flatnonzero(active[i]):
                actual = trace['actions'][i, j]
                answer = session.run(None, {'obs': observed[i, j:j+1]})[0][0]
                np.testing.assert_allclose(actual, answer, atol=2e-5, rtol=2e-5)
                max_error = max(max_error, float(abs(answer-actual).max()))
                replayed += 1
        controller.reset(np.flatnonzero(trace['terminated'][i] | trace['truncated'][i]))
    assert bool(changed_count) == (role == ROLES[-1])
    return dict(replayed_inputs=replayed, maximum_onnx_error=max_error, nonzero_forward_yaw_controls=changed_count)


def verify_first(arms, motors, forward):
    a, b, c = (arms[role] for role in ROLES)
    assert set(motors) == set(ROLES)
    for role in ROLES:
        np.testing.assert_array_equal(motors[role], motors[ROLES[0]])
    for field in ('observations', 'actions', 'requested_joint_targets', 'applied_joint_targets', 'active'):
        np.testing.assert_array_equal(a[field], b[field])
    np.testing.assert_array_equal(a['active'], c['active'])
    np.testing.assert_array_equal(a['observations'][:, :50], c['observations'][:, :50])
    np.testing.assert_array_equal(a['observations'][:, 51:], c['observations'][:, 51:])
    unchanged = ~forward | ~a['active']
    np.testing.assert_array_equal(a['observations'][unchanged], c['observations'][unchanged])
    for field in ('actions', 'requested_joint_targets', 'applied_joint_targets'):
        np.testing.assert_array_equal(a[field][unchanged], c[field][unchanged])
    observed = c['observations'][:, 50] - a['observations'][:, 50]
    targets = c['applied_joint_targets'] - a['applied_joint_targets']
    assert np.any(observed) and np.any(targets)
    return dict(reference_inputs_actions_and_targets_exact=True, non_forward_inputs_and_targets_exact=True,
                first_delayed_motor_command_exact=True, forward_worlds_with_nonzero_yaw=int(np.count_nonzero(observed)),
                maximum_yaw_change_rad_s=float(abs(observed).max()), maximum_projected_target_change_rad=float(abs(targets).max()))


def verify():
    spec, state = read(PLAN), read(OUT / 'coordinator.json')
    assert state['status'] == 'collected-review-required' and not torch.cuda.is_initialized()
    record = ROOT / state['record']
    assert sha(PLAN) == sha(record / 'protocol.json') == state['protocol_sha256']
    check_sources(spec)
    p, k = spec['environment'], spec['phase']['warmup_controls']
    n, h = p['parallel_worlds'], p['maximum_control_steps']
    controls = k + len(ROLES) * (h-k)
    assert state['actor_unchanged'] and state['contact_peak'] < state['contact_slots']
    assert state['actual'] == dict(attempted_controls=controls, completed_controls=controls,
                                  attempted_substeps=controls*5, completed_substeps=controls*5, initial_resets=1)
    assert n*controls == spec['maximum_control_transitions']
    initial_file = record / 'physical-initial.npz'
    assert sha(initial_file) == state['initial_state_sha256']
    initial = arrays(initial_file)
    startup, zero = (read(record / file) for file in ('startup.json', 'zero-snapshot.json'))
    assert startup['status'] == zero['status'] == 'passed' and startup['physical_steps'] == zero['physical_substeps'] == 0
    assert startup['cpu_cuda_rng_unchanged'] and startup['complete_processor_memory_restored']
    inventory = read(record / 'fork-state-fields.json')
    aux = read(record / 'fork-controller-state.json')
    forks = read(record / 'fork-readback.json')
    fork = forks[ROLES[0]]['physical']
    assert list(forks) == list(ROLES) and fork['counters'] == [k, 5*k]
    assert all(receipt['physical'] == fork and receipt['controller_sha256'] == digest(aux) for receipt in forks.values())
    assert fork['stateless_upright_config_sha256'] == zero['snapshot']['stateless_upright_config_sha256']
    assert digest(inventory) == fork['state_sha256'] and len(inventory) == fork['fields']
    for prefix in ('warp.qacc_warmstart', 'warp.contact.', 'obs-delay.actor.', 'actuator.0.delay.', 'rng'):
        assert any(name.startswith(prefix) for name in inventory), prefix
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    groups, desired = groups_and_commands(p)
    prefix_path = record / 'trace-shared-prefix.npz'
    prefix = arrays(prefix_path)
    assert prefix['active'].shape == (k, n)
    reset_prefix = arrays(record / 'resets-shared-prefix.npz')
    scores, first, replays, resets, movies, traces = {}, {}, {}, {}, {}, {}
    for role in ROLES:
        path = record / f'trace-{role}.npz'
        traces[role] = sha(path)
        post = arrays(path)
        assert post['active'].shape == (h-k, n)
        first[role] = {key: post[key][0] for key in ('observations', 'actions', 'requested_joint_targets',
                                                    'applied_joint_targets', 'active')}
        trace = {key: np.concatenate((prefix[key], post[key])) for key in prefix}
        np.testing.assert_allclose(trace['qpos'][..., state['qpos_joint_addresses']], trace['joint_positions'], atol=1e-6, rtol=0)
        replays[role] = verify_inputs(trace, p, spec, desired, role, session)
        resets[role] = verify_resets(reset_prefix, trace) + verify_resets(arrays(record / f'resets-{role}.npz'), trace)
        scores[role] = {name: command_score({key: values[:, ids] for key, values in trace.items()}, initial['xy'][ids], name, p)
                        for name, ids in groups.items()}
        condition = next(item for item in state['conditions'] if item['role'] == role)
        movies[role] = verify_movie(record, condition['video'], trace, initial)
    initial_match = verify_first(first, arrays(record / 'first-motor-commands.npz'), desired[:, 0] > 0.)
    comparison = compare_forward({role: scores[role] for role in ROLES[:2]}, scores[ROLES[-1]], spec['gate'])
    ref_a, ref_b = ([episode['yaw_error'] for episode in scores[role]['forward']['episode_scores']] for role in ROLES[:2])
    spread = np.asarray(ref_a) - ref_b
    result = dict(status='passed-integrity', continued_admission=comparison['passed'], scores=scores, comparison=comparison,
        repeat_forward_yaw_mean_difference_rad_s=float(spread.mean()), repeat_forward_yaw_median_absolute_difference_rad_s=float(np.median(abs(spread))),
        first_control=initial_match, fork=fork, inputs=replays, unscored_resets_including_prefix=resets, videos=movies,
        shared_prefix_sha256=sha(prefix_path), trace_sha256=traces, protocol_sha256=sha(PLAN), record=state['record'],
        actual_control_transitions=controls*n, actual_physics_substeps=controls*5, learning_updates=0, heldout_used=False,
        scope='One new development seed with identical fork state. Separate solver branches after control 50.')
    write(OUT / 'analysis.json', result)
    state.update(status='completed', branch='closed', continuation_gate_passed=comparison['passed'])
    publish(record, state)
    print('FORWARD_PHASE_VERIFIED', {role: (scores[role]['forward']['successes'], scores[role]['forward']['yaw_error'])
                                      for role in ROLES}, 'admission', comparison['passed'], flush=True)
