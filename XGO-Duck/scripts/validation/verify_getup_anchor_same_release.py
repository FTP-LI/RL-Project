"""Audit the paired model effect, original recovery gates and full video files."""
import hashlib

import numpy as np

from getup_anchor_same_release import OUT, PLAN, ROOT, check_sources
from getup_coverage_metrics import POSES, evaluate
from getup_release_fork_state import digest
from record_io import write
from recovery_end_hold import evaluate_end_hold
from s1_reference_inputs import read, sha
from verify_getup_shared_release import load_arrays, same_bytes


def check_trace(t, p, addresses):
    n, h = p['num_envs'], p['maximum_control_steps']
    assert t['qpos'].shape == (h, n, 21) and t['qvel'].shape == (h, n, 20)
    assert t['active'].all() and not t['terminated'].any()
    assert t['truncated'][-1].all() and not t['truncated'][:-1].any()
    assert all(np.isfinite(values).all() for values in t.values() if values.dtype.kind not in ('U', 'S'))
    np.testing.assert_array_equal(t['requested_pose'], [POSES[code] for code in p['split']['pose_codes']])
    np.testing.assert_allclose(t['quaternion'], t['qpos'][..., 3:7], atol=2e-6, rtol=0)
    np.testing.assert_allclose(t['height'], t['qpos'][..., 2], atol=2e-6, rtol=0)
    np.testing.assert_allclose(t['joint_positions'], t['qpos'][..., addresses], atol=2e-6, rtol=0)
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    positions = t['joint_positions']
    in_range = ((positions >= limits[:, 0]-tolerance) & (positions <= limits[:, 1]+tolerance)).all(axis=-1)
    np.testing.assert_array_equal(t['joint_envelope_ok'], in_range)

    actions = t['raw_policy_actions']
    freeze = round(p['criteria']['freeze_seconds'] / .02)
    slew = round(p['handoff_settings']['start_seconds'] / .02)
    assert not actions[:freeze].any()
    history = np.concatenate((np.zeros((1, n, 14), np.float32), actions[:-1]))
    np.testing.assert_array_equal(t['observations'][..., 34:48], history)
    np.testing.assert_array_equal(t['actor_input'][..., 34:48], history)
    np.testing.assert_array_equal(t['actor_input'][..., :34], t['observations'][..., :34])
    np.testing.assert_allclose(t['observations'][..., 48:], 0., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(t['actor_input'][..., 48:], 0.)
    request = np.asarray(p['model_home'], np.float32) + actions
    request[:freeze] = np.asarray(p['simulation_home'], np.float32)
    np.testing.assert_allclose(t['requested_targets'], request, atol=1e-6, rtol=0)
    previous = t['applied_targets'][slew-1:-1]
    lo, hi = np.asarray(p['joint_limits'], np.float32).T
    delta = p['handoff_settings']['rate_rad_s'] * .02
    expected = previous + np.clip(np.clip(request[slew:], lo, hi)-previous, -delta, delta)
    np.testing.assert_allclose(t['applied_targets'][:slew], request[:slew], atol=1e-6, rtol=0)
    np.testing.assert_allclose(t['applied_targets'][slew:], expected, atol=2e-6, rtol=0)
    maximum_step = float(np.abs(t['applied_targets'][slew:]-previous).max())
    assert maximum_step <= .040002
    return maximum_step


def independent_holds(t, criteria):
    """Use interval counts rather than the first-hold search in the scorer."""
    dt = criteria['step_dt']
    release, deadline, count = (round(criteria[key]/dt) for key in
                                ('freeze_seconds', 'deadline_seconds', 'hold_seconds'))
    quat = t['quaternion'].astype(np.float64)
    trunk_vertical = 1-2*(quat[..., 1]**2 + quat[..., 2]**2)
    stable = ((trunk_vertical >= np.cos(np.deg2rad(criteria['tilt_max_deg'])))
              & (abs(t['height']-criteria['nominal_height_m']) <= criteria['height_tolerance_m'])
              & (np.linalg.norm(t['angular_velocity'], axis=-1) <= criteria['angular_speed_max_rad_s'])
              & t['foot_contact'].all(axis=-1) & t['joint_envelope_ok'] & t['active'])
    sums = np.cumsum(stable, axis=0, dtype=np.int16)
    sums = np.concatenate((np.zeros((1, sums.shape[1]), np.int16), sums))
    windows = sums[count:] - sums[:-count] == count
    assert len(windows) == deadline+1
    return windows[release:].any(axis=0), windows[-1]


def verify_policy(t, session):
    max_error, count = 0., 0
    for k in (101, 110, 125, 150, 200, 250, 251, 350, 450, 550):
        # Deterministic, all 320 worlds at the first active decision, and a
        # fixed grid over the subsequent early and late controls.
        indices = np.arange(t['active'].shape[1]) if k == 101 else np.arange(0, t['active'].shape[1], 5)
        for i in indices:
            expected = session.run(None, {'obs': t['actor_input'][k-1, i:i+1]})[0][0]
            actual = t['raw_policy_actions'][k-1, i]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            max_error, count = max(max_error, float(abs(actual-expected).max())), count+1
    return dict(maximum_absolute_error=max_error, actual_inputs_replayed=count, history_checked_all_rows=True)


def verify_video(record, info, t, initial, world):
    import imageio_ffmpeg

    path = record / info['path']
    assert info['environment_index'] == world and info['complete_clip'] and info['extra_physics_steps'] == 0
    assert sha(path) == info['sha256']
    steps = np.arange(0, 551, 2)
    np.testing.assert_allclose(info['frame_times_seconds'], steps*.02, atol=1e-12, rtol=0)
    source = np.concatenate((initial['qpos'][None, world], t['qpos'][:, world]))
    assert [hashlib.sha256(source[k].tobytes()).hexdigest() for k in steps] == info['qpos_sha256_by_frame']
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(frames)
    assert meta['fps'] == 25 and meta['size'] == (640, 480) and sum(1 for _ in frames) == len(steps)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=info['sha256'], frames=len(steps), fps=25,
                start_seconds=0., end_seconds=11., world=world, source_states_verified=True)


def paired(first, second, mask):
    win = np.flatnonzero(mask & second & ~first).tolist()
    loss = np.flatnonzero(mask & first & ~second).tolist()
    return dict(worlds=int(mask.sum()), reference_successes=int((first & mask).sum()),
                comparison_successes=int((second & mask).sum()), wins=win, losses=loss, net=len(win)-len(loss))


def verify():
    import onnxruntime as ort
    import torch

    spec = read(PLAN)
    check_sources(spec)
    state = read(OUT / 'coordinator.json')
    record = ROOT / state['record']
    assert state['status'] == 'collected-review-required' and not torch.cuda.is_initialized()
    assert state['actor_weights_unchanged'] and sha(record / 'protocol.json') == state['protocol_sha256'] == sha(PLAN)
    controls, substeps = spec['maximum_unique_controls'], spec['maximum_physics_substeps']
    assert state['actual'] == dict(controls=controls, attempted_controls=controls,
                                  substeps=substeps, attempted_substeps=substeps, resets=1)
    assert state['contact_peak'] < state['contact_capacity']
    gpu = read(record / 'zero-readback.json')
    assert gpu['status'] == 'passed' and gpu['actual_physics_steps'] == 0
    p, roles = spec['environment'], spec['branches']
    release = spec['prefix_controls']
    receipt = read(record / 'release-readback.json')
    inventory = read(record / 'release-state-fields.json')
    assert set(receipt) == set(roles)
    release_files, first = {}, {}
    for role in roles:
        assert receipt[role] == receipt[roles[0]] and receipt[role]['state_sha256'] == digest(inventory)
        assert receipt[role]['counters'] == [release, 5*release]
        release_files[role] = load_arrays(record / f'release-{role}.npz')
        same_bytes(release_files[roles[0]], release_files[role])
        first[role] = load_arrays(record / f'first-control-{role}.npz')
        np.testing.assert_array_equal(first[role]['actor_input'], first[roles[0]]['actor_input'])
        np.testing.assert_array_equal(first[role]['motor'][0], first[roles[0]]['motor'][0])
        assert first[role]['motor'].shape[:2] == (5, p['num_envs']) and np.isfinite(first[role]['motor']).all()
    for key in ('raw_policy_actions', 'requested_targets', 'applied_targets'):
        np.testing.assert_array_equal(first[roles[0]][key], first[roles[1]][key])
    initial = load_arrays(record / 'initial.npz')
    prefix_path = record / 'trace-shared-prefix.npz'
    prefix = load_arrays(prefix_path)
    np.testing.assert_array_equal(prefix['qpos'][-1], release_files[roles[0]]['qpos'])
    np.testing.assert_array_equal(prefix['qvel'][-1], release_files[roles[0]]['qvel'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reports, combined, release_classes = {}, {}, {}
    videos = []
    for role in roles:
        path = record / f'trace-{role}.npz'
        branch = load_arrays(path)
        assert branch.keys() == prefix.keys()
        t = {key: np.concatenate((prefix[key], branch[key])) for key in prefix}
        t['requested_pose'] = initial['requested_pose']
        for key in ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets'):
            np.testing.assert_array_equal(branch[key][0], first[role][key])
        step = check_trace(t, p, state['qpos_joint_addresses'])
        plan = spec['policies'][role]
        session = ort.InferenceSession(str(ROOT/plan['model']), sess_options=options, providers=['CPUExecutionProvider'])
        numeric = verify_policy(t, session)
        timed = evaluate(t, p)
        end = evaluate_end_hold(t, p, timed['episodes'])
        timing, final = independent_holds(t, p['criteria'])
        np.testing.assert_array_equal(timing, [x['hold_start_seconds'] is not None for x in timed['episodes']])
        np.testing.assert_array_equal(final, end['stable_by_environment'])
        release_classes[role] = np.array([x['release_pose'] for x in timed['episodes']])
        np.testing.assert_array_equal(release_classes[role], release_classes[roles[0]])
        combined[role] = timing & final & np.isin(release_classes[role], POSES)
        for item in (item for item in state['conditions'] if item['role'] == role):
            world = spec['cameras'][role][item['condition']]
            clip = verify_video(record, item['video'], t, initial, world)
            videos.append(dict(clip, role=role, requested_pose=item['condition'],
                               actual_release_pose=str(release_classes[role][world])))
        audits = read(record / f'contact-audits-{role}.json')
        assert audits and all(item['matched'] for item in audits)
        for audit in audits:
            assert int(t['native_head_leg_contact'][audit['control_step']-1].sum()) == audit['contact_worlds']
        reports[role] = dict(timing=timed, end_hold=end, independent_holds_exact=True,
            combined_successes=int(combined[role].sum()), all_motion_gates_passed=timed['functional_coverage_gate']
            and end['functional_coverage_gate'], trace_sha256=sha(path), onnx=numeric, maximum_late_target_step_rad=step)
    assert len(videos) == sum(len(x) for x in spec['cameras'].values()) == 6
    chosen = roles[-1]
    classification = release_classes[chosen]
    all_worlds = np.ones(len(classification), bool)
    supine = classification == 'supine'
    comparisons = {role: dict(all=paired(combined[role], combined[chosen], all_worlds),
                              actual_supine=paired(combined[role], combined[chosen], supine)) for role in roles[:-1]}
    repeat = paired(combined[roles[0]], combined[roles[1]], all_worlds)
    result = dict(status='passed-independent-verification', arms=reports, comparisons=comparisons, official_repeat=repeat,
        candidate_functional_gate_passed=reports[chosen]['all_motion_gates_passed'], release=receipt, videos=videos,
        first_inputs_and_delayed_ctrl_exact=True, record=state['record'], prefix_sha256=sha(prefix_path),
        protocol_sha256=sha(PLAN), actual_control_transitions=controls*p['num_envs'],
        actual_physics_substeps=substeps, physical_resets=1, learning_updates=0, heldout_used=False,
        scope='Fixed-weight, one new development seed, original requested/actual coverage, timely hold and final hold.')
    write(OUT / 'analysis.json', result)
    state.update(status='completed', phase='closed', analysis_sha256=sha(OUT / 'analysis.json'))
    write(OUT / 'coordinator.json', state)
    write(record / 'evaluation.json', state)
    print('ANCHOR_RELEASE_VERIFIED', {role: row['combined_successes'] for role, row in reports.items()},
          'gates', {role: row['all_motion_gates_passed'] for role, row in reports.items()}, flush=True)
