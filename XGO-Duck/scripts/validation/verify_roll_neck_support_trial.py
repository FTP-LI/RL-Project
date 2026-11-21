"""Recompute the original roll score and the neck-only treatment from traces."""
import hashlib

import numpy as np

from roll_neck_support_trial import OUT, PLAN, ROOT, check_sources, read, sha
from getup_release_fork_state import digest
from record_io import write
from verify_roll_reference import independent


def read_trace(path):
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def video_check(record, video, qpos):
    import imageio_ffmpeg

    path = record / video['path']
    assert sha(path) == video['sha256'] and video['complete_clip']
    assert video['extra_physics_steps'] == 0 and video['environment_index'] == 0
    steps = np.arange(0, 451, 2)
    np.testing.assert_allclose(video['frame_times_seconds'], steps * .02, atol=1e-12, rtol=0)
    assert [hashlib.sha256(qpos[k, 0].tobytes()).hexdigest() for k in steps] == video['qpos_sha256_by_frame']
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(frames)
    assert meta['fps'] == 25 and meta['size'] == (640, 480) and sum(1 for _ in frames) == len(steps)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), start_seconds=0., end_seconds=9.,
                frames=len(steps), fps=25, source_qpos_verified=True)


def verify_targets(t, spec, state, candidate):
    cfg = spec['intervention']
    neck = state['joint_names'].index('neck_pitch')
    limits = np.asarray(state['joint_limits'])
    lo, hi = limits[neck]
    w, x, y, z = np.moveaxis(t['quaternion'].astype(np.float64), -1, 0)
    xx, xy = 1-2*(y*y+z*z), 2*(x*y-w*z)
    yx, yy = 2*(x*y+w*z), 1-2*(x*x+z*z)
    lateral = 2*(y*z+w*x)
    horizontal = (xx*yy-yx*xy) / np.maximum(np.hypot(xy, yy), 1e-6)
    angle = np.unwrap(np.arctan2(-2*(x*z-w*y), horizontal), axis=0)
    angle = np.rad2deg(angle-angle[0])
    np.testing.assert_allclose(t['neck_phase_deg'][1:], angle[:-1], atol=1e-9, rtol=0)
    feet = t['foot_contact'][:-1]
    seen = np.zeros_like(feet[:, :, 0])
    seen[1:] = np.logical_or.accumulate(feet[:-1].all(axis=-1) & t['active'][:-2], axis=0)
    request = t['requested_targets'][1:, :, neck]
    left, right = cfg['forward_phase_degrees']
    cue = (t['active'][:-1] & seen & t['head_contact'][:-1] & ~feet.any(axis=-1)
           & (angle[:-1] >= left) & (angle[:-1] <= right)
           & (abs(lateral[:-1]) <= cfg['maximum_body_lateral_projection'])
           & (t['angular_velocity'][:-1, :, 1] > cfg['forward_body_pitch_rate_above_rad_s'])
           & (t['qvel'][:-1, :, state['neck_dof']] < cfg['neck_rate_below_rad_s'])
           & (request >= lo) & (request <= hi))
    np.testing.assert_array_equal(t['neck_cue'][1:], cue)
    assert not cue[:spec['warmup_controls']].any()
    expected = t['requested_targets'][1:].copy()
    if candidate:
        expected[:, :, neck][cue] = np.minimum(request[cue] + cfg['maximum_positive_target_offset_rad'], hi)
    np.testing.assert_allclose(t['applied_targets'][1:], expected, atol=1e-6, rtol=0)
    delta = expected[:, :, neck] - request
    actual = t['applied_targets'][1:, :, neck]
    assert (delta >= 0).all() and float(delta.max()) <= cfg['maximum_positive_target_offset_rad']+1e-6
    assert np.all(actual[delta > 0] <= hi + 1e-6)
    motor = t['motor'][1:]
    old = t['applied_targets'][:-1, :, neck].copy()
    old[0] = actual[0]
    np.testing.assert_allclose(motor[:, 0, :, 0], old, atol=1e-7, rtol=0)
    np.testing.assert_allclose(motor[:, 1:, :, 0], np.broadcast_to(actual[:, None], motor[:, 1:, :, 0].shape),
                               atol=1e-7, rtol=0)
    filtered = np.zeros(actual.shape[1], dtype=np.float32)
    for row in motor[..., :2].reshape(-1, actual.shape[1], 2):
        delayed, smooth = row.T
        filtered = np.clip(delayed, filtered-.4, filtered+.4)
        np.testing.assert_allclose(smooth, filtered, atol=1e-6, rtol=0)
    assert (motor[..., 2] >= 7.0).all() and (motor[..., 2] <= np.float32(7.7)).all()
    return dict(cues=int(cue.sum()), cue_worlds=int(cue.any(axis=0).sum()), modified=int((delta > 0).sum()),
                modified_worlds=np.flatnonzero((delta > 0).any(axis=0)).tolist(),
                maximum_offset_rad=float(delta.max()), minimum_voltage=float(motor[..., 2].min()),
                maximum_voltage=float(motor[..., 2].max()), audited_delayed_motor_writes=motor.shape[0]*5)


def verify_policy(t, state, model):
    home, sim_home = (np.asarray(state[key], np.float32) for key in ('home', 'simulation_home'))
    rebuilt = t['raw_observations'].copy()
    rebuilt[:, :, 6:20] += sim_home-home
    rebuilt[0, :, 34:48] = 0
    rebuilt[1:, :, 34:48] = t['raw_policy_actions'][:-1]
    np.testing.assert_allclose(rebuilt, t['observations'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(rebuilt[:, :, 48:], 0., atol=1e-6, rtol=0)
    np.testing.assert_allclose(t['requested_targets'][1:], home+t['raw_policy_actions'][1:], atol=1e-6, rtol=0)
    error, count = 0., 0
    for k in (1, 10, 15, 20, 25, 30, 40, 100, 300, 450):
        for observation, action in zip(t['observations'][k], t['raw_policy_actions'][k], strict=True):
            expected = model.run(None, {'obs': observation[None]})[0][0]
            np.testing.assert_allclose(expected, action, atol=2e-5, rtol=2e-5)
            error, count = max(error, float(abs(action-expected).max())), count+1
    return dict(maximum_absolute_error=error, observations_replayed=count, raw_history_preserved=True)


def verify():
    import onnxruntime as ort

    spec = read(PLAN)
    check_sources(spec)
    p, roles = spec['environment'], spec['roles']
    state = read(OUT / 'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['actor_unchanged']
    assert state['actual'] == dict(controls=spec['maximum_batched_controls'], substeps=spec['maximum_physics_substeps'], resets=1)
    record = ROOT / state['record']
    assert sha(record / 'protocol.json') == state['protocol_sha256'] == sha(PLAN)
    zero = read(record / 'zero-motor-readback.json')
    runtime = read(record / 'runtime-contract.json')
    assert zero['status'] == 'passed' and zero['physical_substeps'] == 0 and not runtime['capacity_reached']
    assert runtime['control_steps'] == spec['maximum_batched_controls']
    receipts, fields = read(record / 'fork-readback.json'), read(record / 'fork-fields.json')
    assert list(receipts) == roles
    for receipt in receipts.values():
        assert receipt == receipts[roles[0]] and receipt['physical']['state_sha256'] == digest(fields)
        assert receipt['physical']['counters'] == [spec['warmup_controls'], 5*spec['warmup_controls']]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    meta = model.get_modelmeta().custom_metadata_map
    assert meta['joint_names'].split(',') == state['joint_names'] and float(meta['action_scale']) == 1.
    np.testing.assert_array_equal(np.asarray(meta['default_joint_pos'].split(','), np.float32), np.asarray(state['home'], np.float32))
    prefix_path = record / 'trace-shared-prefix.npz'
    prefix = read_trace(prefix_path)
    reports, first, receipts_video = {}, {}, {}
    for arm in roles:
        path = record / f'trace-{arm}.npz'
        rows = read_trace(path)
        assert rows.keys() == prefix.keys()
        t = {key: np.concatenate((prefix[key], rows[key]), axis=0) for key in rows}
        assert t['active'].shape == (451, 128) and t['active'].all()
        assert not t['terminated'].any() and not t['truncated'][:-1].any()
        assert all(np.isfinite(value).all() for value in t.values())
        first[arm] = (rows['observations'][0], rows['raw_policy_actions'][0], rows['applied_targets'][0], rows['motor'][0, 0])
        limits = np.asarray(state['joint_limits'])
        score = independent(t, p['criteria'], limits, np.deg2rad(p['actual_joint_limit_tolerance_deg']))
        original = read(record / f'original-score-{arm}.json')
        assert [row['success'] for row in original['episodes']] == [row['success'] for row in score]
        np.testing.assert_allclose([row['supported_max_deg'] for row in original['episodes']],
                                   [row['supported_max_deg'] for row in score], atol=2e-4, rtol=0)
        audit = verify_targets(t, spec, state, arm == roles[-1])
        parity = verify_policy(t, state, model)
        video = next(item['video'] for item in state['conditions'] if item['role'] == arm)
        receipts_video[arm] = video_check(record, video, t['qpos'])
        reports[arm] = dict(successes=sum(item['success'] for item in score), episodes=score, targets=audit,
                            trace_sha256=sha(path), onnx_parity=parity)
    for arm in roles[1:]:
        for baseline, observed in zip(first[roles[0]], first[arm], strict=True):
            np.testing.assert_array_equal(baseline, observed)

    candidate = reports[roles[-1]]['episodes']
    gate = spec['gate']
    comparisons = {}
    for arm in roles[:-1]:
        reference = reports[arm]['episodes']
        wins = [a['env_index'] for a, b in zip(candidate, reference, strict=True) if a['success'] and not b['success']]
        losses = [a['env_index'] for a, b in zip(candidate, reference, strict=True) if not a['success'] and b['success']]
        cap = p['criteria']['head_penetration_max_m']
        new_penetration = [a['env_index'] for a, b in zip(candidate, reference, strict=True)
                           if a['head_penetration_max_m'] > cap >= b['head_penetration_max_m']]
        new_limits = [a['env_index'] for a, b in zip(candidate, reference, strict=True)
                      if a['joint_violation_samples'] and not b['joint_violation_samples']]
        comparisons[arm] = dict(wins=wins, losses=losses, net_wins=len(wins)-len(losses),
                                new_head_penetration=new_penetration, new_joint_limits=new_limits)
    observed = reports[roles[-1]]
    admitted = (observed['successes'] >= gate['minimum_successes'] and observed['targets']['modified'] > 0
                and all(row['net_wins'] >= gate['minimum_net_wins_against_each_reference']
                        and not row['new_head_penetration'] and not row['new_joint_limits'] for row in comparisons.values()))
    result = dict(status='passed-independent-verification', continued_admission=admitted,
        arms=reports, comparisons=comparisons, fork=receipts, first_post_fork_input_action_and_delayed_motor_exact=True,
        videos=receipts_video, record=state['record'], prefix_sha256=sha(prefix_path), protocol_sha256=sha(PLAN),
        actual_control_transitions=runtime['control_transitions'], actual_physics_substeps=runtime['physics_substeps'],
        initial_resets=1, learning_updates=0, heldout_used=False, hardware_io=False,
        decision='new-development-confirmation-warranted' if admitted else 'close-neck-support-candidate')
    write(OUT / 'analysis.json', result)
    state.update(status='completed', branch='closed', continued_admission=admitted,
                 analysis_sha256=sha(OUT / 'analysis.json'))
    write(OUT / 'coordinator.json', state)
    print('ROLL_NECK_VERIFIED', {name: row['successes'] for name, row in reports.items()},
          comparisons, 'continued admission', admitted, flush=True)
