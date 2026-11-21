"""Independently score the three phases and audit the two model boundaries."""

from collections import Counter
from hashlib import sha256

import imageio_ffmpeg
import numpy as np

import run_routed_getup_sequence as study
from command_input_mapping import transform
from getup_coverage_metrics import evaluate, group_rows
from recovery_end_hold import evaluate_end_hold
from roll_stand_walk_sequence import score_forward
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback
from standup_metrics import projected_gravity
from verify_getup_anchor_same_release import independent_holds


ROOT, read, sha, write = study.ROOT, study.read, study.sha, study.write
PHYSICAL = ('qpos', 'velocity', 'angular_velocity', 'foot_contact', 'foot_positions', 'foot_speed', 'mode')


def audit_targets(spec, state, record, initial):
    p, s, w = spec['environment'], spec['sequence'], spec['walking']
    release, stand_at, forward_at = s['boundary_controls']
    c = s['controller']
    n = p['num_envs']
    modes = np.zeros(n, np.int8)
    dwell = np.zeros(n, np.int32)
    previous_raw = np.zeros((n, 14), np.float32)
    homes = np.asarray([p['model_home'], spec['switch_policies']['stand']['home'], spec['switch_policies']['forward']['home']], np.float32)
    previous_target = np.broadcast_to(np.asarray(p['simulation_home'], np.float32), (n, 14)).copy()
    limits = np.asarray(p['joint_limits'], np.float32)
    command = transform(w['commands']['forward'])
    feedback = ForwardFeedback(w, n)
    models = [spec['common_fallback'], spec['policies']['reference-a'], spec['policies']['candidate'],
              spec['switch_policies']['stand'], spec['switch_policies']['forward']]
    networks = [study.session(ROOT / item['model']) for item in models]
    exact = {name: [] for name in PHYSICAL}
    hashes = [sha256(initial['qpos'][0].tobytes()).hexdigest()]
    scope = np.arange(n)
    right = supine = None
    active_yaw = checked = 0
    jumps = {}
    next_control = 1
    for chunk in state['record_chunks']:
        path = record / chunk['file']
        assert sha(path) == chunk['sha256'] and chunk['first'] == next_control
        with np.load(path, allow_pickle=False) as saved:
            rows = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(rows['control'], np.arange(chunk['first'], chunk['last'] + 1))
        for key in PHYSICAL:
            exact[key].append(rows[key])
        for row, step in enumerate(rows['control']):
            k = int(step) - 1
            sensed, raw = rows['observed'][row], rows['raw'][row]
            g = sensed[:, 3:6]
            ready = ((np.linalg.norm(g, axis=1) > 0.) &
                     (-g[:, 2] >= np.linalg.norm(g, axis=1) * np.cos(np.deg2rad(c['observed_tilt_max_deg']))) &
                     (np.linalg.norm(sensed[:, :3], axis=1) <= c['observed_angular_speed_max_rad_s']))
            dwell = np.where(ready, dwell + 1, 0)
            previous_modes = modes.copy()
            if k == stand_at:
                modes[dwell >= c['standing_dwell_controls']] = STAND
                dwell[:] = 0
            elif k == forward_at:
                modes[(modes == STAND) & (dwell >= c['standing_dwell_controls'])] = FORWARD
            switched = modes != previous_modes
            np.testing.assert_array_equal(rows['mode'][row], modes)
            bridge = (((modes == STAND) & (k < stand_at + c['cross_model_bridge_controls'])) |
                      ((modes == FORWARD) & (k < forward_at + c['cross_model_bridge_controls'])))
            np.testing.assert_array_equal(rows['bridge'][row], bridge)
            x = sensed.copy()
            x[:, 6:20] += homes[GETUP] - homes[modes]
            x[:, 34:48] = np.where(switched[:, None], 0., previous_raw)
            x[:, 48:] = 0.
            x[modes == FORWARD, 48:51] = command
            np.testing.assert_allclose(rows['actor_input'][row], x, atol=2e-7, rtol=0)
            if k < release:
                assert not raw.any()
            if k == release:
                supine, right = study.route.masks(x, spec['dual_route']['alignment'])
            request = raw + homes[modes]
            if k < release:
                request[:] = p['simulation_home']
            projected = np.clip(request, limits[:, 0], limits[:, 1])
            forward = np.flatnonzero(modes == FORWARD)
            yaw = feedback.advance(x[forward], forward, k - forward_at + 1) if len(forward) else None
            yaw_values = np.zeros(n, np.float32)
            if yaw is not None:
                yaw_values[forward] = yaw
                alternate = np.clip(rows['alternate'][row, forward] + homes[FORWARD], limits[:, 0], limits[:, 1])
                diff = alternate - projected[forward]
                scale = np.minimum(1., feedback.maximum_target_change / np.maximum(abs(diff).max(axis=1, keepdims=True), 1e-8))
                projected[forward] += diff * scale
                active_yaw += int(np.count_nonzero(yaw))
            np.testing.assert_allclose(rows['feedback_yaw'][row], yaw_values, atol=2e-7, rtol=0)
            is_getup = modes == GETUP
            limited = bridge | (is_getup & (k >= round(p['handoff_settings']['start_seconds'] / .02)))
            limit = c['bridge_slew_rad_s'] * .02
            target = projected.copy()
            target[limited] = previous_target[limited] + np.clip(projected[limited] - previous_target[limited], -limit, limit)
            if k < round(p['handoff_settings']['start_seconds'] / .02):
                target[:] = request
            for name, expected in (('requested', request), ('projected', projected), ('applied', target), ('motor_joint_target', target)):
                np.testing.assert_allclose(rows[name][row], expected, atol=2e-6, rtol=0)
            if switched.any():
                jumps[str(step)] = dict(worlds=int(switched.sum()), maximum_applied_step_rad=float(abs(target[switched]-previous_target[switched]).max()))
            if k >= release and (k in (release, stand_at, forward_at) or k % 50 == 0):
                model_ids = np.where(is_getup, np.where(right, 2, np.where(supine, 1, 0)), modes + 2)
                for model_id, net in enumerate(networks):
                    sample = scope[model_ids == model_id]
                    if len(sample):
                        for i in sample[np.unique(np.linspace(0, len(sample)-1, min(8, len(sample)), dtype=int))]:
                            answer = net.run(None, {'obs': x[i:i+1]})[0][0]
                            np.testing.assert_allclose(raw[i], answer, atol=2e-5, rtol=2e-5)
                            if model_id == 4 and yaw is not None:
                                extra = x[i:i+1].copy()
                                extra[0, 50] = yaw_values[i]
                                np.testing.assert_allclose(rows['alternate'][row, i], net.run(None, {'obs': extra})[0][0],
                                    atol=2e-5, rtol=2e-5)
                            checked += 1
            assert not rows['terminated'][row].any()
            assert bool(rows['truncated'][row].all() if step == spec['maximum_unique_controls'] else ~rows['truncated'][row].any())
            if step % 2 == 0:
                hashes.append(sha256(rows['qpos'][row, 0].tobytes()).hexdigest())
            previous_raw, previous_target = raw, rows['applied'][row]
        next_control = chunk['last'] + 1
    assert next_control == spec['maximum_unique_controls'] + 1
    masks = record / 'route-masks.npz'
    assert sha(masks) == state['route_masks_sha256']
    with np.load(masks, allow_pickle=False) as saved:
        np.testing.assert_array_equal(right, saved['right'])
        np.testing.assert_array_equal(supine, saved['supine'])
    for step in (stand_at + 1, forward_at + 1):
        with np.load(record / f'boundary-motor-{step}.npz', allow_pickle=False) as data:
            assert data['substeps'].shape == (5, n, 14) and np.isfinite(data['substeps']).all()
    proof = dict(independent_onnx_queries=checked, forward_feedback_nonzero_samples=active_yaw, boundary=jumps,
        exact_getup_route_masks=True, incoming_policy_history_verified=True, physical_joint_targets_verified=True)
    return {key: np.concatenate(arrays) for key, arrays in exact.items()}, hashes, proof


def score(spec, state, record, trace, initial):
    p, s = spec['environment'], spec['sequence']
    _, stand_at, forward_at = s['boundary_controls']
    total = spec['maximum_unique_controls']
    qpos = trace['qpos']
    joints = qpos[..., state['qpos_joint_addresses']]
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    envelope = ((joints >= limits[:, 0]-tolerance) & (joints <= limits[:, 1]+tolerance)).all(axis=-1)
    physical = dict(active=np.ones(qpos.shape[:2], bool), quaternion=qpos[..., 3:7], height=qpos[..., 2],
        joint_envelope_ok=envelope, joint_positions=joints,
        **{name: trace[name] for name in ('velocity', 'angular_velocity', 'foot_contact', 'foot_positions', 'foot_speed')})
    prefix = {key: array[:stand_at] for key, array in physical.items()}
    prefix.update(time=np.arange(1, stand_at+1)*.02, requested_pose=initial['requested_pose'])
    timely = evaluate(prefix, p)
    final = evaluate_end_hold(prefix, p, timely['episodes'])
    t, f = independent_holds(prefix, p['criteria'])
    np.testing.assert_array_equal(t, [row['hold_start_seconds'] is not None for row in timely['episodes']])
    np.testing.assert_array_equal(f, final['stable_by_environment'])
    poses = np.array([row['release_pose'] for row in timely['episodes']])
    with np.load(record / 'release-poses.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(poses, saved['poses'])
    getup = t & f & np.isin(poses, study.base.shared.POSES)
    getup_score = group_rows([dict(row, hold_start_seconds=row['hold_start_seconds'] if keep else None)
        for row, keep in zip(timely['episodes'], getup, strict=True)], p['success_rate_min'], p['actual_release_minimum'])
    c = p['criteria']
    stable = slice(forward_at - s['controller']['standing_dwell_controls'], forward_at)
    standing = (((abs(physical['height'][stable] - c['nominal_height_m']) <= c['height_tolerance_m']) &
        (-projected_gravity(physical['quaternion'][stable])[..., 2] >= np.cos(np.deg2rad(c['tilt_max_deg']))) &
        (np.linalg.norm(physical['angular_velocity'][stable], axis=-1) <= c['angular_speed_max_rad_s']) &
        physical['foot_contact'][stable].all(axis=-1) & envelope[stable] &
        (trace['mode'][stable] == STAND)).all(axis=0))
    began = np.where(trace['mode'][forward_at] == FORWARD, forward_at, -1)
    # This scorer indexes states after each control and includes an initial slot.
    augmented = {key: np.concatenate((value[:1], value)) for key, value in physical.items()}
    forward = score_forward(augmented, began, dict(walking=spec['walking'], forward_controls=total-forward_at))
    forward_passed = np.array([item['success'] for item in forward])
    complete = getup & standing & forward_passed
    entered = began >= 0
    w = spec['walking']['criteria']
    aggregate = dict(passed=False, entered=int(entered.sum()))
    if entered.any():
        rows = [item for item in forward if item['entered_forward']]
        aggregate.update({key: float(np.mean([item[key] for item in rows])) for key in
            ('signed_speed_m_s', 'yaw_error_rad_s', 'block_error_m_s')})
        contact = physical['foot_contact'][forward_at:, entered]
        contact_count = contact.sum(axis=(0, 2))
        slip_sum = np.where(contact, physical['foot_speed'][forward_at:, entered], 0.).sum(axis=(0, 2))
        aggregate['missing_contact'] = int((contact_count == 0).sum())
        aggregate['mean_contact_slip_m_s'] = float((slip_sum / np.maximum(contact_count, 1)).mean())
        hip = [spec['walking']['joint_names'].index(f'{side}_hip_roll') for side in ('left', 'right')]
        angles = np.rad2deg(joints[forward_at + w['warmup_steps']:, entered][..., hip])
        aggregate['inward_fraction'] = float(((angles[..., 0] < -20.) & (angles[..., 1] > 20.)).mean())
        aggregate['joint_envelope_failed_samples'] = int((~envelope[forward_at:, entered]).sum())
        aggregate['episode_success_fraction'] = float(forward_passed[entered].mean())
        aggregate['passed'] = bool(aggregate['signed_speed_m_s'] >= w['signed_speed']['forward'] and
            aggregate['yaw_error_rad_s'] <= w['yaw_error_max'] and aggregate['block_error_m_s'] <= w['linear_error']['forward'] and
            aggregate['mean_contact_slip_m_s'] <= w['contact_slip_max'] and aggregate['missing_contact'] == 0 and
            aggregate['inward_fraction'] <= w['inward_fraction_max'] and aggregate['joint_envelope_failed_samples'] == 0 and
            aggregate['episode_success_fraction'] >= w['episode_success_fraction_min'])
    return dict(getup=getup_score, standing=int(standing.sum()), forward_entries=int((began >= 0).sum()),
        forward_passes=int(forward_passed.sum()), complete=int(complete.sum()), worlds=p['num_envs'],
        original_forward_aggregate=aggregate,
        complete_rate=float(complete.mean()), forward_failures=dict(Counter(check for row in forward for check in row['failed_checks'])),
        forward_worlds=forward, complete_worlds=np.flatnonzero(complete).tolist(),
        passed_development_gate=bool(getup_score['functional_coverage_gate'] and aggregate['passed'] and
            complete.mean() >= s['proposed_minimum_complete_sequence_fraction']))


def main():
    study.phase(study.PLAN)
    spec = read(study.PLAN)
    study.base.shared.assert_sources(spec)
    record = ROOT / spec['record']
    state, view = read(record / 'evaluation.json'), read(record / 'windows-view-verification.json')
    total, n = spec['maximum_unique_controls'], spec['environment']['num_envs']
    assert state['status'] == 'completed' and state['actor_weights_unchanged']
    assert state['protocol_sha256'] == sha(study.PLAN) and not state['heldout_used']
    assert state['actual'] == dict(controls=total, attempted_controls=total, substeps=total*5,
        attempted_substeps=total*5, resets=1)
    assert state['actual_control_transitions'] == total*n and state['contact_peak'] < state['contact_capacity']
    assert view['status'] == 'passed' and view['running_snapshots_with_original_frame'] >= 3
    with np.load(record / 'initial.npz', allow_pickle=False) as saved:
        initial = {key: saved[key] for key in saved.files}
    trace, hashes, boundaries = audit_targets(spec, state, record, initial)
    results = score(spec, state, record, trace, initial)
    clip = state['video']
    video = record / clip['file']
    assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps'] and clip['environment_index'] == 0
    assert clip['common_steps'] == list(range(0, total+1, 2)) and clip['qpos_sha256_by_frame'] == hashes
    assert sha(video) == clip['sha256']
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    metadata = next(frames)
    assert metadata['fps'] == 25 and metadata['size'] == (640, 480)
    assert sum(1 for _ in frames) == len(hashes) == 801
    write(study.OUT / 'analysis.json', dict(status='passed-independent-verification', record=spec['record'],
        protocol_sha256=sha(study.PLAN), physical_resets=1, physical_substeps=total*5, transitions=total*n,
        policy=boundaries, score=results, video=dict(path=study.base.rel(video), sha256=clip['sha256'],
        frames=len(hashes), seconds=32., world=0), learning_updates=0, heldout_used=False, hardware_io=False))
    print('ROUTED_SEQUENCE_VERIFIED', results['getup']['overall']['timely_holds'], results['standing'],
          results['forward_passes'], results['complete'], results['passed_development_gate'], flush=True)


if __name__ == '__main__':
    main()
