"""Verify continuous seven-command acquisition from its saved physical samples."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np


PLAN = Path(__file__).resolve().parents[2]/'configs/experiments/continuous_routed_seven_command.json'


def observe():
    import verify_routed_forward_entry_ramp as windows

    with patch.object(windows, 'PLAN', PLAN):
        windows.observe(windows.read(PLAN))


def audit_targets(spec, state, record, initial):
    from command_input_mapping import transform
    from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback
    from verify_routed_getup_sequence import ROOT, PHYSICAL, sha, study

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
    seven = spec['seven_command']
    requested = seven['commands']['forward']
    command = np.zeros((n, 3), np.float32)
    forward_assigned = np.zeros(n, bool)
    forward_assigned[seven['groups']['forward']] = True
    role_ids = {role: i+3 for i, role in enumerate(seven['policies'])}
    terminal_models = np.empty(n, np.int8)
    for name, ids in seven['groups'].items():
        command[ids] = transform(seven['commands'][name])
        terminal_models[ids] = role_ids[seven['routes'][name]]
    ramp = spec['ramped_forward']['ramp']['ramp_controls']
    audited_ramp_controls = 0
    feedback = ForwardFeedback(w, n)
    models = [spec['common_fallback'], spec['policies']['reference-a'], spec['policies']['candidate'],
              *seven['policies'].values()]
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
            keep_history = ~switched | ((modes == FORWARD) & (terminal_models == role_ids['base']))
            x[:, 34:48] = np.where(keep_history[:, None], previous_raw, 0.)
            x[:, 48:] = 0.
            terminal = modes == FORWARD
            forward_mask = terminal & forward_assigned
            x[terminal, 48:51] = command[terminal]
            elapsed = k - forward_at + 1
            if 1 <= elapsed <= ramp:
                x[forward_mask, 48:51] = transform([requested[0] * (elapsed / ramp), *requested[1:]])
                audited_ramp_controls += 1
            np.testing.assert_allclose(rows['actor_input'][row], x, atol=2e-7, rtol=0)
            if k < release:
                assert not raw.any()
            if k == release:
                supine, right = study.route.masks(x, spec['dual_route']['alignment'])
            request = raw + homes[modes]
            if k < release:
                request[:] = p['simulation_home']
            projected = np.clip(request, limits[:, 0], limits[:, 1])
            nonforward = terminal & ~forward_assigned
            expected_alternate = raw[nonforward] if elapsed > feedback.warmup else np.zeros_like(raw[nonforward])
            np.testing.assert_array_equal(rows['alternate'][row, nonforward], expected_alternate)
            forward = np.flatnonzero(forward_mask)
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
                model_ids = np.where(is_getup, np.where(right, 2, np.where(supine, 1, 0)),
                                     np.where(terminal, terminal_models, role_ids['base']))
                for model_id, net in enumerate(networks):
                    sample = scope[model_ids == model_id]
                    if len(sample):
                        for i in sample[np.unique(np.linspace(0, len(sample)-1, min(8, len(sample)), dtype=int))]:
                            answer = net.run(None, {'obs': x[i:i+1]})[0][0]
                            np.testing.assert_allclose(raw[i], answer, atol=2e-5, rtol=2e-5)
                            if forward_mask[i] and yaw is not None:
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
    assert audited_ramp_controls == ramp
    proof = dict(forward_entry_ramp_verified_controls=audited_ramp_controls, independent_onnx_queries=checked, forward_feedback_nonzero_samples=active_yaw, boundary=jumps,
        exact_getup_route_masks=True, same_onnx_history_verified=True, seven_command_inputs_verified=True,
        incoming_policy_history_verified=True, physical_joint_targets_verified=True)
    return {key: np.concatenate(arrays) for key, arrays in exact.items()}, hashes, proof


def verify():
    import imageio_ffmpeg
    import run_routed_seven_command as study
    from routed_seven_command_score import score

    study.phase(PLAN)
    read, sha, write = study.read, study.sha, study.write
    spec = read(PLAN)
    study.base.shared.assert_sources(spec)
    record = study.ROOT/spec['record']
    state = read(record/'evaluation.json')
    view = read(record/'windows-view-verification.json')
    handoff = read(record/'handoff-entries.json')
    seven = spec['seven_command']
    total, n = spec['maximum_unique_controls'], spec['environment']['num_envs']
    checksum = sha(PLAN)
    assert state['protocol_sha256'] == checksum == sha(record/'protocol.json')
    assert not state['heldout_used'] and not state['hardware_io'] and not state['learning_updates']
    assert handoff['control'] == spec['sequence']['boundary_controls'][-1]
    assert handoff['entrants'] == state['handoff_entries']
    assert view['observer_host'] == 'Windows'
    assert state['contact_peak'] < state['contact_capacity']
    expected = {name:len(ids) for name, ids in seven['groups'].items()}
    assert state['terminal_commands'] == expected and sum(expected.values()) == n
    if state['status'] == 'stopped-review-required':
        assert state['early_stop'] == 'complete-sequence-upper-bound-below-original-gate'
        assert handoff['below_minimum'] or sum(handoff['entrants'].values()) < seven['minimum_overall']
        assert state['actual']['controls'] == state['actual']['attempted_controls'] == handoff['control']
        assert state['actual']['substeps'] == state['actual']['attempted_substeps'] == handoff['control']*5
        assert state['actual']['resets'] == 1 and state['actual_control_transitions'] == handoff['control']*n
        assert all(sha(record/chunk['file']) == chunk['sha256'] for chunk in state['record_chunks'])
        clip, = read(record/'clips.json')
        assert not clip['complete_clip'] and not clip['extra_physics_steps']
        assert clip['common_steps'] == list(range(0, handoff['control']+1, 2))
        video = record/clip['file']
        frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        next(frames)
        assert sum(1 for _ in frames) == len(clip['common_steps'])
        write(study.OUT/'stopped-review.json', dict(status='verified-gate-unreachable-before-command-window',
            record=spec['record'], protocol_sha256=checksum, handoff=handoff, assigned=expected,
            minimum_by_command=seven['minimum_by_command'], minimum_overall=seven['minimum_overall'],
            actual=state['actual'], transitions=state['actual_control_transitions'], terminal_commands_scored=False,
            video=dict(path=study.base.rel(video), sha256=sha(video), complete=False),
            retry_allowed=False, heldout_used=False))
        print('ROUTED_SEVEN_COMMAND_STOP_VERIFIED', handoff, flush=True)
        return
    assert state['status'] == 'completed' and state['actor_weights_unchanged'] and not handoff['below_minimum']
    assert state['actual'] == dict(controls=total, attempted_controls=total, substeps=total*5,
        attempted_substeps=total*5, resets=1)
    assert state['actual_control_transitions'] == seven['maximum_transitions'] == total*n
    assert view['status'] == 'passed' and view['running_snapshots_with_original_frame'] >= 3
    with np.load(record/'initial.npz', allow_pickle=False) as saved:
        initial = dict(saved)
    trace, hashes, policy = audit_targets(spec, state, record, initial)
    result = score(spec, state, record, trace, initial)
    assert {name:row['entered'] for name, row in result['commands'].items()} == handoff['entrants']
    clip = state['video']
    video = record/clip['file']
    assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps'] and clip['environment_index'] == 0
    assert clip['common_steps'] == list(range(0, total+1, 2)) and clip['qpos_sha256_by_frame'] == hashes
    assert sha(video) == clip['sha256']
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    meta = next(frames)
    assert meta['fps'] == 25 and meta['size'] == (640, 480)
    assert sum(1 for _ in frames) == len(hashes) == 801
    target = study.OUT/'analysis.json'
    assert not target.exists()
    write(target, dict(status='passed-independent-verification', protocol_sha256=checksum,
        record=spec['record'], physical_resets=1, transitions=total*n, physical_substeps=total*5, policy=policy,
        score=result, observer_sha256=sha(record/'windows-view-verification.json'),
        video=dict(path=study.base.rel(video), sha256=clip['sha256'], frames=len(hashes), seconds=32.,
                   world=0, terminal_command=seven['video_command']),
        learning_updates=0, heldout_used=False, hardware_io=False))
    print('ROUTED_SEVEN_COMMAND_VERIFIED', result['complete'], result['passed_development_gate'],
          {name:(row['complete'], row['assigned'], row['walking']['successes'])
           for name, row in result['commands'].items()}, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()
