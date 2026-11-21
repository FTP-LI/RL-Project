"""Verify the scheduled command and reuse the original full-sequence score.

The archived fixed-command audit is kept byte-for-byte. This audit follows its
input and actuator readback with one change at the planned startup boundary.
"""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np


PLAN = Path(__file__).resolve().parents[2] / 'configs/experiments/continuous_routed_getup_ramped_forward.json'


def observe():
    import verify_routed_forward_entry_ramp as windows

    with patch.object(windows, 'PLAN', PLAN):
        windows.observe(windows.read(PLAN))


def verify():
    import run_routed_getup_ramped_forward as ramped
    import verify_routed_getup_sequence as original

    with patch.multiple(original.study, PLAN=PLAN, OUT=ramped.OUT, phase=ramped.phase), \
         patch.object(original, 'audit_targets', audit_targets):
        original.main()


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
    requested = w['commands']['forward']
    command = transform(requested)
    ramp = spec['ramped_forward']['ramp']['ramp_controls']
    audited_ramp_controls = 0
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
            forward_mask = modes == FORWARD
            elapsed = k - forward_at + 1
            if 1 <= elapsed <= ramp:
                ramped = transform([requested[0] * (elapsed / ramp), *requested[1:]])
                x[forward_mask, 48:51] = ramped
                np.testing.assert_array_equal(rows['actor_input'][row, forward_mask, 48:51], x[forward_mask, 48:51])
                audited_ramp_controls += 1
            else:
                x[forward_mask, 48:51] = command
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
    assert audited_ramp_controls == ramp
    proof = dict(forward_entry_ramp_verified_controls=audited_ramp_controls, independent_onnx_queries=checked, forward_feedback_nonzero_samples=active_yaw, boundary=jumps,
        exact_getup_route_masks=True, incoming_policy_history_verified=True, physical_joint_targets_verified=True)
    return {key: np.concatenate(arrays) for key, arrays in exact.items()}, hashes, proof


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()
