"""Separate a knee target change from the same-world BAM supply response.

Run five motor writes from identical, unintegrated simulator state. Compare
two ordinary repeats, the knee pulse with normal supply, and the same pulse
with the recorded reference supply. This fixture never advances the plant.
"""
import numpy as np
import torch

from getup_knee_support import KneeSupportPulse, apply_after_processing
from getup_knee_support_screen import cpu
from getup_release_scene_state import SceneReleaseSnapshot


def sample(env, memory, contract, spec):
    joint_ids, limits, knees, motor_ids = contract
    settings = spec['mechanism']
    manager = env.action_manager
    term = manager.get_term('joint_pos')
    robot = env.scene['robot']
    assert len(robot.actuators) == 1
    actuator = robot.actuators[0]
    columns = [cpu(actuator._global_ctrl_ids).tolist().index(motor) for motor in motor_ids]
    inner = actuator._bam_model.actuator
    pulse = KneeSupportPulse(env.num_envs, env.device, settings)
    knee_limits = torch.as_tensor(limits[knees], device=env.device, dtype=torch.float32)
    age = 155
    pulse.trigger[0] = age
    raw = torch.full_like(manager._action, .01)
    snap = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
    initial = snap.verify()
    assert initial['counters'] == [0, 0]
    qpos, qvel, physics_time = (cpu(value) for value in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.time))
    gain, nominal = cpu(actuator.vin_drop_gain), cpu(actuator.vin_tensor)
    minimum = actuator.cfg.vin_min
    assert np.isfinite(gain).all() and (gain > 0).all()
    original_delay = actuator.apply_delay
    delayed = None

    def inspect_delay(command):
        nonlocal delayed
        result = original_delay(command)
        delayed = cpu(result.position_target)
        return result

    def run_path(enabled, reference_voltage=None):
        env.episode_length_buf.fill_(age)
        manager.process_action(raw)
        request = cpu(term.requested_target)
        apply_after_processing(term, pulse, age + 1, knees, knee_limits, enabled)
        np.testing.assert_array_equal(cpu(manager._action), cpu(raw))
        out = dict(request=request, target=cpu(term._processed_actions),
                   history=cpu(torch.stack([manager._action, manager._prev_action, manager._prev_prev_action])))
        steps = []
        if reference_voltage is not None:
            actuator.vin_drop_gain.zero_()
        for write in range(env.cfg.decimation):
            if reference_voltage is not None:
                actuator.vin_tensor.copy_(torch.as_tensor(reference_voltage[write], device=env.device))
            manager.apply_action()
            assert torch.equal(robot.data.joint_pos_target[:, joint_ids], term._processed_actions)
            previous_motor = cpu(actuator._prev_motor_torque)
            env.scene.write_data_to_sim()
            steps.append(dict(delayed=delayed, vin=cpu(inner.vin), previous_motor=previous_motor, ctrl=cpu(env.sim.data.ctrl)))
        for saved, live in zip((qpos, qvel, physics_time), (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.time), strict=True):
            np.testing.assert_array_equal(saved, cpu(live))
        out.update({key: np.stack([entry[key] for entry in steps]) for key in steps[0]})
        assert snap.restore() == initial
        return out

    actuator.apply_delay = inspect_delay
    try:
        with torch.no_grad():
            reference = run_path(False)
            repeat = run_path(False)
            candidate = run_path(True)
            matched = run_path(True, reference['vin'])
    finally:
        del actuator.apply_delay
    meta = dict(snapshot=initial, motor_ids=motor_ids, knee_target_columns=knees, delayed_knee_columns=columns,
                vin_floor=minimum, actual_physics_steps=0, buffered_motor_writes=4 * env.cfg.decimation)
    arrays = dict(gain=gain, nominal=nominal)
    for role, values in (('reference', reference), ('repeat', repeat), ('candidate', candidate), ('matched', matched)):
        arrays.update({f'{role}__{key}': value for key, value in values.items()})
    return meta, arrays


def audit(meta, values, spec):
    role = lambda name, key: values[f'{name}__{key}']
    assert meta['snapshot']['counters'] == [0, 0] and meta['actual_physics_steps'] == 0
    assert meta['buffered_motor_writes'] == 20 and all(np.isfinite(value).all() for value in values.values())
    for key in ('request', 'history', 'target', 'delayed', 'vin', 'previous_motor', 'ctrl'):
        np.testing.assert_array_equal(role('reference', key), role('repeat', key))
    for name in ('candidate', 'matched'):
        for key in ('request', 'history'):
            np.testing.assert_array_equal(role(name, key), role('reference', key))
    settings = spec['mechanism']
    expected = np.zeros_like(role('reference', 'target'))
    knees = meta['knee_target_columns']
    expected[0, knees] = np.asarray(settings['signed_offset']) * settings['offset_rate_rad_s'] * settings['step_dt']
    for name in ('candidate', 'matched'):
        np.testing.assert_allclose(role(name, 'target') - role('reference', 'target'), expected, atol=1e-7, rtol=0)
    delayed = role('candidate', 'delayed') - role('reference', 'delayed')
    expected_delayed = np.zeros_like(delayed)
    expected_delayed[1:, 0, meta['delayed_knee_columns']] = expected[0, knees]
    np.testing.assert_allclose(delayed, expected_delayed, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(role('candidate', 'delayed'), role('matched', 'delayed'))
    for name in ('reference', 'repeat', 'candidate'):
        previous = role(name, 'previous_motor')
        predicted_vin = values['nominal'] - values['gain'] * np.abs(previous).sum(axis=-1, keepdims=True)
        if meta['vin_floor'] is not None:
            predicted_vin = predicted_vin.clip(min=meta['vin_floor'])
        np.testing.assert_allclose(role(name, 'vin'), predicted_vin, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(role('matched', 'vin'), role('reference', 'vin'))
    natural = role('candidate', 'ctrl') - role('reference', 'ctrl')
    fixed_supply = role('matched', 'ctrl') - role('reference', 'ctrl')
    vin_diff = role('candidate', 'vin') - role('reference', 'vin')
    motors = meta['motor_ids']
    other = [i for i in range(natural.shape[-1]) if i not in motors]
    assert not natural[:, 1:].any() and not fixed_supply[:, 1:].any() and not vin_diff[:, 1:].any()
    assert not natural[0].any() and not natural[1, 0, other].any() and natural[1, 0, motors].any()
    assert not vin_diff[:2].any() and not fixed_supply[:, 0, other].any()
    changed_voltage = np.flatnonzero(vin_diff[:, 0].any(axis=-1))
    other_effect = np.flatnonzero(natural[:, 0, other].any(axis=-1))
    assert set(other_effect) <= set(changed_voltage), 'Non-knee motor differences outside the verified supply coupling'
    return dict(status='passed', motor_ids=motors, raw_history_preserved=True,
        only_selected_knee_motors_changed=True, only_selected_knee_motors_changed_scope='voltage-matched diagnostic control',
        first_effective_motor_write=2, same_world_supply_difference_writes=(changed_voltage + 1).tolist(),
        same_world_non_knee_difference_writes=(other_effect + 1).tolist(),
        maximum_knee_motor_difference=float(abs(natural[:, 0, motors]).max()),
        maximum_same_world_non_knee_motor_difference=float(abs(natural[:, 0, other]).max()),
        maximum_supply_voltage_difference=float(abs(vin_diff).max()),
        all_other_worlds_exact=True, actual_physics_steps=0)
