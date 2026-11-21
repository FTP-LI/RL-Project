"""Recompute the actuator correction and score the archived physical branches."""
from unittest.mock import patch

import numpy as np

import common_walk_actuator_yaw_feedback as trial
import verify_common_walk_forward_phase as score
from record_io import write
from s1_reference_inputs import ROOT, read, sha


verify_nominal_inputs = score.verify_inputs


def verify_feedback(trace, p, spec, session, path):
    rows = score.arrays(path)
    assert set(rows) == {'step', 'world', 'yaw', 'alternate_raw', 'fraction'}
    h, n = trace['active'].shape
    fork = spec['phase']['warmup_controls']
    observed = trace['observations']
    eligible = np.zeros((h, n), bool)
    eligible[fork:, p['groups']['forward']] = trace['active'][fork:, p['groups']['forward']]
    step, world = np.nonzero(eligible)
    assert step.size > 0
    np.testing.assert_array_equal(rows['step'], step+1)
    np.testing.assert_array_equal(rows['world'], world)
    assert not observed[step, world, 50].any()

    # Derive the previous usable observation directly from the trace. The
    # sampler's memory state is never used to reconstruct these features.
    left, right = (20+p['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right'))
    sensors = np.stack((observed[..., 2], observed[..., left]+observed[..., right]), axis=-1)
    delta = np.zeros_like(sensors)
    delta[1:] = sensors[1:]-sensors[:-1]
    ended = trace['terminated'] | trace['truncated']
    delta[1:][ended[:-1]] = 0.
    feature = np.concatenate((sensors[step, world], delta[step, world]), axis=-1)
    screen = read(ROOT/spec['feedback_screen'])
    signals = np.column_stack([np.clip(((feature-model['means'])/model['scales']) @ model['weights'] +
                                       model['bias'], -1., 1.) for model in screen['forecast_models']])
    cap = spec['feedback']['maximum_yaw_input_rad_s']
    yaw = (-cap*np.clip((signals @ screen['kernel'])/screen['gradient_scale'], -1., 1.)).astype(np.float32)
    np.testing.assert_allclose(yaw, rows['yaw'], atol=2e-7, rtol=0)
    assert np.isfinite(rows['yaw']).all() and np.count_nonzero(rows['yaw']) > 0
    assert abs(rows['yaw']).max() <= cap + 1e-7

    inputs = observed[step, world].copy()
    inputs[:, 50] = rows['yaw']
    # The official export has a fixed single-row input. Re-run all alternate
    # inputs in the independent CPU ONNX runtime.
    alternate = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
    np.testing.assert_allclose(alternate, rows['alternate_raw'], atol=2e-5, rtol=2e-5)
    home = np.asarray(p['model_home'], np.float32)
    lo, hi = np.asarray(p['joint_limits'], np.float32).T
    normal = np.clip(trace['actions']+home, lo, hi)
    applied = trace['applied_joint_targets']
    np.testing.assert_allclose(applied[~eligible], normal[~eligible], atol=1e-6, rtol=0)
    assert (applied >= lo-1e-7).all() and (applied <= hi+1e-7).all()
    base = normal[step, world]
    change = np.clip(rows['alternate_raw']+home, lo, hi)-base
    maximum = spec['feedback']['maximum_joint_correction_rad']
    fraction = np.minimum(1., maximum/np.maximum(np.max(abs(change), axis=-1), 1e-8))
    np.testing.assert_allclose(fraction, rows['fraction'], atol=2e-7, rtol=0)
    np.testing.assert_allclose(applied[step, world], base+change*fraction[:, None], atol=1e-6, rtol=0)
    actual_max = float(abs(applied[step, world]-base).max())
    assert 0. < actual_max <= maximum+1e-7
    return normal, dict(readback_sha256=sha(path), active_forward_controls=len(step),
                        alternate_onnx_replays=len(alternate), maximum_alternate_onnx_error=float(abs(alternate-rows['alternate_raw']).max()),
                        maximum_actuator_yaw_input_rad_s=float(abs(rows['yaw']).max()),
                        maximum_applied_joint_change_rad=actual_max, scaled_controls=int((fraction < 1.).sum()),
                        all_other_targets_nominal=True)


def verify_first(arms, motors, forward):
    a, b, c = (arms[role] for role in trial.ROLES)
    assert set(motors) == set(trial.ROLES)
    for role in trial.ROLES:
        np.testing.assert_array_equal(motors[role], motors[trial.ROLES[0]])
    for field in ('observations', 'actions', 'requested_joint_targets', 'active'):
        np.testing.assert_array_equal(a[field], b[field])
        np.testing.assert_array_equal(a[field], c[field])
    np.testing.assert_array_equal(a['applied_joint_targets'], b['applied_joint_targets'])
    changed = abs(a['applied_joint_targets']-c['applied_joint_targets']).max(axis=-1)
    assert changed.any() and not changed[~(forward & a['active'])].any()
    return dict(nominal_input_raw_request_and_history_exact=True, first_delayed_motor_command_exact=True,
                corrected_forward_worlds=int(np.count_nonzero(changed)),
                maximum_first_joint_change_rad=float(changed.max()))


def main():
    trial.pulse.check_project()
    location = ROOT/read(trial.OUT/'coordinator.json')['record']

    def verify_inputs(trace, p, spec, desired, role, session):
        candidate = role == trial.ROLES[-1]
        if candidate:
            normal, feedback = verify_feedback(trace, p, spec, session, location/'feedback-readback.npz')
            nominal = dict(trace, applied_joint_targets=normal)
        else:
            nominal, feedback = trace, {}
        # The original verifier checks the nominal policy inputs, previous
        # actions and raw requests. The scorer below still gets the real trace.
        result = verify_nominal_inputs(nominal, p, spec, desired, trial.ROLES[0], session)
        return dict(result, **feedback)

    def publish(record, state):
        write(record/'evaluation.json', state)
        write(trial.OUT/'coordinator.json', state)
        view = read(record/'state.json')
        view.update(status='completed', purpose='前进执行器反馈 · 已核验',
                    updated_at=trial.datetime.now(trial.ZoneInfo('Asia/Shanghai')).isoformat())
        with patch.object(trial.pulse.live, 'PAGE', trial.PAGE):
            trial.pulse.live.publish(record, view)

    with patch.object(score, 'PLAN', trial.PLAN), patch.object(score, 'OUT', trial.OUT), \
         patch.object(score, 'ROLES', trial.ROLES), patch.object(score, 'ForwardPhase', trial.Controller), \
         patch.object(score, 'verify_inputs', verify_inputs), patch.object(score, 'verify_first', verify_first), \
         patch.object(score, 'publish', publish):
        score.verify()


if __name__ == '__main__':
    main()
