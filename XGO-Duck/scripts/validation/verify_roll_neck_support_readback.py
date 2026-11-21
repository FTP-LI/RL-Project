"""Correct the reset-buffer expectation on the completed seed 1023 traces."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

import roll_neck_support_log_recovery as recovery
from record_io import write


BASE = recovery.OUT
OUT = BASE / 'readback'
ROOT = recovery.ROOT
read, sha = recovery.base.read, recovery.base.sha


def zero_reset_buffer(spec):
    from mjlab.envs import ManagerBasedRlEnv

    p = spec['environment']
    cfg, _ = recovery.base.make_config(p)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=p['seed'])
        (actuator,) = env.scene['robot'].actuators
        ring = actuator._delay_buffer._buffer
        # reset() writes scene targets once before the first policy step.
        neck = actuator._target_names.index('neck_pitch')
        assert env.common_step_counter == env._sim_step_counter == 0
        np.testing.assert_array_equal(ring._num_pushes.cpu().numpy(), 1)
        np.testing.assert_array_equal(ring._buffer[:, :, neck].cpu().numpy(), 0.)
        target = actuator._bam_model.actuator.q_target_smooth[:, neck].cpu().numpy()
        np.testing.assert_array_equal(target, 0.)
        assert not torch.cuda.is_initialized()
    finally:
        env.close()
    source = Path(inspect.getfile(ManagerBasedRlEnv))
    return dict(status='passed', reset_motor_writes=1, physical_substeps=0, neck_reset_target_rad=0.,
                neck_smoothed_reset_target_rad=0., checked_worlds=p['episodes'], environment_source=str(source),
                environment_source_sha256=sha(source))


def verify_targets(t, spec, state, candidate):
    cfg = spec['intervention']
    neck = state['joint_names'].index('neck_pitch')
    lo, hi = np.asarray(state['joint_limits'])[neck]
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
    # The zero-step CPU fixture independently reads this value from reset's ring.
    old[0] = 0.
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
                maximum_voltage=float(motor[..., 2].max()), audited_delayed_motor_writes=motor.shape[0]*5,
                initial_delayed_reset_target_rad=0.)


def historical_smoother_check():
    from review_roll_head_motor_envelope import reconstruct_internal_target

    previous = ROOT / 'records/reports/roll-common-confirmation/rng-recovery'
    summary = {}
    for role in ('development-a', 'development-b'):
        report = read(previous / f'{role}-analysis.json')
        assert not report['heldout_used']
        folder = ROOT / report['evaluator']
        trace = folder / 'forward-roll-trace.npz'
        assert sha(trace) == report['trace_sha256']
        order = read(folder / 'evaluation.json')['joint_names']
        columns = [order.index(name) for name in ('neck_pitch', 'head_pitch')]
        with np.load(trace) as stored:
            targets = stored['applied_targets'][:, :, columns]
        old = reconstruct_internal_target(targets, .4)
        fixed = np.zeros_like(targets)
        for k in range(1, len(targets)):
            delayed = targets[k-1] if k > 1 else np.zeros_like(targets[0])
            state = np.clip(delayed, fixed[k-1]-.4, fixed[k-1]+.4)
            for _ in range(4):
                state = np.clip(targets[k], state-.4, state+.4)
            fixed[k] = state
        mismatch = np.flatnonzero(np.any(fixed != old, axis=(1, 2)))
        # The earlier neck cue only starts at 0.44 s; its preceding 4 ms
        # write and the old support-window samples must use identical state.
        assert not np.any(mismatch >= 10)
        summary[role] = dict(differing_control_boundaries=mismatch.tolist(), all_boundaries_after_point_two_exact=True,
                             trace_sha256=sha(trace))
    return summary


def main():
    recovery.bind()
    import verify_roll_neck_support_trial as original

    assert not OUT.exists()
    path = BASE / 'coordinator.json'
    spec = read(recovery.PLAN)
    state = read(path)
    assert state['status'] == 'stopped-review-required' and state['branch'] == 'independent-verification'
    assert state['protocol_sha256'] == sha(recovery.PLAN)
    record = ROOT / state['record']
    archived = {file: sha(file) for file in (path, record / 'evaluation.json', record / 'execution.log')}
    proof = zero_reset_buffer(spec)
    previous = historical_smoother_check()
    captured = {}

    def read_only_view(file):
        value = read(file)
        if file == path:
            return dict(value, status='collected-review-required')
        return value

    def collect_result(file, value):
        assert file in (BASE / 'analysis.json', path)
        captured[file.name] = value

    def result_digest(file):
        if file == BASE / 'analysis.json':
            body = json.dumps(captured[file.name], ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            return hashlib.sha256(body.encode()).hexdigest()
        return sha(file)

    original.read = read_only_view
    original.write = collect_result
    original.sha = result_digest
    original.verify_targets = verify_targets
    original.verify()
    assert archived == {file: sha(file) for file in archived}
    result = captured['analysis.json']
    OUT.mkdir()
    write(OUT / 'analysis.json', result)
    assert sha(OUT / 'analysis.json') == captured['coordinator.json']['analysis_sha256']
    write(OUT / 'verification.json', dict(status='passed', original_verifier_first_error='first delayed neck request expected policy instead of reset zero',
        original_receipts_preserved=True, archived_sha256={file.relative_to(ROOT).as_posix(): digest for file, digest in archived.items()},
        reset_buffer=proof, earlier_development_smoother=previous, original_gates_unchanged=True,
        analysis_sha256=sha(OUT / 'analysis.json'), script_sha256=sha(Path(__file__)),
        new_physical_substeps=0, new_learning_updates=0, heldout_used=False))
    print('ROLL_NECK_READBACK_OK', OUT, flush=True)


if __name__ == '__main__':
    main()
