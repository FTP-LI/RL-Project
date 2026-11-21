"""Read saved development states to check the first routed getup-to-stand request."""

from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write
from s1_reference_inputs import read, sha


DEVELOPMENT = ROOT / 'records/reports/getup-right-robustness/development/analysis.json'
RELEASE = DEVELOPMENT.with_name('release-verification.json')
PROTOCOL = ROOT / 'configs/experiments/getup_right_robustness_development_s1168.json'
WALKING = ROOT / 'artifacts/delivery/walking-simulation.json'
OUT = ROOT / 'records/reports/getup-stand-handoff/routed-development-boundary'


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    assert not OUT.exists()
    score, release, spec, walking = map(read, (DEVELOPMENT, RELEASE, PROTOCOL, WALKING))
    assert release['analysis_sha256'] == sha(DEVELOPMENT)
    assert score['protocol_sha256'] == sha(PROTOCOL) and not score['heldout_used'] and not release['heldout_used']
    assert walking['status'] == 'verified-seven-command-simulation-components'
    arm = score['arms']['candidate']
    assert arm['combined']['functional_coverage_gate']
    record = ROOT / score['record']
    trace = record / 'trace-candidate.npz'
    assert sha(trace) == arm['trace_sha256']
    with np.load(trace, allow_pickle=False) as arrays:
        times = arrays['time']
        observations = arrays['observations'][-25:].copy()
        previous = arrays['applied_targets'][-2].copy()
    p, criteria = spec['environment'], spec['environment']['criteria']
    dt = criteria['step_dt']
    assert np.isclose(times[-1], 11.) and len(observations) * dt == .5
    gravity = observations[..., 3:6]
    norm = np.linalg.norm(gravity, axis=-1)
    assert np.all(norm > 0) and np.isfinite(observations).all()
    tilt = np.rad2deg(np.arccos(np.clip(-gravity[..., 2] / norm, -1., 1.)))
    rate = np.linalg.norm(observations[..., :3], axis=-1)
    observed_ready = ((tilt <= criteria['tilt_max_deg']) &
                      (rate <= criteria['angular_speed_max_rad_s'])).all(axis=0)
    timely = np.array([row['hold_start_seconds'] is not None for row in arm['timely']['episodes']])
    final = np.array(arm['final']['stable_by_environment'], dtype=bool)
    passed = timely & final
    poses = np.array([row['release_pose'] for row in arm['timely']['episodes']])
    assert int(passed.sum()) == arm['combined']['overall']['timely_holds']
    groups, worlds = {}, []
    for pose in ('prone', 'supine', 'left-side', 'right-side'):
        group = poses == pose
        ids = np.flatnonzero(group & passed & observed_ready)
        assert len(ids) >= 32
        chosen = ids[np.linspace(0, len(ids) - 1, 32, dtype=int)]
        worlds.extend(chosen.tolist())
        groups[pose] = dict(worlds=int(group.sum()), passed=int((group & passed).sum()),
            observed_ready=int((group & observed_ready).sum()), ready_and_passed=len(ids),
            ready_but_did_not_pass=int((group & observed_ready & ~passed).sum()))
    model = walking['models'][walking['route_by_command']['stand']]['files']['onnx']
    model_path = ROOT / model['path']
    assert sha(model_path) == model['sha256'] and walking['commands']['stand'] == [0, 0, 0]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata['joint_names'].split(',') == walking['joint_names']
    assert session.get_inputs()[0].shape == [1, 61] and session.get_outputs()[0].shape == [1, 14]
    assert float(metadata['action_scale']) == 1.
    stand_home = np.array(metadata['default_joint_pos'].split(','), dtype=np.float32)
    source_home = np.asarray(p['model_home'], dtype=np.float32)
    sensed = observations[-1, worlds].copy()
    converted = sensed.copy()
    converted[:, 6:20] += source_home - stand_home
    converted[:, 34:] = 0.
    np.testing.assert_allclose(converted[:, 6:20] + stand_home, sensed[:, 6:20] + source_home, atol=2e-7, rtol=0)
    requested = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in converted]) + stand_home
    old = previous[worlds]
    limits = np.asarray(p['joint_limits'], dtype=np.float32)
    projected = np.clip(requested, limits[:, 0], limits[:, 1])
    max_step = p['handoff_settings']['rate_rad_s'] * dt
    applied = old + np.clip(projected - old, -max_step, max_step)
    assert float(abs(applied - old).max()) <= max_step + 1e-7
    old_excess = abs(old - np.clip(old, limits[:, 0], limits[:, 1]))
    new_excess = abs(applied - np.clip(applied, limits[:, 0], limits[:, 1]))
    assert np.all(new_excess <= old_excess + 1e-7)
    for i, group in enumerate(groups.values()):
        subset = slice(i * 32, (i + 1) * 32)
        jumps = abs(requested[subset] - old[subset]).max(axis=1)
        group.update(queried=32, largest_unlimited_jump_rad=float(jumps.max()),
            p95_unlimited_jump_rad=float(np.quantile(jumps, .95)),
            projected_worlds=int((projected[subset] != requested[subset]).any(axis=1).sum()))
    OUT.mkdir()
    data = OUT / 'first-standing-requests.npz'
    np.savez_compressed(data, worlds=np.array(worlds), sensed=sensed, converted=converted, previous=old,
        requested=requested, projected=projected, applied=applied)
    write(OUT / 'analysis.json', dict(status='passed-cpu-request-and-slew-check',
        scope='first hypothetical standing request from sampled development observations',
        reference=ref(DEVELOPMENT), release=ref(RELEASE), protocol=ref(PROTOCOL), trace=ref(trace),
        standing_delivery=ref(WALKING), standing_model=ref(model_path), request_data=ref(data),
        script=ref(Path(__file__)), seconds_before_end=dt, simulated_transitions=0, learning_updates=0,
        heldout_used=False, hardware_io=False, continuous_handoff_verified=False,
        observed_readiness_window_seconds=.5, max_observed_tilt_deg=criteria['tilt_max_deg'],
        max_observed_angular_speed_rad_s=criteria['angular_speed_max_rad_s'],
        query_worlds=len(worlds), unbounded_max_request_jump_rad=float(abs(requested - old).max()),
        bounded_max_first_target_step_rad=float(abs(applied - old).max()), groups=groups))
    print('ROUTED_GETUP_STAND_BOUNDARY', len(worlds), float(abs(applied - old).max()), groups, flush=True)


if __name__ == '__main__':
    main()
