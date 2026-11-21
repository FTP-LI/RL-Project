"""Screen a causal roll-to-stand latch on a saved development rollout."""
from pathlib import Path
import sys

import numpy as np

from roll_metrics import progress, rotation
from s1_reference_inputs import ROOT, read, sha

sys.path.insert(0, str(ROOT/'scripts/training'))
from record_io import write


def first_ready(gravity, gyro, criteria):
    # Reconstruct yaw-independent forward rotation from body-frame gravity.
    # Discard all future samples. A two-second dwell must be earned before switching.
    gx, gy, gz = np.moveaxis(gravity, -1, 0)
    lateral_horizontal = np.maximum(np.sqrt(np.maximum(0., 1.-gy**2)), 1e-6)
    pitch = np.arctan2(gx, -gz/lateral_horizontal)
    delta = (np.diff(pitch, axis=0, prepend=pitch[:1]) + np.pi) % (2*np.pi) - np.pi
    degrees = np.rad2deg(np.cumsum(delta, axis=0))
    low = 360 - criteria['completion_tolerance_deg']
    high = 360 + criteria['completion_tolerance_deg']
    intact = np.logical_and.accumulate(
        (degrees >= -criteria['backward_tolerance_deg']) & (degrees <= high), axis=0)
    stable = ((degrees >= low) & intact
              & (-gz >= np.cos(np.deg2rad(criteria['tilt_max_deg'])))
              & (np.linalg.norm(gyro, axis=-1) <= criteria['angular_speed_max_rad_s']))
    hold = round(criteria['hold_seconds']/criteria['step_dt'])
    count = np.zeros(stable.shape[1], dtype=int)
    first = np.full(stable.shape[1], -1, dtype=int)
    for step, ready in enumerate(stable):
        count = np.where(ready, count + 1, 0)
        first[(first < 0) & (count >= hold)] = step
    return first, degrees


def main():
    source = ROOT/'records/reports/roll-endpoint-confirmation/development-b-readback.json'
    proof = read(source)
    assert proof['status'] == 'passed-independent-verification' and not proof['heldout_used']
    folder = ROOT/proof['record']
    path, plan = folder/'forward-roll-trace.npz', folder/'protocol.json'
    protocol = read(plan)
    criteria = protocol['criteria']
    assert protocol['seed'] == proof['seed'] == 1026 and not protocol['heldout_used']
    assert sha(path) == proof['trace_sha256']
    with np.load(path) as data:
        trace = {key: data[key] for key in ('time', 'quaternion', 'ground_contact', 'angular_velocity')}
    quaternion = np.asarray(trace['quaternion'], dtype=float)
    unit_quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    gravity = -rotation(unit_quaternion)[..., 2, :]
    first, degrees = first_ready(gravity, trace['angular_velocity'], criteria)
    normalized_trace = {**trace, 'quaternion': unit_quaternion}
    np.testing.assert_allclose(degrees, np.rad2deg(progress(normalized_trace)[0]), atol=1e-8, rtol=0)
    archived_difference = float(abs(degrees - np.rad2deg(progress(trace)[0])).max())
    assert archived_difference < 1e-4
    rows = []
    for world, expected in enumerate(proof['fresh_rows']):
        assert expected['env_index'] == world
        ready = float(trace['time'][first[world]]) if first[world] >= 0 else None
        recorded = expected['hold_end_seconds']
        rows.append(dict(world=world, ready_seconds=ready, scored=expected['success'],
                         scored_hold_end_seconds=recorded, reasons=expected['reasons'],
                         premature=bool(ready is not None and recorded is not None and ready < recorded-1e-7)))
    false_ready = [row['world'] for row in rows if not row['scored'] and row['ready_seconds'] is not None]
    missed = [row['world'] for row in rows if row['scored'] and row['ready_seconds'] is None]
    early = [row['world'] for row in rows if row['scored'] and row['premature']]
    output = source.with_name('stand-readiness-development.json')
    assert not output.exists()
    write(output, dict(status='passed-offline-screen' if not (false_ready or missed or early) else 'closed-offline-gate-not-met',
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in (source, path, plan, Path(__file__))},
        criteria=criteria, archived_rotation_maximum_difference_deg=archived_difference,
        passed_reference=proof['fresh_successes'], ready=int((first >= 0).sum()),
        false_ready=false_ready, missed=missed, premature=early, worlds=rows, physical_substeps=0,
        learning_updates=0, heldout_used=False, scope='Ideal orientation and gyro proxies; no switching, noise or hardware tested.'))
    print('ROLL_STAND_READINESS', int((first >= 0).sum()), 'false', false_ready, 'missed', missed, 'early', early, flush=True)


if __name__ == '__main__':
    main()
