"""Replay causal roll-to-stand guards against verified fresh contact labels."""
from pathlib import Path

import numpy as np

from roll_metrics import progress, rotation
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from verify_roll_endpoint_readback import TRACE_FIELDS, score, scoring_view


OUT = ROOT / 'records/reports/roll-endpoint-confirmation'
WIRE = ROOT / 'official_resources/xgoduck_runtime_arduino/python/wire.py'


def readiness(trace, plan):
    angle, supported, rotation_matrix = progress(trace)
    head_vertical = rotation(trace['head_quaternion'])[..., 2, 1]
    lower = np.deg2rad(360 - plan['completion_tolerance_deg'])
    upper = np.deg2rad(360 + plan['completion_tolerance_deg'])
    reverse = np.deg2rad(plan['backward_tolerance_deg'])
    hold = round(plan['hold_seconds'] / plan['step_dt'])
    feet = trace['foot_contact'].all(axis=-1)
    tilt_ok = rotation_matrix[..., 2, 2] >= np.cos(np.deg2rad(plan['tilt_max_deg']))
    height_ok = abs(trace['height'] - plan['nominal_height_m']) <= plan['height_tolerance_m']
    quiet = np.linalg.norm(trace['angular_velocity'], axis=-1) <= plan['angular_speed_max_rad_s']
    active = trace['active']
    count = {name: np.zeros(active.shape[1], dtype=np.int32)
             for name in ('both_feet', 'supported_roll', 'full_simulation')}
    first = {name: np.full(active.shape[1], -1, dtype=np.int32) for name in count}
    seen_feet = np.zeros(active.shape[1], dtype=bool)
    seen_head = seen_feet.copy()
    finished = seen_feet.copy()
    intact = tilt_ok[0] & height_ok[0]
    motion_valid = np.ones_like(seen_feet)
    for k in range(len(active)):
        angle_valid = (angle[k] >= -reverse) & (angle[k] <= upper)
        motion_valid &= angle_valid | ~active[k]
        valid_pose = active[k] & motion_valid & (angle[k] >= lower) & feet[k] & quiet[k] & tilt_ok[k]
        inside_limits = trace['joint_envelope_ok'][k] & (trace['head_penetration_m'][k] <= plan['head_penetration_max_m'])
        intact &= inside_limits | ~active[k]
        ready = dict(both_feet=valid_pose, supported_roll=valid_pose & finished,
                     full_simulation=valid_pose & finished & intact & height_ok[k])
        for name, mask in ready.items():
            count[name] = np.where(mask, count[name] + 1, 0)
            within_deadline = (k - hold) * plan['step_dt'] <= plan['deadline_seconds'] + 1e-8
            fire = (first[name] < 0) & (count[name] >= hold) & within_deadline
            first[name][fire] = k
        # A supported revolution is used only from the following sample.
        seen_head |= (seen_feet & active[k] & trace['head_contact'][k] & (head_vertical[k] < -.3)
                      & (angle[k] > np.deg2rad(20)) & (angle[k] < np.deg2rad(170)))
        seen_feet |= feet[k] & active[k]
        finished |= seen_head & active[k] & (np.minimum(angle[k], supported[k]) >= lower)
    return first


def compare(first, time, rows):
    selected = first >= 0
    expected = np.array([row['success'] for row in rows])
    end = np.array([row['hold_end_seconds'] or np.inf for row in rows])
    triggered = np.flatnonzero(selected)
    earlier = [int(i) for i in triggered if expected[i] and time[first[i]] < end[i] - 1e-7]
    later = [int(i) for i in triggered if expected[i] and time[first[i]] > end[i] + 1e-7]
    return dict(ready=int(selected.sum()), false_ready=np.flatnonzero(selected & ~expected).tolist(),
                missed=np.flatnonzero(~selected & expected).tolist(), early=earlier, late=later,
                ready_time_seconds=[float(time[k]) if k >= 0 else None for k in first])


def main():
    source = OUT / 'development-b-readback.json'
    prior = OUT / 'stand-readiness-development.json'
    proof = read(source)
    record = ROOT / proof['record']
    plan, raw, endpoint = (record / name for name in ('protocol.json', 'forward-roll-trace.npz', 'endpoint-contact.npz'))
    spec = read(plan)
    assert spec['seed'] == proof['seed'] == 1026 and not proof['heldout_used']
    assert sha(raw) == proof['trace_sha256'] and sha(endpoint) == proof['endpoint_sha256']
    assert proof['constraint_agreement_gate_passed'] and not proof['learning_updates']
    with np.load(raw) as data:
        trace = {field: data[field] for field in TRACE_FIELDS}
    with np.load(endpoint) as contacts:
        np.testing.assert_array_equal(contacts['quaternion'], trace['qpos'][..., 3:7])
        np.testing.assert_array_equal(contacts['control'], np.arange(len(trace['time'])))
        assert contacts['fresh'].shape == (451, 128, 4)
        trace = scoring_view(trace, contacts['fresh'])
    config = read(record / 'evaluation.json')
    verified = score(trace, spec['criteria'], np.asarray(config['joint_limits']),
                     np.deg2rad(spec['actual_joint_limit_tolerance_deg']))
    assert verified['successes'] == proof['fresh_successes']
    assert verified['rows'] == proof['fresh_rows']
    results = {name: compare(steps, trace['time'], verified['rows'])
               for name, steps in readiness(trace, spec['criteria']).items()}
    full = results['full_simulation']
    assert not any(full[key] for key in ('false_ready', 'missed', 'early', 'late'))
    destination = OUT / 'stand-contact-gate-development.json'
    assert not destination.exists()
    write(destination, dict(status='passed-simulation-boundary-replay', seed=1026,
        sources={p.relative_to(ROOT).as_posix(): sha(p)
                 for p in (source, prior, plan, raw, endpoint, WIRE, Path(__file__))},
        score_successes=verified['successes'], signals=results, physical_substeps=0,
        learning_updates=0, heldout_used=False,
        scope='Fresh simulation contacts and true poses. Hardware feedback has no direct foot or head contact.'))
    print('ROLL_CONTACT_BOUNDARY', {name: (item['ready'], len(item['false_ready']), len(item['early']))
                                   for name, item in results.items()}, flush=True)


if __name__ == '__main__':
    main()
