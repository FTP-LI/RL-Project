"""Verify the contact clock independently of collection and original scoring."""
from pathlib import Path

import numpy as np

from review_roll_contact_handoff import greatest_air_gap
from roll_metrics import progress
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def aligned_rotation(trace, substeps, meta, spec):
    lo, hi = (round(t/spec['physics_dt']) for t in spec['capture_seconds'])
    clocks = substeps['substep']
    np.testing.assert_array_equal(clocks, np.arange(lo+1, hi+1))
    assert meta['complete'] and meta['recorded_substeps'] == len(clocks)
    assert (meta['first_substep'], meta['last_substep']) == (int(clocks[0]), int(clocks[-1]))
    factor = round(spec['control_dt']/spec['physics_dt'])
    assert lo % factor == hi % factor == 0 and factor == 5
    a, b = lo//factor, hi//factor
    before, after = (substeps[key] for key in ('before', 'after'))
    assert before.shape == after.shape == (hi-lo, spec['episodes'], 4)
    np.testing.assert_array_equal(before[1:], after[:-1])
    np.testing.assert_array_equal(before[::factor], trace['qpos'][a:b, :, 3:7])
    np.testing.assert_array_equal(after[factor-1::factor], trace['qpos'][a+1:b+1, :, 3:7])
    raw = np.concatenate((before[:1], after))
    angle, _, _ = progress(dict(quaternion=raw, ground_contact=np.ones(raw.shape[:2], dtype=bool)))
    original, _, _ = progress(trace)
    np.testing.assert_allclose(np.diff(angle[::factor], axis=0), np.diff(original[a:b+1], axis=0), atol=1e-6, rtol=0.)
    return original, np.rad2deg(np.maximum(0., np.diff(angle, axis=0))), lo, hi


def characterize(trace, substeps, meta, score, spec):
    assert meta['labels'] == ['ground', 'head', 'left_foot', 'right_foot']
    force, points = (substeps[name] for name in ('normal_force_n', 'points'))
    expected = (len(substeps['substep']), spec['episodes'], len(meta['labels']))
    assert force.shape == points.shape == expected and np.isfinite(force).all()
    assert (force >= 0.).all() and (points >= 0).all() and not force[points == 0].any()
    assert (force[..., 0] >= force[..., 1:].sum(-1)-1e-8).all()
    assert (points[..., 0] >= points[..., 1:].sum(-1)).all()
    threshold = meta['force_threshold_n']
    assert threshold == spec['load_threshold_nominal_weight_fraction']*meta['nominal_weight_n'] > 0.
    geometric, raw_forward, lo, hi = aligned_rotation(trace, substeps, meta, spec)
    increments = np.maximum(0., np.diff(geometric, axis=0, prepend=geometric[:1]))
    dt, physics_dt = spec['control_dt'], spec['physics_dt']
    factor = round(dt/physics_dt)
    deadline = round(spec['criteria']['deadline_seconds']/dt)
    raw_normal = force[..., 0]
    rows = []
    for i, event in enumerate(score['rows']):
        assert i == event['env_index']
        row = dict(world=i, success=event['success'], covered=False)
        rows.append(row)
        if event['head_contact_seconds'] is None:
            row['reason'] = 'no-qualified-head-contact'
            continue
        head = round(event['head_contact_seconds']/dt)
        feet = trace['foot_contact'][head+1:deadline+1, i].any(-1)
        found = np.flatnonzero(feet & trace['active'][head+1:deadline+1, i])
        if not len(found):
            row['reason'] = 'no-post-head-sole-contact'
            continue
        foot = head+1+int(found[0])
        gap = greatest_air_gap(trace, i, head, foot, geometric, increments)
        if not gap:
            row['reason'] = 'no-control-sampled-gap'
            continue
        k = round(gap['start_seconds']/dt)
        start, stop = (k-1)*factor, (k-1+gap['air_steps'])*factor
        row['gap'] = gap
        if not lo <= start < stop <= hi:
            row['reason'] = 'gap-outside-predeclared-clock-window'
            continue
        region = slice(start-lo, stop-lo)
        loaded = raw_normal[region, i] > threshold
        turns = raw_forward[region, i]
        coarse_angle = np.rad2deg(increments[k:k+gap['air_steps'], i].sum())
        assert turns.sum()+1e-4 >= coarse_angle and abs(coarse_angle-gap['forward_angle_lost_deg']) < 1e-5
        row.update(covered=True, physics_samples=len(loaded), loaded_substeps=int(loaded.sum()),
            uncovered_normal_fraction=float((~loaded).mean()),
            raw_supported_forward_deg=float(turns[loaded].sum()), raw_unloaded_forward_deg=float(turns[~loaded].sum()),
            raw_forward_deg=float(turns.sum()),
            maximum_normal_n=float(raw_normal[region, i].max()),
            positive_any_contact_points=int((points[region, i, 0] > 0).sum()))
    return rows


def check_alignment_fixture():
    # Known 10-degree unsupported control interval. Two of its ten physics
    # steps have positive force, which the coarse sensor cannot distinguish.
    phase = np.deg2rad(np.arange(31))/2
    quaternion = np.zeros((31, 1, 4), dtype=np.float64)
    quaternion[:, 0, 0], quaternion[:, 0, 2] = np.cos(phase), np.sin(phase)
    ground = np.asarray([1, 1, 1, 0, 0, 1, 1], dtype=bool)[:, None]
    head = np.asarray([0, 1, 1, 0, 0, 1, 0], dtype=bool)[:, None]
    foot = np.zeros((7, 1, 2), dtype=bool)
    foot[-1] = True
    qpos = np.zeros((7, 1, 7))
    qpos[..., 3:7] = quaternion[::5]
    trace = dict(time=np.arange(7)*.02, quaternion=quaternion[::5], qpos=qpos,
                 ground_contact=ground, head_contact=head, foot_contact=foot,
                 active=np.ones_like(ground), angular_velocity=np.zeros((7, 1, 3)))
    force = np.zeros((30, 1, 4))
    force[10:12, 0, :2] = 1.
    steps = dict(substep=np.arange(1, 31), before=quaternion[:-1], after=quaternion[1:],
                 points=(force > 0).astype(int), normal_force_n=force)
    meta = dict(labels=['ground', 'head', 'left_foot', 'right_foot'], force_threshold_n=.05,
                nominal_weight_n=5., complete=True, recorded_substeps=30, first_substep=1, last_substep=30)
    spec = dict(capture_seconds=[0., .12], episodes=1, control_dt=.02, physics_dt=.004,
                load_threshold_nominal_weight_fraction=.01, criteria={'deadline_seconds': .12})
    score = dict(rows=[dict(env_index=0, success=False, head_contact_seconds=.02)])
    row, = characterize(trace, steps, meta, score, spec)
    assert row['covered'] and row['physics_samples'] == 10 and row['loaded_substeps'] == 2
    np.testing.assert_allclose([row['uncovered_normal_fraction'], row['raw_supported_forward_deg'],
                                row['raw_unloaded_forward_deg']], [.8, 2., 8.], atol=1e-8, rtol=0.)


def verify(spec, out, score):
    folder = ROOT/score['evaluator']
    state, runtime = read(folder/'evaluation.json'), read(folder/'runtime-contract.json')
    meta = read(folder/'contact-clock.json')
    trace_path, clock_path = folder/'forward-roll-trace.npz', folder/'contact-clock.npz'
    assert sha(trace_path) == score['trace_sha256'] and sha(clock_path) == meta['sha256']
    assert state['completed_physical_substeps'] == state['attempted_physical_substeps'] == runtime['physics_substeps'] == spec['maximum_batched_substeps']
    assert score['actual_control_transitions'] == spec['maximum_control_transitions']
    assert not score['heldout_used'] and score['learning_updates'] == 0
    assert read(folder/'contact-clock-zero.json')['rng_and_state_exact']
    names = ('time', 'quaternion', 'qpos', 'head_contact', 'foot_contact', 'ground_contact', 'active', 'angular_velocity')
    with np.load(trace_path) as saved:
        trace = {name: saved[name] for name in names}
    with np.load(clock_path) as data:
        rows = characterize(trace, {key: data[key] for key in data.files}, meta, score, spec)
    limits = spec['minimum_unloaded_substep_fraction']
    groups = {}
    for label, success in (('success', True), ('failure', False)):
        selected = [row for row in rows if row['success'] == success]
        covered = [row for row in selected if row['covered']]
        groups[label] = dict(total=len(selected), covered=len(covered),
            sustained_loss=sum(row['uncovered_normal_fraction'] >= limits for row in covered),
            some_raw_load=sum(row['loaded_substeps'] > 0 for row in covered),
            median_unloaded_fraction=float(np.median([row['uncovered_normal_fraction'] for row in covered])) if covered else None,
            median_raw_supported_forward_deg=float(np.median([row['raw_supported_forward_deg'] for row in covered])) if covered else None)
    failed = groups['failure']
    if failed['covered'] < spec['minimum_covered_failures']:
        decision = 'insufficient-covered-failures'
    elif failed['sustained_loss']/failed['covered'] >= spec['minimum_fraction_failures_with_sustained_loss']:
        decision = 'prioritize-physical-support-loss'
    else:
        decision = 'inspect-short-contact-and-clock-alignment-before-modifying-policy'
    result = dict(status='passed-independent-diagnostic-verification', decision=decision, record=score['evaluator'],
        groups=groups, rows=rows, recorded_substeps=meta['recorded_substeps'], force_threshold_n=meta['force_threshold_n'],
        original_score=dict(successes=score['successes'], attempts=score['episodes'], wilson95=score['wilson95']),
        original_score_unchanged=True, trace_sha256=score['trace_sha256'], substeps_sha256=meta['sha256'],
        verifier_sha256=sha(Path(__file__)), actual_control_transitions=score['actual_control_transitions'],
        physical_substeps=runtime['physics_substeps'], complete_video=score['complete_video'],
        decoded_frames=score['decoded_frames'], heldout_used=False, learning_updates=0)
    write(out/'analysis.json', result)
    print('ROLL_CONTACT_CLOCK_VERIFIED', decision, groups, flush=True)
    return result
