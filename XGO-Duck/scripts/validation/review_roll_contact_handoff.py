"""Identify which airborne interval costs the most supported roll progress."""
from pathlib import Path

import numpy as np

from roll_metrics import progress, summarize
from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / 'records/reports/roll-common-confirmation/rng-recovery'


def support(trace, step, world):
    if step is None:
        return 'unobserved'
    feet = int(trace['foot_contact'][step, world].sum())
    head = bool(trace['head_contact'][step, world])
    if feet or head:
        return ('head+' if head and feet else 'head' if head else '') + (f'{feet}-foot' if feet else '')
    return 'other-body' if trace['ground_contact'][step, world] else 'air'


def greatest_air_gap(trace, world, head, last, geometric, forward_delta):
    air = ~trace['ground_contact'][head + 1:last + 1, world]
    air &= trace['active'][head + 1:last + 1, world]
    changes = np.diff(air.astype(np.int8), prepend=0, append=0)
    start = np.flatnonzero(changes == 1) + head + 1
    stop = np.flatnonzero(changes == -1) + head + 1
    if not len(start):
        return None
    losses = [float(np.rad2deg(forward_delta[a:b, world].sum())) for a, b in zip(start, stop, strict=True)]
    i = int(np.argmax(losses))
    a, b = int(start[i]), int(stop[i])
    landed = b if b <= last and trace['active'][b, world] and trace['ground_contact'][b, world] else None
    return dict(air_steps=b-a, start_seconds=float(trace['time'][a]), start_after_head_seconds=float(trace['time'][a] - trace['time'][head]),
        last_supported_type=support(trace, a - 1, world), return_support_type=support(trace, landed, world),
        before_takeoff_deg=float(np.rad2deg(geometric[a - 1, world])),
        last_airborne_deg=float(np.rad2deg(geometric[b - 1, world])),
        takeoff_pitch_rate_rad_s=float(trace['angular_velocity'][a - 1, world, 1]), forward_angle_lost_deg=losses[i],
        all_airborne_loss_deg=sum(losses))


def batch(role, timing):
    proof_file = OUT / f'{role}-analysis.json'
    proof = read(proof_file)
    assert proof['status'] == 'passed' and not proof['heldout_used']
    folder = ROOT / proof['evaluator']
    plan, path = folder / 'protocol.json', folder / 'forward-roll-trace.npz'
    criteria = read(plan)['criteria']
    assert sha(path) == proof['trace_sha256']
    with np.load(path) as values:
        trace = {key: values[key] for key in ('time', 'active', 'height', 'quaternion', 'head_quaternion',
            'angular_velocity', 'foot_contact', 'head_contact', 'ground_contact', 'head_penetration_m', 'joint_envelope_ok')}
    checked = summarize(trace, criteria)
    assert checked['attempts'] == 128 and checked['successes'] == proof['successes']
    geometric, supported, _ = progress(trace)
    forward_delta = np.maximum(0., np.diff(geometric, axis=0, prepend=geometric[:1]))
    rows = []
    for event, previous in zip(checked['episodes'], proof['rows'], strict=True):
        i = event['env_index']
        assert all(event[k] == previous[k] for k in ('success', 'head_contact_seconds', 'completion_seconds'))
        source = timing[i]
        head_time = event['head_contact_seconds']
        gap = None
        if head_time is not None:
            h = round(head_time / criteria['step_dt'])
            last = round(source['endpoint_seconds'] / criteria['step_dt'])
            np.testing.assert_allclose(np.rad2deg(min(geometric[last, i], supported[last, i])),
                                       source['endpoint_qualified_deg'], atol=1e-8, rtol=0)
            gap = greatest_air_gap(trace, i, h, last, geometric, forward_delta)
            np.testing.assert_allclose(0. if gap is None else gap['all_airborne_loss_deg'],
                                       source['air_after_head_deg'], atol=1e-8, rtol=0)
        rows.append(dict(world=i, success=event['success'], head_seconds=head_time, gap=gap, reasons=event['reasons']))
    return rows, {p.relative_to(ROOT).as_posix(): sha(p) for p in (proof_file, plan, path)}


def stats(rows):
    result = {}
    for role, success in (('success', True), ('failure', False)):
        with_head = [row for row in rows if row['success'] == success and row['head_seconds'] is not None]
        gaps = [row['gap'] for row in with_head if row['gap'] is not None]
        def median(key):
            return None if not gaps else float(np.median([row[key] for row in gaps]))
        result[role] = dict(with_head=len(with_head), with_air_gap=len(gaps),
            median_start_after_head_s=median('start_after_head_seconds'), median_air_steps=median('air_steps'),
            median_takeoff_pitch_deg=median('before_takeoff_deg'), median_takeoff_pitch_rate=median('takeoff_pitch_rate_rad_s'),
            median_largest_airborne_loss_deg=median('forward_angle_lost_deg'),
            previous_support={key: sum(row['last_supported_type'] == key for row in gaps)
                              for key in sorted({row['last_supported_type'] for row in gaps})},
            return_support={key: sum(row['return_support_type'] == key for row in gaps)
                            for key in sorted({row['return_support_type'] for row in gaps})})
    return result


def main():
    timing_file = OUT / 'development-support-timing.json'
    timing = read(timing_file)
    assert timing['status'] == 'passed-saved-development-data-only' and not timing['heldout_used']
    sources = {timing_file.relative_to(ROOT).as_posix(): sha(timing_file)}
    results = {}
    for role in ('development-a', 'development-b'):
        rows, fingerprints = batch(role, {row['env_index']: row for row in timing['rows'] if row['role'] == role})
        sources.update(fingerprints)
        result = stats(rows)
        results[role] = dict(statistics=result, rows=rows)
        print('ROLL_CONTACT_HANDOFF', role, result, flush=True)
    write(OUT / 'development-contact-handoff.json', dict(status='passed-saved-development-data-only',
        cohorts=results, source_sha256=sources, script_sha256=sha(Path(__file__)), heldout_used=False,
        scope='Largest contiguous airborne positive angle loss, using original score endpoint; contact type from archived sensors.',
        new_physical_steps=0, learning_updates=0, original_scores_unchanged=True))


if __name__ == '__main__':
    main()
