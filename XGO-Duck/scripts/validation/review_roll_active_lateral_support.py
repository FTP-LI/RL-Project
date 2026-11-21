"""Explain saved roll failures at matched geometric angle, without stepping physics."""

import numpy as np

from review_roll_active_lateral_losses import OUT, PLAN, ROOT, read, sha
from roll_metrics import progress


def load_trace(path, verified_hashes):
    assert sha(path) == verified_hashes[path.relative_to(ROOT).as_posix()]
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ('quaternion', 'ground_contact', 'active')}


def contributions(trace):
    geometric, supported, rotation = progress(trace)
    positive = np.maximum(np.diff(geometric, axis=0, prepend=0.), 0.)
    flat = np.clip((.866 - np.abs(rotation[..., 2, 1])) / (.866 - .5), 0., 1.)
    flat = flat**2 * (3. - 2. * flat)
    ground = trace['ground_contact']
    airborne = np.cumsum(positive * ~ground, axis=0)
    lateral = np.cumsum(positive * ground * (1. - flat), axis=0)
    np.testing.assert_allclose(geometric - supported, airborne + lateral, atol=2e-12, rtol=0.)
    return tuple(np.rad2deg(data) for data in (geometric, supported, airborne, lateral))


def branch(folder, shared, role, analysis, first, last, dt, threshold):
    suffix = load_trace(folder/'clips'/role/'trace.npz', analysis['source_sha256'])
    trace = {key: np.concatenate((value, suffix[key][1:]))[first:last + 1]
             for key, value in shared.items()}
    geometric, supported, airborne, lateral = contributions(trace)
    rows = {}
    for score in analysis['branches'][role]['roll_outcomes']:
        world, hold_end = score['env_index'], score['hold_end_seconds']
        active = trace['active'][:, world].copy()
        if hold_end is not None:
            active[round(hold_end / dt) + 1:] = False
        np.testing.assert_allclose(
            [geometric[active, world].max(initial=0.), supported[active, world].max(initial=0.)],
            [score['geometric_max_deg'], score['supported_max_deg']], atol=2e-9, rtol=0.)
        crossing = np.flatnonzero(active & (geometric[:, world] >= threshold))
        if crossing.size:
            k = crossing[0]
            rows[world] = dict(seconds=float(k * dt), geometric_deg=geometric[k, world],
                supported_deg=supported[k, world], airborne_loss_deg=airborne[k, world],
                lateral_loss_deg=lateral[k, world])
    return rows


def main():
    import json

    analysis_path, loss_path = OUT/'analysis.json', OUT/'stage-loss-review.json'
    analysis, loss = read(analysis_path), read(loss_path)
    assert loss['analysis_sha256'] == sha(analysis_path)
    assert analysis['protocol_sha256'] == loss['protocol_sha256'] == sha(PLAN)
    plan = read(PLAN)
    folder = ROOT/analysis['record']
    shared = load_trace(folder/'shared-trace.npz', analysis['source_sha256'])
    criteria = plan['roll']['criteria']
    first = plan['initial_stand_controls']
    last, dt = first + plan['roll_controls'], criteria['step_dt']
    threshold = 360. - criteria['completion_tolerance_deg']
    reference = branch(folder, shared, 'reference', analysis, first, last, dt, threshold)
    groups = {}
    for role in ('left-push', 'right-push'):
        disturbed = branch(folder, shared, role, analysis, first, last, dt, threshold)
        lost = [row['world'] for row in loss['groups'][role]['roll_lost']]
        assert all(world in reference for world in lost)
        matched = [dict(world=world, reference=reference[world], disturbed=disturbed[world],
            **{f'delta_{key}': disturbed[world][key] - reference[world][key]
               for key in ('airborne_loss_deg', 'lateral_loss_deg')})
            for world in lost if world in disturbed]
        groups[role] = dict(lost=len(lost), without_geometric_crossing=[w for w in lost if w not in disturbed],
            matched=matched, median_delta_deg={key: float(np.median([row[f'delta_{key}_loss_deg']
                for row in matched])) for key in ('airborne', 'lateral')} if matched else {})
    result = dict(status='verified-saved-trace-support-decomposition', physical_substeps=0, heldout_used=False,
        threshold_deg=threshold, analysis_sha256=sha(analysis_path), stage_loss_sha256=sha(loss_path),
        protocol_sha256=sha(PLAN), comparison='first geometric threshold crossing independently in each branch',
        groups=groups)
    target = OUT/'support-loss-review.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('ROLL_ACTIVE_LATERAL_SUPPORT', {role: (len(group['matched']), group['without_geometric_crossing'],
        group['median_delta_deg']) for role, group in groups.items()})


if __name__ == '__main__':
    main()
