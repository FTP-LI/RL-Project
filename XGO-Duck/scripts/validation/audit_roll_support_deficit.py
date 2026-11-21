"""Recount the support-weighted turn deficit before the first geometric 330-degree crossing."""
from pathlib import Path
import json

import numpy as np

from roll_metrics import progress, rotation
from summarize_roll_factor_continuation import ROOT, OUT, read, sha


def angular_stats(rows):
    available = [r for r in rows if r['first_geometric_threshold_control'] is not None]
    keys = ('loss_without_contact_degrees', 'loss_with_contact_lateral_degrees', 'lost_after_head_degrees',
            'total_loss_degrees', 'supported_degrees')
    result = dict(worlds=len(rows), crossed_geometry_threshold=len(available),
        contact_gap_dominant=sum(r[keys[0]] > r[keys[1]] for r in available),
        post_head_loss_dominant=sum(r[keys[2]] > .5 * r[keys[3]] for r in available))
    result.update({f'median_{key}':float(np.median([r[key] for r in available])) for key in keys} if available else {})
    return result


def main():
    source = OUT / 'matched-prefix-analysis.json'
    target = OUT / 'support-deficit-audit.json'
    if target.exists():
        raise FileExistsError('Keep the existing audit')
    paired = read(source)
    if paired['status'] != 'independently-replayed-common-prefix':
        raise ValueError('A verified prefix is required')
    a, b, low = (paired['arms'][role] for role in ('nominal-a', 'nominal-b', 'friction-low'))
    selected = low['impossible_worlds']
    both = set(a['roll_certificate_worlds']) & set(b['roll_certificate_worlds']) & set(selected)
    roles = {}
    for role, arm in paired['arms'].items():
        plan = read(ROOT / arm['protocol'])
        path = ROOT / 'records/evaluations' / arm['run_id'] / 'trace.npz'
        if sha(path) != arm['trace_sha256'] or sha(ROOT / arm['protocol']) != arm['protocol_sha256']:
            raise ValueError('Evidence changed: ' + role)
        start, end = plan['initial_stand_controls'], paired['control_step']
        with np.load(path, allow_pickle=False) as data:
            trace = {name: data[name][start:end+1, selected] for name in
                     ('quaternion', 'head_quaternion', 'ground_contact', 'head_contact', 'foot_contact', 'active')}
        geometric, supported, matrix = progress(trace)
        positive = np.maximum(np.diff(geometric, axis=0, prepend=0.), 0.)
        lateral = np.clip((.866-np.abs(matrix[...,2,1]))/(.866-.5), 0., 1.)
        ground = trace['ground_contact']
        gap_contact = positive * ~ground
        gap_lateral = positive * ground * (1.-lateral*lateral*(3.-2.*lateral))
        loss = gap_contact + gap_lateral
        np.testing.assert_allclose(np.cumsum(loss, axis=0), geometric-supported, atol=1e-10, rtol=0.)
        min_turn = np.deg2rad(360 - plan['roll']['criteria']['completion_tolerance_deg'])
        feet = np.logical_and(trace['foot_contact'].all(axis=-1), trace['active'])
        previously_on_feet = np.pad(np.logical_or.accumulate(feet[:-1], axis=0), ((1,0),(0,0)))
        head = (previously_on_feet & trace['active'] & trace['head_contact'] & (rotation(trace['head_quaternion'])[...,2,1] < -.3)
                & (geometric > np.deg2rad(20.)) & (geometric < np.deg2rad(170.)))
        rows = []
        for j, world in enumerate(selected):
            reached = np.flatnonzero(geometric[:,j] >= min_turn)
            hits = np.flatnonzero(head[:,j])
            h = int(hits[0]) if len(hits) else None
            row = dict(world=world, first_qualifying_head_contact_control=None if h is None else start+h,
                       first_geometric_threshold_control=None)
            if len(reached):
                k = int(reached[0])
                if h is None or h >= k:
                    raise ValueError('Expected qualifying head contact before geometric rotation: '+role+'/'+str(world))
                row.update(first_geometric_threshold_control=start+k,
                    supported_degrees=float(np.rad2deg(supported[k,j])),
                    total_loss_degrees=float(np.rad2deg(loss[:k+1,j].sum())),
                    loss_without_contact_degrees=float(np.rad2deg(gap_contact[:k+1,j].sum())),
                    loss_with_contact_lateral_degrees=float(np.rad2deg(gap_lateral[:k+1,j].sum())),
                    lost_after_head_degrees=float(np.rad2deg(loss[h+1:k+1,j].sum())))
            rows.append(row)
        roles[role] = dict(all_low_exclusions=angular_stats(rows), both_nominals_certified=angular_stats([r for r in rows if r['world'] in both]), rows=rows)
    result = dict(status='read-only-support-deficit-decomposition', source_sha256=sha(source),
        script_sha256=sha(Path(__file__)), compared_selected_worlds=selected, both_nominal_certified_worlds=sorted(both),
        scope='Selection by observed low-friction roll exclusion, descriptive diagnostic only; losses are terms of the existing score, not identified force or contact causes.',
        event='Start of the roll to the first common definition of reaching the geometric minimum, if reached in each arm',
        roles=roles, new_physics_steps=0, new_learning_updates=0, hardware_io=False)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print('ROLL_SUPPORT_DEFICIT_AUDIT', json.dumps({k:v['both_nominals_certified'] for k,v in roles.items()}))


if __name__ == '__main__':
    main()
