"""Read timing and actual IMU samples before the first meaningful loss of support."""

import json
from collections import Counter

import numpy as np

from review_roll_active_lateral_support import OUT, PLAN, ROOT, contributions, read, sha


def arrays(file, names, count, hashes):
    assert sha(file) == hashes[file.relative_to(ROOT).as_posix()]
    with np.load(file, allow_pickle=False) as data:
        return {key: data[key][:count] for key in names}


def first_gap(ground, airborne, start, end):
    gap = None
    for k in range(start, end+1):
        if ground[k]:
            gap = None
        elif gap is None:
            gap = k
        if gap is not None and airborne[k]-airborne[gap-1] >= 5.:
            return gap, k
    raise AssertionError('Missing a five-degree airborne support gap before crossing')


def stats(positive, negative):
    differences = positive[:, None]-negative[None, :]
    return dict(failure_median=float(np.median(positive)), success_median=float(np.median(negative)),
        rank_fraction=float((differences > 0).mean()+.5*(differences == 0).mean()))


def main():
    analysis_path, support_path = OUT/'analysis.json', OUT/'support-loss-review.json'
    analysis, support = read(analysis_path), read(support_path)
    assert support['analysis_sha256'] == sha(analysis_path)
    assert support['protocol_sha256'] == sha(PLAN)
    plan = read(PLAN)
    root = ROOT/analysis['record']/'clips'
    dt = plan['roll']['criteria']['step_dt']
    offset = analysis['shared_controls']-plan['initial_stand_controls']
    control = round(.36/dt)
    probe = control-offset
    source = analysis['source_sha256']
    names = ('gyro_x', 'gyro_pitch', 'gyro_z', 'gravity_x', 'gravity_y', 'gravity_z')
    results = {}
    for role in ('left-push', 'right-push'):
        matches = support['groups'][role]['matched']
        end = max(round(row['disturbed']['seconds']/dt)-offset+1 for row in matches)
        physical = arrays(root/role/'trace.npz', ('quaternion', 'ground_contact', 'head_contact'), end, source)
        observed = arrays(root/role/'commands.npz', ('observed',), end, source)['observed'][..., :6]
        _, _, airborne, _ = contributions(physical)
        events = []
        rolls = analysis['branches'][role]['roll_outcomes']
        for row in matches:
            world = row['world']
            head = round(rolls[world]['head_contact_seconds']/dt)-offset
            end = round(row['disturbed']['seconds']/dt)-offset
            assert physical['head_contact'][head, world] and physical['ground_contact'][head, world]
            first, total = first_gap(physical['ground_contact'][:, world], airborne[:, world], head+1, end)
            assert first > probe and physical['ground_contact'][probe, world]
            events.append(dict(world=world, first_airborne_control=first+offset, five_degree_control=total+offset,
                lead_milliseconds=int(round((first-probe)*dt*1000)), observed=observed[probe, world].tolist()))
        good = [row['env_index'] for row in rolls if row['success']]
        bad = [item['world'] for item in events]
        values = observed[probe]
        results[role] = dict(events=events, successes_used=len(good),
            gap_onset_controls=dict(Counter(item['first_airborne_control'] for item in events)),
            minimum_lead_ms=min(item['lead_milliseconds'] for item in events),
            maximum_lead_ms=max(item['lead_milliseconds'] for item in events),
            features={name: stats(values[bad, i], values[good, i]) for i, name in enumerate(names)})
    proof = dict(status='verified-descriptive-single-seed-support-signal-review', physical_substeps=0, heldout_used=False,
        protocol_sha256=sha(PLAN), analysis_sha256=sha(analysis_path), support_review_sha256=sha(support_path),
        probe_seconds=control*dt, airborne_gap_minimum_deg=5.,
        groups=results, threshold_selected=False, controller_admitted=False)
    with (OUT/'support-signal-review.json').open('x', encoding='utf-8') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')
    print('ROLL_SUPPORT_SIGNAL_TIMING', {role: (data['minimum_lead_ms'], data['maximum_lead_ms'],
          {name:round(row['rank_fraction'],3) for name,row in data['features'].items()}) for role,data in results.items()})


if __name__ == '__main__':
    main()
