"""Partition the original supported-angle gap at saved contact events."""
from pathlib import Path

import numpy as np

from review_roll_contact_handoff import OUT
from roll_metrics import progress, summarize
from s1_reference_inputs import ROOT, read, sha
from record_io import write


EVENTS = ('head_top', 'departure', 'head_recontact', 'first_sole')
SEGMENTS = ('before_departure', 'through_recontact', 'after_recontact_to_sole')
LOSSES = ('air_loss_deg', 'lateral_loss_deg', 'total_loss_deg')


def loss_terms(geometric, credited, rotation, ground):
    # Match the frozen scorer: the end-of-control contact mask credits only
    # forward angle, while reverse angle is deducted even without support.
    positive = np.maximum(0., np.diff(geometric, axis=0, prepend=geometric[:1]))
    flatness = np.clip((.866-np.abs(rotation[..., 2, 1]))/(.866-.5), 0., 1.)
    flatness = flatness**2*(3.-2.*flatness)
    air = np.rad2deg(np.cumsum(positive*~ground, axis=0))
    lateral = np.rad2deg(np.cumsum(positive*ground*(1.-flatness), axis=0))
    np.testing.assert_allclose(air+lateral, np.rad2deg(geometric-credited), atol=1e-8, rtol=0.)
    return air, lateral


def episode(trace, angles, losses, old, event):
    world = old['world']
    assert event['env_index'] == world and event['success'] == old['success']
    assert old['return_support'] == 'head' and old['first_foot'] is not None
    times = trace['time']
    head_time = event['head_contact_seconds']
    ordered = (head_time, old['departure']['time_s'], old['return']['time_s'], old['first_foot']['time_s'])
    dt = times[1]-times[0]
    h, d, r, f = [round(value/dt) for value in ordered]
    np.testing.assert_allclose(times[[h, d, r, f]], ordered, atol=1e-8, rtol=0.)
    assert 0 < h < d < r-1 < f and trace['active'][:f+1, world].all()
    head = trace['head_contact'][:, world]
    feet = trace['foot_contact'][:, world].any(axis=-1)
    ground = trace['ground_contact'][:, world]
    assert head[[h, d, r]].all() and ground[[h, d, r, f]].all()
    assert not ground[d+1:r].any() and not feet[d:f].any() and feet[f]
    geometric, credited = (value[:, world] for value in angles)
    air, lateral = (value[:, world] for value in losses)

    def at(k):
        return dict(time_s=float(times[k]), phase_deg=float(geometric[k]), supported_deg=float(credited[k]),
                    air_loss_deg=float(air[k]), lateral_loss_deg=float(lateral[k]),
                    total_loss_deg=float(air[k]+lateral[k]))

    def change(start, stop):
        a, b = at(start), at(stop)
        return dict(**{key: b[key]-a[key] for key in ('phase_deg', 'supported_deg', *LOSSES)},
                    air_frames=int((~ground[start+1:stop+1]).sum()),
                    head_frames=int(head[start+1:stop+1].sum()),
                    total_frames=stop-start)

    points = {label: at(k) for label, k in zip(EVENTS, (h, d, r, f), strict=True)}
    for label, previous in (('departure', 'departure'), ('head_recontact', 'return'), ('first_sole', 'first_foot')):
        np.testing.assert_allclose([points[label]['phase_deg'], points[label]['supported_deg']],
                                   [old[previous]['phase_deg'], old[previous]['supported_progress_deg']],
                                   atol=1e-8, rtol=0.)
    parts = {label: change(a, b) for label, a, b in zip(SEGMENTS, (0, d, r), (d, r, f), strict=True)}
    assert abs(sum(part['total_loss_deg'] for part in parts.values())-points['first_sole']['total_loss_deg']) < 1e-8
    return dict(world=world, success=event['success'], events=points, segments=parts, max_air_frames=r-d-1)


def difference(failed, passed):
    def compare(a, b):
        # A positive number is a loss for the failed roll in either view.
        return dict(supported_deficit_deg=b['supported_deg']-a['supported_deg'],
                    geometric_deficit_deg=b['phase_deg']-a['phase_deg'],
                    **{key: a[key]-b[key] for key in LOSSES})

    result = dict(failed_world=failed['world'], successful_world=passed['world'],
        events={key: compare(failed['events'][key], passed['events'][key]) for key in EVENTS},
        segments={key: compare(failed['segments'][key], passed['segments'][key]) for key in SEGMENTS})
    for values in (*result['events'].values(), *result['segments'].values()):
        assert abs(values['supported_deficit_deg']-values['geometric_deficit_deg']-values['total_loss_deg']) < 1e-8
    return result


def report(pairs, bucket, labels):
    draws = np.random.default_rng(7076).integers(len(pairs), size=(8192, len(pairs)))
    stats = {}
    for label in labels:
        selected = [pair[bucket][label] for pair in pairs]
        stats[label] = {}
        for key in selected[0]:
            values = np.asarray([row[key] for row in selected])
            stats[label][key] = dict(median=float(np.median(values)), mean=float(values.mean()),
                minimum=float(values.min()), maximum=float(values.max()), positive=int((values > 0).sum()),
                median_bootstrap_95=np.quantile(np.median(values[draws], axis=1), [.025, .975]).tolist())
    return stats


def main():
    transfer_path = OUT/'development-foot-transfer.json'
    transfer = read(transfer_path)
    target = OUT/'development-head-recontact.json'
    assert not target.exists(), 'Preserve the original mechanism receipt'
    assert transfer['status'] == 'passed-saved-development-data-only' and not transfer['heldout_used']
    sources = {transfer_path.relative_to(ROOT).as_posix(): sha(transfer_path)}
    cohorts = {}
    for role, seed in (('development-a', 1020), ('development-b', 1021)):
        proof_path = OUT/f'{role}-analysis.json'
        proof = read(proof_path)
        assert proof['status'] == 'passed' and not proof['heldout_used']
        folder = ROOT/proof['evaluator']
        plan_path, trace_path = folder/'protocol.json', folder/'forward-roll-trace.npz'
        for path in (proof_path, plan_path, trace_path):
            name = path.relative_to(ROOT).as_posix()
            assert sha(path) == transfer['source_sha256'][name], name
            sources[name] = transfer['source_sha256'][name]
        plan = read(plan_path)
        assert plan['seed'] == seed and sources[trace_path.relative_to(ROOT).as_posix()] == proof['trace_sha256']
        keys = ('time', 'active', 'height', 'quaternion', 'head_quaternion', 'angular_velocity',
                'foot_contact', 'head_contact', 'ground_contact', 'head_penetration_m', 'joint_envelope_ok')
        with np.load(trace_path) as saved:
            trace = {name: saved[name] for name in keys}
        score = summarize(trace, plan['criteria'])
        assert score['attempts'] == 128 and score['successes'] == proof['successes']
        old = transfer['cohorts'][role]
        assert [row['success'] for row in score['episodes']] == [row['success'] for row in proof['rows']]
        geometric, supported, rotation = progress(trace)
        losses = loss_terms(geometric, supported, rotation, trace['ground_contact'])
        angles = np.rad2deg(geometric), np.rad2deg(supported)
        indexed = {row['world']: row for row in old['rows']}
        pairs, worlds = [], []
        for pair in old['matched_head_recontacts']:
            a, b = pair['failed_world'], pair['successful_world']
            failed, passed = (episode(trace, angles, losses, indexed[i], score['episodes'][i]) for i in (a, b))
            assert not failed['success'] and passed['success']
            worlds.extend((failed, passed))
            pairs.append(difference(failed, passed))
        events, segments = report(pairs, 'events', EVENTS), report(pairs, 'segments', SEGMENTS)
        cohorts[role] = dict(seed=seed, original_attempts=128, original_successes=score['successes'],
                             pairs=pairs, worlds=worlds, events=events, segments=segments)
        print('ROLL_RECONTACT', seed, 'pairs', len(pairs),
              'event_medians', {key: (v['supported_deficit_deg']['median'], v['total_loss_deg']['median']) for key, v in events.items()},
              'air_loss_means', {key: v['air_loss_deg']['mean'] for key, v in segments.items()}, flush=True)
    assert [len(row['pairs']) for row in cohorts.values()] == [6, 8]
    write(target, dict(status='passed-saved-development-data-only', cohorts=cohorts,
          original_scores_unchanged=True, matched_pairs=14, new_physical_steps=0, new_learning_updates=0,
          heldout_used=False, source_sha256=sources, script_sha256=sha(Path(__file__)),
          scope='Paired descriptive differences under the original 20 ms end-of-control support rule. The bootstrap resamples matched pairs within a single development seed.'))


if __name__ == '__main__':
    main()
