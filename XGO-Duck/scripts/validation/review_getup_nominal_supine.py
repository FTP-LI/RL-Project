"""Compare failed and successful specified supine releases without advancing physics."""
from pathlib import Path
import sys

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity

sys.path.insert(0, str(ROOT/'scripts/training'))
from record_io import write


REPORTS = (
    'getup-early-support-ppo-functional',
    'getup-side-to-supine-replay/evaluation',
)
ROLES = ('fixed-weighted', 'control', 'candidate')
FIELDS = ('time', 'quaternion', 'foot_contact', 'native_head_leg_contact', 'height')


def summarize_trace(trace, worlds, gate):
    time = trace['time']
    gravity = projected_gravity(trace['quaternion'][:, worlds])
    tilt = np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1., 1.)))
    feet = trace['foot_contact'][:, worlds].sum(axis=-1)
    head = trace['native_head_leg_contact'][:, worlds]
    windows = ((2., 2.3), (2.3, 3.), (3., 4.), (8., 11.))
    result = {}
    for index, world in enumerate(worlds):
        sections = {}
        for start, end in windows:
            selected = (time > start + 1e-7) & (time <= end + 1e-7)
            sections[f'{start:g}-{end:g}s'] = dict(
                samples=int(selected.sum()), minimum_tilt_deg=float(tilt[selected, index].min()),
                maximum_height_m=float(trace['height'][selected, world].max()),
                one_foot=int((feet[selected, index] == 1).sum()),
                both_feet=int((feet[selected, index] == 2).sum()),
                head_or_leg_contact=int(head[selected, index].sum()),
            )
        after_release = time > 2. + 1e-7
        crossing = {}
        for degrees in (90, 60, 30, gate):
            hit = np.flatnonzero(after_release & (tilt[:, index] <= degrees))
            crossing[str(degrees)] = float(time[hit[0]] - 2.) if len(hit) else None
        release = np.argmin(abs(time - 2.))
        result[str(world)] = dict(release_tilt_deg=float(tilt[release, index]),
                                 first_crossing_after_release_seconds=crossing, windows=sections)
    return result


def review(folder):
    source = ROOT/'records/reports'/folder/'analysis.json'
    proof = read(source)
    record = ROOT/proof['record']
    assert proof['status'] == 'passed-independent-verification'
    assert proof['learning_updates'] == 0 and not proof['heldout_used']
    assert proof['protocol_sha256'] == sha(record/'protocol.json')
    criteria = read(record/'protocol.json')['environment']['criteria']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix) == proof['prefix_sha256']
    with np.load(prefix) as archive:
        initial = {field: archive[field] for field in FIELDS}
    scores = {}
    worlds = None
    for role in ROLES:
        arm = proof['arms'][role]
        episodes = arm['timing']['episodes']
        specified = [ep['env_index'] for ep in episodes if ep['requested_pose'] == 'supine']
        assert len(specified) == 32 and all(episodes[world]['release_pose'] == 'supine' for world in specified)
        if worlds is None:
            worlds = specified
        assert worlds == specified
        scores[role] = np.array([episodes[world]['success'] and arm['end_hold']['stable_by_environment'][world]
                                 for world in worlds], dtype=bool)
        assert scores[role].sum() == arm['timing']['requested']['supine']['timely_holds']
    common = np.logical_and.reduce(list(scores.values()))
    failed = ~np.logical_or.reduce(list(scores.values()))
    focus = np.asarray(worlds)[~common]
    arms = {}
    for role in ROLES:
        trace_path = record/f'trace-{role}.npz'
        assert sha(trace_path) == proof['arms'][role]['trace_sha256']
        with np.load(trace_path) as archive:
            trace = {field: np.concatenate((initial[field], archive[field])) for field in FIELDS}
        assert trace['time'].shape == (550,)
        samples = summarize_trace(trace, worlds, criteria['tilt_max_deg'])
        successful = [sample for success, sample in zip(scores[role], samples.values(), strict=True) if success]
        arms[role] = dict(
            failures=[int(world) for world, success in zip(worlds, scores[role], strict=True) if not success],
            paired_worlds={str(world): samples[str(world)] for world in focus},
            success_median_seconds_to_60deg=float(np.median([
                sample['first_crossing_after_release_seconds']['60'] for sample in successful])),
        )
    return dict(record=proof['record'], evaluation_sha256=sha(source), all_three_failed=np.asarray(worlds)[failed].tolist(),
                all_three_succeeded=int(common.sum()), arms=arms)


def main():
    output = ROOT/'records/reports/getup-side-to-supine-replay/evaluation/nominal-supine-review.json'
    assert not output.exists()
    batches = {folder: review(folder) for folder in REPORTS}
    write(output, dict(status='passed-saved-trace-review', batches=batches, source_sha256=sha(Path(__file__)),
                      physical_substeps=0, learning_updates=0, heldout_used=False))
    print('NOMINAL_SUPINE_REVIEW', {name: row['all_three_failed'] for name, row in batches.items()}, flush=True)


if __name__ == '__main__':
    main()
