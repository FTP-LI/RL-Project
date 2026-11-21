"""Decompose the original supported-roll score on saved nominal failure scenes."""

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL


def saved_window(folder, chunks, first, last):
    names = ('quaternion', 'ground_contact', 'head_contact', 'foot_contact', 'roll_phase')
    values = {name: [] for name in names}
    next_control = first
    for part in chunks:
        start, stop = max(first, part['first']), min(last, part['last'])
        if start > stop:
            continue
        path = folder / part['file']
        assert start == next_control and sha(path) == part['sha256']
        span = slice(start - part['first'], stop - part['first'] + 1)
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['control'][span], np.arange(start, stop + 1))
            for name in names:
                values[name].append(data[name][span])
        next_control = stop + 1
    assert next_control == last + 1
    return {name: np.concatenate(segments) for name, segments in values.items()}


def losses(raw, first, destinations):
    r = rotation(raw['quaternion'])
    lateral = np.maximum(np.linalg.norm(r[..., :2, 1], axis=-1), 1e-6)
    forward = (r[..., 0, 0] * r[..., 1, 1] - r[..., 1, 0] * r[..., 0, 1]) / lateral
    pitch = np.arctan2(-r[..., 2, 0], forward)
    delta = (np.diff(pitch, axis=0, prepend=pitch[:1]) + np.pi) % (2 * np.pi) - np.pi
    flat = np.clip((.866 - abs(r[..., 2, 1])) / (.866 - .5), 0., 1.)
    weight = flat**2 * (3. - 2. * flat)
    contact = raw['ground_contact']
    positive = np.maximum(delta, 0.)
    ungrounded = positive * ~contact
    lateral_discount = positive * contact * (1. - weight)
    supported = np.cumsum(np.where(delta < 0, delta, positive * weight * contact), axis=0)
    geometric = np.cumsum(delta, axis=0)
    rolling = np.concatenate((np.zeros_like(raw['roll_phase'][:1], dtype=bool), raw['roll_phase'][:-1] == ROLL))
    records = []
    for command, case in destinations:
        world = case['world']
        mask = rolling[:, world]
        peak = int(np.argmax(np.where(mask, supported[:, world], -np.inf)))
        np.testing.assert_allclose(np.rad2deg(supported[peak, world]), case['supported_peak_deg'], atol=1e-6, rtol=0.)
        np.testing.assert_allclose(np.rad2deg(np.max(geometric[mask, world])), case['geometric_peak_deg'], atol=1e-6, rtol=0.)
        degrees = np.rad2deg(ungrounded[:peak + 1, world])
        skipped = np.flatnonzero(degrees > 0.)
        no_contact = float(degrees.sum())
        discounted = float(np.rad2deg(lateral_discount[:peak + 1, world].sum()))
        np.testing.assert_allclose(no_contact + discounted,
            np.rad2deg(geometric[peak, world] - supported[peak, world]), atol=1e-6, rtol=0.)
        head = np.flatnonzero(raw['head_contact'][:peak + 1, world])
        on_ground = np.flatnonzero(raw['foot_contact'][:peak + 1, world].all(axis=-1) & (np.arange(peak + 1) > (head[0] if head.size else 0)))
        records.append(dict(command=command, world=world, supported_peak_control=first + peak,
            uncounted_no_contact_deg=no_contact, uncounted_lateral_discount_deg=discounted,
            skipped_controls=[dict(sequence_control=first + int(i), time_from_roll_s=round(int(i) * .02, 2),
                positive_deg=float(degrees[i])) for i in skipped],
            first_roll_head_contact=None if not head.size else first + int(head[0]),
            first_dual_foot_contact_after_head=None if not on_ground.size else first + int(on_ground[0])))
    return records


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_pre_roll_stand_pair']
    collection, handoff = ROOT / state['collection'], ROOT / state['handoff_review']
    raw, audit = read(collection, state['collection_sha256']), read(handoff, state['handoff_review_sha256'])
    assert audit['physical_collection_sha256'] == sha(collection)
    folder = collection.parent
    plan = read(folder / 'protocol.json', state['protocol_sha256'])
    d = plan['composition']
    first, last = d['roll_start_control'], d['roll_deadline_control']
    arms = {}
    for role, arm in audit['arms'].items():
        loss_cases = [(command, case) for command, report in arm['commands'].items() for case in report['missed_worlds']
                      if case['reason'] == 'insufficient-ground-supported-rotation']
        raw_window = saved_window(folder / role, raw['branches'][role]['chunks'], first, last)
        cases = losses(raw_window, first, loss_cases)
        arms[role] = dict(cases=cases, case_count=len(cases),
            median_missing_contact_deg=float(np.median([row['uncounted_no_contact_deg'] for row in cases])),
            median_lateral_discount_deg=float(np.median([row['uncounted_lateral_discount_deg'] for row in cases])))
    output = handoff.with_name('supported-rotation-review.json')
    assert not output.exists()
    write(output, dict(contact_timescale='Actual endpoint readings every 20 ms', arms=arms,
        collection_sha256=sha(collection), handoff_sha256=sha(handoff), additional_physical_steps=0))
    print('NOMINAL_SUPPORTED_ROTATION_REVIEW', sha(output),
        {role: (arm['case_count'], arm['median_missing_contact_deg'], arm['median_lateral_discount_deg'])
         for role, arm in arms.items()})


if __name__ == '__main__':
    main()
