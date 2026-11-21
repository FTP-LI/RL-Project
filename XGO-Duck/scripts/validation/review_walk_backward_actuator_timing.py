"""Trace when the last actuator perturbation did and did not create a step."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from review_walk_backward_support_transfer import first_support


OUT = ROOT / 'records/reports/common-walk-backward-actuator-support'
PLAN = ROOT / 'configs/experiments/common_walk_backward_actuator_support.json'
FIELDS = ('active', 'contact', 'velocity', 'joint_positions', 'applied_joint_targets', 'foot_positions')


def checked_trace(path, digest):
    assert sha(path) == digest
    with np.load(path) as archive:
        return {field: archive[field] for field in FIELDS}


def timing(trace, world, fork, deadline, onset, pitch_joints):
    stop = min(len(trace['active']), 50*onset) if onset else len(trace['active'])
    contact = trace['contact'][:, world]
    active = trace['active'][:, world]
    first = first_support(trace, world, fork, stop)
    first_after_onset = first_support(trace, world, stop, len(active))
    stance = lambda row: dict(start_control=int(row[0] + 1), confirmed_control=int(row[0] + 4),
                              side=['left', 'right'][row[1]]) if row else None
    forward = -trace['velocity'][:, world, 0]
    peak = int(np.argmax(forward[fork:stop])) + fork if stop > fork else None
    one = np.flatnonzero(active[fork:stop] & (contact[fork:stop].sum(axis=-1) == 1))
    around = slice(fork, deadline)
    return dict(
        onset_block_end_seconds=onset,
        first_single_support_before_onset_or_end=stance(first),
        first_single_support_after_onset=stance(first_after_onset),
        single_foot_samples_by_second=((contact.sum(axis=-1) == 1) & active).reshape(-1, 50).sum(1).tolist(),
        first_single_foot_sample_control=int(fork + one[0] + 1) if len(one) else None,
        max_backward_speed_before_onset_or_end_m_s=float(forward[peak]) if peak is not None else None,
        max_backward_speed_control=peak + 1 if peak is not None else None,
        early_pitch_joint_excursion_rad=np.ptp(trace['joint_positions'][around, world][:, pitch_joints], axis=0).tolist(),
        early_pitch_target_excursion_rad=np.ptp(trace['applied_joint_targets'][around, world][:, pitch_joints], axis=0).tolist(),
        early_maximum_height_asymmetry_m=float(np.max(abs(np.diff(trace['foot_positions'][around, world, :, 2], axis=-1)))),
    )


def review(seed, spec, combined):
    receipt = OUT / f'seed{seed}-analysis.json'
    proof = read(receipt)
    assert sha(receipt) == combined['sources'][receipt.name]
    assert proof['status'] == 'passed-integrity' and proof['protocol_sha256'] == sha(PLAN)
    assert not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT / proof['record']
    assert sha(record / 'protocol.json') == sha(PLAN)
    prefix = checked_trace(record / 'trace-shared-prefix.npz', proof['shared_prefix_sha256'])
    fork = spec['phase']['warmup_controls']
    deadline = fork + spec['support']['support_deadline_controls']
    joints = spec['environment']['joint_names']
    pitch_joints = [joints.index(f'{side}_{axis}') for side in ('left', 'right') for axis in ('hip_pitch', 'knee')]
    rescues, losses = (proof['directions']['body-right'][label] for label in ('rescued', 'lost'))
    worlds = sorted(set(proof['common_near_static'] + losses))
    data = {str(world): dict(was_stalled=world in proof['common_near_static'],
                             rescued=world in rescues, lost=world in losses, arms={}) for world in worlds}
    for role in spec['roles']:
        post = checked_trace(record / f'trace-{role}.npz', proof['trace_sha256'][role])
        trace = {field: np.concatenate((prefix[field], post[field])) for field in FIELDS}
        assert trace['contact'].shape == (300, 224, 2)
        for world in worlds:
            onset = proof['scores'][role]['worlds'][world]['onset_block_end_seconds']
            row = timing(trace, world, fork, deadline, onset, pitch_joints)
            early = row['first_single_support_before_onset_or_end']
            if role == 'body-right' and world in rescues:
                assert onset and onset > 2 and (early is None or early['confirmed_control'] > deadline)
            data[str(world)]['arms'][role] = row
    return dict(seed=seed, record=proof['record'], receipt_sha256=sha(receipt), worlds=data)


def main():
    combined_path = OUT / 'analysis.json'
    combined = read(combined_path)
    assert combined['status'] == 'closed-mechanism-gate-not-met' and not combined['heldout_used']
    spec = read(PLAN)
    assert combined['protocol_sha256'] == sha(PLAN)
    batches = [review(int(seed), spec, combined) for seed in spec['runs']]
    examples = [(batch['seed'], world, row['arms']['body-right']['first_single_support_before_onset_or_end'])
                for batch in batches for world, row in batch['worlds'].items() if row['rescued']]
    output = OUT / 'onset-timing-review.json'
    assert not output.exists()
    write(output, dict(status='passed-saved-timing-review', source_sha256=sha(Path(__file__)),
        combined_sha256=sha(combined_path), protocol_sha256=sha(PLAN), batches=batches, rescued_worlds=examples,
        physical_substeps=0, learning_updates=0, heldout_used=False))
    print('BACKWARD_ACTUATOR_TIMING', examples, flush=True)


if __name__ == '__main__':
    main()
