"""Compare the first post-fork yaw response with the two official repeats."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / 'records/reports/common-walk-forward-phase'
OFFSETS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 25)


def main():
    source = OUT / 'analysis.json'
    proof = read(source)
    assert proof['status'] == 'passed-integrity' and not proof['heldout_used']
    record = ROOT / proof['record']
    worlds = proof['comparison']['forward_worlds']
    roles = ('reference-a', 'reference-b', 'phase-yaw')
    data = {}
    for role in roles:
        path = record / f'trace-{role}.npz'
        assert sha(path) == proof['trace_sha256'][role]
        with np.load(path) as trace:
            assert trace['active'][:max(OFFSETS), worlds].all()
            data[role] = dict(yaw=trace['yaw_rate'][:max(OFFSETS), worlds].astype(float),
                              yaw_input=trace['observations'][:max(OFFSETS), worlds, 50].astype(float))
    a, b, c = (data[role] for role in roles)
    assert not a['yaw_input'].any() and not b['yaw_input'].any()
    command = c['yaw_input']
    sign = np.sign(command[0])
    keep = np.logical_and.accumulate(np.sign(command) == sign, axis=0) & (sign != 0)
    rows = []
    for offset in OFFSETS:
        i = offset - 1
        ref = (a['yaw'][i] + b['yaw'][i]) / 2
        repeat = abs(a['yaw'][i] - b['yaw'][i])
        change = c['yaw'][i] - ref
        aligned = keep[i]
        signed = change[aligned] * sign[aligned]
        rows.append(dict(control_after_fork=offset, elapsed_seconds=offset*.02,
            median_absolute_repeat_spread=float(np.median(repeat)),
            median_absolute_candidate_change=float(np.median(abs(change))),
            median_absolute_yaw_gain=float(np.median(abs(ref)-abs(c['yaw'][i]))),
            same_input_sign_worlds=int(aligned.sum()),
            same_sign_command_response_median=None if not signed.size else float(np.median(signed)),
            same_sign_command_response_positive=int((signed > 0.).sum())))
    write(OUT / 'early-response.json', dict(status='passed-saved-data-only', source_sha256=sha(source),
        script_sha256=sha(Path(__file__)), rows=rows, world_order=worlds,
        first_yaw_input_rad_s=command[0].tolist(),
        data={role: {key: value.tolist() for key, value in entry.items()} for role, entry in data.items()},
        scope='Physical yaw after the first changed control from a common fork. The signed response uses only worlds where the internal command has not switched sign. Closed-loop branches are not an isolated impulse.',
        new_physical_steps=0, learning_updates=0, heldout_used=False))


if __name__ == '__main__':
    main()
