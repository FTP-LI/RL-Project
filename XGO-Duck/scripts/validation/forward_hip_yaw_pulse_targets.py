"""Check a one-control hip-yaw pulse against the original joint envelope."""

from pathlib import Path

import numpy as np

from run_roll_stand_forward_diagnostic import ROOT, read, require, sha


SIDES = ('left', 'right')
ROLES = {'zero-a': 0, 'positive': 1, 'negative': -1, 'zero-b': 0}


def eligible(plan, requested, normal_target, contacts, side):
    walk = plan['walking']
    low, high = np.asarray(walk['joint_limits'], dtype=np.float32).T
    cap = plan['feedback']['maximum_joint_correction_rad']
    joint = walk['joint_names'].index(f'{SIDES[side]}_hip_yaw')
    baseline = np.clip(requested, low, high)
    require(np.isfinite(normal_target).all() and np.abs(normal_target-baseline).max() <= cap+1e-6,
            'The original projected feedback exceeds its correction cap')
    lower = np.maximum(low[joint], baseline[..., joint]-cap)
    upper = np.minimum(high[joint], baseline[..., joint]+cap)
    pulse = plan['pulse']['delta_rad']
    current = normal_target[..., joint]
    matched = contacts[..., side] & ~contacts[..., 1-side] & (current-pulse >= lower) & (current+pulse <= upper)
    return matched, joint


def apply_pulse(target, selected, joint, delta):
    changed = target.copy()
    changed[selected, joint] += np.float32(delta)
    return changed


def archived_coverage(reference, plan):
    receipt = ROOT / reference['path']
    proof = read(receipt)
    folder = ROOT / 'records/evaluations' / proof['run_id']
    manifest = read(folder / 'guarded-evaluation.json')
    commands_path, trace_path = (folder / name for name in ('commands.npz', 'trace.npz'))
    require(sha(receipt) == reference['sha256'] and sha(commands_path) == manifest['command_sha256']
            and sha(trace_path) == manifest['trace_sha256'] and proof['full_route'] and not proof['holdout_used'],
            'Archived state or command hash changed')
    pulse = plan['pulse']
    start = pulse['earliest_fork_control']
    stop = pulse['shared_controls']
    window = slice(start, stop)
    results = {}
    with np.load(commands_path, allow_pickle=False) as commands, np.load(trace_path, allow_pickle=False) as trace:
        requested = commands['requested_targets'][window]
        applied = commands['applied_targets'][window]
        contacts = trace['foot_contact'][window]
        for side, label in enumerate(SIDES):
            cells, joint = eligible(plan, requested, applied, contacts, side)
            counts = cells.sum(axis=1)
            candidates = np.flatnonzero(counts >= pulse['minimum_worlds_per_fork'])
            require(len(candidates), f'No archived common fork has enough {label} support worlds')
            k = int(candidates[0])
            idx = np.flatnonzero(cells[k])
            positive = apply_pulse(applied[k], idx, joint, pulse['delta_rad'])
            negative = apply_pulse(applied[k], idx, joint, -pulse['delta_rad'])
            np.testing.assert_array_equal(positive[:, np.arange(applied.shape[-1]) != joint],
                                          negative[:, np.arange(applied.shape[-1]) != joint])
            np.testing.assert_allclose(positive[idx, joint]-negative[idx, joint], 2*pulse['delta_rad'], atol=2e-7, rtol=0)
            results[label] = dict(first_eligible_control=start+k, count=int(counts[k]), maximum=int(counts.max()))
    return dict(seed=plan['seed'], friction=plan['parameter_factors']['foot_friction'], support=results,
                sources={path.relative_to(ROOT).as_posix(): sha(path) for path in (receipt, commands_path, trace_path)})
