"""Compare each routed development trajectory with its selected pure branch."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from evaluate_getup_pose_router import OUT, PLAN, ROLES, SEEDS
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def outcomes(report):
    timely = np.array([row['hold_start_seconds'] is not None for row in report['timing']['episodes']])
    final = np.array(report['end_hold']['stable_by_environment'])
    return timely & final


def compare(seed):
    proof_path = OUT/f'seed{seed}-analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-independent-verification' and not proof['heldout_used']
    assert proof['master_sha256'] == sha(PLAN) and proof['seed'] == seed
    record = ROOT/proof['record']
    spec = read(record/'protocol.json')
    n, first = spec['environment']['num_envs'], spec['prefix_controls']+1
    mask = np.zeros(n, dtype=bool)
    mask[proof['route']['specialist_worlds']] = True
    arms = proof['arms']
    source_paths = {role: record/f'trace-{role}.npz' for role in ROLES}
    checks = {}
    with ExitStack() as stack:
        archives = []
        for role, path in source_paths.items():
            assert sha(path) == arms[role]['trace_sha256']
            archives.append(stack.enter_context(np.load(path)))
        for field in ('actor_input', 'raw_policy_actions', 'qpos', 'qvel'):
            left, right, actual = (data[field] for data in archives)
            expected = np.where(mask[None, :, None], right, left)
            diff = np.abs(actual-expected)
            changed = np.any(actual != expected, axis=-1)
            earliest = np.where(changed.any(axis=0), changed.argmax(axis=0)+first, 0)
            checks[field] = dict(first_control_exact=int((~changed[0]).sum()),
                                 first_control_maximum=float(diff[0].max()),
                                 exact_all_controls=int((~changed.any(axis=0)).sum()),
                                 first_difference_control=earliest.tolist())
    parent, expert, routed = (outcomes(arms[role]) for role in ROLES)
    chosen = np.where(mask, expert, parent)
    assert all(int(result.sum()) == arms[role]['combined_successes']
               for role, result in zip(ROLES, (parent, expert, routed), strict=True))
    return dict(evaluation_sha256=sha(proof_path), physical_branch_rows=checks,
                selected_pure_successes=int(chosen.sum()), routed_successes=int(routed.sum()),
                routed_only_successes=np.flatnonzero(routed & ~chosen).tolist(),
                selected_pure_only_successes=np.flatnonzero(chosen & ~routed).tolist())


def main():
    rows = {str(seed): compare(seed) for seed in SEEDS}
    write(OUT/'development-fork-consistency.json', dict(status='passed-saved-development-integrity-check',
          script_sha256=sha(Path(__file__)), master_sha256=sha(PLAN), rows=rows,
          physical_substeps=0, learning_updates=0, heldout_used=False))
    print('POSE_ROUTED_DEV_FORKS', {seed: {
        'first_actions_exact': row['physical_branch_rows']['raw_policy_actions']['first_control_exact'],
        'first_qpos_exact': row['physical_branch_rows']['qpos']['first_control_exact'],
        'selected_pure_only': len(row['selected_pure_only_successes']),
        'route_only': len(row['routed_only_successes'])} for seed, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
