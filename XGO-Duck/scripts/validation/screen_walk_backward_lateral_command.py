"""Screen an internal lateral command against natural pre-step requests."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from s1_reference_inputs import ROOT, read, sha
from record_io import write


HISTORY = ROOT / 'records/reports/common-walk-backward-start-pulse/support-transfer-review.json'
OUT = ROOT / 'records/reports/common-walk-backward-actuator-support'
PLAN = ROOT / 'configs/experiments/common_walk_backward_actuator_support.json'
DOSES = (-.24, -.18, -.12, -.06, -.03, .03, .06, .12, .18, .24)
CAP = .03


def cosine(a, b):
    length = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(np.sum(a*b, axis=-1), length, out=np.zeros_like(length), where=length > 1e-9)


def donor(role):
    selected = [row['preceding']['applied'] for row in role['rows']
                if row['support'] and row['support']['side'] == 'left']
    result = np.array([row['stance_delta_rad'] + row['swing_delta_rad'] for row in selected])
    assert len(result) >= 2 and result.shape[1:] == (8,)
    return result


def review(seed, session, plan, protocol_sha, directions, indices):
    source = OUT / f'seed{seed}-analysis.json'
    proof = read(source)
    assert proof['protocol_sha256'] == protocol_sha and not proof['heldout_used']
    path = ROOT / proof['record'] / 'trace-backward.npz'
    assert sha(path) == proof['trace_sha256']['backward']
    world_ids = proof['common_near_static']
    with np.load(path) as data:
        observed = data['observations'][0, world_ids].copy()
        saved_action = data['actions'][0, world_ids]
    np.testing.assert_allclose(observed[:, 48:51], np.tile([-.14, 0., 0.], (len(observed), 1)), atol=2e-8, rtol=0)
    run = lambda values: np.concatenate([session.run(None, {'obs': row[None]})[0] for row in values])
    base = run(observed)
    np.testing.assert_allclose(base, saved_action, atol=2e-5, rtol=2e-5)
    home = np.asarray(plan['model_home'], np.float32)
    low, high = np.asarray(plan['joint_limits'], np.float32).T
    nominal = np.clip(base + home, low, high)
    doses = {}
    for dose in DOSES:
        candidate = observed.copy()
        candidate[:, 49] = dose
        desired = np.clip(run(candidate) + home, low, high)
        delta = np.clip(desired - nominal, -CAP, CAP)
        legs = delta[:, indices]
        changes = np.linalg.norm(legs, axis=1)
        agreement = {str(key): cosine(legs, value).tolist() for key, value in directions.items()}
        passed = (np.median(changes) >= .012
                  and all(np.median(v) >= .5 and (np.asarray(v) > 0).mean() >= 2/3 for v in agreement.values()))
        doses[str(dose)] = dict(passed=bool(passed), world_cosines=agreement, leg_delta_norm_rad=changes.tolist(),
                                maximum_target_change_rad=float(abs(delta).max()))
    return dict(seed=seed, worlds=world_ids, doses=doses, trace_sha256=sha(path), proof_sha256=sha(source))


def main():
    source = read(HISTORY)
    protocol = read(PLAN)
    assert source['status'] == 'passed-saved-development-review' and not source['heldout_used']
    p = protocol['environment']
    axes = source['axes']
    indices = [p['joint_names'].index(f'{side}_{axis}') for side in ('left', 'right') for axis in axes]
    directions = {}
    donor_counts = {}
    for row in source['cohorts']:
        path = ROOT / row['source']
        assert sha(path) == row['source_sha256']
        values = donor(row['roles']['backward'])
        directions[row['seed']] = np.median(values, axis=0)
        donor_counts[row['seed']] = len(values)
    assert tuple(directions) == (1071, 1072, 1073, 1074)
    assert all(np.linalg.norm(v) > 1e-4 for v in directions.values())
    path = ROOT / p['model']
    assert sha(path) == p['model_sha256']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    protocol_sha = sha(PLAN)
    batches = [review(int(seed), session, p, protocol_sha, directions, indices) for seed in protocol['runs']]
    survivors = [dose for dose in DOSES if all(batch['doses'][str(dose)]['passed'] for batch in batches)]
    path = OUT / 'lateral-command-target-screen.json'
    assert not path.exists()
    write(path, dict(status='passed-static-target-screen' if survivors else 'closed-static-target-screen',
        source_sha256=sha(Path(__file__)), history_sha256=sha(HISTORY), protocol_sha256=protocol_sha,
        required_minimum_leg_delta_norm_rad=.012, required_median_cosine=.5,
        required_positive_cosine_fraction=2/3, target_cap_rad=CAP, axes=axes, donor_counts=donor_counts,
        directions={str(k): v.tolist() for k, v in directions.items()}, batches=batches, admissible_doses=survivors,
        physical_substeps=0, learning_updates=0, heldout_used=False,
        scope='Static projected request alignment with natural left-foot support. No causal or dynamic claim.'))
    print('BACKWARD_LATERAL_COMMAND_TARGETS', donor_counts, survivors, flush=True)


if __name__ == '__main__':
    main()
