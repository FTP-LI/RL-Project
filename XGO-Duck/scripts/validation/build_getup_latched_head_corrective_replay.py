"""Build equal-input replay for recovering lost supine cases without losing gains."""

import json
from pathlib import Path

import numpy as np

from build_getup_archived_policy_teacher_replay import pack, read_rows, save
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import actor, predict, success


PLAN = ROOT / 'configs/experiments/getup_latched_head_corrective_replay.json'
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/corrective-replay'


def seed_rows(seed, pointer, failure, plan):
    score = pinned(pointer)
    assert score['seed'] == seed and not score['heldout_used']
    record = ROOT / score['record']
    protocol = record / 'protocol.json'
    assert sha(protocol) == score['protocol_sha256']
    spec = read(protocol)
    baseline, fitted = plan['corrective_teacher'], 'candidate'
    a, actual = success(score['arms'][baseline])
    b, _ = success(score['arms']['reference-b'], actual)
    c, _ = success(score['arms'][fitted], actual)
    requested = np.array([episode['requested_pose'] for episode in score['arms'][baseline]['timely']['episodes']])
    both_supine = (requested == 'supine') & (actual == 'supine')
    losses, gains = (np.flatnonzero(both_supine & mask) for mask in (a & b & ~c, ~a & ~b & c))
    groups = failure['groups']
    assert losses.tolist() == groups['candidate_only_failure']['worlds']
    assert gains.tolist() == groups['candidate_only_success']['worlds']
    assert min(len(losses), len(gains)) >= plan['minimum_worlds_each_role_and_seed']
    p, joints = actor(spec, baseline)
    q, candidate_joints = actor(spec, fitted)
    assert joints == candidate_joints and len(joints) == plan['required_joint_count']
    first, last = plan['action_window_controls']
    controls = np.arange(first, last + 1)
    x, y, wanted, parent_sha = read_rows(record, baseline, score, losses, controls)
    gx, gy, _, candidate_sha = read_rows(record, fitted, score, gains, controls)
    stored = predict(q, x)
    e_parent = float(abs(predict(p, x) - y).max())
    e_gain = float(abs(predict(q, gx) - gy).max())
    assert max(e_parent, e_gain) <= plan['maximum_action_readback_error_rad']
    env = spec['environment']
    home = np.asarray(env['model_home'], dtype=np.float32)
    limits = np.asarray(env['joint_limits'], dtype=np.float32)
    np.testing.assert_allclose(home + y, wanted, atol=1e-6, rtol=0)
    old = stored + home
    leg = np.array([name.startswith(('left_', 'right_')) for name in joints])
    available = (leg & (abs(y - stored) <= plan['maximum_joint_delta_rad']) &
                 (wanted >= limits[:, 0]) & (wanted <= limits[:, 1]) &
                 (old >= limits[:, 0]) & (old <= limits[:, 1]))
    replacement = np.where(available, y, stored)
    np.testing.assert_array_equal(replacement[..., ~leg], stored[..., ~leg])
    delta = abs(replacement - stored)
    assert float(delta.max()) <= plan['maximum_joint_delta_rad']
    enough = (delta >= plan['material_action_delta_rad']).any(axis=-1).sum(axis=0) >= plan['minimum_material_controls_per_corrective_world']
    correction = pack(x, controls, losses, seed, control_targets=stored, candidate_targets=replacement, mask=available)
    retention = pack(gx, controls, gains, seed, control_targets=gy, candidate_targets=gy, mask=np.zeros_like(gy, dtype=bool))
    stats = dict(corrective_worlds=len(losses), retained_worlds=len(gains), qualifying_corrective_worlds=int(enough.sum()),
                 max_correction_delta_rad=float(delta.max()), parent_readback_rad=e_parent, retention_readback_rad=e_gain,
                 trace_sha256={baseline: parent_sha, fitted: candidate_sha},
                 starting_policy_sha256=spec['policies'][fitted]['model_sha256'],
                 passed=float(enough.mean()) >= plan['minimum_material_world_fraction_each_seed'])
    return correction, retention, stats


def compact(data, controls, plan):
    rng = np.random.default_rng(plan['compact_sampling_seed'])
    cells, chosen = {}, []
    count = plan['compact_time_bins_per_role']
    size, rem = divmod(plan['compact_rows_per_role'], count)
    for kind in ('corrective', 'retention'):
        for i, times in enumerate(np.array_split(controls, count)):
            pool = np.flatnonzero((data['kind'] == kind) & np.isin(data['control'], times))
            rows = np.sort(rng.choice(pool, size=size + (i < rem), replace=False))
            cells[f'{kind}|{i:02}'] = list(range(len(chosen), len(chosen) + len(rows)))
            chosen.extend(rows.tolist())
    assert len(set(chosen)) == len(chosen) and len(chosen) == 900 and len(cells) == 24
    result = {key: value[chosen] for key, value in data.items()}
    assert len(np.unique(np.column_stack((result['seed'], result['world'])), axis=0)) == len(
        np.unique(np.column_stack((data['seed'], data['world'])), axis=0))
    for role in ('corrective', 'retention'):
        assert int((result['kind'] == role).sum()) == plan['compact_rows_per_role']
    unchanged = result['kind'] == 'retention'
    np.testing.assert_array_equal(result['candidate_targets'][unchanged], result['control_targets'][unchanged])
    return result, cells


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    review = pinned(plan['failure_review'])
    rows, receipts = [], {}
    for seed, pointer in plan['sources'].items():
        a, b, receipts[seed] = seed_rows(int(seed), pointer, review['seeds'][seed], plan)
        for kind, group in (('corrective', a), ('retention', b)):
            group['kind'] = np.repeat(kind, len(group['world']))
            rows.append(group)
    assert len({value['starting_policy_sha256'] for value in receipts.values()}) == 1
    merged = {key: np.concatenate([chunk[key] for chunk in rows]) for key in rows[0]}
    start, end = plan['action_window_controls']
    prepared, cells = compact(merged, np.arange(start, end + 1), plan)
    OUT.mkdir()
    pool = save(OUT / 'paired-replay.npz', prepared)
    cell_file = OUT / 'sampling-cells.json'
    cell_file.write_text(json.dumps(cells, indent=2) + '\n', encoding='utf-8')
    changed = np.any(prepared['control_targets'] != prepared['candidate_targets'], axis=1)
    passed = all(value['passed'] for value in receipts.values()) and changed.any()
    result = dict(status='completed-paired-corrective-replay', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        data=pool, cells=cell_file.relative_to(ROOT).as_posix(), cells_sha256=sha(cell_file), cells_count=len(cells),
        raw_rows=len(merged['world']), corrective_changed_compact_rows=int(changed.sum()), seeds=receipts,
        admitted_to_training_preparation=bool(passed), heldout_used=False, physical_substeps=0, learning_updates=0)
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('TEACHER_HEAD_CORRECTIVE_REPLAY', bool(passed), 'rows', pool['rows'], 'changed', int(changed.sum()), flush=True)
    for seed, value in receipts.items():
        print(seed, value['qualifying_corrective_worlds'], '/', value['corrective_worlds'],
              'retained', value['retained_worlds'], 'max', value['max_correction_delta_rad'], flush=True)
