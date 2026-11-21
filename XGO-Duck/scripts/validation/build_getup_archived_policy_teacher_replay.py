"""Build bounded teacher supervision and a four-pose parent retention pool."""

import json
from pathlib import Path

import numpy as np

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import actor, predict, success


PLAN = ROOT / 'configs/experiments/getup_archived_policy_teacher_replay_pool.json'
OUT = ROOT / 'records/reports/getup-archived-policy-comparison/teacher-replay'
POSES = ('prone', 'supine', 'left-side', 'right-side')


def read_rows(record, role, score, worlds, controls):
    path = record / f'trace-{role}.npz'
    expected = score['arms'][role]['trace_sha256']
    assert sha(path) == expected
    indices = np.asarray(controls) - 101
    with np.load(path, allow_pickle=False) as source:
        np.testing.assert_allclose(source['time'][indices], np.asarray(controls) * .02, atol=1e-12, rtol=0)
        x = source['actor_input'][np.ix_(indices, worlds)]
        y = source['raw_policy_actions'][np.ix_(indices, worlds)]
        requested = source['requested_targets'][np.ix_(indices, worlds)]
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(requested).all()
    return x, y, requested, expected


def pack(x, controls, worlds, seed, **arrays):
    count, batch = x.shape[:2]
    return dict(observations=x.reshape(-1, 61), seed=np.full(count * batch, seed, np.int32),
                world=np.tile(worlds, count), control=np.repeat(controls, batch),
                **{name: value.reshape(count * batch, *value.shape[2:]) for name, value in arrays.items()})


def one_seed(seed, pointer, plan, random):
    score = pinned(pointer)
    assert score['seed'] == seed and not score['heldout_used'] and score['learning_updates'] == 0
    record = ROOT / score['record']
    protocol_path = record / 'protocol.json'
    assert sha(protocol_path) == score['protocol_sha256']
    spec = read(protocol_path)
    p, t = plan['parent_role'], plan['teacher_role']
    parent, joints = actor(spec, p)
    teacher, names = actor(spec, t)
    assert names == joints and spec['prefix_controls'] == 100
    parent_pass, actual = success(score['arms'][p])
    teacher_pass, _ = success(score['arms'][t], actual)
    parents = {pose: np.sort(random.choice(np.flatnonzero(parent_pass & (actual == pose)),
        plan['retention_worlds_per_seed_and_actual_pose'], replace=False)) for pose in POSES}
    retain_worlds = np.concatenate(list(parents.values()))
    rescues = np.flatnonzero((actual == 'supine') & teacher_pass & ~parent_pass)
    assert np.intersect1d(rescues, retain_worlds).size == 0
    assert len(rescues) >= plan['minimum_teacher_only_worlds_per_seed']
    first, last = plan['teaching_command_controls']
    controls = np.arange(first, last + 1)
    x, y, desired, teacher_sha = read_rows(record, t, score, rescues, controls)
    base = predict(parent, x)
    reproduce = predict(teacher, x)
    check = plan['maximum_actor_replay_error_rad']
    err = float(abs(reproduce - y).max())
    assert err <= check
    env = spec['environment']
    home = np.asarray(env['model_home'], dtype=np.float32)
    limits = np.asarray(env['joint_limits'], dtype=np.float32)
    np.testing.assert_allclose(home + y, desired, atol=1e-6, rtol=0)
    leg = np.array([name.startswith(('left_', 'right_')) for name in joints])
    assert int(leg.sum()) == 10
    baseline_target = home + base
    proposed = (leg & (abs(y - base) <= plan['maximum_joint_delta_rad']) &
                (desired >= limits[:, 0]) & (desired <= limits[:, 1]) &
                (baseline_target >= limits[:, 0]) & (baseline_target <= limits[:, 1]))
    label = np.where(proposed, y, base)
    np.testing.assert_array_equal(label[..., ~leg], base[..., ~leg])
    assert np.max(abs(label - base)) <= plan['maximum_joint_delta_rad']
    material = (abs(label - base) >= plan['material_action_delta_rad']).any(axis=-1)
    valid_worlds = material.sum(axis=0) >= plan['minimum_material_controls_per_world']
    target_steps = np.asarray(plan['retention_command_controls'])
    rx, ry, _, parent_sha = read_rows(record, p, score, retain_worlds, target_steps)
    replay_error = float(abs(predict(parent, rx) - ry).max())
    assert replay_error <= check
    training = pack(x, controls, rescues, seed, teacher_targets=label, control_targets=base, mask=proposed)
    retention = pack(rx, target_steps, retain_worlds, seed, targets=ry,
                     actual_pose=np.broadcast_to(actual[retain_worlds], rx.shape[:2]))
    stats = dict(worlds=len(rescues), rows=len(training['world']), material_worlds=int(valid_worlds.sum()),
        material_controls=int(material.sum()), median_material_controls=float(np.median(material.sum(axis=0))),
        material_world_fraction=float(valid_worlds.mean()), retained_worlds_by_pose={pose: ids.tolist() for pose, ids in parents.items()},
        retention_rows=len(retention['world']), teacher_replay_error_rad=err, parent_retention_error_rad=replay_error,
        maximum_action_delta_rad=float(abs(label - base).max()),
        mask_counts_by_joint={name: int(proposed[..., j].sum()) for j, name in enumerate(joints)},
        source_model_sha256={role: spec['policies'][role]['model_sha256'] for role in (p, t)},
        trace_sha256={p: parent_sha, t: teacher_sha},
        passed=float(valid_worlds.mean()) >= plan['minimum_material_world_fraction_per_seed'])
    return training, retention, stats


def save(path, data):
    keys = list(zip(data['seed'], data['world'], data['control'], strict=True))
    assert len(keys) == len(set(keys)) and data['observations'].shape == (len(keys), 61)
    np.savez_compressed(path, **data)
    with np.load(path, allow_pickle=False) as saved:
        for key, values in data.items():
            np.testing.assert_array_equal(saved[key], values)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), rows=len(keys))


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    assert not pinned(plan['closed_blend'])['admitted_to_dynamic_preparation']
    rng = np.random.default_rng(plan['retention_seed'])
    sets = {int(seed): one_seed(int(seed), pointer, plan, rng) for seed, pointer in plan['sources'].items()}
    folds = plan['folds']
    assert set(folds.values()) == set(sets)
    OUT.mkdir()
    files = {}
    for label, seed in folds.items():
        files[label] = {name: save(OUT / f'{label}-{name}.npz', sets[seed][i])
                        for i, name in enumerate(('teacher', 'retention'))}
    stats = {seed: pieces[2] for seed, pieces in sets.items()}
    passed = all(row['passed'] for row in stats.values())
    result = dict(status='completed-bounded-teacher-replay-pool', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        folds=folds, results=stats, files=files, admitted_to_offline_fit_preparation=passed,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('ARCHIVED_TEACHER_REPLAY_POOL', passed, flush=True)
    for seed, row in stats.items():
        print(seed, 'teacher', row['rows'], 'retention', row['retention_rows'], 'material worlds',
              row['material_worlds'], '/', row['worlds'], 'max delta', row['maximum_action_delta_rad'], flush=True)
