"""Check a fixed state-dependent policy blend on recorded, real observations."""

import json
from pathlib import Path

import numpy as np

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import actor, predict, success


PLAN = ROOT / 'configs/experiments/getup_archived_policy_blend_preflight.json'
OUT = ROOT / 'records/reports/getup-archived-policy-comparison/blend-preflight.json'


def blend(inputs, parent, teacher, spec):
    gravity = inputs[..., 3:6]
    norm = np.maximum(np.linalg.norm(gravity, axis=-1), 1e-6)
    supine = -gravity[..., 0] / norm
    start, full = (spec[key] for key in ('observed_supine_start', 'observed_supine_full'))
    mix = spec['maximum_teacher_fraction'] * np.clip((supine - start) / (full - start), 0., 1.)
    return np.where(mix[..., None] > 0, parent + mix[..., None] * (teacher - parent), parent), mix


def read_observed_trace(path, proof, indices, controls):
    assert sha(path) == proof
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved['time'][indices], np.array(controls) * .02, atol=1e-12, rtol=0)
        return {key: saved[key][indices] for key in ('actor_input', 'raw_policy_actions', 'quaternion', 'foot_contact')}


def excess(request, limits):
    return np.maximum(0, np.maximum(limits[:, 0] - request, request - limits[:, 1]))


def review(seed, pointer, plan):
    result = pinned(pointer)
    assert result['seed'] == int(seed) and not result['heldout_used'] and result['learning_updates'] == 0
    record = ROOT / result['record']
    protocol = record / 'protocol.json'
    assert sha(protocol) == result['protocol_sha256']
    spec = read(protocol)
    roles = (plan['parent_role'], plan['teacher_role'])
    parent, names = actor(spec, roles[0])
    teacher, teacher_names = actor(spec, roles[1])
    assert names == teacher_names
    parent_pass, actual = success(result['arms'][roles[0]])
    teacher_pass, _ = success(result['arms'][roles[1]], actual)
    supine = actual == 'supine'
    rescue = supine & teacher_pass & ~parent_pass
    controls = plan['probe_controls']
    indices = np.asarray(controls) - spec['prefix_controls'] - 1
    assert indices[0] == 0 and 155 in controls
    early = np.asarray(controls) <= 155
    env = spec['environment']
    limits = np.asarray(env['joint_limits'], dtype=np.float32)
    home = np.asarray(env['model_home'], dtype=np.float32)
    gate = plan['gate']
    paths = [record / f'trace-{role}.npz' for role in roles]
    arm_results, release_inputs = {}, []
    for role, path in zip(roles, paths, strict=True):
        trace = read_observed_trace(path, result['arms'][role]['trace_sha256'], indices, controls)
        x = trace['actor_input']
        a, b = predict(parent, x), predict(teacher, x)
        sent, alpha = blend(x, a, b, plan)
        release_inputs.append(x[0])
        err = float(abs((a if role == roles[0] else b) - trace['raw_policy_actions']).max())
        assert np.isfinite(sent).all() and (alpha >= 0).all() and (alpha <= plan['maximum_teacher_fraction']).all()
        assert ((sent >= np.minimum(a, b) - 1e-7) & (sent <= np.maximum(a, b) + 1e-7)).all()
        np.testing.assert_array_equal(sent[alpha == 0], a[alpha == 0])
        quat = trace['quaternion'].astype(np.float64)
        vertical = 1 - 2 * np.sum(quat[..., 1:3]**2, axis=-1)
        upright = (vertical >= np.cos(np.deg2rad(env['criteria']['tilt_max_deg']))) & trace['foot_contact'].all(axis=-1)
        delta = abs(sent - a).max(axis=-1)
        world_delta = delta[early][:, rescue].max(axis=0)
        material = int((world_delta >= gate['material_absolute_action_shift_rad']).sum())
        extra = max(0., float((excess(sent + home, limits) - excess(a + home, limits)).max()))
        release_covered = int((alpha[0, supine] > 0).sum())
        release_other = int((alpha[0, ~supine] > 0).sum())
        upright_uses = int((alpha[upright] != 0).sum())
        tests = dict(original_readback=err <= gate['maximum_actual_action_reproduction_error_rad'],
            supine_coverage=release_covered / int(supine.sum()) >= gate['minimum_actual_supine_release_coverage'],
            other_release=release_other / int((~supine).sum()) <= gate['maximum_other_pose_release_blend_fraction'],
            upright_exact=upright_uses == 0 and int(upright.sum()) > 0,
            material_early=material / int(rescue.sum()) >= gate['minimum_teacher_only_rescues_with_material_early_change_fraction'],
            bounded_shift=float(delta.max()) <= gate['maximum_absolute_action_shift_rad'],
            bounded_limit_excess=extra <= gate['maximum_added_target_limit_excess_rad'])
        arm_results[role] = dict(readback_error_rad=err, actual_inputs=int(x.shape[0] * x.shape[1]),
            applied_teacher_fraction_by_control={step: int((alpha[i] > 0).sum()) for i, step in enumerate(controls)},
            actual_supine_release_worlds=int(supine.sum()), actual_supine_release_eligible=release_covered,
            other_pose_release_worlds=int((~supine).sum()), other_pose_release_eligible=release_other,
            upright_inputs=int(upright.sum()), upright_blended=upright_uses, rescue_worlds=int(rescue.sum()),
            rescue_material_early=material, median_rescue_early_max_delta_rad=float(np.median(world_delta)),
            maximum_delta_rad=float(delta.max()), maximum_added_target_excess_rad=extra, tests=tests, passed=all(tests.values()))
    np.testing.assert_array_equal(*release_inputs)
    return arm_results


def boundary(plan):
    x = np.zeros((1, 4, 61), dtype=np.float32)
    magnitude = np.asarray([0., plan['observed_supine_start'],
                           (plan['observed_supine_start'] + plan['observed_supine_full']) / 2, 1.])
    x[0, :, 3] = -magnitude
    x[0, :, 5] = -np.sqrt(1 - magnitude**2)
    p = np.zeros((1, 4, 14), dtype=np.float32)
    t = np.ones_like(p)
    output, alpha = blend(x, p, t, plan)
    top = plan['maximum_teacher_fraction']
    np.testing.assert_allclose(alpha[0], [0., 0., top/2, top], rtol=0, atol=5e-7)
    np.testing.assert_array_equal(output[..., 0], alpha)


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    assert not pinned(plan['teacher_timing'])['admitted_to_learning_preparation']
    boundary(plan)
    results = {seed: review(seed, pointer, plan) for seed, pointer in plan['sources'].items()}
    passed = all(row['passed'] for seed in results.values() for row in seed.values())
    output = dict(status='completed-fixed-blend-cpu-preflight', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        results=results, admitted_to_dynamic_preparation=passed, boundary_fixture_passed=True,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    OUT.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('ARCHIVED_POLICY_BLEND_CPU', passed, flush=True)
    for seed, arms in results.items():
        for role, row in arms.items():
            print(seed, role, row['rescue_material_early'], '/', row['rescue_worlds'],
                  'max delta', row['maximum_delta_rad'], 'limit extra', row['maximum_added_target_excess_rad'], row['tests'], flush=True)
