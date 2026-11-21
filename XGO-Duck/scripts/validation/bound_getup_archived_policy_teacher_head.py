"""Export and verify a single fixed shrink of the saved teacher residual."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fit_getup_archived_policy_teacher_head import artifact, infer, load_pool, model_with_delta, score, session, write
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha


PLAN = ROOT / 'configs/experiments/getup_archived_policy_teacher_head_bound.json'


def main():
    plan = read(PLAN)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == plan['stage'] and ROOT / project['goal_execution']['current_protocol'] == PLAN
    assert not any(plan[k] for k in ('physical_substeps', 'learning_updates', 'heldout_used', 'hardware_io'))
    origin, result = (pinned(plan[key]) for key in ('fit_protocol', 'fit_result'))
    assert result['protocol_sha256'] == plan['fit_protocol'][1] and result['solves'] == 2
    assert not result['admitted_to_native_preparation'] and not result['heldout_used']
    run = ROOT / origin['record']
    out = run / 'bounded-head'
    assert not out.exists()
    model, checkpoint = (artifact(origin[key]) for key in ('parent_model', 'parent_checkpoint'))
    delta_path = artifact(((run / 'offline-head-delta.npz').relative_to(ROOT).as_posix(), result['delta_sha256']))
    with np.load(delta_path, allow_pickle=False) as saved:
        assert not saved['control'].any()
        d = saved['candidate'] * plan['delta_fraction']
    old = onnx.load(model)
    feature, weight, bias = old.graph.node[-1].input
    graph = model_with_delta(model, feature, weight, bias, d, origin['run_id'] + '-bounded')
    a = {p.name: numpy_helper.to_array(p) for p in old.graph.initializer}
    b = {p.name: numpy_helper.to_array(p) for p in graph.graph.initializer}
    assert a.keys() == b.keys() and [n.SerializeToString() for n in old.graph.node] == [n.SerializeToString() for n in graph.graph.node]
    leg = np.asarray([name.startswith(('left_', 'right_')) for name in session(str(model)).get_modelmeta().custom_metadata_map['joint_names'].split(',')])
    for name in a.keys() - {weight, bias}:
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(a[weight][~leg], b[weight][~leg])
    np.testing.assert_array_equal(a[bias][~leg], b[bias][~leg])
    out.mkdir()
    target = out / 'policy.onnx'
    onnx.save_model(graph, target)
    candidate = session(str(target))
    old.graph.output.append(helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, 128]))
    base = session(old.SerializeToString())
    data = load_pool(pinned(origin['replay']))
    gate = origin['gate']
    env = pinned(origin['native_environment'])['environment']
    home, limits = (np.asarray(env[name], dtype=np.float32) for name in ('model_home', 'joint_limits'))
    checks = {}
    for fold, rows in data.items():
        t, r = (rows[key] for key in ('teacher', 'retention'))
        parent = infer(base, r['observations'], 'actions')
        control, _ = score(base, parent, t, r, limits, home)
        found, outputs = score(candidate, parent, t, r, limits, home)
        errors = []
        for row, actual in zip((t, r), outputs, strict=True):
            hidden = infer(base, row['observations'], feature).astype(np.float64)
            expected = hidden @ b[weight].T + b[bias]
            errors.append(float(abs(expected - actual).max()))
            np.testing.assert_array_equal(actual[:, ~leg], infer(base, row['observations'], 'actions')[:, ~leg])
        gain = 1 - found['cross_teacher_mse'] / control['cross_teacher_mse']
        tests = dict(imitation=gain >= gate['minimum_cross_teacher_masked_mse_improvement_fraction'],
            retained_poses=max(found['cross_retention_rmse_by_pose_rad'].values()) <= gate['maximum_cross_retention_rmse_each_pose_rad'],
            retained_max=found['cross_retention_max_abs_rad'] <= gate['maximum_cross_retention_absolute_delta_rad'],
            bounded_shift=found['max_candidate_delta_rad'] <= gate['maximum_candidate_absolute_delta_rad'],
            bounded_limit=found['maximum_added_target_limit_excess_rad'] <= gate['maximum_added_target_limit_excess_rad'],
            onnx=max(errors) <= gate['maximum_onnx_readback_error_rad'])
        checks[fold] = dict(result=found, improvement_fraction=gain, maximum_readback_error_rad=max(errors),
                            tests=tests, passed=all(tests.values()))
    np.savez_compressed(out / 'bounded-head.npz', weight=b[weight], bias=b[bias], scaled_delta=d)
    receipt = dict(status='completed-fixed-head-shrink-verification', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        model=target.relative_to(ROOT).as_posix(), model_sha256=sha(target), parameter_sha256=sha(out / 'bounded-head.npz'),
        parent_checkpoint_sha256=sha(checkpoint), folds=checks,
        admitted_to_native_preparation=all(row['passed'] for row in checks.values()), original_fit_preserved=True,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(out / 'analysis.json', receipt)
    print('ARCHIVED_TEACHER_HEAD_BOUNDED', receipt['admitted_to_native_preparation'], flush=True)
    for fold, row in checks.items():
        print(fold, row['improvement_fraction'], row['result']['cross_retention_max_abs_rad'],
              row['result']['maximum_added_target_limit_excess_rad'], row['tests'], flush=True)


if __name__ == '__main__':
    main()
