"""Find one common action envelope for the already fitted residual."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fit_getup_archived_policy_teacher_head import artifact, infer, load_pool, model_with_delta, score, session, write
from review_getup_archived_policy_blend import excess
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha


PLAN = ROOT / 'configs/experiments/getup_archived_policy_teacher_head_envelope.json'


def cap_for_rows(raw, delta, limits, home, gate, margin, retention):
    request = raw + home
    outside = excess(request, limits)
    allowance = outside + gate['maximum_added_target_limit_excess_rad'] - margin
    room = np.where(delta > 0, limits[:, 1] + allowance - request, request - (limits[:, 0] - allowance))
    assert (room >= 0).all()
    max_scale = np.divide(room, abs(delta), out=np.full_like(delta, np.inf), where=delta != 0)
    i, j = np.unravel_index(max_scale.argmin(), max_scale.shape)
    shift_limit = gate['maximum_cross_retention_absolute_delta_rad'] if retention else gate['maximum_candidate_absolute_delta_rad']
    shift_scale = (shift_limit - margin) / float(abs(delta).max())
    return dict(limit_scale=float(max_scale[i, j]), shift_scale=shift_scale, row=int(i), joint=int(j))


def main():
    plan = read(PLAN)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == plan['stage'] and ROOT / project['goal_execution']['current_protocol'] == PLAN
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    shrink, failed = (pinned(plan[key]) for key in ('shrink_protocol', 'failed_fixed_shrink'))
    assert failed['protocol_sha256'] == plan['shrink_protocol'][1] and not failed['admitted_to_native_preparation']
    fit, fit_result = (pinned(shrink[key]) for key in ('fit_protocol', 'fit_result'))
    run = ROOT / fit['record']
    out = run / 'enveloped-head'
    assert not out.exists()
    with np.load(artifact(((run / 'offline-head-delta.npz').relative_to(ROOT).as_posix(), fit_result['delta_sha256'])), allow_pickle=False) as data:
        d = data['candidate']
    model = artifact(fit['parent_model'])
    graph = onnx.load(model)
    feature, w, b = graph.graph.node[-1].input
    graph.graph.output.append(helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, 128]))
    parent = session(graph.SerializeToString())
    leg = np.asarray([name.startswith(('left_', 'right_')) for name in parent.get_modelmeta().custom_metadata_map['joint_names'].split(',')])
    assert int(leg.sum()) == 10 and not d[:, ~leg].any()
    data = load_pool(pinned(fit['replay']))
    env = pinned(fit['native_environment'])['environment']
    home, limits = (np.asarray(env[key], np.float64) for key in ('model_home', 'joint_limits'))
    gate, margin = fit['gate'], plan['margin_rad']
    caps, baseline, features = {}, {}, {}
    for fold, kinds in data.items():
        for kind, rows in kinds.items():
            name = f'{fold}/{kind}'
            x = rows['observations']
            y = infer(parent, x, 'actions')
            hidden = infer(parent, x, feature).astype(np.float64)
            delta = hidden @ d[:-1] + d[-1]
            caps[name] = cap_for_rows(y, delta, limits, home, gate, margin, kind == 'retention')
            caps[name].update(world=int(rows['world'][caps[name]['row']]), control=int(rows['control'][caps[name]['row']]))
            baseline[name], features[name] = y, hidden
    raw_cap = min(1., *(min(c['limit_scale'], c['shift_scale']) for c in caps.values()))
    scale = float(np.floor(raw_cap * 1000) / 1000)
    assert 0 < scale < shrink['delta_fraction']
    merged = model_with_delta(model, feature, w, b, d * scale, fit['run_id'] + '-enveloped')
    tensors = {item.name: numpy_helper.to_array(item) for item in merged.graph.initializer}
    out.mkdir()
    target = out / 'policy.onnx'
    onnx.save_model(merged, target)
    actor = session(str(target))
    results = {}
    for fold, kinds in data.items():
        t, r = (kinds[name] for name in ('teacher', 'retention'))
        base = baseline[f'{fold}/retention']
        control, _ = score(parent, base, t, r, limits, home)
        candidate, output = score(actor, base, t, r, limits, home)
        errors = []
        for kind, y in zip(('teacher', 'retention'), output, strict=True):
            name = f'{fold}/{kind}'
            prediction = features[name] @ tensors[w].T + tensors[b]
            errors.append(float(abs(y - prediction).max()))
            np.testing.assert_array_equal(y[:, ~leg], baseline[name][:, ~leg])
        improvement = 1 - candidate['cross_teacher_mse'] / control['cross_teacher_mse']
        passed = (improvement >= gate['minimum_cross_teacher_masked_mse_improvement_fraction'] and
            max(candidate['cross_retention_rmse_by_pose_rad'].values()) <= gate['maximum_cross_retention_rmse_each_pose_rad'] and
            candidate['cross_retention_max_abs_rad'] <= gate['maximum_cross_retention_absolute_delta_rad'] and
            candidate['max_candidate_delta_rad'] <= gate['maximum_candidate_absolute_delta_rad'] and
            candidate['maximum_added_target_limit_excess_rad'] <= gate['maximum_added_target_limit_excess_rad'] and
            max(errors) <= gate['maximum_onnx_readback_error_rad'])
        results[fold] = dict(result=candidate, improvement_fraction=improvement, max_onnx_error=max(errors), passed=passed)
    arrays = out / 'head.npz'
    np.savez_compressed(arrays, weight=tensors[w], bias=tensors[b])
    receipt = dict(status='completed-analytic-head-envelope', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        scale=scale, raw_upper_bound=raw_cap, caps=caps, folds=results, model=target.relative_to(ROOT).as_posix(),
        model_sha256=sha(target), weight_archive=arrays.relative_to(ROOT).as_posix(), weight_archive_sha256=sha(arrays),
        admitted_to_native_preparation=all(row['passed'] for row in results.values()), learning_updates=0,
        physical_substeps=0, heldout_used=False)
    write(out / 'analysis.json', receipt)
    print('ARCHIVED_TEACHER_HEAD_ENVELOPE', scale, receipt['admitted_to_native_preparation'], flush=True)
    for fold, row in results.items():
        print(fold, row['improvement_fraction'], row['result']['cross_retention_max_abs_rad'],
              row['result']['maximum_added_target_limit_excess_rad'], row['passed'], flush=True)


if __name__ == '__main__':
    main()
