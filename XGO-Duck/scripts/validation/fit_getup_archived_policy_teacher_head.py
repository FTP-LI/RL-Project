"""Fit one bounded leg-output residual per arm and verify the exported graph."""

from argparse import ArgumentParser
from datetime import datetime
from html import escape
import os
from pathlib import Path
import json
from time import perf_counter
from urllib.parse import quote

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from review_getup_archived_policy_blend import excess
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha


PLAN = ROOT / 'configs/experiments/getup_archived_policy_teacher_head_fit.json'
POSES = ('prone', 'supine', 'left-side', 'right-side')


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def current(plan):
    p = read(ROOT / 'project.json')
    state = p['staged_training_request']
    assert p['goal_execution']['status'] == 'active' and not state['pause_requested']
    assert ROOT / state['current_protocol'] == PLAN and state['current_protocol_frozen']
    assert p['current_phase'] == plan['stage']


def artifact(pointer):
    name, checksum = pointer
    path = ROOT / name
    assert sha(path) == checksum, name
    return path


def publication(plan, state):
    folder, page = (ROOT / plan[key] for key in ('record', 'page'))
    state['updated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    write(folder / 'state.json', state)
    def url(path):
        return quote(Path(os.path.relpath(path, page.parent)).as_posix())
    refresh = '<meta http-equiv="refresh" content="2">' if state['status'] in ('starting', 'running') else ''
    steps = ''.join(f'<li>{escape(item)}</li>' for item in state['events'])
    rows = ''.join(f'<tr><td>{escape(role)}</td><td>{data["cross_teacher_mse"]:.7g}</td>'
                   f'<td>{data["cross_retention_max_abs_rad"]:.5f}</td></tr>' for role, data in state.get('scores', {}).items())
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<title>XGO-Duck 起身末层拟合</title><style>body{{max-width:980px;margin:30px auto;padding:0 16px;font:16px/1.6 system-ui}}img{{max-width:100%}}td,th{{padding:5px 12px;text-align:left}}</style>
<h1>起身末层拟合</h1><p>{escape(state['status'])} · {escape(plan['run_id'])} · {state['updated_at']}</p>
<p>学习求解 {state['solves']} / 2 · 物理步 0</p><ol>{steps}</ol>
<table><tr><th>策略</th><th>另一批动作均方误差</th><th>保留输入最大偏差（rad）</th></tr>{rows}</table>
<p><a href="{url(folder/'state.json')}">过程数据</a> · <a href="{url(folder/'execution.log')}">日志</a> ·
<a href="{url(ROOT/plan['archive_clip'])}">教师原始录像</a></p>
<p>归档参考画面</p><img src="{url(ROOT/plan['archive_frame'])}" alt="两份教师策略来源对照"></html>'''
    temp = page.with_suffix('.tmp')
    temp.write_text(html, encoding='utf-8')
    temp.replace(page)


def open_page(plan, mode):
    folder, page = (ROOT / plan[key] for key in ('record', 'page'))
    if mode == 'prepare':
        assert not folder.exists() and not page.exists()
        assert all((ROOT / plan[key]).is_file() for key in ('archive_clip', 'archive_frame'))
        folder.mkdir()
        publication(plan, dict(status='starting', events=['已固定数据、模型和单次求解预算'], solves=0))
        return
    state = read(folder / 'state.json')
    html = page.read_text(encoding='utf-8')
    assert state['status'] == 'starting' and state['solves'] == 0
    for key in ('archive_clip', 'archive_frame'):
        path = ROOT / plan[key]
        assert path.stat().st_size > 0 and quote(Path(os.path.relpath(path, page.parent)).as_posix()) in html
    assert plan['run_id'] in html and 'http-equiv="refresh"' in html
    path = folder / 'windows-observer-ready.json'
    assert os.name == 'nt' and not path.exists()
    write(path, dict(status='windows-file-observation-ready', page_sha256=sha(page), state_sha256=sha(folder / 'state.json'),
                     protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__))))
    print('ARCHIVED_TEACHER_HEAD_PAGE_READY', plan['page'], flush=True)


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])


def infer(actor, x, output):
    return np.concatenate([actor.run([output], {'obs': row[None]})[0] for row in x])


def load_pool(replay):
    data = {}
    for fold, entries in replay['files'].items():
        data[fold] = {}
        for kind, row in entries.items():
            file = artifact((row['path'], row['sha256']))
            with np.load(file, allow_pickle=False) as archive:
                data[fold][kind] = {key: archive[key] for key in archive.files}
            assert len(data[fold][kind]['observations']) == row['rows']
            assert np.all(data[fold][kind]['seed'] == replay['folds'][fold])
    return data


def model_with_delta(path, feature_key, weight_key, bias_key, delta, run_id):
    graph = onnx.load(path)
    tensors = {item.name: item for item in graph.graph.initializer}
    assert graph.graph.node[-1].input[0] == feature_key
    for name, diff in ((weight_key, delta[:-1].T), (bias_key, delta[-1])):
        values = numpy_helper.to_array(tensors[name])
        tensors[name].CopyFrom(numpy_helper.from_array((values + diff).astype(np.float32), name=name))
    names = {row.key: row for row in graph.metadata_props}
    names['run_name'].value = run_id
    names['training_source'].value = 'Bounded offline leg-target ridge fit on two archived policies'
    onnx.checker.check_model(graph)
    return graph


def score(actor, base, teacher, retention, limits, home):
    t = infer(actor, teacher['observations'], 'actions')
    r = infer(actor, retention['observations'], 'actions')
    delta = r - retention['targets']
    mask = teacher['mask']
    err = (t[mask] - teacher['teacher_targets'][mask])**2
    rmse = {pose: float(np.sqrt(np.mean(delta[retention['actual_pose'] == pose]**2))) for pose in POSES}
    shift = max(float(abs(t - teacher['control_targets']).max()), float(abs(delta).max()))
    extra = max(0., *(float((excess(result + home, limits) - excess(parent + home, limits)).max())
        for result, parent in ((t, teacher['control_targets']), (r, base))))
    return dict(cross_teacher_mse=float(err.mean()), cross_retention_rmse_by_pose_rad=rmse,
        cross_retention_max_abs_rad=float(abs(delta).max()), max_candidate_delta_rad=shift,
        maximum_added_target_limit_excess_rad=extra), (t, r)


def run(plan):
    folder, page = (ROOT / plan[key] for key in ('record', 'page'))
    state = read(folder / 'state.json')
    assert state['status'] == 'starting' and not (folder / 'analysis.json').exists()
    ready = read(folder / 'windows-observer-ready.json')
    assert ready['status'] == 'windows-file-observation-ready' and ready['page_sha256'] == sha(page)
    assert ready['state_sha256'] == sha(folder / 'state.json') and ready['protocol_sha256'] == sha(PLAN)
    assert ready['script_sha256'] == sha(Path(__file__))
    start = perf_counter()
    state['status'] = 'running'

    def progress(message):
        state['events'].append(message)
        publication(plan, state)
        print(message, flush=True)

    progress('已核验观察页，正在读取固定回放')
    try:
        replay = pinned(plan['replay'])
        assert replay['admitted_to_offline_fit_preparation'] and not replay['heldout_used']
        model_path = artifact(plan['parent_model'])
        artifact(plan['parent_checkpoint'])
        data = load_pool(replay)
        fit = data['fit_archive']
        graph = onnx.load(model_path)
        last = graph.graph.node[-1]
        assert last.op_type == 'Gemm' and tuple(last.output) == ('actions',)
        feature, weight, bias = last.input
        params = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
        w0, b0 = params[weight], params[bias]
        assert w0.shape == (14, 128) and b0.shape == (14,)
        graph.graph.output.append(helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, 128]))
        reference = session(graph.SerializeToString())
        x, xr = (infer(reference, fit[kind]['observations'], feature).astype(np.float64) for kind in ('teacher', 'retention'))
        np.testing.assert_allclose(x @ w0.T + b0, fit['teacher']['control_targets'], atol=2e-5, rtol=0)
        np.testing.assert_allclose(xr @ w0.T + b0, fit['retention']['targets'], atol=2e-5, rtol=0)
        joined = np.concatenate((x, xr))
        mean = joined.mean(axis=0)
        std = np.maximum(joined.std(axis=0), plan['minimum_feature_std'])
        def features(a):
            return np.column_stack(((a - mean) / std, np.ones(len(a))))
        z, zr = features(x), features(xr)
        k = z.T @ z / len(z) + plan['retain_loss_weight'] * (zr.T @ zr) / len(zr)
        k += plan['ridge_penalty'] * np.eye(k.shape[0])
        parent, teacher = fit['teacher']['control_targets'], fit['teacher']['teacher_targets']
        progress(f'已复算 {len(x)} 条训练输入及 {len(xr)} 条保留输入')
        cross = data['cross_archive']
        cx, cr = (infer(reference, cross[kind]['observations'], feature).astype(np.float64) for kind in ('teacher', 'retention'))
        base = infer(reference, cross['retention']['observations'], 'actions')
        residuals, scores, hashes, maxerr = {}, {}, {}, 0.
        source_protocol = pinned(plan['native_environment'])
        limits = np.asarray(source_protocol['environment']['joint_limits'], dtype=np.float32)
        home = np.asarray(source_protocol['environment']['model_home'], dtype=np.float32)
        for arm, label in (('control', parent), ('candidate', teacher)):
            current(plan)
            scaled = np.linalg.solve(k, z.T @ (label.astype(np.float64) - parent) / len(z))
            d = np.vstack((scaled[:-1] / std[:, None], scaled[-1] - (mean / std) @ scaled[:-1]))
            state['solves'] += 1
            assert plan['maximum_solves_per_arm'] == 1 and state['solves'] <= 2
            leg = np.asarray([n.startswith(('left_', 'right_')) for n in reference.get_modelmeta().custom_metadata_map['joint_names'].split(',')])
            assert int(leg.sum()) == 10 and not d[:, ~leg].any()
            if arm == 'control':
                assert not d.any()
            path = folder / f'{arm}.onnx'
            onnx.save_model(model_with_delta(model_path, feature, weight, bias, d, f'{plan["run_id"]}-{arm}'), path)
            exported = session(str(path))
            archived, computed = score(exported, base, cross['teacher'], cross['retention'], limits, home)
            for output, hidden in zip(computed, (cx, cr), strict=True):
                prediction = hidden @ (w0.T + d[:-1]) + b0 + d[-1]
                maxerr = max(maxerr, float(abs(output - prediction).max()))
            hashes[arm] = sha(path)
            residuals[arm] = d
            scores[arm] = archived
            state['scores'] = scores
            progress(f'{arm} 已求解，已复算另一批档案')
        np.savez_compressed(folder / 'offline-head-delta.npz', **residuals)
        gate = plan['gate']
        a, b = scores['control'], scores['candidate']
        improvement = 1 - b['cross_teacher_mse'] / a['cross_teacher_mse']
        tests = dict(masked_imitation=improvement >= gate['minimum_cross_teacher_masked_mse_improvement_fraction'],
            retained_poses=max(b['cross_retention_rmse_by_pose_rad'].values()) <= gate['maximum_cross_retention_rmse_each_pose_rad'],
            retained_max=b['cross_retention_max_abs_rad'] <= gate['maximum_cross_retention_absolute_delta_rad'],
            bounded_shift=b['max_candidate_delta_rad'] <= gate['maximum_candidate_absolute_delta_rad'],
            bounded_limit=b['maximum_added_target_limit_excess_rad'] <= gate['maximum_added_target_limit_excess_rad'],
            onnx=maxerr <= gate['maximum_onnx_readback_error_rad'])
        write(folder / 'analysis.json', dict(status='completed-offline-teacher-head-fit', protocol_sha256=sha(PLAN),
            script_sha256=sha(Path(__file__)), replay_sha256=plan['replay'][1], parent_checkpoint_sha256=plan['parent_checkpoint'][1],
            delta_sha256=sha(folder / 'offline-head-delta.npz'), onnx_sha256=hashes, scores=scores,
            cross_teacher_improvement_fraction=improvement, onnx_maximum_readback_error_rad=maxerr, tests=tests,
            admitted_to_native_preparation=all(tests.values()), solves=state['solves'], physical_substeps=0,
            heldout_used=False, elapsed_seconds=perf_counter() - start))
        state['status'] = 'completed'
        progress(f'独立档案误差下降 {improvement:.1%}。组件门槛为 {all(tests.values())}')
        print('ARCHIVED_TEACHER_HEAD_FIT', improvement, tests, flush=True)
    except BaseException as error:
        state['status'] = 'stopped-review-required'
        progress(f'停止原因 {error}')
        raise


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'ready', 'fit'))
    args = parser.parse_args()
    plan = read(PLAN)
    current(plan)
    assert not any(plan[k] for k in ('physical_substeps', 'heldout_used', 'hardware_io'))
    if args.mode == 'fit':
        run(plan)
    else:
        open_page(plan, args.mode)
