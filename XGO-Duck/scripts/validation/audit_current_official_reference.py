"""Audit a separately pinned official walking model with saved feedback on CPU.

Change: use the newly fetched official ONNX and its own policy/rl_core sources,
without replacing the archived reference or importing the hardware controller.
Why: the published model, home pose and right-leg filter slice changed. Shape
compatibility alone does not establish correct observation or action mapping.
Expected: 61-to-14 inference, embedded normalization and runtime math agree with
independently reconstructed saved observations, filter coefficients and IDs.
Verification: source Git hashes, ONNX checker and numerical CPU replay. These
are conditional software checks, not new rollouts, gait results or training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'records/reports/official-reference-refresh/source-manifest.json')
    # This completed baseline record binds both its report and traces by hash.
    # Older reference verifications only bind traces, so do not silently treat
    # their report as equivalently authenticated by that older verifier.
    parser.add_argument('--feedback-run', type=Path, default=ROOT/'records/evaluations/custom_onnx_reference_20261101T144436_s1001')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    source = (ROOT/manifest['source_directory']).resolve()
    assert source.parent == ROOT/'workspace/official-reference'
    inputs = {str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    for item in manifest['files']:
        path = (source/item['path']).resolve()
        assert path.is_relative_to(source)
        data = path.read_bytes()
        assert len(data) == item['bytes'] and sha(path) == item['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob']
        inputs[str(path.relative_to(ROOT))] = sha(path)

    # Only the inspected numeric adapter is imported, never controller/main or
    # serial/firmware code. Policy.load explicitly chooses one CPU thread.
    runtime = source/'python'
    sys.path.insert(0, str(runtime))
    import rl_core
    import policy
    assert Path(rl_core.__file__).resolve().parent == runtime
    assert Path(policy.__file__).resolve().parent == runtime
    model = runtime/'xgoduck_walk.onnx'
    graph = onnx.load(model)
    onnx.checker.check_model(graph)
    metadata = {v.key: v.value for v in graph.metadata_props}
    names = metadata['joint_names'].split(',')
    home = np.asarray([float(v) for v in metadata['default_joint_pos'].split(',')], dtype=np.float32)
    assert home.shape == (14,) and np.isfinite(home).all()
    ids = [10, 11, 12, 13, 14, 30, 31, 32, 33, 20, 21, 22, 23, 24]
    assert rl_core.POLICY_IDS == ids and len(names) == len(ids)
    assert metadata['observation_names'].split(',') == ['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command']
    assert float(metadata['action_scale']) == rl_core.ACTION_SCALE == 1.0
    default = policy.Policy()
    matched = policy.Policy()
    ok, message = default.load(str(model)); assert ok, message
    ok, message = matched.load(str(model), use_model_home=True); assert ok, message
    assert default.session.get_providers() == ['CPUExecutionProvider']
    assert default.session.get_inputs()[0].shape == [1, 61]
    assert default.session.get_outputs()[0].shape == [1, 14]
    initial = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    sub, div = graph.graph.node[:2]
    assert sub.op_type == 'Sub' and sub.input[0] == default.in_name
    assert div.op_type == 'Div' and div.input[0] == sub.output[0]
    mean = initial[sub.input[1]].reshape(61)
    divisor = initial[div.input[1]].reshape(61)
    assert np.isfinite(mean).all() and np.isfinite(divisor).all() and (divisor > 0).all()

    feedback = args.feedback_run.resolve()
    saved = read(feedback/'evaluation.json')
    verified = read(feedback/'verification.json')
    assert verified['status'] == 'passed' and verified['report_sha256'] == sha(feedback/'evaluation.json')
    assert saved['model_metadata']['joint_names'].split(',') == names
    old_home = np.asarray(saved['policy_home'], dtype=np.float32)
    inputs[str((feedback/'evaluation.json').relative_to(ROOT))] = sha(feedback/'evaluation.json')
    inputs[str((feedback/'verification.json').relative_to(ROOT))] = sha(feedback/'verification.json')
    # Assign coefficients by physical ID, independently of runtime slicing.
    alphas = np.asarray([rl_core.ACTION_ALPHA_HEAD if 30 <= i <= 33 else rl_core.ACTION_ALPHA for i in ids], dtype=np.float32)
    rows = []
    for condition in saved['conditions']:
        trace = feedback/(condition['condition']+'-trace.npz')
        check = next(v for v in verified['checks'] if v['condition'] == condition['condition'])
        assert check['trace_sha256'] == sha(trace)
        inputs[str(trace.relative_to(ROOT))] = sha(trace)
        with np.load(trace, allow_pickle=False) as data:
            obs, active = data['observations'], data['active']
            assert obs.shape[1:] == (32, 61)
            for mode, adapter in [('runtime_default', default), ('model_home', matched)]:
                adapter.reset_action_filter()
                previous_filter = np.zeros(14, dtype=np.float32)
                errors = dict(observation=0.0, raw_history=0.0, filtered_action=0.0, motor_target_degrees=0.0)
                count = 0
                for step in (0, 100, 500, 999):
                    for env in np.flatnonzero(active[step]):
                        x = obs[step, env].copy()
                        angles = np.zeros(15, dtype=np.float32)
                        velocities = np.zeros(15, dtype=np.float32)
                        angles[rl_core.POLICY_INDEX] = np.rad2deg(x[6:20]+old_home)
                        velocities[rl_core.POLICY_INDEX] = np.rad2deg(x[20:34])
                        expected_obs = x.copy()
                        expected_obs[6:20] = x[6:20]+old_home-np.deg2rad(adapter.home_deg)
                        np.testing.assert_allclose(x[55:61], 0, atol=0, rtol=0)
                        adapter.prev_action[:] = x[34:48]
                        actual_obs = adapter.build_obs(x[3:6], x[:3], angles, velocities, x[48:51], x[51:55]).copy()
                        errors['observation'] = max(errors['observation'], float(np.max(np.abs(expected_obs-actual_obs))))
                        expected_raw = adapter.session.run(adapter._outputs, {adapter.in_name: expected_obs[None]})[0][0]
                        assert np.isfinite(expected_raw).all()
                        filtered = adapter.infer(x[3:6], x[:3], angles, velocities, x[48:51], x[51:55])
                        previous_filter = alphas*previous_filter+(1-alphas)*expected_raw
                        errors['raw_history'] = max(errors['raw_history'], float(np.max(np.abs(adapter.prev_action-expected_raw))))
                        errors['filtered_action'] = max(errors['filtered_action'], float(np.max(np.abs(filtered-previous_filter))))
                        expected_targets = np.zeros(15, dtype=np.float32)
                        for j, motor_id in enumerate(ids):
                            expected_targets[rl_core.MOTOR_IDS.index(motor_id)] = adapter.home_deg[j]+np.rad2deg(float(metadata['action_scale'])*float(previous_filter[j]))
                        actual_targets = np.asarray(adapter.action_to_deg(filtered))
                        errors['motor_target_degrees'] = max(errors['motor_target_degrees'], float(np.max(np.abs(expected_targets-actual_targets))))
                        count += 1
                assert errors['observation'] < 2e-6, errors
                assert errors['raw_history'] < 2e-5 and errors['filtered_action'] < 2e-5, errors
                assert errors['motor_target_degrees'] < 2e-3, errors
                rows.append(dict(condition=condition['condition'], home_mode=mode, samples=count, max_absolute_errors=errors))

    original = ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_walk.onnx'
    old_graph = onnx.load(original)
    old_metadata = {v.key: v.value for v in old_graph.metadata_props}
    old_model_home = np.rad2deg([float(v) for v in old_metadata['default_joint_pos'].split(',')])
    inputs[str(original.relative_to(ROOT))] = sha(original)
    output = manifest_path.parent
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], source_commit=manifest['commit'],
        model_sha256=sha(model), original_model_sha256=sha(original),
        source_sha256=sha(Path(__file__)), inputs_sha256=inputs,
        model_metadata=metadata, cpu_provider=default.session.get_providers(),
        samples=sum(r['samples'] for r in rows), distinct_saved_samples=sum(r['samples'] for r in rows if r['home_mode']=='runtime_default'), probes=rows,
        embedded_normalization=dict(command_mean=mean[48:51].tolist(), command_divisor=divisor[48:51].tolist(), external_normalization_required=False),
        joints=[dict(name=name, motor_id=ids[i], original_model_home_deg=float(old_model_home[i]), current_model_home_deg=float(np.rad2deg(home[i])), runtime_default_home_deg=float(default.home_deg[i]), action_filter_alpha=float(alphas[i])) for i,name in enumerate(names)],
        maximum_model_vs_runtime_home_delta_deg=float(np.max(np.abs(np.rad2deg(home)-default.home_deg))),
        hardware_controller_imported=False, simulation_run=False, training_started=False, gait_verified=False,
        training_provenance_verified=False,
        limitations=['The samples are reconstructed noisy/delayed saved simulator feedback, not real sensors.',
            'Filter replay spans selected saved states with forced old action history; it is not a closed-loop rollout.',
            'Embedded run_path is a provenance clue, not proof of training configuration, seed or full budget.',
            'Interface agreement and home differences do not establish gait improvement or hardware readiness.'])
    (output/'interface-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, output/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ['status','model_sha256','samples','distinct_saved_samples','maximum_model_vs_runtime_home_delta_deg','embedded_normalization','gait_verified']},indent=2))


if __name__ == '__main__':
    main()
