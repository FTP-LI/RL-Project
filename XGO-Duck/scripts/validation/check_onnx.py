"""Check ONNX graph/CPU loading and finite output; never enables hardware."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def check_model(path: Path) -> dict:
    result = {'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        result['ir_version'] = graph.ir_version
        result['opsets'] = {op.domain or 'ai.onnx': op.version for op in graph.opset_import}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        result['inputs'] = [{'name': item.name, 'shape': item.shape, 'type': item.type} for item in inputs]
        result['outputs'] = [{'name': item.name, 'shape': item.shape, 'type': item.type} for item in outputs]
        assert len(inputs) == len(outputs) == 1, 'Expected one input and one output'
        assert inputs[0].shape == [1, 61] and inputs[0].type == 'tensor(float)', 'Observation interface mismatch'
        assert outputs[0].shape == [1, 14] and outputs[0].type == 'tensor(float)', 'Action interface mismatch'
        metadata = session.get_modelmeta().custom_metadata_map
        result['metadata'] = metadata
        joint_names = metadata.get('joint_names', '').split(',')
        result['joint_name_count'] = len(joint_names)
        if metadata.get('default_joint_pos'):
            home = np.array([float(item) for item in metadata['default_joint_pos'].split(',')])
            assert home.shape == (14,) and np.isfinite(home).all(), 'Invalid default_joint_pos metadata'
            result['model_home_degrees'] = np.rad2deg(home).tolist()
        rng = np.random.default_rng(42)
        # Synthetic interface probes, not trajectories and not evidence of locomotion.
        observations = rng.normal(0, 0.05, (128, 61)).astype(np.float32)
        observations[:, 3:6] = [0, 0, -1]
        observations[:, 55:61] = 0
        observations[0] = 0
        observations[0, 5] = -1
        magnitudes = []
        for observation in observations:
            action = session.run(None, {inputs[0].name: observation[None, :]})[0]
            assert action.shape == (1, 14) and np.isfinite(action).all(), 'Invalid action output'
            magnitudes.append(float(np.abs(action).max()))
        timings = []
        for _ in range(200):
            start = time.perf_counter_ns()
            session.run(None, {inputs[0].name: observations[:1]})
            timings.append((time.perf_counter_ns() - start) / 1e6)
        result['synthetic_observations_tested'] = len(observations)
        result['maximum_absolute_raw_action'] = max(magnitudes)
        result['pc_cpu_latency_ms'] = {'median': float(np.median(timings)), 'p99': float(np.percentile(timings, 99))}
        result['status'] = 'passed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True, help='One ONNX file or directory of ONNX files')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.models.glob('*.onnx')) if args.models.is_dir() else [args.models]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No readable ONNX models found')
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'onnx', 'onnxruntime']},
        'scope': 'PC CPU graph/interface/synthetic inference only; no simulation, trajectory parity, training or hardware validation',
        'hardware_deployment_ready': False,
        'seed': 42,
        'models': [check_model(path) for path in paths],
    }
    report['passed'] = all(model['status'] == 'passed' for model in report['models'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for model in report['models']:
        print(Path(model['file']).name, model['status'], model.get('error', model.get('pc_cpu_latency_ms')))
    print('Report:', args.report.resolve())
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
