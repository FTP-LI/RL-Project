"""Independently check saved CUDA zero-step scene and actuator evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl').resolve()
RUN = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T135953'
PRIOR = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T134504'
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_scene_zero_nominal_s1242_20261116.json'
WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_scene_native_nominal_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    output = REPORT / 'scene-nominal-zero-verification.json'
    assert not output.exists()
    plan, wrapper = read(PLAN), read(WRAPPER)
    result, receipt = read(RUN / 'native-zero-run/result.json'), read(RUN / 'native-zero-execution.json')
    runtime, startup = read(RUN / 'zero/runtime-contract.json'), read(RUN / 'zero/startup.json')
    proof = read(RUN / 'native-zero-run/ui-proof.json')
    actual = read(RUN / 'zero/actuator-start-state.json')
    metadata = read(RUN / 'zero/actuator-observed-start.json')
    state_path = RUN / 'zero/start-fingerprints.json'
    state = read(state_path)
    cpu = read(REPORT / 'initialized-scene-fingerprint-cpu.json')
    cpu_run = read(REPORT / 'initialized-scene-inventory-cpu.json')
    assert result['status'] == 'zero-completed-awaiting-independent-verification'
    assert result['cuda_initialized'] and result['physics_substeps'] == 0
    assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    assert receipt['log_sha256'] == sha(RUN / 'native-zero-execution.log')
    assert result['protocol_sha256'] == wrapper['upstream_plan_sha256'] == sha(PLAN)
    assert result['native_wrapper_sha256'] == sha(WRAPPER)
    assert result['native_ui_proof_sha256'] == sha(RUN / 'native-zero-run/ui-proof.json')
    assert proof['run_id'] == plan['run_id']
    assert proof['screenshot_sha256'] == sha(RUN / 'native-zero-run/ui-check.jpg')
    assert runtime['control_steps'] == runtime['physics_substeps'] == 0
    assert actual['physical_steps'] == metadata['physical_steps'] == 0 and actual['worlds'] == 128
    assert not metadata['errors'] and not metadata['missing']
    assert actual['observed_metadata_sha256'] == sha(RUN / 'zero/actuator-observed-start.json')
    arrays = RUN / 'zero/actuator-observed-start.npz'
    assert actual['arrays_sha256'] == result['actuator_start_state'] == metadata['arrays_sha256'] == sha(arrays)
    assert actual['fingerprint_sha256'] == result['start_fingerprint'] == sha(state_path)
    assert state['status'] == 'captured-zero-step-fingerprints' and state['run_id'] == plan['run_id']
    assert state['protocol_sha256'] == sha(PLAN) and state['counters'] == [0, 0] and state['rng_unchanged']
    assert state['worlds'] == 128 and state['field_count'] == len(state['fields']) == 310
    assert state['fields_digest'] == digest(state['fields']) and state['rng_digest'] == digest(state['rng'])
    assert cpu_run['fingerprint_sha256'] == sha(REPORT / 'initialized-scene-fingerprint-cpu.json')
    assert set(state['fields']) == set(cpu['fields'])
    builtin = [name for name, kind in cpu_run['sensor_types'].items() if kind == 'BuiltinSensor']
    assert len(builtin) == 13 and all('sensor.' + name + '._data_view' in state['fields'] for name in builtin)
    assert 'terrain.env_origins' in state['fields'] and any(x.startswith('terrain.data.') for x in state['fields'])
    assert max(startup['actual_onnx_parity'].values()) < 2e-5
    with np.load(arrays, allow_pickle=False) as observed, np.load(PRIOR/'zero/actuator-observed-start.npz', allow_pickle=False) as prior:
        assert set(observed.files) == set(prior.files)
        assert all(np.array_equal(observed[name], prior[name]) for name in observed.files)
        assert observed['vin_tensor'].shape == (128, 1) and np.allclose(observed['vin_tensor'], 7.7, rtol=0, atol=1e-6)
        torque = observed['previous_motor_torque']
        assert torque.shape == (128, 14) and np.isfinite(torque).all()
        limits = observed['actuator_forcerange']
        assert limits.shape == (14, 2) and np.all(observed['actuator_forcelimited'])
        np.testing.assert_allclose(limits[:, 0], -0.9781615622273244, rtol=1e-7)
        np.testing.assert_allclose(limits[:, 1], 0.9781615622273244, rtol=1e-7)
        torque_summary = {'minimum': float(torque.min()), 'maximum': float(torque.max()),
                          'nonzero_count': int(np.count_nonzero(torque)), 'size': int(torque.size)}
    counts = {}
    for key, base in [('frozen_sha256', ROOT), ('training_source_sha256', WORK), ('library_source_sha256', Path('/'))]:
        files = plan[key]
        for name, expected in files.items():
            path = (base / name).resolve()
            assert path.is_relative_to(ROOT if key == 'frozen_sha256' else WORK)
            assert path.is_file() and sha(path) == expected, str(path)
        counts[key] = len(files)
    for name, expected in wrapper['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    verification = {'status': 'passed-independent-nominal-scene-zero-readback',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'run_id': plan['run_id'],
        'protocol_sha256': sha(PLAN), 'wrapper_sha256': sha(WRAPPER),
        'field_count': state['field_count'], 'builtin_sensor_count': len(builtin),
        'cpu_gpu_field_names_equal': True, 'worlds': 128, 'physics_substeps': 0, 'learning_updates': 0,
        'rng_unchanged_during_capture': True, 'actuator_arrays_equal_previous_zero': True,
        'torque': torque_summary, 'actual_onnx_parity': startup['actual_onnx_parity'],
        'source_file_counts': counts, 'wrapper_source_count': len(wrapper['source_sha256']),
        'ui_screenshot_sha256': proof['screenshot_sha256'], 'hardware_io': False,
        'scope': 'Saved nominal zero-step readback only; other factor arms, paired starts and dynamics remain unverified.'}
    output.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
