"""Compare saved nominal and low-friction zero-step evidence without sampling."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
from selective_friction_guard import compare_reset_fields, compare_static

WORK = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl').resolve()
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
NOMINAL = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T135953'
LOW = ROOT / 'records/reports/duck_roll-factor_friction-low_s1242_20261116T140434'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_scene_zero_friction-low_s1242_20261116.json'
WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_scene_native_friction-low_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def equal_archive(name):
    first, second = arrays(NOMINAL / 'zero' / name), arrays(LOW / 'zero' / name)
    assert set(first) == set(second)
    return {key: bool(np.array_equal(first[key], second[key])) for key in sorted(first)}


def main():
    output = REPORT / 'friction-low-scene-zero-comparison.json'
    assert not output.exists()
    baseline = read(REPORT / 'scene-nominal-zero-verification.json')
    plan, wrapper = read(PLAN), read(WRAPPER)
    result, receipt = read(LOW / 'native-zero-run/result.json'), read(LOW / 'native-zero-execution.json')
    runtime, proof = read(LOW / 'zero/runtime-contract.json'), read(LOW / 'native-zero-run/ui-proof.json')
    a, b = read(NOMINAL / 'zero/start-fingerprints.json'), read(LOW / 'zero/start-fingerprints.json')
    assert baseline['status'] == 'passed-independent-nominal-scene-zero-readback'
    assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    assert receipt['log_sha256'] == sha(LOW / 'native-zero-execution.log')
    assert result['status'] == 'zero-completed-awaiting-independent-verification' and result['cuda_initialized']
    assert result['protocol_sha256'] == wrapper['upstream_plan_sha256'] == sha(PLAN)
    assert result['native_wrapper_sha256'] == sha(WRAPPER)
    assert proof['run_id'] == plan['run_id'] and proof['screenshot_sha256'] == sha(LOW/'native-zero-run/ui-check.jpg')
    assert result['native_ui_proof_sha256'] == sha(LOW / 'native-zero-run/ui-proof.json')
    assert runtime['control_steps'] == runtime['physics_substeps'] == 0
    assert result['start_fingerprint'] == sha(LOW / 'zero/start-fingerprints.json')
    assert result['actuator_start_state'] == sha(LOW / 'zero/actuator-observed-start.npz')
    for state in (a, b):
        assert state['field_count'] == len(state['fields']) == 310 and state['worlds'] == 128
        assert state['counters'] == [0, 0] and state['rng_unchanged']
        assert state['fields_digest'] == digest(state['fields']) and state['rng_digest'] == digest(state['rng'])
    assert set(a['fields']) == set(b['fields'])
    changed = sorted(name for name in a['fields'] if a['fields'][name] != b['fields'][name])
    unexpected = [name for name in changed if name != 'model.geom_friction']
    static_check = compare_static(arrays(LOW / 'zero/scene-static-fields.npz'),
                                  arrays(NOMINAL / 'zero/scene-static-fields.npz'))
    reset_check = compare_reset_fields(arrays(LOW / 'zero/expanded-fields-before-physics.npz'),
        arrays(NOMINAL / 'zero/expanded-fields-before-physics.npz'),
        foot_geom_ids=result['foot_geom_ids'], friction=0.8, worlds=128)
    archive_checks = {name: equal_archive(name) for name in
        ('startup.npz', 'reset-0-interface.npz', 'endpoint-contact.npz', 'actuator-observed-start.npz')}
    assert all(all(check.values()) for check in archive_checks.values())
    assert a['rng'] == b['rng']
    for key, base in [('frozen_sha256', ROOT), ('training_source_sha256', WORK), ('library_source_sha256', Path('/'))]:
        for name, expected in plan[key].items():
            path = (base / name).resolve()
            assert path.is_relative_to(ROOT if key == 'frozen_sha256' else WORK)
            assert path.is_file() and sha(path) == expected, str(path)
    for name, expected in wrapper['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    status = 'passed-matched-friction-zero-comparison' if not unexpected else 'review-required-internal-state-differences'
    result = {'status': status, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'nominal_run': a['run_id'], 'friction_run': b['run_id'], 'seed': plan['seed'], 'worlds': 128,
        'static_model_check': static_check, 'expanded_factor_check': reset_check,
        'saved_startup_and_interface_fields_equal': archive_checks, 'rng_equal': True,
        'fingerprint_field_count': len(a['fields']), 'changed_fingerprint_fields': changed,
        'changed_beyond_declared_friction': unexpected,
        'interpretation': 'Different internal fingerprints require review; the saved digests alone do not prove cause or size.',
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'dynamic_execution_authorized': False, 'high_voltage_started': False,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), PLAN, WRAPPER,
            NOMINAL/'zero/start-fingerprints.json', LOW/'zero/start-fingerprints.json',
            LOW/'native-zero-execution.log', LOW/'native-zero-run/ui-proof.json']}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
