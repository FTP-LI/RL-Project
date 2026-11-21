"""Read back the nominal repeat and compare saved same-factor fingerprints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl').resolve()
REPORT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'
BASE = ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T135953'
RUN = ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T141555'
PLAN = ROOT/'configs/experiments/roll_factor_fp_raw_zero_nominal_s1242_20261116.json'
WRAPPER = ROOT/'configs/experiments/roll_factor_fp_raw_native_nominal_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(array):
    return [array.dtype.str, list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]


def main():
    output = REPORT/'raw-nominal-zero-verification.json'
    assert not output.exists()
    plan, wrapper = read(PLAN), read(WRAPPER)
    result, receipt = read(RUN/'native-zero-run/result.json'), read(RUN/'native-zero-execution.json')
    proof, runtime = read(RUN/'native-zero-run/ui-proof.json'), read(RUN/'zero/runtime-contract.json')
    a, b = read(BASE/'zero/start-fingerprints.json'), read(RUN/'zero/start-fingerprints.json')
    meta = read(RUN/'zero/selected-zero-numeric.json')
    arrays = RUN/'zero/selected-zero-numeric.npz'
    assert result['status'] == 'zero-completed-awaiting-independent-verification' and result['cuda_initialized']
    assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    assert receipt['log_sha256'] == sha(RUN/'native-zero-execution.log')
    assert result['protocol_sha256'] == wrapper['upstream_plan_sha256'] == sha(PLAN)
    assert result['native_wrapper_sha256'] == sha(WRAPPER) and proof['run_id'] == plan['run_id']
    assert proof['screenshot_sha256'] == sha(RUN/'native-zero-run/ui-check.jpg')
    assert result['native_ui_proof_sha256'] == sha(RUN/'native-zero-run/ui-proof.json')
    assert runtime['physics_substeps'] == runtime['control_steps'] == 0
    assert b['counters'] == [0, 0] and b['rng_unchanged'] and a['rng'] == b['rng']
    assert result['start_fingerprint'] == meta['fingerprint_sha256'] == sha(RUN/'zero/start-fingerprints.json')
    assert meta['arrays_sha256'] == sha(arrays) and meta['protocol_sha256'] == sha(PLAN)
    assert set(a['fields']) == set(b['fields']) and len(b['fields']) == 310
    changed = sorted(name for name in a['fields'] if a['fields'][name] != b['fields'][name])
    prior = read(REPORT/'friction-low-scene-zero-comparison.json')['changed_beyond_declared_friction']
    assert set(changed) == set(prior)
    with np.load(arrays, allow_pickle=False) as values:
        assert len(values.files) == 34 and set(values.files) == set(meta['fields'])
        assert all(fingerprint(values[name]) == b['fields'][name] for name in values.files)
        array_sizes = {name: {'shape': list(values[name].shape), 'dtype': str(values[name].dtype)} for name in values.files}
    comparisons = {}
    for name in ('startup.npz', 'reset-0-interface.npz', 'endpoint-contact.npz', 'actuator-observed-start.npz'):
        with np.load(BASE/'zero'/name, allow_pickle=False) as x, np.load(RUN/'zero'/name, allow_pickle=False) as y:
            assert set(x.files) == set(y.files)
            comparisons[name] = all(np.array_equal(x[key], y[key]) for key in x.files)
    assert all(comparisons.values())
    cache_a, cache_b = a['fields']['env._ground_support_cache'], b['fields']['env._ground_support_cache']
    assert len(cache_a) == len(cache_b) == 1 and next(iter(cache_a.values())) == next(iter(cache_b.values()))
    counts = {}
    for key, base in [('frozen_sha256', ROOT), ('training_source_sha256', WORK), ('library_source_sha256', Path('/'))]:
        for name, expected in plan[key].items():
            path = (base/name).resolve()
            assert path.is_relative_to(ROOT if key == 'frozen_sha256' else WORK)
            assert path.is_file() and sha(path) == expected, str(path)
        counts[key] = len(plan[key])
    for name, expected in wrapper['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    report = {'status': 'passed-nominal-raw-zero-readback', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'run_id': plan['run_id'], 'same_factor_previous_run': a['run_id'], 'seed': 1242, 'worlds': 128,
        'same_factor_changed_fields': changed, 'same_factor_changes_equal_prior_friction_nonparameter_changes': True,
        'cache_values_equal_without_process_key': True, 'startup_and_interface_arrays_equal': comparisons,
        'rng_equal': True, 'numeric_field_count': len(array_sizes), 'numeric_arrays_sha256': sha(arrays),
        'numeric_field_shapes': array_sizes, 'source_file_counts': counts,
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'scope': 'Saved same-factor repeat; numerical differences between new nominal and low-friction arrays still require a separate run.'}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'run_id', 'same_factor_changed_fields',
        'same_factor_changes_equal_prior_friction_nonparameter_changes', 'rng_equal', 'numeric_field_count')}))


if __name__ == '__main__':
    main()
