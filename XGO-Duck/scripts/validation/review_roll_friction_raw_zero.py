"""Compare saved numeric zero-step states, including active constraint order."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl').resolve()
REPORT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'
A = ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T141555'
B = ROOT/'records/reports/duck_roll-factor_friction-low_s1242_20261116T141926'
PLAN = ROOT/'configs/experiments/roll_factor_fp_raw_zero_friction-low_s1242_20261116.json'
WRAPPER = ROOT/'configs/experiments/roll_factor_fp_raw_native_friction-low_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(array):
    return [array.dtype.str, list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]


def statistics(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.dtype.kind in 'biuf'
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {'shape': list(left.shape), 'dtype': str(left.dtype),
        'exactly_equal': bool(np.array_equal(left, right)),
        'different_elements': int(np.count_nonzero(left != right)),
        'nonfinite_pairs': int(finite.size - np.count_nonzero(finite)),
        'maximum_finite_absolute_difference': float(delta[finite].max()) if finite.any() else None}


def check_run(folder, plan_hash, wrapper_hash):
    result, receipt = read(folder/'native-zero-run/result.json'), read(folder/'native-zero-execution.json')
    proof, runtime = read(folder/'native-zero-run/ui-proof.json'), read(folder/'zero/runtime-contract.json')
    meta, state = read(folder/'zero/selected-zero-numeric.json'), read(folder/'zero/start-fingerprints.json')
    data_path = folder/'zero/selected-zero-numeric.npz'
    assert result['status'] == 'zero-completed-awaiting-independent-verification' and result['cuda_initialized']
    assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    assert receipt['log_sha256'] == sha(folder/'native-zero-execution.log')
    assert result['protocol_sha256'] == meta['protocol_sha256'] == plan_hash
    assert result['native_wrapper_sha256'] == proof['wrapper_sha256'] == wrapper_hash
    assert result['native_ui_proof_sha256'] == sha(folder/'native-zero-run/ui-proof.json')
    assert proof['screenshot_sha256'] == sha(folder/'native-zero-run/ui-check.jpg')
    assert result['start_fingerprint'] == meta['fingerprint_sha256'] == sha(folder/'zero/start-fingerprints.json')
    assert meta['arrays_sha256'] == sha(data_path) and runtime['physics_substeps'] == runtime['control_steps'] == 0
    assert state['worlds'] == 128 and state['counters'] == [0, 0] and state['rng_unchanged']
    with np.load(data_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    assert len(values) == 34 and set(values) == set(meta['fields'])
    assert all(fingerprint(value) == state['fields'][name] for name, value in values.items())
    return state, values


def main():
    output = REPORT/'friction-raw-zero-comparison.json'
    assert not output.exists()
    nominal = read(REPORT/'raw-nominal-zero-verification.json')
    assert nominal['status'] == 'passed-nominal-raw-zero-readback'
    plan, wrapper = read(PLAN), read(WRAPPER)
    assert wrapper['upstream_plan_sha256'] == sha(PLAN) and plan['factor_arm'] == 'friction-low'
    nominal_plan = ROOT/'configs/experiments/roll_factor_fp_raw_zero_nominal_s1242_20261116.json'
    nominal_wrapper = ROOT/'configs/experiments/roll_factor_fp_raw_native_nominal_s1242_20261116.json'
    a, x = check_run(A, sha(nominal_plan), sha(nominal_wrapper))
    b, y = check_run(B, sha(PLAN), sha(WRAPPER))
    assert set(x) == set(y) and a['rng'] == b['rng']
    summary = {name: statistics(x[name], y[name]) for name in sorted(x)}
    assert not any(item['nonfinite_pairs'] for item in summary.values())
    assert np.array_equal(x['warp.nacon'], y['warp.nacon']) and int(x['warp.nacon'][0]) == 0
    assert np.array_equal(x['warp.nefc'], y['warp.nefc']) and np.all(x['warp.nefc'] == 14)
    # Each active ID is unique within a world. Sorting by ID makes row-order
    # effects visible; this does not by itself prove cross-run physical parity.
    ids_a, ids_b = x['warp.efc.id'][:, :14], y['warp.efc.id'][:, :14]
    order_a, order_b = np.argsort(ids_a, axis=1), np.argsort(ids_b, axis=1)
    assert all(len(set(row)) == 14 for row in ids_a) and all(len(set(row)) == 14 for row in ids_b)
    assert np.array_equal(np.take_along_axis(ids_a, order_a, axis=1), np.take_along_axis(ids_b, order_b, axis=1))
    aligned = {}
    for name in ('warp.efc.id', 'warp.efc.D', 'warp.efc.frictionloss', 'warp.efc.force', 'warp.efc.state'):
        aligned[name] = statistics(np.take_along_axis(x[name][:, :14], order_a, axis=1),
                                   np.take_along_axis(y[name][:, :14], order_b, axis=1))
    aligned['warp.efc.J'] = statistics(np.take_along_axis(x['warp.efc.J'][:, :14, :], order_a[:, :, None], axis=1),
                                       np.take_along_axis(y['warp.efc.J'][:, :14, :], order_b[:, :, None], axis=1))
    for name in ('warp.efc.id', 'warp.efc.D', 'warp.efc.frictionloss', 'warp.efc.state', 'warp.efc.J'):
        assert aligned[name]['exactly_equal'], name
    startup = {}
    for name in ('startup.npz', 'reset-0-interface.npz', 'endpoint-contact.npz', 'actuator-observed-start.npz'):
        with np.load(A/'zero'/name, allow_pickle=False) as first, np.load(B/'zero'/name, allow_pickle=False) as second:
            assert set(first.files) == set(second.files)
            startup[name] = all(np.array_equal(first[k], second[k]) for k in first.files)
    assert all(startup.values())
    for key, base in [('frozen_sha256', ROOT), ('training_source_sha256', WORK), ('library_source_sha256', Path('/'))]:
        for name, expected in plan[key].items():
            path = (base/name).resolve()
            assert path.is_relative_to(ROOT if key == 'frozen_sha256' else WORK)
            assert path.is_file() and sha(path) == expected, str(path)
    for name, expected in wrapper['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    result = {'status': 'raw-comparison-completed-dynamic-review-required',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'worlds': 128, 'seed': 1242,
        'nominal_run': a['run_id'], 'low_friction_run': b['run_id'],
        'startup_and_interface_arrays_equal': startup, 'rng_equal': True,
        'detected_contact_count_all_worlds': 0, 'active_constraints_each_world': 14,
        'raw_fields': summary, 'active_constraints_sorted_by_unique_id': aligned,
        'constraint_sort_scope': 'Diagnostic alignment by unique active ID; it does not establish dynamic trajectory equivalence.',
        'same_factor_repeat_showed_same_prior_fingerprint_difference_names': True,
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'dynamic_execution_authorized': False, 'high_voltage_started': False,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), PLAN, WRAPPER,
            A/'zero/selected-zero-numeric.npz', B/'zero/selected-zero-numeric.npz',
            A/'zero/start-fingerprints.json', B/'zero/start-fingerprints.json', B/'native-zero-execution.log']}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'contacts': result['detected_contact_count_all_worlds'],
        'qacc_max_delta': summary['warp.qacc']['maximum_finite_absolute_difference'],
        'cacc_max_delta': summary['warp.cacc']['maximum_finite_absolute_difference'],
        'imu_max_delta': summary['sensor.robot/imu_accel._data_view']['maximum_finite_absolute_difference'],
        'aligned_force_max_delta': aligned['warp.efc.force']['maximum_finite_absolute_difference']}))


if __name__ == '__main__':
    main()
