"""Explain the support-cache key difference using saved fingerprints and code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'
CPU = REPORT/'cpu-friction-repeat-review'
GPU_A = ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T135953/zero/start-fingerprints.json'
GPU_B = ROOT/'records/reports/duck_roll-factor_friction-low_s1242_20261116T140434/zero/start-fingerprints.json'
MDP = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def values_without_process_key(state):
    cache = state['fields']['env._ground_support_cache']
    assert isinstance(cache, dict) and len(cache) == 1
    return next(iter(cache.values()))


def main():
    output = REPORT/'friction-cache-identity-review.json'
    assert not output.exists()
    analysis, receipt = read(CPU/'analysis.json'), read(CPU/'execution.json')
    assert analysis['status'] == 'completed-cpu-repeat-diagnostic' and receipt['status'] == 'completed'
    assert receipt['exit_code'] == 0 and receipt['log_sha256'] == sha(CPU/'execution.log')
    assert analysis['physical_steps'] == 0 and analysis['learning_updates'] == 0
    for name, expected in analysis['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    source = MDP.read_text(encoding='utf-8')
    assert 'cache = env.__dict__.setdefault("_ground_support_cache", {})' in source
    assert 'if id(asset) not in cache:' in source
    a, b, low = (read(CPU/(name+'-fingerprint.json')) for name in ('nominal-a', 'nominal-b', 'friction-low'))
    gpu_a, gpu_b = read(GPU_A), read(GPU_B)
    assert values_without_process_key(a) == values_without_process_key(b) == values_without_process_key(low)
    assert values_without_process_key(gpu_a) == values_without_process_key(gpu_b)
    assert a['rng'] == b['rng'] == low['rng'] and gpu_a['rng'] == gpu_b['rng']
    gpu_differences = read(REPORT/'friction-low-scene-zero-comparison.json')['changed_beyond_declared_friction']
    assert 'env._ground_support_cache' in gpu_differences
    remaining = [name for name in gpu_differences if name != 'env._ground_support_cache']
    result = {'status': 'saved-cache-identity-and-cpu-repeat-reviewed',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'cpu_same_factor_changed_fields': analysis['same_factor_repeat']['changed_fields'],
        'cpu_low_friction_changed_fields': analysis['low_friction_comparison']['changed_fields'],
        'cpu_rng_equal': True, 'gpu_rng_equal': True, 'cpu_cache_values_equal_after_key_removal': True,
        'gpu_cache_values_equal_after_key_removal': True,
        'source_reason': 'The support cache is keyed by id(asset), which differs between independently created objects.',
        'gpu_other_changed_field_count': len(remaining), 'gpu_other_changed_fields': remaining,
        'scope': 'Cache-key explanation is supported by source and saved values. CPU repeats do not establish GPU field cause or magnitude.',
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'dynamic_execution_authorized': False, 'high_voltage_started': False,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__),
            CPU/'analysis.json', CPU/'execution.json', CPU/'nominal-a-fingerprint.json',
            CPU/'nominal-b-fingerprint.json', CPU/'friction-low-fingerprint.json', GPU_A, GPU_B]},
        'training_source': str(MDP), 'training_source_sha256': sha(MDP)}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
