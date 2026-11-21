"""Inspect saved actuator arrays and source after a zero-step stop."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T134504'
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
SCENE = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/lib/python3.12/site-packages/mjlab/scene/scene.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = REPORT / 'logged-zero-stop-analysis.json'
    assert not out.exists()
    result = read(RUN / 'native-zero-run/result.json')
    runtime = read(RUN / 'zero/runtime-contract.json')
    actuator = read(RUN / 'zero/actuator-observed-start.json')
    arrays = RUN / 'zero/actuator-observed-start.npz'
    log = (RUN / 'native-zero-execution.log').read_text(encoding='utf-8')
    source = SCENE.read_text(encoding='utf-8')
    config = (RUN / 'zero/env.yaml').read_text(encoding='utf-8')
    assert result['status'] == 'stopped-review-required' and result['error_type'] == 'AssertionError'
    assert runtime['control_steps'] == runtime['physics_substeps'] == 0
    assert actuator['arrays_sha256'] == sha(arrays) and not actuator['errors'] and not actuator['missing']
    assert "assert set(env.scene.entities)=={'robot'}" in log
    assert 'self._entities["terrain"] = terrain' in source
    assert 'self._entities[ent_name] = ent' in source and '\n  entities:\n    robot:' in config
    with np.load(arrays, allow_pickle=False) as data:
        summary = {}
        for name in data.files:
            value = data[name]
            summary[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                'minimum': float(value.min()), 'maximum': float(value.max()),
                'nonzero_count': int(np.count_nonzero(value)), 'size': int(value.size),
                'finite': bool(np.isfinite(value).all())}
    events = [json.loads(line) for line in (RUN / 'native-zero-run/guard-rejections.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['original_decision']['reason'] == 'observer-health-unavailable'
    assert events[0]['checked_at'] < read(RUN / 'native-zero-run/ui-proof.json')['observed_at']
    proof = read(RUN / 'native-zero-run/ui-proof.json')
    assert proof['screenshot_sha256'] == sha(RUN / 'native-zero-run/ui-check.jpg')
    analysis = {'status': 'saved-zero-stop-and-source-reviewed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'run_id': actuator['run_id'], 'control_steps': runtime['control_steps'],
        'physics_substeps': runtime['physics_substeps'], 'learning_updates': 0, 'hardware_io': False,
        'actuator_fields': summary, 'actuator_arrays_sha256': sha(arrays),
        'actual_actuator_validation_errors': actuator['errors'],
        'scene_assertion': "set(env.scene.entities)=={'robot'}",
        'source_based_scene_cause': 'Configured terrain is inserted into scene.entities as terrain, alongside robot.',
        'scene_source_sha256': sha(SCENE), 'actual_entity_set_saved': False,
        'observer_log': 'Only the expected pre-publication missing-health event was logged; later UI proof was captured.',
        'previous_transient_refusal_cause_known': False,
        'first_actuator_stop_cause_proven': False,
        'no_pair_or_dynamics_claim': True,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__),
            RUN / 'native-zero-execution.log', RUN / 'zero/runtime-contract.json',
            RUN / 'zero/actuator-observed-start.json', RUN / 'native-zero-run/ui-proof.json',
            RUN / 'native-zero-run/guard-rejections.jsonl']}}
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == '__main__':
    main()
