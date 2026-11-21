"""Recover reset-aliased raw actions from an independent, unfiltered copy.

Only for the already completed foot-contact mechanism diagnosis. Preserve the
original NPZ and every non-action array; no simulation, policy call or changed
online result. This refuses filtered actions because inversion is not exact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    folder = parser.parse_args().run.resolve()
    assert folder.parent == ROOT/'records/evaluations'
    report = read(folder/'evaluation.json')
    assert report['status'] == 'completed'
    assert report['mechanism_variant'] == 'foot_contact'
    assert np.all(np.asarray(report['action_filter_alpha']) == 0)
    source = folder/'evaluate_fixed_policy.py'
    assert sha(source) == report['evaluator_sha256']
    code = source.read_text()
    assert 'trace[key].append(value)' in code
    assert 'applied_actions=filtered.copy()' in code
    assert 'previous = actions' in code and 'previous[reset_ids.cpu().numpy()] = 0' in code
    backup = folder/'trace-repair'
    assert not backup.exists(), 'Existing repair must be inspected, not overwritten'
    prepared = []
    for item in report['conditions']:
        path = folder/(item['condition']+'-trace.npz')
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key:saved[key].copy() for key in saved.files}
        raw, applied = arrays['actions'], arrays['applied_actions']
        # alpha=0 means applied_actions is an exact copy of the original raw
        # action; check target reconstruction as a separate saved constraint.
        restored = applied.astype(raw.dtype)
        np.testing.assert_array_equal(applied, restored)
        home = np.asarray(report['policy_home'])
        scale = float(report['model_metadata']['action_scale'])
        np.testing.assert_allclose(arrays['policy_joint_targets'], home+scale*applied, atol=1e-5)
        changed = np.any(raw != restored, axis=-1)
        assert np.all(raw[changed] == 0)
        assert np.all((arrays['terminated'] | arrays['truncated'])[changed])
        # Restrict repair to the exact known alias failure; this does not alter
        # trajectory, termination, videos, timing or the original metrics.
        evidence = dict(condition=item['condition'], original_sha256=sha(path),
                        changed_frame_env_indices=np.argwhere(changed).tolist(),
                        changed_active_rows=int((changed & arrays['active']).sum()))
        prepared.append((path, arrays, restored, evidence))
    assert any(row[3]['changed_frame_env_indices'] for row in prepared)
    backup.mkdir()
    shutil.copy2(Path(__file__), backup/Path(__file__).name)
    result = dict(status='repairing', source_evaluator_sha256=sha(source),
                  report_sha256=sha(folder/'evaluation.json'),
                  reason='Raw action array aliased previous and was zeroed at reset; alpha=0 independently saved applied_actions recovers exact values',
                  simulation_rerun=False, online_metrics_changed=False, conditions=[])
    for path, arrays, restored, evidence in prepared:
        shutil.copy2(path, backup/path.name)
        arrays['actions'] = restored
        temp = path.with_name(path.stem+'-repair.tmp.npz')
        np.savez_compressed(temp, **arrays)
        with np.load(backup/path.name) as original, np.load(temp) as repaired:
            assert set(original.files) == set(repaired.files)
            for key in original.files:
                if key != 'actions':
                    np.testing.assert_array_equal(original[key], repaired[key])
            np.testing.assert_array_equal(repaired['actions'], repaired['applied_actions'])
        temp.replace(path)
        evidence['repaired_sha256'] = sha(path)
        evidence['unchanged_non_action_arrays_verified'] = True
        result['conditions'].append(evidence)
    result.update(status='repaired-awaiting-independent-verification',
                  completed_at=datetime.now(timezone.utc).isoformat())
    (backup/'repair.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
