"""Score saved runs while reporting physical drift after the shared first substep."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_getup_archived_policy_qualified as trial
import verify_getup_archived_policy_comparison as scoring
from verify_getup_shared_release import same_bytes


def verify():
    spec = trial.read(trial.PLAN)
    record = trial.ROOT / spec['record']
    original_log = trial.OUT / 'verify.log'
    assert 'AssertionError: motor' in original_log.read_text(encoding='utf-8')
    qual = trial.read(record / 'zero-qualification.json')
    assert qual['status'] == 'passed-process-local-zero' and qual['physical_substeps'] == 0
    assert trial.read(trial.OUT / 'native-zero.json')['qualification_sha256'] == trial.sha(record / 'zero-qualification.json')
    evidence = trial.OUT / 'reference-motor-drift.json'
    differences = []

    def check_pair(first, second):
        if 'motor' not in first:
            return same_bytes(first, second)
        assert first.keys() == second.keys()
        same_bytes({key: value for key, value in first.items() if key != 'motor'},
                   {key: value for key, value in second.items() if key != 'motor'})
        a, b = first['motor'], second['motor']
        assert a.shape == b.shape and a.shape[:2] == (5, spec['environment']['num_envs'])
        same_bytes({'motor': a[0]}, {'motor': b[0]})
        assert not np.array_equal(a[1:], b[1:]) and np.isfinite(a).all() and np.isfinite(b).all()
        for k, delta in enumerate(np.abs(a.astype(np.float64) - b)):
            differences.append(dict(physical_substep=k + 1, max_absolute=float(delta.max()),
                rms=float(np.sqrt(np.mean(delta**2))), worlds_changed=int((delta != 0).any(axis=1).sum()),
                entries_above_1e_4=int((delta > 1e-4).sum())))
        trial.write(evidence, dict(status='measured-native-post-actuation-drift',
            protocol_sha256=trial.sha(trial.PLAN), first_controller_arrays_exact=True, first_substep_exact=True,
            first_five_substeps_byte_exact=False, physical_substeps_added=0, per_substep=differences,
            original_strict_check_log_sha256=trial.sha(original_log)))

    def save(path, result):
        assert path == trial.OUT / 'analysis.json' and len(differences) == 5
        result.update(first_five_motor_substeps_exact_between_repeats=False,
                      reference_motor_drift_sha256=trial.sha(evidence), original_strict_check_stopped=True,
                      verifier_sha256=trial.sha(Path(__file__)), added_physical_substeps=0)
        trial.write(path, result)

    with patch.object(scoring, 'files', lambda seed: (trial.PLAN, trial.OUT, trial.PAGE)), \
            patch.object(scoring, 'same_bytes', check_pair), patch.object(scoring, 'write', save):
        scoring.verify(trial.SEED)


if __name__ == '__main__':
    verify()
