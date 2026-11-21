"""Provide the saved termination field to the frozen seven-command scorer."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

import routed_seven_command_score as scorer
import run_routed_seven_command as study
import verify_routed_seven_command as frozen


ORIGINAL_SCORE = scorer.score


def score(spec, state, record, trace, initial):
    assert 'terminated' not in trace
    pieces = []
    for item in state['record_chunks']:
        path = record/item['file']
        assert study.sha(path) == item['sha256']
        with np.load(path, allow_pickle=False) as saved:
            pieces.append(saved['terminated'])
    terminated = np.concatenate(pieces)
    assert terminated.shape == trace['mode'].shape and terminated.dtype == np.bool_
    assert not terminated.any(), 'The frozen target audit asserts this invariant separately'
    return ORIGINAL_SCORE(spec, state, record, dict(trace, terminated=terminated), initial)


def main():
    log = study.OUT/'verify.log'
    text = log.read_text(encoding='utf-8')
    assert "KeyError: 'terminated'" in text and 'result = score(' in text
    target = study.OUT/'readback-recovery.json'
    assert not target.exists() and not (study.OUT/'analysis.json').exists()
    with patch.object(scorer, 'score', score):
        frozen.verify()
    result = study.OUT/'analysis.json'
    study.write(target, dict(status='verified-saved-termination-field-recovery',
        original_error="KeyError: 'terminated'", initial_verifier_log_sha256=study.sha(log),
        recovery_script_sha256=study.sha(Path(__file__)), protocol_sha256=study.sha(study.PLAN),
        analysis_sha256=study.sha(result), original_sources_unchanged=True, physical_substeps=0))


if __name__ == '__main__':
    main()
