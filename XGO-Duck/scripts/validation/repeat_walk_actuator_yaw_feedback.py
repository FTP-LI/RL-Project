"""Use new releases with the previously admitted actuator feedback and scorer."""
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import common_walk_actuator_yaw_feedback as trial
import verify_walk_actuator_yaw_feedback as verifier
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback_replication_draft.json'
PLAN = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback_replication.json'
OUT = ROOT/'records/reports/common-walk-actuator-yaw-feedback-replication'
PAGE = ROOT/'records/walk-actuator-feedback-replication-live.html'
STAGE = 'S2-forward-yaw-actuator-feedback-replication'


def freeze():
    attempt = read(DRAFT)
    base = ROOT/attempt['base_draft']
    previous_plan = ROOT/attempt['prior_development_protocol']
    previous_result = ROOT/attempt['prior_development_result']
    protocol, result = read(previous_plan), read(previous_result)
    assert result['status'] == 'passed-integrity' and result['continued_admission']
    assert not result['heldout_used'] and not protocol['heldout_used']
    assert result['protocol_sha256'] == sha(previous_plan)
    trial.rollout.check_sources(protocol)

    def load(path):
        if path == DRAFT:
            return dict(read(base), development_seed=attempt['development_seed'], stage=STAGE)
        if path == trial.rollout.PLAN:
            # Start with the admitted controller and frozen thresholds. Only
            # scene assignment and the paired bootstrap seed change.
            for file in (base, previous_plan, previous_result):
                protocol['reference_sha256'][file.relative_to(ROOT).as_posix()] = sha(file)
            source = Path(__file__).resolve()
            protocol['source_sha256'][source.relative_to(ROOT).as_posix()] = sha(source)
            return protocol
        return read(path)

    with patch.object(trial, 'read', load):
        trial.freeze()


def main():
    trial.pulse.check_project()
    commands = {'--freeze': freeze, '--sample': trial.sample, '--verify': verifier.main}
    assert len(sys.argv) == 2 and sys.argv[1] in commands
    with ExitStack() as scope:
        for name, value in dict(DRAFT=DRAFT, PLAN=PLAN, OUT=OUT, PAGE=PAGE, STAGE=STAGE).items():
            scope.enter_context(patch.object(trial, name, value))
        commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
