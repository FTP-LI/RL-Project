"""Confirm the frozen feedback on a reserved seed after two passing developments."""
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import common_walk_actuator_yaw_feedback as trial
import repeat_walk_actuator_yaw_feedback as replication
import verify_walk_actuator_yaw_feedback as verifier
from record_io import write
from s1_reference_inputs import ROOT, read, sha


BASE_DRAFT = trial.DRAFT
DRAFT = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback_heldout_draft.json'
PLAN = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback_heldout.json'
OUT = ROOT/'records/reports/common-walk-actuator-yaw-feedback-heldout'
PAGE = ROOT/'records/walk-actuator-feedback-heldout-live.html'
STAGE = 'S2-forward-yaw-actuator-feedback-heldout'


def freeze():
    draft = read(DRAFT)
    assert draft['status'] == 'reserved-not-executable' and not draft['heldout_used']
    first, second = (ROOT/path for path in draft['require_completed_development_results'])
    plans = (ROOT/draft['controller_and_gates'], replication.PLAN)
    protocols = [read(path) for path in plans]
    for path, protocol, result_path in zip(plans, protocols, (first, second), strict=True):
        result = read(result_path)
        assert result['status'] == 'passed-integrity' and result['continued_admission']
        assert not result['heldout_used'] and not protocol['heldout_used']
        assert result['protocol_sha256'] == sha(path)
        trial.rollout.check_sources(protocol)
    baseline, repeated = protocols
    assert baseline['feedback'] == repeated['feedback']
    assert baseline['feedback_screen'] == repeated['feedback_screen'] and baseline['phase'] == repeated['phase']
    assert {key: value for key, value in baseline['gate'].items() if key != 'bootstrap_seed'} == {
        key: value for key, value in repeated['gate'].items() if key != 'bootstrap_seed'}
    assert draft['maximum_transitions'] == repeated['maximum_control_transitions']
    assert draft['maximum_batched_substeps'] == repeated['maximum_physics_substeps']
    assert draft['episodes_per_condition'] == repeated['environment']['episodes_per_command']
    assert draft['seconds_per_episode'] == repeated['environment']['seconds']
    assert draft['shared_prefix_controls'] == repeated['phase']['warmup_controls']
    assert draft['branches'] == repeated['roles']
    for path in (*plans, first, second, BASE_DRAFT):
        repeated['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    source = Path(__file__).resolve()
    repeated['source_sha256'][source.relative_to(ROOT).as_posix()] = sha(source)

    def load(path):
        if path == DRAFT:
            return dict(read(BASE_DRAFT), development_seed=draft['heldout_seed'], stage=STAGE)
        if path == trial.rollout.PLAN:
            return repeated
        return read(path)

    def save(path, data):
        if path == PLAN:
            data.update(cohort='heldout', scope='One reserved heldout seed. Controller and gates fixed by two independent developments.')
            plant = data['environment']
            plant.update(cohort='heldout', run_id=plant['run_id'].replace('_paired_', '_heldout_'))
        write(path, data)

    with patch.object(trial, 'read', load), patch.object(trial, 'write', save):
        trial.freeze()


def sample():
    assert read(PLAN)['cohort'] == 'heldout'
    publish_live = trial.pulse.live.publish

    def save(path, state):
        if path.name in ('evaluation.json', 'coordinator.json'):
            state.update(cohort='heldout', heldout_used=bool(state['actual']['attempted_controls']))
        write(path, state)

    def publish(record, state):
        # The inherited recorder issues a receipt only after closing its
        # video. Show the latest complete clip while the solver continues.
        if state['latest_clip'] is None:
            for role in reversed(trial.ROLES):
                folder = record/f'evaluation-{role}'
                receipt = folder/'clips.json'
                if not receipt.is_file():
                    continue
                clip = next((row for row in reversed(read(receipt)) if row['complete_clip']), None)
                if clip is not None:
                    path = folder/clip['file']
                    assert path.is_file() and path.stat().st_size > 48
                    state['latest_clip'] = path.relative_to(record).as_posix()
                    break
        publish_live(record, state)

    with patch.object(trial.pulse, 'write', save), patch.object(trial.pulse.live, 'publish', publish):
        trial.sample()


def verify():
    state = read(OUT/'coordinator.json')
    assert state['cohort'] == 'heldout' and state['heldout_used']

    def save(path, result):
        if path == OUT/'analysis.json':
            result.update(cohort='heldout', heldout_used=True, heldout_gate_passed=result['continued_admission'],
                          scope='One reserved heldout cohort with a shared release and the original seven command and paired yaw gates.')
        write(path, result)

    with patch.object(verifier.score, 'write', save):
        verifier.main()


def main():
    trial.pulse.check_project()
    commands = {'--freeze': freeze, '--sample': sample, '--verify': verify}
    assert len(sys.argv) == 2 and sys.argv[1] in commands
    with ExitStack() as scope:
        for name, value in dict(DRAFT=DRAFT, PLAN=PLAN, OUT=OUT, PAGE=PAGE, STAGE=STAGE).items():
            scope.enter_context(patch.object(trial, name, value))
        commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
