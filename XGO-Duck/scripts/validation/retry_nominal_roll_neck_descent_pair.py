"""Permit one same-seed release qualification after the original pre-branch stop."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_nominal_roll_neck_descent_pair as trial


ROOT, PROJECT = trial.ROOT, trial.PROJECT
read, rel, sha, write = trial.read, trial.rel, trial.sha, trial.write
KEY, STAGE = trial.KEY, trial.STAGE
PLAN = trial.PLAN.with_stem(trial.PLAN.stem + '_release_retry')
REPORT = trial.REPORT / 'release-retry'
ENTRY = Path(__file__).resolve()


def freeze_retry():
    project = read(PROJECT)
    task, state = project['staged_training_request'], project['goal_execution'][KEY]
    assert trial.base.allowed(project, STAGE) and state['status'] == 'review-required' and not task['simulation_running']
    assert task['current_protocol'] == rel(trial.PLAN) and state['protocol_sha256'] == sha(trial.PLAN)
    assert not REPORT.exists() and not PLAN.exists() and 'first_release_attempt' not in state
    original = ROOT / state['collection']
    audit = ROOT / state['release_admission_audit']
    receipt = read(original)
    judgement = read(audit)
    assert sha(original) == state['collection_sha256'] == judgement['collection_sha256']
    assert sha(audit) == state['release_admission_audit_sha256'] and not judgement['candidate_exercised']
    assert receipt['status'] == 'stopped-before-fork' and receipt['reason'] == 'actual-release-coverage'
    assert receipt['controls'] == 100 and receipt['transitions'] == 89600 and not receipt['branches']
    plan = read(trial.PLAN)
    assert plan['environment']['seed'] == 1257 and not trial.torch.cuda.is_initialized()
    archived_qual = ROOT / state['neck_rule_qualification']
    assert sha(archived_qual) == state['neck_rule_qualification_sha256'] and read(archived_qual)['physical_substeps'] == 0
    original_log = trial.REPORT.parent / 'nominal-neck-descent-pair-run.log'
    assert 'SCORE_EXIT_CODE=1' in original_log.read_text(encoding='utf-8')
    assert not (ROOT / plan['record'] / 'fork.json').exists()
    run = 'duck_nominal_roll_neck_descent_retry_s1257_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['environment']['run_id'] = run
    plan.update(record=f'records/evaluations/{run}', release_retry=dict(allowed_attempts_after_first=1,
        same_seed=True, same_action_rule=True, same_release_gate=True,
        first_attempt_transitions=receipt['transitions'], first_collection_sha256=sha(original),
        maximum_total_transitions=receipt['transitions'] + plan['maximum_control_transitions']))
    plan['references'].update({rel(path): sha(path) for path in (trial.PLAN, original, audit, archived_qual, original_log)})
    plan['source_sha256'][rel(ENTRY)] = sha(ENTRY)
    trial.base.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN, plan)
    preflight = REPORT / 'preflight.json'
    write(preflight, dict(protocol_sha256=sha(PLAN), retry=plan['release_retry'], prior_protocol_sha256=sha(trial.PLAN),
        original_reason=receipt['reason'], observer=trial.base.preflight(), simulation_steps_in_registration=0))
    state.update(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run, preflight=rel(preflight),
        first_release_attempt=dict(collection=rel(original), collection_sha256=sha(original), transitions=receipt['transitions'],
                                   protocol=rel(trial.PLAN), protocol_sha256=sha(trial.PLAN), reason=receipt['reason']))
    task.update(current_protocol=rel(PLAN), current_protocol_execution='nominal-neck-descent-release-retry-frozen',
                current_evidence=rel(preflight))
    write(PROJECT, project)
    print('NOMINAL_NECK_RELEASE_RETRY_FROZEN', sha(PLAN), run, flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'zero', 'sample', 'score'))
    action = cli.parse_args().action
    if action == 'freeze':
        return freeze_retry()
    with (
        patch.multiple(trial, PLAN=PLAN, REPORT=REPORT),
        patch('sys.argv', [str(ENTRY), action]),
    ):
        return trial.main()


if __name__ == '__main__':
    main()
