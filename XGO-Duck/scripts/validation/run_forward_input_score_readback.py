"""Read the full-route bootstrap seed using the frozen master field."""

import argparse
from pathlib import Path
from unittest.mock import patch

import run_forward_input_full_route as full
from run_forward_stride_score_readback import command
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


ENTRY = Path(__file__).resolve()
RECEIPT = full.REPORT/'score-readback.json'


def register():
    project, master = read(P), read(full.MASTER)
    task, state = project['staged_training_request'], project['goal_execution'][full.KEY]
    first, *pending = full.ORDER
    case = full.REPORT/first
    failed = case/'score-command.log'
    worker = read(case/'collection-result.json')
    folder = ROOT/'records/evaluations'/worker['run_id']
    manifest = folder/'guarded-evaluation.json'
    require(full.direct.pair.trial.allowed(project, full.STAGE) and state['status'] == 'score-pending'
            and not (state['scored'] or task['simulation_running'] or task['learning_running'] or RECEIPT.exists())
            and sha(full.MASTER) == state['master_sha256']
            and 'bootstrap_seed' not in master and master['paired_bootstrap_seed'] == 981247
            and "KeyError: 'bootstrap_seed'" in failed.read_text(encoding='utf-8')
            and sha(manifest) == worker['evidence_sha256']
            and read(folder/'runtime-contract.json')['control_steps'] == master['controls_per_case']
            and all(not (full.REPORT/name/'collect-command.log').exists() for name in pending),
            'The first acquired full route or frozen seed does not match the score failure')
    put(RECEIPT, dict(status='frozen-score-field-readback', score_seed_field='paired_bootstrap_seed',
         frozen_score_seed=master['paired_bootstrap_seed'], master_sha256=sha(full.MASTER), runner_sha256=sha(ENTRY),
         failure_sha256=sha(failed), actual_first_run=worker['run_id'], actual_first_manifest_sha256=sha(manifest),
         new_physical_steps_for_first_case=0, learning_updates=0))
    state['score_readback'] = dict(path=rel(RECEIPT), sha256=sha(RECEIPT))
    task.update(current_protocol_execution='full-route-score-field-readback', current_evidence=rel(RECEIPT))
    full.direct.pair.trial.write(P, project)


def score(name):
    p, evidence = read(P), read(RECEIPT)
    require(sha(RECEIPT) == p['goal_execution'][full.KEY]['score_readback']['sha256']
            and sha(ENTRY) == evidence['runner_sha256'] and sha(full.MASTER) == evidence['master_sha256'],
            'The score field readback was not frozen')

    def read_master(path):
        row = read(path)
        if path == full.MASTER:
            return dict(row, bootstrap_seed=row[evidence['score_seed_field']])
        return row

    bindings = dict(STAGE=full.STAGE, KEY=full.KEY, PLANS=full.PLANS, REPORT=full.REPORT, MASTER=full.MASTER, ENTRY=full.ENTRY)
    with patch.multiple(full.direct, **bindings), patch.object(full.paired, 'read', read_master):
        full.score_case(name)


def resume():
    register()
    first, *pending = full.ORDER
    command(ENTRY, 'score', first, full.REPORT/first/'score-readback.log')
    print('FORWARD_FULL_FIRST_SAVED_SCORE_VERIFIED', first, flush=True)
    cases = {first:read(full.REPORT/first/'independent.json')}
    for name in pending:
        full.show(cases, name)
        command(full.ENTRY, 'collect', name, full.REPORT/name/'collect-command.log')
        print('FORWARD_FULL_REMAINING_CASE_COLLECTED', name, flush=True)
        command(ENTRY, 'score', name, full.REPORT/name/'score-readback.log')
        cases[name] = read(full.REPORT/name/'independent.json')
    full.review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('resume', 'score'))
    parser.add_argument('case', nargs='?', choices=full.ORDER)
    args = parser.parse_args()
    if args.action == 'resume':
        resume()
    elif args.case:
        score(args.case)
    else:
        parser.error('Select an acquired case')
