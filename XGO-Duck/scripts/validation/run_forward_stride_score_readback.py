"""Score the acquired stride data after correcting a verifier context expression."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import run_forward_stride_response as batch
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


ENTRY = Path(__file__).resolve()
SOURCE = ENTRY.with_name('verify_forward_stride_response.py')
RECEIPT = batch.REPORT/'score-recovery.json'


def checked_code():
    project, master = read(P), read(batch.MASTER)
    state = project['goal_execution'][batch.KEY]
    require(sha(batch.MASTER) == state['master_sha256']
            and sha(SOURCE) == master['source_sha256'][rel(SOURCE)], 'The frozen scorer changed')
    original = SOURCE.read_bytes()
    erroneous = b", +             patch.object(policy_reader, 'transform',"
    require(original.count(erroneous) == 1, 'The diagnostic does not match the frozen source')
    corrected = original.replace(erroneous, b", patch.object(policy_reader, 'transform',")
    return project, state, corrected


def register():
    project, state, corrected = checked_code()
    task = project['staged_training_request']
    first, second = (batch.REPORT/name for name in batch.CASES)
    worker = read(first/'collection-result.json')
    run = ROOT/'records/evaluations'/worker['run_id']
    failed_log = first/'score-command.log'
    actual = read(run/'runtime-contract.json')
    require(batch.direct.pair.trial.allowed(project, batch.STAGE) and state['status'] == 'score-pending'
            and not state['scored'] and not task['simulation_running'] and not task['learning_running']
            and not RECEIPT.exists() and not (second/'collect-command.log').exists()
            and worker['evidence_sha256'] == sha(run/'guarded-evaluation.json') and actual['control_steps'] == 1000
            and "bad operand type for unary +: '_patch'" in failed_log.read_text(encoding='utf-8'),
            'The original batch does not have the expected post-collection score failure')
    compile(corrected, str(SOURCE), 'exec')
    put(RECEIPT, dict(status='frozen-score-only-correction', correction='remove the unintended unary plus on the patch context manager',
                      original_source=rel(SOURCE), original_source_sha256=sha(SOURCE),
                      corrected_in_memory_sha256=hashlib.sha256(corrected).hexdigest(),
                      runner_sha256=sha(ENTRY), failed_log=rel(failed_log), failed_log_sha256=sha(failed_log),
                      original_run=worker['run_id'], original_transitions=actual['control_transitions'],
                      new_physical_controls_for_original_case=0, learning_updates=0, repeated_first_collection=False))
    state['score_recovery'] = dict(path=rel(RECEIPT), sha256=sha(RECEIPT))
    task.update(current_protocol_execution='stride-score-readback', current_evidence=rel(RECEIPT))
    batch.direct.pair.trial.write(P, project)


def score_case(name):
    _, state, corrected = checked_code()
    receipt = read(RECEIPT)
    require(sha(RECEIPT) == state['score_recovery']['sha256']
            and sha(ENTRY) == receipt['runner_sha256']
            and hashlib.sha256(corrected).hexdigest() == receipt['corrected_in_memory_sha256'], 'Unverified score overlay')
    module = ModuleType(SOURCE.stem)
    module.__file__ = str(SOURCE)
    exec(compile(corrected, str(SOURCE), 'exec'), vars(module))
    bindings = dict(STAGE=batch.STAGE, KEY=batch.KEY, PLANS=batch.PLANS, REPORT=batch.REPORT,
                    MASTER=batch.MASTER, ENTRY=batch.ENTRY)
    with patch.dict(sys.modules, {SOURCE.stem:module}), patch.multiple(batch.direct, **bindings):
        batch.score_case(name)


def command(script, action, name, log):
    arguments = [sys.executable, '-B', '-u', str(script), action, name]
    with log.open('x', encoding='utf-8') as stream:
        stream.write('COMMAND '+subprocess.list2cmdline(arguments)+'\n')
        stream.flush()
        subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=540)


def resume():
    register()
    first, second = batch.CASES
    command(ENTRY, 'score', first, batch.REPORT/first/'score-readback.log')
    print('STRIDE_SAVED_FIRST_CASE_SCORED', first, flush=True)
    batch.publish({first:read(batch.REPORT/first/'independent.json')}, second)
    command(batch.ENTRY, 'collect', second, batch.REPORT/second/'collect-command.log')
    print('STRIDE_REMAINING_CASE_COLLECTED', second, flush=True)
    command(ENTRY, 'score', second, batch.REPORT/second/'score-readback.log')
    batch.review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('resume', 'score'))
    parser.add_argument('case', nargs='?', choices=batch.CASES)
    args = parser.parse_args()
    if args.action == 'resume':
        resume()
    elif args.case:
        score_case(args.case)
    else:
        parser.error('Select an acquired case')
