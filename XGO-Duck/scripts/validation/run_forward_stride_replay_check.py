"""Verify the reverse-arm history with its actual shared-prefix boundary."""

import argparse
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np

import run_forward_stride_score_readback as prior
from collect_forward_stride_response import StrideControl
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


BATCH = prior.batch
ENTRY = Path(__file__).resolve()
RECEIPT = BATCH.REPORT/'replay-boundary-recovery.json'


class ReplayStrideControl(StrideControl):
    def __init__(self, plan, design):
        super().__init__(plan, design)
        self.branch_at = plan['stride_response']['shared_controls']

    def next(self, k, command, observed, enabled):
        # The eligibility sensor runs throughout the route; the nonzero
        # intervention is enabled only after the physical restore.
        return super().next(k, command, observed, enabled and k >= self.branch_at)


def register():
    project, state, corrected = prior.checked_code()
    task = project['staged_training_request']
    first, second = BATCH.CASES
    marker = BATCH.REPORT/first/'score-readback.log'
    worker = read(BATCH.REPORT/first/'collection-result.json')
    run = ROOT/'records/evaluations'/worker['run_id']
    raw = read(run/'guarded-evaluation.json')
    commands = run/'shared/commands.npz'
    existing = read(prior.RECEIPT)
    require(BATCH.direct.pair.trial.allowed(project, BATCH.STAGE) and state['status'] == 'score-pending' and not state['scored']
            and not RECEIPT.exists() and not (BATCH.REPORT/second/'collect-command.log').exists()
            and not task['simulation_running'] and not task['learning_running']
            and sha(prior.RECEIPT) == state['score_recovery']['sha256'] and sha(prior.ENTRY) == existing['runner_sha256']
            and sha(run/'guarded-evaluation.json') == worker['evidence_sha256']
            and sha(commands) == raw['prefix_sha256']['commands.npz']
            and 'right_extension_applied' in marker.read_text(encoding='utf-8'), 'The acquired prefix or initial score changed')
    with np.load(commands, allow_pickle=False) as recorded:
        eligible = recorded['right_extension_eligible']
        applied = recorded['right_extension_applied']
        world = 120
        k = int(np.flatnonzero(eligible[:, world])[0])
        require(len(applied) == raw['shared_controls'] and k == 151 and not applied.any(),
                'The discrepant score does not match the unchanged physical prefix')
    put(RECEIPT, dict(status='frozen-score-boundary-correction', runner_sha256=sha(ENTRY),
        score_source_sha256=hashlib.sha256(corrected).hexdigest(), first_failed_score_sha256=sha(marker),
        source_prefix_sha256=sha(commands), observed_eligible_world=world, observed_eligible_control=k,
        actual_reverse_enabled_from_control=raw['shared_controls'], first_case_additional_physical_steps=0,
        learning_updates=0, prior_recovery_sha256=sha(prior.RECEIPT)))
    state['replay_recovery'] = dict(path=rel(RECEIPT), sha256=sha(RECEIPT))
    task.update(current_protocol_execution='stride-score-at-actual-boundary', current_evidence=rel(RECEIPT))
    BATCH.direct.pair.trial.write(P, project)


def score_case(name):
    _, state, corrected = prior.checked_code()
    evidence = read(RECEIPT)
    require(sha(RECEIPT) == state['replay_recovery']['sha256'] and sha(ENTRY) == evidence['runner_sha256']
            and hashlib.sha256(corrected).hexdigest() == evidence['score_source_sha256'], 'Readback code changed')
    module = ModuleType(prior.SOURCE.stem)
    module.__file__ = str(prior.SOURCE)
    exec(compile(corrected, str(prior.SOURCE), 'exec'), vars(module))
    module.StrideControl = ReplayStrideControl

    def keep(path, value):
        if path.exists():
            require(read(path) == value, 'A prior partial raw score changed under the corrected replay')
        else:
            put(path, value)

    module.put = keep
    bindings = dict(STAGE=BATCH.STAGE, KEY=BATCH.KEY, PLANS=BATCH.PLANS, REPORT=BATCH.REPORT,
                    MASTER=BATCH.MASTER, ENTRY=BATCH.ENTRY)
    with patch.dict(sys.modules, {prior.SOURCE.stem:module}), patch.multiple(BATCH.direct, **bindings):
        BATCH.score_case(name)


def resume():
    register()
    first, second = BATCH.CASES
    prior.command(ENTRY, 'score', first, BATCH.REPORT/first/'score-paired-readback.log')
    print('STRIDE_SAVED_FIRST_CASE_VERIFIED', first, flush=True)
    BATCH.publish({first:read(BATCH.REPORT/first/'independent.json')}, second)
    prior.command(BATCH.ENTRY, 'collect', second, BATCH.REPORT/second/'collect-command.log')
    print('STRIDE_UNSTARTED_CASE_COLLECTED', second, flush=True)
    prior.command(ENTRY, 'score', second, BATCH.REPORT/second/'score-paired-readback.log')
    BATCH.review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('resume', 'score'))
    parser.add_argument('case', nargs='?', choices=BATCH.CASES)
    args = parser.parse_args()
    if args.action == 'resume':
        resume()
    elif args.case:
        score_case(args.case)
    else:
        parser.error('Select an acquired case')
