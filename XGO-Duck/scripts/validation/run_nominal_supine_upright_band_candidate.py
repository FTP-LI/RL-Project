"""Run the zero-work candidate again with direct authorization refreshes."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from functools import partial
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import recover_nominal_supine_upright_band_observation as observed
from run_roll_ppo_capacity_live import SynchronousAuthorization


ROOT, PROJECT = observed.ROOT, observed.PROJECT
read, rel, sha, write = observed.read, observed.rel, observed.sha, observed.write
KEY = 'nominal_supine_upright_band_candidate_authorization'
STAGE = 'nominal-supine-upright-band-candidate-direct-authorization'
PLAN = observed.PLAN.with_name(KEY + '.json')
OUT = observed.OUT / 'authorization-recovery'
ENTRY = Path(__file__).resolve()


class RecordedAuthorization(SynchronousAuthorization):
    def __init__(self, predicate, *, report):
        super().__init__(predicate)
        self.report = report

    def close(self):
        super().close()
        write(self.report, self.summary())


def freeze():
    assert not PLAN.exists() and not OUT.exists()
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    previous = goal[observed.KEY]
    training = read(observed.PLAN, previous['protocol_sha256'])
    observed.original.sources_checked(training)
    assert observed.original.parent.allowed(project, observed.STAGE) and not task['simulation_running']
    assert previous['status'] == 'review-required' and KEY not in goal
    control, stopped = (previous['results'][label] for label in observed.original.ROLES)
    assert control['complete'] and not stopped['complete']
    assert stopped['reason'] == "PPOObservationStop('project-pause-or-protocol-changed')"
    candidate = ROOT / stopped['record']
    halt = read(candidate/'state.json')
    assert halt['status']=='stopped-review-required' and halt['error']==stopped['reason']
    assert not any(halt[key] for key in ('controls','attempted_controls','substeps',
                                        'attempted_substeps','completed_updates'))
    assert halt['paired_zero_start_verified']
    first = ROOT / control['record']
    assert sha(first/'model-final.pt') == control['checkpoint_sha256']
    assert sha(first/'learning-metrics.jsonl') == control['metrics_sha256']
    admission_path = ROOT / previous['zero']
    admission = read(admission_path, previous['zero_sha256'])
    assert sha(candidate/'matched-state.json') == admission['proof']['candidate']['matched_state_sha256']
    assert sha(first/'matched-state.json') == admission['proof']['control']['matched_state_sha256']
    assert read(ROOT / stopped['result'])['reason'] == stopped['reason']

    # Retain the tested pause predicate and verify that a denial latches.
    answers = iter((True, False))
    guard = SynchronousAuthorization(lambda: next(answers)).start()
    guard.last = None
    assert not guard.check() and not guard.check() and guard.failure == 'authorization-denied'
    guard.close()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    runs = dict(training['runs'],
                candidate=f'duck-getup-upright-band-candidate-direct-s{training["training_seed"]}-{stamp}')
    plan = dict(training, training_stage=STAGE, runs=runs, retained_control_protocol=rel(observed.PLAN),
                retained_control_protocol_sha256=sha(observed.PLAN), retained_control_report=rel(observed.OUT/'control'),
                authorization_change='Refresh the same predicate directly before worker callbacks. Use the existing 0.1 s cache and discard checks slower than 0.5 s.')
    source = Path(SynchronousAuthorization.check.__code__.co_filename).resolve()
    paths = (ENTRY, source, observed.PLAN, admission_path, first/'state.json', first/'model-final.pt',
             first/'learning-metrics.jsonl', ROOT/control['result'], ROOT/stopped['result'], candidate/'state.json',
             candidate/'matched-state.json', candidate/'stopped-partial.pt')
    plan['integrity'] = {**training['integrity'], **{rel(path):sha(path) for path in paths}}
    destination = OUT / 'cuda-zero' / 'candidate'
    destination.mkdir(parents=True)
    copyfile(candidate/'matched-state.json', destination/'matched-state.json')
    write(PLAN, plan)
    reused = OUT/'cuda-zero'/'matched.json'
    write(reused, dict(admission, protocol_sha256=sha(PLAN), reused_prior_cuda_proof=rel(admission_path),
                       reused_prior_cuda_proof_sha256=sha(admission_path), per_arm_runtime_signature_required=True))
    receipt = OUT/'authorization-preflight.json'
    write(receipt, dict(status='verified-zero-work-candidate-stop', policy_check_stop=stopped['reason'],
                        original_monitor_detail_saved=False, pause_flags_clear_at_review=True,
                        actual_control_steps=0, actual_batched_substeps=0, completed_updates=0,
                        sticky_denial_test=True, protocol_sha256=sha(PLAN), previous_run=rel(candidate),
                        retained_control=rel(first), reused_cuda_zero=rel(reused)))
    goal[KEY] = dict(status='paired-training-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), runs=runs,
                     preflight=rel(receipt), zero=rel(reused), zero_sha256=sha(reused), results={'control':control},
                     hardware_io=False)
    previous.update(status='control-complete-candidate-zero-work-stop', candidate_recovery=rel(receipt))
    info = dict(current_protocol_execution='nominal-supine-upright-band-candidate-start-ready',
                current_evidence=rel(receipt))
    project.update(info)
    task.update(info, current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Use the recorded control arm and complete the identically seeded candidate with direct authorization readback.')
    write(PROJECT, project)
    print('UPRIGHT_BAND_CANDIDATE_DIRECT_FROZEN', sha(PLAN), runs, flush=True)


def train():
    base = observed.original
    authorization = partial(RecordedAuthorization, report=OUT/'candidate'/'authorization-monitor.json')
    with (patch.object(base,'KEY',KEY), patch.object(base,'OUT',OUT), patch.object(base,'PLAN',PLAN),
          patch.object(base,'RunObserver',observed.JsonRunObserver),
          patch.object(base,'AuthorizationMonitor',authorization)):
        base.train('candidate')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','train'))
    args = parser.parse_args()
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if args.action == 'freeze' else train)()


if __name__ == '__main__':
    main()
