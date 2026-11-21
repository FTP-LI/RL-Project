"""Complete the remaining nominal cohorts with a separately checked observer."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_right_support_continuation as previous
import run_roll_stand_forward_diagnostic as direct
import native_roll_observation as observer
import native_roll_observation_bounded as bounded
from native_nominal_baseline import verify_sources
from native_observer_guard import check_health
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


ENTRY = Path(__file__).resolve()
KEY = 'forward_right_support_nominal_verification'
STAGE = 'forward-right-support-nominal-verification'
REPORT = previous.REPORT/'nominal-verification'
PLANS = ROOT/'configs/experiments'/KEY
MASTER = PLANS/'master.json'
ORDER = previous.ORDER[2:]
READ_WAIT_SECONDS = .45


def read_health(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(child, health_file, **criteria):
    # Windows-mounted files can briefly be unavailable while publishing an
    # atomic snapshot. Hold the simulator at this boundary during the reread.
    deadline = monotonic()+READ_WAIT_SECONDS
    while child.poll() is None:
        try:
            health = read_health(Path(health_file))
        except (FileNotFoundError, PermissionError) as error:
            remaining = deadline-monotonic()
            if remaining <= 0 or not error.filename or Path(error.filename) != Path(health_file):
                return dict(allowed=False, reason='observer-health-unavailable')
            sleep(min(.025, remaining))
        except (OSError, UnicodeError, ValueError):
            return dict(allowed=False, reason='observer-health-invalid')
        else:
            return check_health(health, expected_pid=child.pid, process_alive=child.poll() is None, **criteria)
    return dict(allowed=False, reason='observer-exited')


class RecordedObserver(bounded.NativeRollObservationBounded):
    def check(self, *, force=False):
        with patch.object(observer, 'check_child', check):
            allowed = super().check(force=force)
        if not allowed and self.in_collect:
            output = self.report/'observer-terminal-refusal.json'
            if not output.exists():
                try:
                    health = read_health(self.view/'native-health.json')
                except (OSError, ValueError):
                    health = None
                event = dict(reason=self.refusal, publisher_error=self.error,
                             child_status=self.child.poll() if self.child is not None else None,
                             published_controls=self.last_control, health_before_close=health,
                             at=datetime.now(timezone.utc).isoformat())
                print('NATIVE_OBSERVER_REFUSAL', json.dumps(event, ensure_ascii=False), flush=True)
                put(output, event)
        return allowed


def preflight():
    class ObserverHandle:
        pid = 87

        def poll(self):
            return None

    child, path = ObserverHandle(), Path('native-health.json')
    at = datetime.now(timezone.utc).isoformat()
    health = dict(kind='native-run-observer', mode='live', health='fresh', observer_pid=child.pid, run_id='diagnostic',
                  source='test.json', source_status='running', source_sha256='0'*64, heartbeat_at=at, source_updated_at=at)
    missing = FileNotFoundError(2, 'missing', str(path))
    criteria = dict(expected_run_id='diagnostic', expected_source='test.json', max_heartbeat_seconds=15, max_source_seconds=30)
    with patch(__name__+'.read_health', side_effect=[missing, health]) as reads:
        assert check(child, path, **criteria)['allowed'] and reads.call_count == 2
    with patch(__name__+'.read_health', side_effect=missing) as reads:
        assert check(child, path, **criteria)['reason'] == 'observer-health-unavailable' and reads.call_count > 1
    with patch(__name__+'.read_health', side_effect=ValueError()) as reads:
        assert check(child, path, **criteria)['reason'] == 'observer-health-invalid' and reads.call_count == 1
    with patch(__name__+'.read_health', return_value=dict(health, observer_pid=-1)) as reads:
        assert check(child, path, **criteria)['reason'] == 'observer-process-mismatch' and reads.call_count == 1
    stale = (datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat()
    with patch(__name__+'.read_health', return_value=dict(health, heartbeat_at=stale)) as reads:
        assert check(child, path, **criteria)['reason'] == 'observer-heartbeat-stale' and reads.call_count == 1
    with patch.object(child, 'poll', return_value=1), patch(__name__+'.read_health') as reads:
        assert check(child, path, **criteria)['reason'] == 'observer-exited' and reads.call_count == 0
    return dict(transient_missing_recovers=True, persistent_missing_refused=True,
                invalid_content_refused=True, wrong_process_refused=True, stale_heartbeat_refused=True, exited_child_refused=True,
                maximum_reread_seconds=READ_WAIT_SECONDS, new_physical_steps=0, learning_updates=0)


def freeze():
    project, parent = read(P), read(previous.MASTER)
    goals, task = project['goal_execution'], project['staged_training_request']
    old = goals[previous.KEY]
    require(direct.pair.trial.allowed(project, previous.STAGE) and old['status'] == 'review-required'
            and set(old['scored']) == set(previous.ORDER[:2]) and sha(previous.MASTER) == old['master_sha256']
            and not (task['simulation_running'] or task['learning_running'] or REPORT.exists() or PLANS.exists()),
            'An idle stopped nominal stage and two completed low-friction scores are required')
    stopped = previous.REPORT/ORDER[0]
    recorded = read(stopped/'collection-result.json')
    runtime = ROOT/'records/evaluations'/recorded['run_id']/'runtime-contract.json'
    stop = read(runtime)
    peak = max(sample['used_mib']/sample['total_mib'] for sample in recorded['gpu_samples'])
    require(recorded['status'] == 'review-required' and 'guard stopped' in recorded['reason']
            and 0 < stop['control_steps'] < parent['unique_controls_per_case'] and not stop['capacity_reached']
            and peak < .85 and not (previous.REPORT/ORDER[1]/'collect-command.log').exists(),
            'The interrupted nominal route could not be accounted for')
    for receipt in old['scored'].values():
        require(sha(ROOT/receipt['path']) == receipt['sha256'], 'Completed evidence differs')
    tests = preflight()
    REPORT.mkdir()
    receipt = REPORT/'observer-preflight.json'
    evidence = [runtime, stopped/'collection-result.json', stopped/'collect-command.log', stopped/'native-view/native-health.json']
    put(receipt, dict(tests, stopped_run=recorded['run_id'], stopped_controls=stop['control_steps'],
                      interrupted_transitions=stop['control_transitions'], peak_gpu_fraction=peak,
                      historical_detailed_refusal_available=False, source_sha256={rel(path):sha(path) for path in evidence}))
    PLANS.mkdir()
    protocols = {}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    for name in ORDER:
        base = previous.PLANS/f'{name}.json'
        require(sha(base) == parent['protocols'][name], 'Nominal protocol differs')
        current = read(base)
        out = REPORT/name
        out.mkdir()
        current.update(stage=STAGE, run_id=f'duck_forward_support_nominal_verification_{name}_{stamp}',
                       guarded_output=rel(out), page=rel(out/'route-view.html'))
        current['frozen_sha256'].update({rel(path):sha(path) for path in (base, ENTRY, receipt)})
        verify_sources(current)
        path = PLANS/f'{name}.json'
        put(path, current)
        protocols[name] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=sha(path), evidence='native-file-tracker'))
    put(MASTER, dict(parent, stage=STAGE, order=list(ORDER), protocols=protocols,
                     maximum_transitions=parent['unique_controls_per_case']*128*len(ORDER), observer_preflight=rel(receipt),
                     interrupted_transitions=stop['control_transitions'], parent_master_sha256=sha(previous.MASTER)))
    old.update(status='nominal-verification', interrupted_transitions=stop['control_transitions'],
               interrupted_runtime=rel(runtime))
    goals[KEY] = dict(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='nominal-observer-verified', current_evidence=rel(receipt),
                next_gate='Complete only the two nominal matched-route measurements with recorded observer refusals.')
    direct.pair.trial.write(P, project)
    print('NOMINAL_OBSERVER_PREFLIGHT_AND_FREEZE', sha(MASTER), stop['control_steps'], flush=True)


def cases(project):
    goals = project['goal_execution']
    all_scores = {**goals[previous.KEY]['scored'], **goals[KEY]['scored']}
    result = {}
    for name in previous.ORDER:
        if name not in all_scores:
            continue
        mark = all_scores[name]
        path = ROOT/mark['path']
        require(sha(path) == mark['sha256'], 'An independently verified score changed')
        result[name] = read(path)
    return result


def publish(results, active=None):
    previous.page(results, active)
    page = previous.REPORT/'current-evaluation.md'
    lines = page.read_text()
    for name in ORDER:
        lines = lines.replace(f']({name}/', f'](nominal-verification/{name}/')
    page.write_text(lines, encoding='utf-8')


def review():
    project, master = read(P), read(MASTER)
    goals = project['goal_execution']
    status, original = goals[KEY], goals[previous.KEY]
    require(direct.pair.trial.allowed(project, STAGE) and status['status'] == 'ready'
            and set(status['scored']) == set(ORDER) and sha(MASTER) == status['master_sha256'],
            'Both nominal cohorts must be independently scored')
    rows = cases(project)
    transitions = sum(row['physical_transitions'] for row in rows.values())
    substeps = sum(row['physical_substeps'] for row in rows.values())
    require(transitions == read(previous.MASTER)['maximum_transitions'], 'Combined verified budget differs')
    candidates = [row['arms']['right-support-continuation'] for row in rows.values()]
    eligible = all(row['action_gate_passed'] and row['controls']['actual_right_support_precision'] >= master['observer_precision_gate']
                   for row in candidates)
    output = previous.REPORT/'closed-loop-comparison.json'
    put(output, dict(cases=rows, candidate_passes_full_range=eligible, physical_transitions=transitions,
                     physical_substeps=substeps, interrupted_transitions=original['interrupted_transitions'],
                     paired_yaw_basis='Median of within-world changes in 20-second time-mean absolute yaw',
                     learning_updates=0, heldout_used=False, hardware_io=False))
    for item in (original, status):
        item.update(status='reviewed', result=rel(output), result_sha256=sha(output), candidate_passes_full_range=eligible)
    original.update(physical_transitions=transitions, physics_steps=substeps)
    project['staged_training_request'].update(current_evidence=rel(output), current_protocol_execution='same-snapshot-forward-pair-reviewed',
        next_gate='Use both full gait cycles and matched repeated reference traces to measure durable phase-resolved yaw changes.')
    direct.pair.trial.write(P, project)
    publish(rows)
    print('FORWARD_SUPPORT_NOMINAL_VERIFIED', eligible, transitions, original['interrupted_transitions'], flush=True)


def batch():
    for name in ORDER:
        publish(cases(read(P)), name)
        for action in ('collect', 'score'):
            command = [sys.executable, '-B', '-u', str(ENTRY), action, name]
            with (REPORT/name/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=540)
            print('FORWARD_SUPPORT_NOMINAL_CASE', name, action, flush=True)
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        if args.case is None:
            parser.error('Select a nominal cohort')
        binds = dict(STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY)
        with patch.multiple(direct, **binds), patch.multiple(previous, **binds):
            if args.action == 'score':
                previous.score_case(args.case)
            else:
                import guarded_roll_collector
                from collect_forward_right_support_continuation import make_collector
                with patch.object(guarded_roll_collector, 'make_guarded_collector', make_collector), \
                     patch.object(bounded, 'NativeRollObservationBounded', RecordedObserver):
                    direct.collect(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
