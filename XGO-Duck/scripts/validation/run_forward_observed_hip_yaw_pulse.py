"""Repeat the matched pulse in support that the ordinary sensors identify."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_forward_hip_yaw_pulse as original
from check_forward_hip_yaw_sign_selection import OBSERVER, OBSERVER_PLAN
from forward_hip_yaw_pulse_targets import SIDES, eligible
from walking_stance_feedback import StanceObserver


D, ROOT, PROJECT = original.direct, original.ROOT, original.PROJECT
ENTRY = Path(__file__).resolve()
PLANS = ROOT/'configs/experiments/forward_observed_hip_yaw_pulse'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/forward-observed-hip-yaw-pulse'
STAGE, KEY = 'forward-observed-hip-yaw-pulse', 'forward_observed_hip_yaw_pulse'
TITLE = '# 可识别支撑下的髋偏航脉冲'
ORDER = original.ORDER


def observer_for(plan):
    receipt, source = plan['pulse_observer']['receipt'], plan['pulse_observer']['config']
    return StanceObserver(D.read(ROOT/receipt)['model'], D.read(ROOT/source)['minimum_score_separation'])


def coverage(parent, plan):
    folder = ROOT/'records/evaluations'/parent['run_id']
    manifest = D.read(folder/'guarded-evaluation.json')
    commands, trace = (folder/name for name in ('shared-commands.npz', 'shared-trace.npz'))
    D.require(D.sha(commands) == manifest['shared_commands_sha256'] and D.sha(trace) == manifest['shared_trace_sha256'],
              'Originating shared-prefix telemetry changed')
    design = plan['pulse']
    start = design['earliest_fork_control']
    span = slice(start, design['shared_controls'])
    with np.load(commands, allow_pickle=False) as saved, np.load(trace, allow_pickle=False) as physical:
        inferred = observer_for(plan).predict(saved['observed'][span])
        rows = {}
        for index, side in enumerate(SIDES):
            mask, _ = eligible(plan, saved['requested_targets'][span], saved['applied_targets'][span],
                               physical['foot_contact'][span], index)
            counts = (mask & (inferred == index+1)).sum(axis=1)
            candidates = np.flatnonzero(counts >= design['minimum_worlds_per_fork'])
            D.require(len(candidates), f'Settled {side} support has insufficient observable coverage')
            k = int(candidates[0])
            rows[side] = dict(earliest_archived_fork_control=start+k, matched_worlds=int(counts[k]), maximum=int(counts.max()))
    return rows, {D.rel(p): D.sha(p) for p in (commands, trace, folder/'guarded-evaluation.json')}


def freeze():
    project = D.read(PROJECT)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, previous = goals[KEY], goals[original.KEY]
    transfer, readback, result = (ROOT/state[k] for k in ('transfer_evidence', 'sign_evidence', 'previous_physical_evidence'))
    D.require(D.pair.trial.allowed(project, STAGE+'-preflight') and state['status'] == 'planned-preflight'
              and not task['learning_running'] and not task['simulation_running'] and not PLANS.exists()
              and not (REPORT/'cpu-coverage.json').exists() and previous['status'] == 'reviewed'
              and D.sha(original.MASTER) == previous['master_sha256'] and D.sha(result) == previous['result_sha256']
              and D.sha(transfer) == state['transfer_evidence_sha256'] and D.sha(readback) == state['sign_evidence_sha256']
              and D.read(transfer)['passed_same_observer_gates'] and D.sha(OBSERVER) == D.read(transfer)['sources'][D.rel(OBSERVER)]
              and state['earliest_fork_control'] == 150, 'Observer and pulse evidence are required')
    old_master = D.read(original.MASTER)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared, cases = {}, {}
    for key in ORDER:
        previous_plan = original.PLANS/(key+'.json')
        D.require(D.sha(previous_plan) == old_master['protocols'][key], 'First pulse protocol changed')
        parent = D.read(previous_plan)
        out = REPORT/key
        design = dict(parent['pulse'], earliest_fork_control=state['earliest_fork_control'])
        plan = dict(parent, stage=STAGE, run_id=f'duck_observed_hip_yaw_{key}_{stamp}', pulse=design,
                    pulse_observer=dict(receipt=D.rel(OBSERVER), config=D.rel(OBSERVER_PLAN)),
                    guarded_output=D.rel(out), page=D.rel(out/'route-view.html'),
                    route_scope='Official stand and forward with one pulse on each verified settled contact phase')
        cases[key], archive = coverage(parent, plan)
        files = (ENTRY, original.ENTRY, previous_plan, result, transfer, readback, OBSERVER, OBSERVER_PLAN,
                 ENTRY.with_name('walking_stance_feedback.py'))
        plan['frozen_sha256'].update({D.rel(path): D.sha(path) for path in files})
        plan['frozen_sha256'].update(archive)
        prepared[key] = plan
    REPORT.mkdir(exist_ok=True)
    PLANS.mkdir()
    evidence = REPORT/'cpu-coverage.json'
    D.put(evidence, dict(cases=cases, observation_columns='same frozen encoder and IMU model', physics_steps=0, learning_updates=0))
    checks = {}
    for key, plan in prepared.items():
        path, out = PLANS/(key+'.json'), REPORT/key
        out.mkdir()
        plan['frozen_sha256'][D.rel(evidence)] = D.sha(evidence)
        original.verify_sources(plan)
        D.put(path, plan)
        checks[key] = D.sha(path)
        D.put(out/'windows-entry.json', dict(protocol_sha256=checks[key], evidence='native-file-tracker'))
    D.put(MASTER, dict(old_master, stage=STAGE, protocols=checks, shared=design))
    state.update(status='ready', master=D.rel(MASTER), master_sha256=D.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=D.rel(MASTER), current_protocol_frozen=True,
        current_evidence=D.rel(evidence), current_protocol_execution='observed-support-pulse-ready',
        next_gate='Score the matched zero, positive and negative pulses within the observable support windows.')
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_OBSERVED_HIP_YAW_PULSE_FROZEN', state['master_sha256'], cases, flush=True)


def collect(key):
    import guarded_roll_collector as guard
    import collect_forward_hip_yaw_pulse as pulse

    plan = D.read(PLANS/(key+'.json'))
    observer = observer_for(plan)

    def factory(base, may_continue):
        def run(env, *args):
            def observable(design, requested, target, contacts, side):
                physics, joint = eligible(design, requested, target, contacts, side)
                label = observer.predict(base.cpu(env.obs_buf['actor']))
                return physics & (label == side+1), joint

            with patch.object(pulse, 'eligible', observable):
                return pulse.make_pulse_collector(base, may_continue)(env, *args)
        return run

    with patch.object(guard, 'make_guarded_collector', factory):
        D.collect(key)


def score(key):
    import verify_forward_hip_yaw_pulse as verifier

    verify_side = verifier.verify_side

    def checked_side(folder, plan, side, shared, commands, fork):
        model = observer_for(plan)
        k = fork['control']
        predicted = model.predict(commands['observed'][k])
        earlier = model.predict(commands['observed'][plan['pulse']['earliest_fork_control']:k])

        def observable(design, requested, target, contacts, index):
            physics, joint = eligible(design, requested, target, contacts, index)
            labels = earlier if target.ndim == 3 else predicted
            return physics & (labels == index+1), joint

        with patch.object(verifier, 'eligible', observable):
            return dict(verify_side(folder, plan, side, shared, commands, fork), identical_sensor_mask_verified=True)

    with patch.object(verifier, 'verify_side', checked_side):
        original.score_case(key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        bindings = dict(ENTRY=ENTRY, KEY=KEY, STAGE=STAGE, PLANS=PLANS, MASTER=MASTER, REPORT=REPORT)
        with patch.multiple(original, TITLE=TITLE, **bindings), patch.multiple(D, **bindings):
            if args.action in ('collect', 'score'):
                (collect if args.action == 'collect' else score)(args.case)
            else:
                (original.batch if args.action == 'batch' else original.review)()
