"""Find matched mid-right support forks in the recorded low-friction routes."""

from pathlib import Path

import numpy as np

from audit_forward_yaw_by_support_age import contact_age
from forward_support_joint_pulse import common_mask, observer_for
from readback_forward_support_pulse_context import observation_age
from run_roll_stand_forward_diagnostic import P, ROOT, pair, put, read, rel, require, sha


KEY = 'right_support_hip_pitch_duration'
STAGE = 'right-support-hip-pitch-duration-preflight'
REPORT = ROOT/'records/reports/right-support-hip-pitch-duration'


def inspect(plan, draft):
    folder = ROOT/'records/evaluations'/plan['run_id']
    trace, command, manifest_path = (folder/name for name in ('trace.npz', 'commands.npz', 'guarded-evaluation.json'))
    manifest = read(manifest_path)
    require(sha(trace) == manifest['trace_sha256'] and sha(command) == manifest['command_sha256']
            and manifest['actual_control_steps'] == 1050 and not manifest['hardware_io'], 'The closed-loop traces differ')
    with np.load(trace, allow_pickle=False) as data, np.load(command, allow_pickle=False) as motor:
        contact = data['foot_contact'][:-1]
        _, physical_age = contact_age(contact)
        observed = motor['observed']
        targets, original = motor['requested_targets'], motor['applied_targets']
        recorded = motor['hip_pitch_pulse_side']
    observer_plan = dict(plan, pulse_observer=plan['phase_hip_pitch']['observer'], pulse=dict(delta_rad=draft['single_control_delta_rad']))
    label = observer_for(observer_plan).predict(observed)
    estimated_age = observation_age(label)
    earliest, latest = draft['recognized_right_support_age_controls']
    tick = np.arange(len(contact))[:, None]
    last_right = np.maximum.accumulate(np.where(recorded == 2, tick, -1), axis=0)
    in_episode = (last_right >= tick-physical_age+1) & (last_right < tick)
    eligible = common_mask(observer_plan, targets, original, contact, label, 1)
    eligible &= (tick >= draft['first_zero_based_control']) & in_episode
    eligible &= (estimated_age >= earliest) & (estimated_age <= latest)
    eligible &= (physical_age >= earliest) & (physical_age <= latest)
    eligible &= observed[..., plan['phase_hip_pitch']['pre_action_gyro_column']] > 0
    counts = eligible.sum(axis=1)
    windows = np.flatnonzero(counts >= draft['minimum_common_worlds'])
    result = dict(first_matched=None, maximum_matched_at_one_control=int(counts.max()), distinct_eligible_worlds=int(eligible.any(axis=0).sum()))
    if windows.size:
        k = int(windows[0])
        selected = np.flatnonzero(eligible[k])
        later = eligible[k:k+latest-earliest+1, selected]
        streak = np.logical_and.accumulate(later, axis=0).sum(axis=0)
        result['first_matched'] = dict(control=k, worlds=len(selected), selected_worlds=selected.tolist(),
                                      consecutive_eligible_controls_distribution=np.bincount(streak, minlength=latest-earliest+2)[1:].tolist(),
                                      worlds_with_at_least_three_eligible_controls=int((streak >= 3).sum()))
    return result, {rel(path): sha(path) for path in (trace, command, manifest_path)}


def main():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, prior = goals[KEY], goals['forward_phase_hip_pitch']
    draft_path, master_path = ROOT/state['draft'], ROOT/prior['master']
    output = REPORT/'archived-common-window.json'
    draft, master = read(draft_path), read(master_path)
    require(pair.trial.allowed(project, STAGE) and state['status'] == 'pending-archived-matched-window'
            and prior['status'] == 'reviewed' and not output.exists() and not task['simulation_running'] and not task['learning_running']
            and not draft['physical_steps_authorized_by_draft'] and sha(master_path) == prior['master_sha256'],
            'The extension preflight is not the current task')
    sources = {rel(path): sha(path) for path in (Path(__file__).resolve(), draft_path, master_path)}
    for key in ('result', 'support_age_effect', 'prefix_pairing'):
        source = ROOT/prior[key]
        digest = sha(source)
        require(digest == prior[key+'_sha256'], 'A source for the longer correction changed')
        sources[rel(source)] = digest
    require(prior['result_sha256'] == state['parent_result_sha256'], 'Expected the frozen closed-loop result')
    cases = {}
    for name in draft['development_cohorts']:
        protocol = master_path.parent/f'{name}.json'
        plan = read(protocol)
        require(sha(protocol) == master['protocols'][name] and plan['parameter_factors']['foot_friction'] == .8
                and plan['feedback']['maximum_joint_correction_rad'] == draft['maximum_joint_correction_rad']
                and plan['phase_hip_pitch']['single_control_delta_rad'] == draft['single_control_delta_rad'],
                'The matched actuator or route scope changed')
        for source in plan['phase_hip_pitch']['observer'].values():
            require(sha(ROOT/source) == plan['frozen_sha256'][source], 'The support observer changed')
            sources[source] = plan['frozen_sha256'][source]
        cases[name], artifacts = inspect(plan, draft)
        sources.update({rel(protocol): master['protocols'][name], **artifacts})
    allowed = all(row['first_matched'] is not None for row in cases.values())
    REPORT.mkdir(exist_ok=True)
    put(output, dict(cases=cases, matched_windows_verified=allowed, source_sha256=sources, physical_steps=0,
                     hardware_io=False, heldout_used=False, learning_updates=0))
    state.update(status='archived-matched-window-verified' if allowed else 'archived-coverage-needs-review',
                 result=rel(output), result_sha256=sha(output))
    task.update(current_evidence=rel(output), current_protocol_execution=state['status'],
                next_gate='Freeze the bounded same-state continuation with a shared sensor latch and the original motor envelope.'
                if allowed else 'Review the joint and sensor overlap before choosing simultaneous right support forks.')
    pair.trial.write(P, project)
    print('RIGHT_SUPPORT_DURATION_ARCHIVED_MATCH', allowed, {
        key: {k: v for k, v in row['first_matched'].items() if k != 'selected_worlds'} if row['first_matched'] else None
        for key, row in cases.items()}, flush=True)


if __name__ == '__main__':
    main()
