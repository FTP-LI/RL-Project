"""Replay a sensor-only continuation across the original 20-second forward routes."""

from pathlib import Path

import numpy as np

from audit_forward_yaw_by_support_age import contact_age
from forward_phase_hip_pitch import HipPitchSchedule
from right_hip_pitch_duration_control import targets
from run_roll_stand_forward_diagnostic import P, ROOT, pair, put, read, rel, require, sha


KEY = 'forward_right_support_continuation'
STAGE = 'forward-right-support-continuation-preflight'
REPORT = ROOT/'records/reports/forward-right-support-continuation'
CONTINUATION_FIELDS = ('continuation_active', 'continuation_attempted')


class RightSupportContinuation:
    def __init__(self, plan, design):
        self.onset = HipPitchSchedule(plan, plan['phase_hip_pitch'])
        self.design = design
        self.continuation_active = np.zeros(plan['worlds'], dtype=bool)
        self.continuation_attempted = np.zeros(plan['worlds'], dtype=bool)

    def next(self, k, command, observed, enabled):
        allowed = (~self.continuation_attempted | self.continuation_active) if enabled else None
        target, row = targets(self.onset, self.design, k, command, observed, allowed)
        self.continuation_attempted[row['observer_label'] == 1] = False
        # An unknown label ends an ongoing correction without rearming it.
        self.continuation_active = row['right_extension_applied'].copy()
        self.continuation_attempted |= self.continuation_active
        return target, row


def inspect(plan, design, folder):
    command_path, trace_path, manifest_path = (folder/name for name in ('commands.npz', 'trace.npz', 'guarded-evaluation.json'))
    manifest = read(manifest_path)
    sources = {rel(path):sha(path) for path in (command_path, trace_path, manifest_path)}
    require(sources[rel(command_path)] == manifest['command_sha256'] and sources[rel(trace_path)] == manifest['trace_sha256']
            and manifest['actual_control_steps'] == 1050 and not manifest['hardware_io'], 'The verified 20-second route changed')
    with np.load(command_path, allow_pickle=False) as snapshots, np.load(trace_path, allow_pickle=False) as physics:
        fields = ('requested_targets', 'applied_targets', 'observed', 'phase_used', 'actual_motor_target', 'hip_pitch_pulse_side')
        command = {key: snapshots[key] for key in fields}
        supports, ages = contact_age(physics['foot_contact'][:-1])
    original, candidate = (RightSupportContinuation(plan, design) for _ in range(2))
    contract = candidate.onset
    hits = np.zeros_like(supports, dtype=bool)
    streak = np.zeros(plan['worlds'], dtype=np.int32)
    maximum_controls = 0
    for k, observed in enumerate(command['observed']):
        current = {key:command[key][k] for key in ('requested_targets', 'applied_targets', 'phase_used')}
        parent, reference = original.next(k, current, observed, False)
        alternative, row = candidate.next(k, current, observed, True)
        np.testing.assert_array_equal(reference['hip_pitch_pulse_side'], command['hip_pitch_pulse_side'][k])
        np.testing.assert_array_equal(row['hip_pitch_pulse_side'], reference['hip_pitch_pulse_side'])
        np.testing.assert_allclose(parent, command['actual_motor_target'][k], atol=1e-6, rtol=0)
        hits[k] = row['right_extension_applied']
        change = np.zeros_like(parent)
        change[hits[k], contract.pitch[1]] = design['single_control_delta_rad']
        np.testing.assert_allclose(alternative-parent, change, atol=2e-7, rtol=0)
        neutral = np.clip(current['requested_targets'], contract.lower, contract.upper)
        require(np.isfinite(alternative).all() and (alternative >= contract.lower-1e-6).all()
                and (alternative <= contract.upper+1e-6).all() and np.abs(alternative-neutral).max() <= contract.cap+1e-6,
                'A scheduled command exceeded the original actuator envelope')
        streak = np.where(hits[k], streak+1, 0)
        maximum_controls = max(maximum_controls, int(streak.max()))
    low, high = design['recognized_right_support_age_controls']
    require(not hits[:design['first_zero_based_control']].any() and maximum_controls <= high-low+1,
            'A continuation started before its budget or remained active too long')
    match = hits & (supports == 2)
    mid = match & (ages >= low) & (ages <= high)
    starts = hits & ~np.vstack((np.zeros_like(hits[:1]), hits[:-1]))
    k, w = np.nonzero(starts & match)
    unique = len(np.unique(np.column_stack((w, k-ages[k, w]+1)), axis=0))
    count = int(hits.sum())
    return dict(continuation_controls=count, actual_right_support_controls=int(match.sum()),
                true_mid_right_support_controls=int(mid.sum()), right_support_agreement=float(match.sum()/count) if count else None,
                starts=int(starts.sum()), duplicate_starts_on_same_actual_support=len(k)-unique,
                affected_worlds=int(hits.any(axis=0).sum()), maximum_consecutive_controls=maximum_controls), sources


def main():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, duration, parent = (goals[key] for key in (KEY, 'right_support_hip_pitch_duration', 'forward_phase_hip_pitch'))
    draft_path, output = ROOT/state['draft'], REPORT/'archived-controller-replay.json'
    draft = read(draft_path)
    require(pair.trial.allowed(project, STAGE) and state['status'] == 'pending-archived-controller-replay'
            and not output.exists() and not task['simulation_running'] and not task['learning_running']
            and duration['status'] == parent['status'] == 'reviewed' and duration['duration_gate_passed']
            and not draft['physical_steps_authorized_by_draft'], 'The measured duration and active route preflight are required')
    evidence, routes = ROOT/draft['duration_result'], ROOT/draft['onset_route_result']
    duration_master, parent_master = ROOT/duration['master'], ROOT/parent['master']
    require(sha(evidence) == duration['result_sha256'] == state['duration_source_sha256']
            and sha(routes) == parent['result_sha256'] and sha(duration_master) == duration['master_sha256']
            and sha(parent_master) == parent['master_sha256'], 'The measured route or its parent changed')
    duration_plan = duration_master.parent/'s1247-friction-low.json'
    require(sha(duration_plan) == read(duration_master)['protocols'][duration_plan.stem], 'The tested continuation setup changed')
    design = read(duration_plan)['duration_probe']
    require(draft['right_support_age_controls'] == design['recognized_right_support_age_controls']
            and draft['conditional_target_delta_rad'] == design['single_control_delta_rad']
            and draft['first_control_zero_based'] == draft['shared_prefix_controls'] == design['first_zero_based_control']
            and draft['maximum_joint_correction_rad'] == design['maximum_joint_correction_rad'], 'The tested duration conditions changed')
    entry = Path(__file__).resolve()
    used = (entry.with_name('right_hip_pitch_duration_control.py'), entry.with_name('forward_phase_hip_pitch.py'))
    sources = {rel(path):sha(path) for path in (entry, draft_path, evidence, routes, duration_plan, duration_master, parent_master, *used)}
    master, cases = read(parent_master), {}
    for key in draft['development_cohorts']:
        protocol = parent_master.parent/f'{key}.json'
        plan = read(protocol)
        require(sha(protocol) == master['protocols'][key] and plan['worlds'] == draft['worlds_per_cohort']
                and plan['maximum_controls'] == draft['standing_controls']+draft['forward_controls']
                and plan['maximum_controls'] == draft['shared_prefix_controls']+draft['branch_controls']
                and plan['walking']['criteria']['yaw_error_max'] == draft['full_forward_gate']['maximum_mean_absolute_yaw_rad_s']
                and plan['minimum_continuous_successes'] == draft['full_forward_gate']['minimum_successes_per_128'], 'The original route gates changed')
        for src in plan['phase_hip_pitch']['observer'].values():
            require(sha(ROOT/src) == plan['frozen_sha256'][src], 'The sensor observer changed')
            sources[src] = plan['frozen_sha256'][src]
        cases[key], archived = inspect(plan, design, ROOT/'records/evaluations'/plan['run_id'])
        sources.update({rel(protocol):master['protocols'][key], **archived})
    observer_config = read(ROOT/plan['phase_hip_pitch']['observer']['config'])
    criterion = observer_config['minimum_single_side_precision']
    accepted = all(row['affected_worlds'] == draft['worlds_per_cohort'] and row['right_support_agreement'] >= criterion
                   for row in cases.values())
    REPORT.mkdir(exist_ok=True)
    put(output, dict(cases=cases, original_actuator_limits_verified=True, archived_recovery_ready=accepted,
                     original_observer_precision_gate=criterion, source_sha256=sources,
                     physical_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    state.update(status='archived-controller-verified' if accepted else 'archived-controller-review-required',
                 result=rel(output), result_sha256=sha(output))
    task.update(current_evidence=rel(output), current_protocol_execution=state['status'],
                next_gate='Freeze the complete forward comparison with both arms restored at control 150.'
                if accepted else 'Use the observed scope to refine the timing before the full-route comparison.')
    pair.trial.write(P, project)
    print('RIGHT_SUPPORT_CONTINUATION_ARCHIVED', accepted, cases, flush=True)


if __name__ == '__main__':
    main()
