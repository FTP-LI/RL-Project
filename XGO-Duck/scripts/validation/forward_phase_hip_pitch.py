"""Apply and audit a single observed-phase hip-pitch correction per support interval."""

from pathlib import Path

import numpy as np

from audit_forward_yaw_by_support_age import contact_age
from run_forward_observed_hip_yaw_pulse import observer_for
from run_roll_stand_forward_diagnostic import P, ROOT, pair, put, read, rel, require, sha
from roll_stand_walk_sequence import FORWARD


KEY = 'forward_phase_hip_pitch'
STAGE = 'forward-phase-hip-pitch-preflight'
REPORT = ROOT/'records/reports/forward-phase-hip-pitch'
SIDES = ('left', 'right')


class HipPitchSchedule:
    def __init__(self, plan, draft):
        names = plan['walking']['joint_names']
        self.lower, self.upper = np.asarray(plan['walking']['joint_limits'], dtype=np.float32).T
        self.pitch = [names.index(f'{side}_{draft["joint"]}') for side in SIDES]
        self.headroom_joints = [[names.index(f'{side}_{joint}') for joint in draft['paired_headroom_joints']] for side in SIDES]
        self.sign = [draft['direction_by_observed_support'][side] for side in SIDES]
        self.delta, self.cap = draft['single_control_delta_rad'], draft['maximum_joint_correction_rad']
        self.start, self.latest_age = draft['first_zero_based_control'], draft['latest_recognized_support_age_controls']
        self.gyro = draft['pre_action_gyro_column']
        self.observer = observer_for({'pulse_observer': draft['observer']})
        n = plan['worlds']
        self.previous_label = np.zeros(n, dtype=np.int8)
        self.last_known_support = np.zeros(n, dtype=np.int8)
        self.age = np.zeros(n, dtype=np.int32)
        self.fired = np.zeros(n, dtype=bool)

    def apply(self, k, requested, original, observed, forward):
        labels = np.where(forward, self.observer.predict(observed), 0)
        known = labels != 0
        changed_support = known & (labels != self.last_known_support)
        self.fired[changed_support] = False
        self.last_known_support[known] = labels[known]
        self.age = np.where(known, np.where(labels == self.previous_label, self.age+1, 1), 0)
        self.previous_label = labels
        selected = np.zeros(len(forward), dtype=np.int8)
        if k < self.start:
            return original, selected

        baseline = np.clip(requested, self.lower, self.upper)
        require(np.abs(original-baseline).max() <= self.cap+1e-6, 'The parent correction exceeds the original envelope')
        floor = np.maximum(self.lower, baseline-self.cap)
        ceiling = np.minimum(self.upper, baseline+self.cap)
        target = original.copy()
        for index, (axis, joints, direction) in enumerate(zip(self.pitch, self.headroom_joints, self.sign, strict=True), 1):
            current = original[:, joints]
            fits = ((current-self.delta >= floor[:, joints]) & (current+self.delta <= ceiling[:, joints])).all(axis=1)
            pulse = (labels == index) & (self.age <= self.latest_age) & ~self.fired & fits & (direction*observed[:, self.gyro] > 0)
            target[pulse, axis] += np.float32(direction*self.delta)
            self.fired[pulse] = True
            selected[pulse] = index
        return target, selected


def inspect_case(plan, draft, folder):
    command_path, trace_path = folder/'commands.npz', folder/'trace.npz'
    manifest_path = folder/'guarded-evaluation.json'
    manifest = read(manifest_path)
    require(sha(command_path) == manifest['command_sha256'] and sha(trace_path) == manifest['trace_sha256']
            and manifest['actual_control_steps'] == plan['maximum_controls'] and not manifest['artifact_errors'],
            'The complete official route evidence changed')
    schedule = HipPitchSchedule(plan, draft)
    with np.load(command_path, allow_pickle=False) as samples, np.load(trace_path, allow_pickle=False) as states:
        requested, original, observed, phase = (samples[key] for key in ('requested_targets', 'applied_targets', 'observed', 'phase_used'))
        np.testing.assert_allclose(samples['actual_motor_target'], original, rtol=0, atol=1e-6)
        true_support, true_age = contact_age(states['foot_contact'])
    controls = plan['maximum_controls']
    require(original.shape == requested.shape == (controls, plan['worlds'], len(schedule.lower))
            and phase.shape == observed.shape[:2] == (controls, plan['worlds']), 'The archived action inputs are incomplete')

    hits = np.zeros_like(phase)
    for k, row in enumerate(original):
        target, hits[k] = schedule.apply(k, requested[k], row, observed[k], phase[k] == FORWARD)
        change = np.zeros_like(row)
        for i, (axis, sign) in enumerate(zip(schedule.pitch, schedule.sign, strict=True), 1):
            change[hits[k] == i, axis] = sign*schedule.delta
        np.testing.assert_allclose(target-row, change, rtol=0, atol=2e-7)
        nominal = np.clip(requested[k], schedule.lower, schedule.upper)
        require((target >= schedule.lower-1e-6).all() and (target <= schedule.upper+1e-6).all() and
                np.abs(target-nominal).max() <= schedule.cap+1e-6, 'A scheduled target falls outside the original envelope')
    require((hits[:draft['first_zero_based_control']] == 0).all() and not ((hits != 0) & (phase != FORWARD)).any(),
            'The archived schedule changed a command outside the permitted forward window')
    rows = {}
    for index, side in enumerate(SIDES, 1):
        k, world = np.nonzero(hits == index)
        correct = true_support[k, world] == index
        ages = true_age[k[correct], world[correct]]
        episodes = np.column_stack((world[correct], k[correct]-ages+1))
        unique = len(np.unique(episodes, axis=0))
        count = len(k)
        rows[side] = dict(pulses=count, distinct_worlds=len(np.unique(world)), actual_support_matches=int(correct.sum()),
                          actual_support_precision=float(correct.mean()) if count else None,
                          actual_support_age_first_three=int((ages <= 3).sum()), repeated_in_same_physical_support=len(ages)-unique)
    return rows, {rel(path): sha(path) for path in (command_path, trace_path, manifest_path)}


def main():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, parent, original = (goals[key] for key in (KEY, 'forward_support_joint_pulse', 'roll_friction_stand_forward'))
    draft_path, output = ROOT/state['draft'], REPORT/'archived-schedule.json'
    draft = read(draft_path)
    require(pair.trial.allowed(project, STAGE) and not task['simulation_running'] and not task['learning_running']
            and state['status'] == 'pending-archived-schedule-readback' and not output.exists()
            and parent['status'] == original['status'] == 'reviewed' and draft['matching_gyro_sign_required']
            and draft['physical_steps_authorized_by_draft'] == 0, 'The schedule preflight is not the active stage')
    result, timing, original_master = (ROOT/path for path in (state['parent_result'], state['support_timing'], original['master']))
    require(sha(result) == state['parent_result_sha256'] and sha(timing) == state['support_timing_sha256']
            and sha(original_master) == original['master_sha256'], 'The matched pulse or route evidence changed')
    pulses = read(result)['low_friction_advance_candidates']
    stages = read(timing)['cases']
    require(pulses == {'left': [{'joint': 'hip_pitch', 'sign': 'negative'}], 'right': [{'joint': 'hip_pitch', 'sign': 'positive'}]}
            and draft['direction_by_observed_support'] == {'left': -1, 'right': 1}
            and all(row[side]['observed_age']['controls_1_to_3'] == row[side]['worlds']
                    and row[side]['preceding_gyro_positive'] == (row[side]['worlds'] if side == 'right' else 0)
                    for row in stages.values() for side in SIDES), 'The schedule does not match the observed intervention context')
    observer_paths = [ROOT/path for path in draft['observer'].values()]
    source_plan = ROOT/parent['master']
    require(sha(source_plan) == parent['master_sha256'], 'The matched support protocol changed')
    first = source_plan.parent/(read(source_plan)['order'][0]+'.json')
    require(sha(first) == read(source_plan)['protocols'][first.stem] and all(
        read(first)['frozen_sha256'][rel(path)] == sha(path) for path in observer_paths), 'The frozen observing model changed')
    sources = {rel(path): sha(path) for path in (Path(__file__).resolve(), draft_path, result, timing, original_master, first, *observer_paths)}
    cases = {}
    for key, digest in read(original_master)['protocols'].items():
        path = original_master.parent/f'{key}.json'
        receipt = original['scored'][key]
        independent = ROOT/receipt['path']
        plan, verified = read(path), read(independent)
        require(sha(path) == digest == verified['protocol_sha256'] and sha(independent) == receipt['sha256']
                and verified['full_route'] and not verified['holdout_used']
                and plan['forward_controls'] == draft['full_forward_controls']
                and plan['feedback']['maximum_joint_correction_rad'] == draft['maximum_joint_correction_rad']
                and plan['walking']['criteria']['yaw_error_max'] == draft['forward_yaw_limit_rad_s']
                and plan['minimum_continuous_successes'] == draft['minimum_forward_successes'], 'The original forward gate changed')
        folder = ROOT/'records/evaluations'/plan['run_id']
        cases[key], artifacts = inspect_case(plan, draft, folder)
        sources.update({rel(path): digest, rel(independent): receipt['sha256'], **artifacts})
    minimum_precision = read(ROOT/draft['observer']['config'])['minimum_single_side_precision']
    accepted = all(item['distinct_worlds'] == 128 and item['actual_support_precision'] >= minimum_precision
                   for case in cases.values() for item in case.values())
    REPORT.mkdir(exist_ok=True)
    put(output, dict(cases=cases, target_and_timing_verified=True, original_observer_precision_gate=minimum_precision,
                     schedule_ready_for_route_design=accepted, source_sha256=sources, physical_steps=0,
                     learning_updates=0, heldout_used=False, hardware_io=False))
    state.update(status='archived-schedule-verified' if accepted else 'archived-schedule-needs-timing-review',
                 result=rel(output), result_sha256=sha(output))
    task.update(current_evidence=rel(output), current_protocol_execution=state['status'],
                next_gate='Freeze and compare the timed hip-pitch candidate against the original 20-second route and joint, posture, speed and contact requirements.'
                if accepted else 'Use the saved trigger errors to refine support timing before the dynamic comparison.')
    pair.trial.write(P, project)
    print('HIP_PITCH_ARCHIVED_SCHEDULE', accepted, cases, flush=True)


if __name__ == '__main__':
    main()
