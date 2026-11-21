"""Find a common observable support set and define one-control joint perturbations."""

from pathlib import Path

import numpy as np

from forward_hip_yaw_pulse_targets import SIDES, apply_pulse
from run_forward_observed_hip_yaw_pulse import observer_for
from run_roll_stand_forward_diagnostic import P, ROOT, pair, put, read, rel, require, sha


STAGE = 'forward-support-joint-pulse'
KEY = 'forward_support_joint_pulse'
REPORT = ROOT/'records/reports'/STAGE
JOINTS = ('hip_roll', 'hip_pitch')
ROLES = {
    'zero-a': None,
    'hip-roll-positive': ('hip_roll', 1),
    'hip-roll-negative': ('hip_roll', -1),
    'hip-pitch-positive': ('hip_pitch', 1),
    'hip-pitch-negative': ('hip_pitch', -1),
    'zero-b': None,
}


def common_mask(plan, requested, target, contact, observed_side, side):
    walk = plan['walking']
    joint_indices = [walk['joint_names'].index(f'{SIDES[side]}_{joint}') for joint in JOINTS]
    lower_limit, upper_limit = np.asarray(walk['joint_limits'], dtype=np.float32).T
    baseline = np.clip(requested, lower_limit, upper_limit)
    cap = plan['feedback']['maximum_joint_correction_rad']
    radius = plan['pulse']['delta_rad']
    require(np.isfinite(target).all() and np.abs(target-baseline).max() <= cap+1e-6,
            'The original projected target exceeded its correction cap')
    lower = np.maximum(baseline-cap, lower_limit)[..., joint_indices]
    upper = np.minimum(baseline+cap, upper_limit)[..., joint_indices]
    value = target[..., joint_indices]
    both_joints = ((value-radius >= lower) & (value+radius <= upper)).all(axis=-1)
    return both_joints & contact[..., side] & ~contact[..., 1-side] & (observed_side == side+1)


def branch_target(plan, target, selected, side, role):
    intervention = ROLES[role]
    if intervention is None:
        return target
    joint, sign = intervention
    index = plan['walking']['joint_names'].index(f'{SIDES[side]}_{joint}')
    return apply_pulse(target, selected, index, sign*plan['pulse']['delta_rad'])


def inspect_archive(plan, archive_folder):
    manifest_path, commands, trace = (archive_folder/name for name in (
        'guarded-evaluation.json', 'shared-commands.npz', 'shared-trace.npz'))
    manifest = read(manifest_path)
    require(sha(commands) == manifest['shared_commands_sha256'] and sha(trace) == manifest['shared_trace_sha256'],
            'The original shared commands or physical states changed')
    design = plan['pulse']
    start, stop, minimum = (design[key] for key in ('earliest_fork_control', 'shared_controls', 'minimum_worlds_per_fork'))
    window = slice(start, stop)
    observer = observer_for(plan)
    with np.load(commands, allow_pickle=False) as saved, np.load(trace, allow_pickle=False) as physical:
        requested, target = saved['requested_targets'][window], saved['applied_targets'][window]
        observed, contact = observer.predict(saved['observed'][window]), physical['foot_contact'][window]
    results = {}
    for side, label in enumerate(SIDES):
        mask = common_mask(plan, requested, target, contact, observed, side)
        counts = mask.sum(axis=1)
        available = np.flatnonzero(counts >= minimum)
        results[label] = dict(peak_matched_worlds=int(counts.max()), distinct_worlds=int(mask.any(axis=0).sum()),
                              first_fork=None)
        if available.size:
            k = int(available[0])
            selected = np.flatnonzero(mask[k])
            results[label]['first_fork'] = dict(control=start+k, matched_worlds=len(selected),
                                                selected_worlds=selected.tolist())
    return results, {rel(item): sha(item) for item in (manifest_path, commands, trace)}


def preflight():
    project = read(P)
    goal = project['goal_execution'][KEY]
    previous = project['goal_execution']['forward_observed_hip_yaw_pulse']
    task = project['staged_training_request']
    draft, master = ROOT/goal['draft'], ROOT/previous['master']
    result, headroom = ROOT/goal['parent_result'], ROOT/goal['individual_joint_headroom']
    receipt = REPORT/'common-mask.json'
    require(pair.trial.allowed(project, STAGE+'-preflight') and not task['simulation_running']
            and not task['learning_running'] and goal['status'] == 'archived-command-headroom-verified'
            and previous['status'] == 'reviewed' and not receipt.exists()
            and sha(master) == previous['master_sha256'] and sha(result) == goal['parent_result_sha256']
            and sha(headroom) == goal['individual_joint_headroom_sha256'], 'The archived study or next stage changed')
    draft_data, protocol, review = read(draft), read(master), read(result)
    require(tuple(draft_data['comparison_joints_on_support_side']) == JOINTS
            and draft_data['physical_steps_authorized_by_draft'] == 0 and not review['heldout_used'],
            'The comparison scope or development-only source changed')
    sources = {rel(item): sha(item) for item in (Path(__file__).resolve(), draft, master, result, headroom)}
    cases = {}
    for case, digest in protocol['protocols'].items():
        path = master.parent/f'{case}.json'
        plan = read(path)
        checks = {rel(ROOT/name): sha(ROOT/name) for name in plan['pulse_observer'].values()}
        require(sha(path) == digest and plan['worlds'] == 128 and plan['initial_stand_controls'] == 50
                and plan['pulse']['earliest_fork_control'] == draft_data['earliest_fork_control']
                and plan['pulse']['minimum_worlds_per_fork'] == draft_data['minimum_same_worlds_per_side_and_cohort']
                and plan['pulse']['delta_rad'] == draft_data['single_control_delta_rad']
                and plan['feedback']['maximum_joint_correction_rad'] == draft_data['maximum_joint_correction_rad']
                and all(plan['frozen_sha256'][name] == current for name, current in checks.items()),
                'The frozen observation, comparison action or actuator constraints changed')
        folder = (ROOT/review['cases'][case]['video']['path']).parent
        require(folder == ROOT/'records/evaluations'/plan['run_id'], 'The archived run does not match the plan')
        cases[case], trace_sources = inspect_archive(plan, folder)
        sources.update({rel(path): digest, **checks, **trace_sources})
    ready = all(side['first_fork'] is not None for row in cases.values() for side in row.values())
    REPORT.mkdir(exist_ok=True)
    put(receipt, dict(common_mask_verified=ready, cases=cases, source_sha256=sources,
                      physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    goal.update(status='archived-common-mask-verified' if ready else 'archived-common-mask-below-minimum',
                common_mask_verified=ready, common_mask=rel(receipt), common_mask_sha256=sha(receipt))
    task.update(current_evidence=rel(receipt), current_protocol_execution=goal['status'])
    pair.trial.write(P, project)
    print('FORWARD_SUPPORT_COMMON_MASK', ready, {
        case: {side: value['first_fork']['matched_worlds'] if value['first_fork'] else 0 for side, value in row.items()}
        for case, row in cases.items()}, flush=True)


if __name__ == '__main__':
    preflight()
