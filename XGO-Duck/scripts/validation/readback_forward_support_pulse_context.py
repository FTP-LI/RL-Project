"""Verify the pulse summaries and locate the actual interventions within support phases."""

from pathlib import Path

import imageio_ffmpeg
import numpy as np

from audit_forward_yaw_by_support_age import contact_age
from forward_support_joint_pulse import JOINTS, KEY, REPORT, ROLES, SIDES, observer_for
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


def observation_age(labels):
    age = np.ones_like(labels, dtype=np.int32)
    for control in range(1, len(labels)):
        age[control] = np.where(labels[control] == labels[control-1], age[control-1]+1, 1)
    return age


def age_masks(age):
    return {'controls_1_to_3': age <= 3, 'controls_4_to_8': (age >= 4) & (age <= 8), 'controls_9_and_later': age >= 9}


def read_case(case, plan, candidates):
    video = ROOT/case['video']['path']
    folder = video.parent
    manifest = read(folder/'guarded-evaluation.json')
    forks, command, trace = (folder/name for name in ('pulse-forks.json', 'shared-commands.npz', 'shared-trace.npz'))
    for file, field in ((forks, 'forks_sha256'), (command, 'shared_commands_sha256'), (trace, 'shared_trace_sha256')):
        require(sha(file) == manifest[field], 'The named support-state source changed')
    with np.load(command, allow_pickle=False) as inputs, np.load(trace, allow_pickle=False) as physics:
        observed = inputs['observed']
        observed_age = observation_age(observer_for(plan).predict(observed))
        _, actual_age = contact_age(physics['foot_contact'])
    raw = read(forks)
    results = {}
    for name in SIDES:
        row, fork = case['sides'][name], raw[name]
        k, scene = fork['control'], fork['selected_worlds']
        assert row['all_six_starts_exact'] and row['identical_sensor_mask_verified']
        assert row['fork_control'] == k and row['worlds'] == len(scene)
        pulse_yaw = {}
        for role in ROLES:
            branch = folder/name/f'{role}.npz'
            require(sha(branch) == fork['branches'][role], 'An intervention archive changed')
            with np.load(branch, allow_pickle=False) as samples:
                pulse_yaw[role] = np.abs(samples['state_angular_velocity'][1:, scene, 2].astype(np.float64))
        zero_a, zero_b = pulse_yaw['zero-a'], pulse_yaw['zero-b']
        baseline = (zero_a+zero_b)*.5
        for joint in JOINTS:
            for label, n in (('primary', 3), ('complete', 8)):
                check = row['joints'][joint][label]
                gap = np.abs(zero_a[:n].mean(axis=0)-zero_b[:n].mean(axis=0))
                np.testing.assert_allclose(np.percentile(gap, 95), check['zero_pair_gap_p95_rad_s'], rtol=0, atol=1e-14)
                for sign in ('positive', 'negative'):
                    role = joint.replace('_', '-')+'-'+sign
                    median = np.median((pulse_yaw[role][:n]-baseline[:n]).mean(axis=0))
                    np.testing.assert_allclose(median, check['effects'][sign]['median_change_in_mean_abs_yaw_rad_s'], rtol=0, atol=1e-14)
        predicted_age, physical_age = observed_age[k, scene], actual_age[k, scene]
        result = dict(worlds=len(scene), control=k,
                      preceding_gyro_positive=int((observed[k, scene, 2] > 0).sum()),
                      observed_age={label: int(mask.sum()) for label, mask in age_masks(predicted_age).items()},
                      physical_age={label: int(mask.sum()) for label, mask in age_masks(physical_age).items()},
                      qualifying_arm_by_observed_age={})
        for option in candidates[name]:
            role = option['joint'].replace('_', '-')+'-'+option['sign']
            branch_change = (pulse_yaw[role]-baseline).mean(axis=0)
            zero_gap = np.abs((zero_a-zero_b).mean(axis=0))
            result['qualifying_arm_by_observed_age'][role] = {
                label: dict(worlds=int(mask.sum()), median_change_rad_s=float(np.median(branch_change[mask])),
                            zero_pair_gap_p95_rad_s=float(np.percentile(zero_gap[mask], 95)))
                for label, mask in age_masks(predicted_age).items() if mask.any()}
        results[name] = result
    assert sha(video) == case['video']['sha256']
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    assert next(frames)['fps'] == 25
    assert sum(1 for _ in frames) == case['video']['frames'] == 126
    return results


def main():
    state = read(P)['goal_execution'][KEY]
    source, master_path = ROOT/state['result'], ROOT/state['master']
    out = REPORT/'pulse-support-timing.json'
    require(state['status'] == 'reviewed' and not out.exists() and sha(source) == state['result_sha256']
            and sha(master_path) == state['master_sha256'], 'The pulse proof is not closed or its sources changed')
    scores, master = read(source), read(master_path)
    cases = {}
    for name, case in scores['cases'].items():
        path = master_path.parent/f'{name}.json'
        assert sha(path) == case['protocol_sha256'] == master['protocols'][name]
        cases[name] = read_case(case, read(path), scores['low_friction_advance_candidates'])
    put(out, dict(cases=cases, fully_decoded_videos=len(cases), rechecked_window_controls=[3, 8],
                  physical_transitions_verified=scores['physical_transitions'],
                  source_sha256={rel(p): sha(p) for p in (source, master_path, Path(__file__).resolve())},
                  physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print('SUPPORT_JOINT_RAW_READBACK', {
        name: {side: value['observed_age'] for side, value in case.items()} for name, case in cases.items()}, flush=True)


if __name__ == '__main__':
    main()
