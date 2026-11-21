"""Read back two observation-only steering signs using the matched physical branches."""

from pathlib import Path

import numpy as np

from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from walking_stance_feedback import StanceObserver


REPORT = ROOT/'records/reports/forward-hip-yaw-pulse'
OBSERVER = ROOT/'records/reports/common-walk-right-support-yaw-training/forward-stance-observation-cpu.json'
OBSERVER_PLAN = ROOT/'configs/experiments/common_walk_forward_stance_observation.json'
TRANSFER = REPORT/'stance-observer-transfer.json'
OUTPUT = REPORT/'pulse-adaptive-sign-readback.json'


def summaries(absolute, baseline, controls):
    change = np.mean(absolute[:controls]-baseline[:controls], axis=0)
    return dict(controls=controls, median_change_rad_s=float(np.median(change)), improved=int((change < 0).sum()))


def main():
    state = read(P)['goal_execution']['forward_hip_yaw_pulse']
    require(state['status'] == 'reviewed' and sha(ROOT/state['result']) == state['result_sha256']
            and sha(ROOT/state['master']) == state['master_sha256'], 'Expected verified pulse comparison')
    receipt, config, transfer = (read(path) for path in (OBSERVER, OBSERVER_PLAN, TRANSFER))
    require(transfer['passed_same_observer_gates'] and receipt['protocol_sha256'] == sha(OBSERVER_PLAN)
            and transfer['sources'][rel(OBSERVER)] == sha(OBSERVER), 'Same observer did not qualify on current friction cohorts')
    observer = StanceObserver(receipt['model'], config['minimum_score_separation'])
    master_path = ROOT/state['master']
    master = read(master_path)
    controls = master['shared']['primary_response_controls']
    result, sources = {}, {rel(p): sha(p) for p in (OBSERVER, OBSERVER_PLAN, TRANSFER, Path(__file__).resolve(), master_path)}
    for case, checksum in master['protocols'].items():
        path = master_path.parent/(case+'.json')
        require(sha(path) == checksum, 'Pulse protocol changed')
        plan = read(path)
        root = ROOT/'records/evaluations'/plan['run_id']
        manifest = read(root/'guarded-evaluation.json')
        mark = state['scored'][case]
        independent = ROOT/mark['path']
        require(sha(independent) == mark['sha256'] and read(independent)['protocol_sha256'] == checksum,
                'Independent score does not match the collector')
        fork_path = root/'pulse-forks.json'
        require(sha(fork_path) == manifest['forks_sha256'], 'Fork record changed')
        sources[rel(fork_path)] = manifest['forks_sha256']
        hips = [20+plan['walking']['joint_names'].index(f'{leg}_hip_pitch') for leg in ('left', 'right')]
        result[case] = {}
        for side, label in enumerate(('left', 'right'), 1):
            saved = read(fork_path)[label]
            selected = saved['selected_worlds']
            series, observed = {}, None
            for role, digest in saved['branches'].items():
                trace = root/label/f'{role}.npz'
                require(sha(trace) == digest, 'Candidate trace changed')
                sources[rel(trace)] = digest
                with np.load(trace, allow_pickle=False) as branch:
                    features = branch['command_observed'][0, selected]
                    series[role] = branch['state_angular_velocity'][1:, selected, 2].astype(np.float64)
                    if observed is None:
                        observed = features
                    else:
                        np.testing.assert_array_equal(observed, features)
            baseline = (np.abs(series['zero-a'])+np.abs(series['zero-b']))*.5
            phase = observer.predict(observed)
            results = {}
            for rule, evidence in (('gyro_z', observed[:, 2]), ('sum_hip_pitch_rate', observed[:, hips].sum(axis=1))):
                select = np.where(phase == side, -np.sign(evidence), 0.)
                coupled = np.where(select[None] > 0, np.abs(series['positive']),
                                   np.where(select[None] < 0, np.abs(series['negative']), baseline))
                results[rule] = dict(selected_count=int((select != 0).sum()), primary=summaries(coupled, baseline, controls),
                                     full_response=summaries(coupled, baseline, len(baseline)))
            result[case][label] = dict(worlds=len(selected), local_first_control_odd_signed_yaw_rad_s=
                float(np.median((series['positive'][0]-series['negative'][0])*.5)), rules=results)
    put(OUTPUT, dict(cases=result, method='Select only from actually sampled intervention branches using pre-action inputs',
                     observer_only_selects_contact=True, physics_steps=0, learning_updates=0, hardware_io=False,
                     heldout_used=False, sources=sources))
    print('HIP_YAW_CAUSAL_SIGN_READBACK', len(result), flush=True)


if __name__ == '__main__':
    main()
