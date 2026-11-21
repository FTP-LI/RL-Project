"""Verify the frozen encoder/IMU contact observer on both friction settings."""

from pathlib import Path

import numpy as np

from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from check_walk_forward_stance_observation import PLAN as OBSERVER_PLAN, report_metrics
from walking_stance_feedback import StanceObserver


RECEIPT = ROOT/'records/reports/common-walk-right-support-yaw-training/forward-stance-observation-cpu.json'
OUTPUT = ROOT/'records/reports/forward-hip-yaw-pulse/stance-observer-transfer.json'


def main():
    reference = read(P)['goal_execution']['roll_friction_stand_forward']
    model, plan = read(RECEIPT), read(OBSERVER_PLAN)
    require(reference['status'] == 'reviewed' and len(reference['scored']) == 4 and model['admitted_for_control_design']
            and model['protocol_sha256'] == sha(OBSERVER_PLAN) and not model['heldout_used'], 'Prior observer provenance differs')
    observer = StanceObserver(model['model'], plan['minimum_score_separation'])
    cases, sources = {}, {rel(path): sha(path) for path in (OBSERVER_PLAN, RECEIPT, Path(__file__).resolve())}
    for key, mark in reference['scored'].items():
        independence = ROOT/mark['path']
        proof = read(independence)
        folder = ROOT/'records/evaluations'/proof['run_id']
        manifest = read(folder/'guarded-evaluation.json')
        commands, trace, route = (folder/name for name in ('commands.npz', 'trace.npz', 'protocol.json'))
        files = {independence: mark['sha256'], trace: manifest['trace_sha256'], commands: manifest['command_sha256'],
                 route: proof['protocol_sha256']}
        require(all(sha(path) == digest for path, digest in files.items()) and proof['full_route'] and not proof['holdout_used'],
                'Archived walking trace differs')
        sources.update({rel(path): digest for path, digest in files.items()})
        archived = read(route)
        start, duration = archived['initial_stand_controls'], archived['forward_controls']
        window = slice(start, start+duration)
        ids = [index for index, name in enumerate(archived['walking']['joint_names']) if name.startswith(('left_', 'right_'))]
        require(observer.columns.tolist() == [0, 1, 3, 4, 5]+[6+i for i in ids]+[20+i for i in ids],
                'Encoder observation columns differ')
        with np.load(commands, allow_pickle=False) as saved, np.load(trace, allow_pickle=False) as states:
            require(states['active'][window].all(), 'Incomplete contact labels')
            predicted = observer.predict(saved['observed'][window])
            contact = states['foot_contact'][window]
        actual = contact[..., 0].astype(int)+2*contact[..., 1]
        cases[key] = report_metrics(np.where(actual == 3, 0, actual), predicted, plan)
    passed = all(row['passed'] for row in cases.values())
    put(OUTPUT, dict(cases=cases, passed_same_observer_gates=passed, sources=sources, fitted_new_weights=False,
                     aligned_same_control_pre_action=True, physics_steps=0, learning_updates=0, hardware_io=False, heldout_used=False))
    print('FORWARD_HIP_YAW_STANCE_TRANSFER', passed, {key: row['passed'] for key, row in cases.items()}, flush=True)


if __name__ == '__main__':
    main()
