"""Compare the pulse-based prediction with the verified 20-second motion."""
import csv
from pathlib import Path

import numpy as np

from common_walk_actuator_yaw_feedback import OUT, PLAN, ROLES
from record_io import write
from review_walk_yaw_pulse_capacity import apply_kernel
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import ActuatorYawFeedback


def main():
    result_path = OUT/'analysis.json'
    result, spec = read(result_path), read(PLAN)
    assert result['status'] == 'passed-integrity' and result['protocol_sha256'] == sha(PLAN)
    assert not result['heldout_used'] and not spec['heldout_used']
    record = ROOT/result['record']
    p, fork = spec['environment'], spec['phase']['warmup_controls']
    worlds = p['groups']['forward']
    hip = [20+p['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
    screen = read(ROOT/spec['feedback_screen'])
    law = ActuatorYawFeedback(screen, spec['feedback']['maximum_yaw_input_rad_s'])
    prefix_file = record/'trace-shared-prefix.npz'
    assert sha(prefix_file) == result['shared_prefix_sha256']
    with np.load(prefix_file) as prefix:
        last_obs = prefix['observations'][-1, worlds]

    traces = {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        assert sha(path) == result['trace_sha256'][role]
        with np.load(path) as data:
            yaw = data['yaw_rate'][:, worlds].astype(float)
            active = data['active'][:, worlds]
            observed = np.concatenate((last_obs[None], data['observations'][:, worlds]))
        sensors = np.stack((observed[..., 2], observed[..., hip].sum(axis=-1)), axis=-1)
        features = np.concatenate((sensors[1:], np.diff(sensors, axis=0)), axis=-1)
        traces[role] = dict(yaw=yaw, active=active, command=np.where(active, law.commands(features), 0.))

    candidate = traces[ROLES[-1]]
    correction_file = record/'feedback-readback.npz'
    assert sha(correction_file) == result['inputs'][ROLES[-1]]['readback_sha256']
    with np.load(correction_file) as readback:
        local = {world: index for index, world in enumerate(worlds)}
        rows = readback['step']-fork-1, [local[int(world)] for world in readback['world']]
        effective_command = np.zeros_like(candidate['command'])
        effective_command[rows] = readback['yaw']*readback['fraction']
    realized_proxy = apply_kernel(effective_command, law.kernel)
    comparisons = {}
    for role in ROLES[:2]:
        reference = traces[role]
        both = reference['active'] & candidate['active']
        predicted = apply_kernel(reference['command'], law.kernel)
        base, achieved = reference['yaw'], candidate['yaw']
        measured_gain = abs(base)-abs(achieved)
        proxy_gain = abs(base)-abs(base+predicted)
        samples = {}
        for start, end in ((1, 4), (4, 8), (8, 12), (12, 20)):
            block = slice(start*50-fork, end*50-fork)
            mask = both[block]
            assert mask.any()
            samples[f'{start}-{end}s'] = dict(
                active_control_pairs=int(mask.sum()), measured_yaw_gain_rad_s=float(measured_gain[block][mask].mean()),
                archived_linear_gain_rad_s=float(proxy_gain[block][mask].mean()))
        comparisons[role] = dict(matched_active_control_pairs=int(both.sum()),
            measured_yaw_gain_rad_s=float(measured_gain[both].mean()),
            archived_linear_gain_rad_s=float(proxy_gain[both].mean()),
            linear_gain_using_actual_capped_commands_rad_s=float((abs(base)-abs(base+realized_proxy))[both].mean()),
            intervals=samples)

    gpu_path = OUT/'gpu.csv'
    with gpu_path.open(newline='') as csv_file:
        reader = csv.reader(csv_file)
        header = [value.strip() for value in next(reader)]
        assert header[2:8] == ['utilization.gpu [%]', 'utilization.memory [%]', 'memory.used [MiB]',
                              'memory.total [MiB]', 'power.draw [W]', 'temperature.gpu']
        rows = [[int(row[2]), int(row[4]), int(row[5])] for row in reader if len(row) == len(header)]
    load = np.asarray(rows)
    info = dict(status='passed-saved-trace-review', protocol_sha256=sha(PLAN), source_sha256=sha(result_path),
                script_sha256=sha(Path(__file__)), comparisons=comparisons,
                resource=dict(samples=len(load), median_gpu_utilization_percent=float(np.median(load[:, 0])),
                              peak_gpu_utilization_percent=int(load[:, 0].max()), peak_vram_mib=int(load[:, 1].max()),
                              total_vram_mib=int(load[:, 2].max()), source_sha256=sha(gpu_path)),
                physical_substeps=0, learning_updates=0, heldout_used=False,
                scope='Same-seed open-loop linear proxy and actual closed-loop traces. Aligned time points are not identical physical states.')
    write(OUT/'closed-loop-review.json', info)
    print('WALK_FEEDBACK_CLOSED_LOOP_REVIEW', comparisons, info['resource'], flush=True)


if __name__ == '__main__':
    main()
