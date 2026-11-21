"""Find reachable same-state comparison windows for other support-leg joints."""

from pathlib import Path

import numpy as np

from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from run_forward_observed_hip_yaw_pulse import observer_for


REPORT = ROOT/'records/reports/forward-observed-hip-yaw-pulse'
OUTPUT = REPORT/'support-joint-headroom.json'
JOINTS = ('hip_roll', 'hip_pitch', 'knee', 'ankle')


def case(plan, path, sources):
    folder = ROOT/'records/evaluations'/plan['run_id']
    manifest_path, commands, trace = (folder/name for name in ('guarded-evaluation.json', 'shared-commands.npz', 'shared-trace.npz'))
    manifest = read(manifest_path)
    require(sha(commands) == manifest['shared_commands_sha256'] and sha(trace) == manifest['shared_trace_sha256'],
            'The measured original control sequence changed')
    sources.update({rel(item): sha(item) for item in (commands, trace, manifest_path, path)})
    start, stop = 150, plan['pulse']['shared_controls']
    window = slice(start, stop)
    classifier = observer_for(plan)
    with np.load(commands, allow_pickle=False) as saved, np.load(trace, allow_pickle=False) as physics:
        requested, target = saved['requested_targets'][window], saved['applied_targets'][window]
        identified = classifier.predict(saved['observed'][window])
        contacts = physics['foot_contact'][window]
    bounds = np.asarray(plan['walking']['joint_limits'], dtype=np.float32)
    cap, pulse = plan['feedback']['maximum_joint_correction_rad'], plan['pulse']['delta_rad']
    base = np.clip(requested, bounds[:, 0], bounds[:, 1])
    lower = np.maximum(base-cap, bounds[:, 0])
    upper = np.minimum(base+cap, bounds[:, 1])
    reachable = (target-pulse >= lower) & (target+pulse <= upper)
    output = {}
    for side, label in enumerate(('left', 'right')):
        observed = (identified == side+1) & contacts[..., side] & ~contacts[..., 1-side]
        output[label] = {}
        for joint in JOINTS:
            index = plan['walking']['joint_names'].index(f'{label}_{joint}')
            mask = reachable[..., index] & observed
            counts = mask.sum(axis=1)
            eligible = np.flatnonzero(counts >= 64)
            output[label][joint] = dict(first_fork_at_least_64=int(start+eligible[0]) if len(eligible) else None,
                peak_matched_worlds=int(counts.max()), distinct_worlds=int(mask.any(axis=0).sum()))
    return output


def main():
    project = read(P)
    first = project['goal_execution']['forward_hip_yaw_pulse']
    settled = project['goal_execution']['forward_observed_hip_yaw_pulse']
    master_path = ROOT/first['master']
    require(first['status'] == 'reviewed' and sha(master_path) == first['master_sha256']
            and sha(ROOT/settled['transfer_evidence']) == settled['transfer_evidence_sha256'], 'Expected the original verified archives')
    transfer = read(ROOT/settled['transfer_evidence'])
    model = next(path for path in transfer['sources'] if path.endswith('/forward-stance-observation-cpu.json'))
    conf = 'configs/experiments/common_walk_forward_stance_observation.json'
    require(all(sha(ROOT/path) == transfer['sources'][path] for path in (model, conf)), 'Observer files changed')
    observer = dict(receipt=model, config=conf)
    sources = {rel(path): sha(path) for path in (Path(__file__).resolve(), master_path, ROOT/model, ROOT/conf)}
    data = {}
    for key, digest in read(master_path)['protocols'].items():
        original = master_path.parent/(key+'.json')
        require(sha(original) == digest, 'Frozen pulse protocol changed')
        data[key] = case(dict(read(original), pulse_observer=observer), original, sources)
    ready = {joint: all(row[side][joint]['first_fork_at_least_64'] is not None
                        for row in data.values() for side in ('left', 'right')) for joint in JOINTS}
    put(OUTPUT, dict(cases=data, all_cohorts_have_matched_windows=ready, delta_rad=.01,
                     source_sha256=sources, physics_steps=0, learning_updates=0, hardware_io=False, heldout_used=False))
    print('FORWARD_SUPPORT_JOINT_HEADROOM', ready, flush=True)


if __name__ == '__main__':
    main()
