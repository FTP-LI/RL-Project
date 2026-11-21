"""Measure paired physical differences before the closed-loop hip-pitch intervention."""

from pathlib import Path

import numpy as np

from forward_phase_hip_pitch import KEY, REPORT
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


def pair_prefix(case, controls, sources):
    old, new = (ROOT/'records/evaluations'/case[key] for key in ('official_run_id', 'candidate_run_id'))
    for folder in (old, new):
        path = folder/'guarded-evaluation.json'
        manifest = read(path)
        sources[rel(path)] = sha(path)
        for filename, field in (('trace.npz', 'trace_sha256'), ('commands.npz', 'command_sha256')):
            data = folder/filename
            digest = sha(data)
            require(digest == manifest[field], 'A matched run changed')
            sources[rel(data)] = digest
    with np.load(old/'commands.npz', allow_pickle=False) as a, np.load(new/'commands.npz', allow_pickle=False) as b:
        require(not b['hip_pitch_pulse_side'][:controls].any(), 'An intervention appears before the frozen start')
        initial_commands = {name: np.array_equal(a[name][0], b[name][0])
                            for name in ('observed', 'actor_input', 'raw_policy_actions', 'actual_motor_target')}
    with np.load(old/'trace.npz', allow_pickle=False) as a, np.load(new/'trace.npz', allow_pickle=False) as b:
        before = slice(None, controls+1)
        first, second = a['qpos'][before], b['qpos'][before]
        np.testing.assert_array_equal(first[0], second[0])
        difference = np.abs(first-second)
        np.testing.assert_allclose(difference.max(), case['maximum_pre_pulse_qpos_difference'], rtol=0, atol=1e-14)
        divergent = np.flatnonzero(np.any(difference != 0, axis=(1, 2)))
        translation = np.linalg.norm(first[..., :3]-second[..., :3], axis=-1)
        qa, qb = (row['quaternion'][before].astype(np.float64) for row in (a, b))
        dot = np.abs(np.sum(qa*qb, axis=-1)/(np.linalg.norm(qa, axis=-1)*np.linalg.norm(qb, axis=-1)))
        angle = np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.)))
        joint = np.abs(a['joint_positions'][before]-b['joint_positions'][before])
        contact = (a['foot_contact'][before] == b['foot_contact'][before]).all(axis=-1)
    return dict(first_control_input_and_target_equal=initial_commands,
                first_nonidentical_qpos_after_control=int(divergent[0]) if len(divergent) else None,
                body_translation_max_m=float(np.max(translation)), body_rotation_max_degrees=float(np.max(angle)),
                joint_position_max_rad=float(np.max(joint)), paired_contact_equal_fraction=float(contact.mean()),
                measured_through_control=controls)


def main():
    state = read(P)['goal_execution'][KEY]
    route, master, out = ROOT/state['result'], ROOT/state['master'], REPORT/'pre-pulse-pairing.json'
    require(state['status'] == 'reviewed' and not out.exists() and sha(route) == state['result_sha256']
            and sha(master) == state['master_sha256'], 'The paired route evidence is not ready')
    sources = {rel(path): sha(path) for path in (route, master, Path(__file__).resolve())}
    protocols = read(master)['protocols']
    cases = {}
    for name, case in read(route)['cases'].items():
        protocol = master.parent/f'{name}.json'
        require(sha(protocol) == protocols[name], 'A case description changed')
        sources[rel(protocol)] = protocols[name]
        cases[name] = pair_prefix(case, read(protocol)['phase_hip_pitch']['first_zero_based_control'], sources)
    put(out, dict(cases=cases, source_sha256=sources, physical_steps=0, learning_updates=0, hardware_io=False, heldout_used=False))
    print('HIP_PITCH_PRE_PULSE_PAIRING', cases, flush=True)


if __name__ == '__main__':
    main()
