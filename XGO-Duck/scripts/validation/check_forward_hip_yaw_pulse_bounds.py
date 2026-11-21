"""Measure paired hip-yaw command headroom on archived official forward gait."""

from pathlib import Path

import numpy as np

import run_roll_stand_forward_diagnostic as direct
from roll_stand_walk_sequence import FORWARD


ROOT = direct.ROOT
DRAFT = ROOT / 'configs/experiments/forward_hip_yaw_pulse_draft.json'
OUTPUT = ROOT / 'records/reports/forward-output-layer-pair/hip-yaw-pulse-bounds.json'


def measure(reference, draft, sources):
    receipt = ROOT / reference['path']
    score = direct.read(receipt)
    run = ROOT / 'records/evaluations' / score['run_id']
    plan_path, command_path, trace_path = (run / name for name in ('protocol.json', 'commands.npz', 'trace.npz'))
    manifest = direct.read(run/'guarded-evaluation.json')
    files = {receipt: reference['sha256'], plan_path: score['protocol_sha256'],
             command_path: manifest['command_sha256'], trace_path: manifest['trace_sha256']}
    direct.require(score['full_route'] and not score['holdout_used'] and not any(score['safety'].values())
                   and all(direct.sha(path) == sha for path, sha in files.items()), 'Official control source changed')
    sources.update({direct.rel(path): digest for path, digest in files.items()})
    plan = direct.read(plan_path)
    start, duration = plan['initial_stand_controls'], draft['forward_controls_to_inspect']
    window = slice(start, start + duration)
    walking = plan['walking']
    joints = [walking['joint_names'].index(name) for name in draft['pulse_joints']]
    low, high = np.asarray(walking['joint_limits'], dtype=np.float32).T
    home = np.asarray(walking['model_home'], dtype=np.float32)
    cap = plan['feedback']['maximum_joint_correction_rad']
    direct.require(cap == draft['original_joint_correction_cap_rad'], 'Original correction cap changed')
    with np.load(command_path, allow_pickle=False) as commands, np.load(trace_path, allow_pickle=False) as trace:
        np.testing.assert_array_equal(commands['phase_used'][window], FORWARD)
        np.testing.assert_array_equal(trace['active'][window], True)
        baseline = (commands['raw_policy_actions'][window]+home).clip(low, high)
        target = commands['applied_targets'][window]
        np.testing.assert_allclose(commands['actual_motor_target'][window], target, rtol=0, atol=1e-6)
        contact = trace['foot_contact'][window]
    direct.require(np.abs(target-baseline).max() <= cap + 1e-6, 'Recorded controller already exceeds its correction cap')
    lower = np.maximum(low[joints], baseline[..., joints] - cap)
    upper = np.minimum(high[joints], baseline[..., joints] + cap)
    delta = draft['paired_target_delta_rad']
    selected = target[..., joints]
    both_signs = (selected - delta >= lower) & (selected + delta <= upper)
    single_contact = contact & ~contact[..., ::-1]
    eligible = both_signs & single_contact
    worlds = eligible.any(axis=0)
    return dict(run_id=run.name, seed=plan['seed'], friction=plan['parameter_factors']['foot_friction'],
                total_worlds=worlds.shape[0], both_contact_phases_have_a_pulse_worlds=int(worlds.all(axis=1).sum()),
                eligible_worlds_by_joint=dict(zip(draft['pulse_joints'], worlds.sum(axis=0).tolist())),
                eligible_samples_by_joint=dict(zip(draft['pulse_joints'], eligible.sum(axis=(0, 1)).tolist())))


def main():
    draft = direct.read(DRAFT)
    reference = direct.read(direct.P)['goal_execution'][draft['official_source']]
    direct.require(draft['physical_budget'] == draft['learning_updates'] == 0 and reference['status'] == 'reviewed',
                   'The preflight must use archived official states')
    sources = {direct.rel(path): direct.sha(path) for path in (DRAFT, Path(__file__).resolve())}
    rows = {case: measure(entry, draft, sources) for case, entry in reference['scored'].items()}
    direct.require({row['seed'] for row in rows.values()} == set(draft['seeds']) and
                   {row['friction'] for row in rows.values()} == set(draft['foot_sliding_friction']), 'Unexpected cohorts')
    direct.put(OUTPUT, dict(cohorts=rows, sources=sources, pulse_rad=draft['paired_target_delta_rad'],
                           forward_controls=draft['forward_controls_to_inspect'], physics_steps=0,
                           learning_updates=0, hardware_io=False, heldout_used=False))
    print('FORWARD_HIP_YAW_PULSE_HEADROOM', {case: row['both_contact_phases_have_a_pulse_worlds']
          for case, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
