"""Check the previously selected encoder lead in nominal and low-friction walks."""

from pathlib import Path

import numpy as np

import run_roll_stand_forward_diagnostic as direct
from review_common_walk_phase import correlation


ROOT = direct.ROOT
SOURCE = ROOT / 'records/reports/common-walk-confirmation/development-phase-review.json'
OUTPUT = ROOT / 'records/reports/forward-output-layer-pair/encoder-phase-readback.json'


def check(case, mark, prior, source_hashes):
    receipt = ROOT / mark['path']
    proof = direct.read(receipt)
    folder = ROOT / 'records/evaluations' / proof['run_id']
    manifest = direct.read(folder / 'guarded-evaluation.json')
    plan_path, trace, commands = (folder / key for key in ('protocol.json', 'trace.npz', 'commands.npz'))
    paths = {receipt: mark['sha256'], plan_path: proof['protocol_sha256'], trace: manifest['trace_sha256'],
             commands: manifest['command_sha256']}
    direct.require(proof['full_route'] and not proof['holdout_used'] and not any(proof['safety'].values())
                   and all(direct.sha(path) == sha for path, sha in paths.items()), 'The official trajectory changed')
    source_hashes.update({direct.rel(path): sha for path, sha in paths.items()})
    plan = direct.read(plan_path)
    left, right = [20 + plan['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
    start = plan['initial_stand_controls']
    lead = prior['lead_controls']
    # The command input precedes the first physics step; trace index 0 is the initial state.
    time = np.arange(start, start+plan['forward_controls']+1-lead)
    with np.load(trace, allow_pickle=False) as states, np.load(commands, allow_pickle=False) as model:
        direct.require(states['active'][time+lead].all(), 'Incomplete phase correlation window')
        inputs = model['actor_input'][time]
        future = states['angular_velocity'][time+lead, :, 2]
    encoder = correlation(inputs[..., left]+inputs[..., right], future) * prior['polarity']
    gyro = correlation(inputs[..., 2], future)
    return dict(case=case, samples=int(future.size), worlds=len(encoder),
                median_encoder_correlation=float(np.median(encoder)), same_sign_worlds=int((encoder > 0).sum()),
                median_gyro_correlation=float(np.median(gyro)), encoder_world_correlations=encoder.tolist())


def main():
    reference = direct.read(direct.P)['goal_execution']['roll_friction_stand_forward']
    prior = direct.read(SOURCE)
    direct.require(reference['status'] == 'reviewed' and len(reference['scored']) == 4
                   and prior['encoder_channel'] == 'sum_hip_pitch' and prior['association_replicated'],
                   'Expected the previously replicated fixed encoder feature')
    sources = {direct.rel(path): direct.sha(path) for path in (SOURCE, Path(__file__).resolve())}
    rows = {case: check(case, mark, prior, sources) for case, mark in reference['scored'].items()}
    direct.put(OUTPUT, dict(feature=prior['encoder_channel'], lead_controls=prior['lead_controls'],
                           lead_seconds=prior['lead_seconds'], polarity=prior['polarity'], studies=rows,
                           sources=sources, physics_steps=0, learning_updates=0, heldout_used=False,
                           scope='Archived correlation of encoder observations and subsequent yaw rate'))
    print('OFFICIAL_FORWARD_ENCODER_PHASE', {case: round(row['median_encoder_correlation'], 4)
          for case, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
