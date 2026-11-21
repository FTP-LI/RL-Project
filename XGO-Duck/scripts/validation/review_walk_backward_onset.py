"""Measure backward gait onset on verified development-only traces."""
from pathlib import Path

import numpy as np

from common_walk_actuator_yaw_feedback import OUT, PLAN, ROLES
from repeat_walk_actuator_yaw_feedback import OUT as REPEAT_OUT, PLAN as REPEAT_PLAN
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def inspect(plan, directory):
    result_path = directory/'analysis.json'
    result, spec = read(result_path), read(plan)
    assert result['status'] == 'passed-integrity' and result['protocol_sha256'] == sha(plan)
    assert not result['heldout_used'] and not spec['heldout_used']
    record = ROOT/result['record']
    p = spec['environment']
    ids = p['groups']['backward']
    gate = p['criteria']['signed_speed']['backward']
    prefix_path = record/'trace-shared-prefix.npz'
    assert sha(prefix_path) == result['shared_prefix_sha256']
    keys = ('observations', 'velocity', 'active', 'contact')
    with np.load(prefix_path) as source:
        prefix = {key: source[key][:, ids] for key in keys}
    branches = {}
    for role in ROLES[:2]:
        path = record/f'trace-{role}.npz'
        assert sha(path) == result['trace_sha256'][role]
        with np.load(path) as source:
            trace = {key: np.concatenate((prefix[key], source[key][:, ids])) for key in keys}
        assert trace['active'].shape == (1000, 32) and trace['active'].all()
        verified = result['scores'][role]['backward']['episode_scores']
        assert [row['world'] for row in verified] == ids
        reverse = -trace['velocity'][..., 0].astype(float)
        np.testing.assert_allclose(reverse.mean(axis=0), [row['signed_speed'] for row in verified], atol=1e-7, rtol=0)
        blocks = reverse.reshape(20, 50, 32).mean(axis=1)
        qualifies = blocks >= gate
        onset = np.where(qualifies.any(axis=0), qualifies.argmax(axis=0)+1, 0)
        walking = np.logical_xor(trace['contact'][..., 0], trace['contact'][..., 1])
        single_support = walking.reshape(20, 50, 32).mean(axis=1)
        commands = np.unique(trace['observations'][..., 48:51].reshape(-1, 3), axis=0)
        assert commands.shape == (1, 3)
        episodes = [dict(world=world, onset_block_end_seconds=int(onset[i]),
                         first_four_seconds_signed_speed=blocks[:4, i].tolist(),
                         first_four_seconds_single_support=single_support[:4, i].tolist(),
                         score_success=verified[i]['success'], original_mean_speed_m_s=verified[i]['signed_speed'])
                    for i, world in enumerate(ids)]
        branches[role] = dict(mapped_command=commands[0].tolist(),
                              no_qualifying_second=int((onset == 0).sum()),
                              qualifying_by_two_seconds=int(((onset > 0) & (onset <= 2)).sum()),
                              qualifying_after_two_seconds=int((onset > 2).sum()),
                              earliest_block_end_seconds=int(onset[onset > 0].min()) if onset.any() else None,
                              latest_block_end_seconds=int(onset.max()), episodes=episodes)
    return dict(seed=p['seed'], source_sha256=sha(result_path), protocol_sha256=sha(plan),
                external_command=p['commands']['backward'], signed_speed_threshold_m_s=gate, branches=branches)


def main():
    cases = [inspect(PLAN, OUT), inspect(REPEAT_PLAN, REPEAT_OUT)]
    assert [row['seed'] for row in cases] == [1047, 1048]
    result = dict(status='passed-saved-development-review', cohorts=cases, script_sha256=sha(Path(__file__)),
                  learning_updates=0, physical_substeps=0, heldout_used=False,
                  scope='First non-overlapping second meeting the original signed speed threshold. This is a diagnostic and leaves the original 20-second score unchanged.')
    target = ROOT/'records/reports/common-walk-actuator-yaw-feedback/backward-development-onset.json'
    write(target, result)
    print('WALK_BACKWARD_ONSET', {case['seed']: {role: {key: value for key, value in row.items() if key != 'episodes'}
                            for role, row in case['branches'].items()} for case in cases}, flush=True)


if __name__ == '__main__':
    main()
