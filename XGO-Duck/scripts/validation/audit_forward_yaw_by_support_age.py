"""Locate the low-friction yaw gap within each actual foot-contact phase."""

from pathlib import Path

import numpy as np

from forward_support_joint_pulse import REPORT
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


def contact_age(contacts):
    codes = contacts[..., 0].astype(np.int8) + 2*contacts[..., 1].astype(np.int8)
    age = np.ones_like(codes, dtype=np.int32)
    for control in range(1, len(codes)):
        age[control] = np.where(codes[control] == codes[control-1], age[control-1]+1, 1)
    return codes, age


def summarize(marker, sources):
    independent = ROOT/marker['path']
    score = read(independent)
    run = ROOT/'records/evaluations'/score['run_id']
    manifest_path, trace, raw = (run/name for name in ('guarded-evaluation.json', 'trace.npz', 'raw-score.json'))
    manifest = read(manifest_path)
    expected = {independent: marker['sha256'], trace: manifest['trace_sha256'], raw: score['raw_score_sha256']}
    require(score['full_route'] and not any(score['safety'].values()) and not score['holdout_used']
            and all(sha(path) == digest for path, digest in expected.items()), 'An official baseline archive changed')
    sources.update({rel(path): digest for path, digest in expected.items()})
    sources[rel(manifest_path)] = sha(manifest_path)
    data = read(raw)
    require(len(data['forward_at']) == 128 and set(data['forward_at']) == {50}, 'The original forward interval differs')
    forward = slice(51, 1051)
    with np.load(trace, allow_pickle=False) as saved:
        yaw = np.abs(saved['angular_velocity'][forward, :, 2].astype(np.float64))
        contact, age = contact_age(saved['foot_contact'])
        np.testing.assert_array_equal(saved['active'][forward], True)
    np.testing.assert_allclose(yaw.mean(axis=0), [row['yaw_error_rad_s'] for row in data['forward']], rtol=0, atol=1e-6)
    contact, age = contact[forward], age[forward]
    groups = {'double_support': contact == 3, 'no_support': contact == 0}
    for side, code in (('left', 1), ('right', 2)):
        groups[f'{side}_controls_1_to_3'] = (contact == code) & (age <= 3)
        groups[f'{side}_controls_4_to_8'] = (contact == code) & (age >= 4) & (age <= 8)
        groups[f'{side}_controls_9_and_later'] = (contact == code) & (age >= 9)
    np.testing.assert_array_equal(np.sum(list(groups.values()), axis=0), 1)
    rows = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        rows[name] = dict(samples=count, sample_fraction=count/yaw.size,
                          mean_abs_yaw_rad_s=float(yaw[selected].mean()) if count else None,
                          contribution_rad_s=float(yaw[selected].sum()/yaw.size))
    return dict(run_id=run.name, samples=yaw.size, mean_abs_yaw_rad_s=float(yaw.mean()), groups=rows)


def main():
    original = read(P)['goal_execution']['roll_friction_stand_forward']
    expected = {f's{seed}-{condition}' for seed in (1247, 1248) for condition in ('nominal', 'friction-low')}
    require(original['status'] == 'reviewed' and set(original['scored']) == expected, 'Four verified development archives are required')
    source, output = Path(__file__).resolve(), REPORT/'archived-yaw-by-support-age.json'
    require(not output.exists(), 'Preserve the existing support-age analysis')
    sources = {rel(source): sha(source)}
    cases = {name: summarize(marker, sources) for name, marker in original['scored'].items()}
    pairs = {}
    for seed in (1247, 1248):
        low, nominal = (cases[f's{seed}-{label}'] for label in ('friction-low', 'nominal'))
        gap = low['mean_abs_yaw_rad_s']-nominal['mean_abs_yaw_rad_s']
        parts = {part: row['contribution_rad_s']-nominal['groups'][part]['contribution_rad_s']
                 for part, row in low['groups'].items()}
        np.testing.assert_allclose(sum(parts.values()), gap, atol=1e-12, rtol=0)
        pairs[str(seed)] = dict(pooled_mean_abs_yaw_gap_rad_s=gap, contribution_by_contact_and_age_rad_s=parts)
    put(output, dict(cases=cases, pairs=pairs, source_sha256=sources, samples=sum(row['samples'] for row in cases.values()),
                     physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print('OFFICIAL_FORWARD_SUPPORT_AGE', pairs, flush=True)


if __name__ == '__main__':
    main()
