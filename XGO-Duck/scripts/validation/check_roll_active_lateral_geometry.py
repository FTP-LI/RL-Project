"""Check the roll-time intervention geometry from an existing nominal run."""

from pathlib import Path

import numpy as np

from check_continuous_lateral_push_geometry import ROOT, lateral_world, load, sha


DRAFT = ROOT / 'configs/experiments/continuous_roll_early_lateral_push_draft.json'
OUT = ROOT / 'records/reports/continuous-roll-early-lateral-push'


def pinned(pair):
    path, expected = pair
    item = ROOT / path
    assert sha(item) == expected
    return item


def main():
    import json

    d = load(DRAFT)
    assert d['status'] == 'draft-no-physical-execution'
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used',
                                      'hardware_io', 'automatic_retry', 'automatic_extension'))
    nominal, previous = [pinned(d[key]) for key in ('nominal_result', 'previous_lateral_result')]
    a, b = load(nominal), load(previous)
    assert a['passed_development_gate'] and not a['heldout_used']
    assert not b['heldout_used'] and all(row['passed_gate'] for row in b['branches'].values())
    run = ROOT / a['record']
    path = run / 'protocol.json'
    assert sha(path) == a['protocol_sha256']
    p = load(path)
    n = d['worlds']
    reference = p['initial_stand_controls']
    inject = d['injection_control']
    end = p['maximum_controls']
    assert n == p['worlds'] and reference == d['standing_heading_control'] < inject < reference + p['roll_controls']
    assert end == d['maximum_controls_each_trajectory']
    assert p['minimum_continuous_successes'] == d['minimum_complete_original_sequence'] == p['minimum_roll_completion']
    unique = inject + len(d['groups_horizontal_velocity_m_s']) * (end - inject)
    assert unique == d['maximum_unique_controls'] and unique*n == d['maximum_transitions']
    assert unique*5 == d['maximum_batched_physics_substeps']
    data_path = run / 'trace.npz'
    assert sha(data_path) == a['source_sha256'][data_path.relative_to(ROOT).as_posix()]
    with np.load(data_path, allow_pickle=False) as data:
        active = data['active'][inject]
        rolling = data['phase_next'][inject] == 1  # ROLL on the next control
        before_head = ~data['head_contact'][reference:inject+1].any(axis=0)
        eligible = active & rolling & before_head
        heading_quat = data['qpos'][reference, eligible, 3:7]
        np.testing.assert_allclose(data['quaternion'][reference, eligible], heading_quat, rtol=0., atol=2e-7)
        original = data['qvel'][inject, eligible]
    assert int(eligible.sum()) >= d['minimum_eligible_active_roll']
    assert d['groups_horizontal_velocity_m_s'] == {'reference': 0., 'left-push': .08, 'right-push': -.08}
    sides = {}
    for role, magnitude in d['groups_horizontal_velocity_m_s'].items():
        yaw, delta = lateral_world(heading_quat, magnitude)
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        np.testing.assert_allclose((forward * delta[:, :2]).sum(axis=1), 0., rtol=0., atol=1e-12)
        after = original.copy()
        after[:, :2] += delta[:, :2].astype(after.dtype)
        np.testing.assert_array_equal(after[:, 2:], original[:, 2:])
        achieved = np.linalg.norm((after - original)[:, :2], axis=1)
        np.testing.assert_allclose(achieved, abs(magnitude), rtol=0., atol=1e-7)
        sides[role] = dict(requested_lateral_m_s=magnitude, maximum_velocity_error_m_s=float(abs(achieved-abs(magnitude)).max()))
    result = dict(status='passed-saved-roll-disturbance-geometry', previous_nominal_continuous=a['counts']['continuous_successes'],
        rollout_worlds=n, worlds_in_active_roll=int(rolling.sum()), worlds_already_at_head_support=int((~before_head).sum()),
        eligible=int(eligible.sum()), standing_heading_control=reference, injection_control=inject, roles=sides,
        original_gate=d['minimum_complete_original_sequence'], physical_substeps=0, heldout_used=False, hardware_io=False,
        sources={file.relative_to(ROOT).as_posix():sha(file) for file in (DRAFT, nominal, previous, path, data_path, Path(__file__).resolve())})
    assert not OUT.exists()
    OUT.mkdir()
    (OUT / 'geometry-preflight.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('ROLL_DISTURBANCE_GEOMETRY', result['eligible'], inject, sides, flush=True)


if __name__ == '__main__':
    main()
