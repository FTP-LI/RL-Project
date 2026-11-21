"""Verify the per-world forward input timing on archived rolling routes."""

import numpy as np

from command_input_mapping import transform
from forward_input_timing import post_entry_forward_input
from plot_forward_support_continuation import ROOT, read, sha, write


PLANS = ROOT/'configs/experiments/roll_friction_paired_development'
REPORT = ROOT/'records/reports/forward-input-full-route'


def measure(case, master, design, candidate):
    path = PLANS/f'{case}.json'
    plan = read(path)
    assert sha(path) == master['arms'][case]['plan_sha256'] and plan['run_id'] == master['arms'][case]['run_id']
    folder = ROOT/'records/evaluations'/plan['run_id']
    manifest_path = folder/'guarded-evaluation.json'
    raw = folder/'raw-score.json'
    raw_data = read(raw)
    command_path = folder/'commands.npz'
    manifest = read(manifest_path)
    assert sha(command_path) == manifest['command_sha256'] and manifest['actual_control_steps'] == 1550
    with np.load(command_path, allow_pickle=False) as stored:
        external, phase, inputs = (stored[key] for key in ('external_command', 'phase_used', 'actor_input'))
    forward = phase == 3
    enters = forward.any(axis=0)
    entry = np.where(enters, forward.argmax(axis=0), -1)
    np.testing.assert_array_equal(entry, raw_data['forward_at'])
    command, target = design['external_forward_command_m_s'], design['internal_forward_input_m_s'][candidate]
    np.testing.assert_array_equal(external[forward], np.broadcast_to(np.float32([command, 0, 0]), (int(forward.sum()), 3)))
    control = np.arange(len(phase), dtype=np.float32)[:, None]
    wait = design['shared_controls']-design['standing_controls']
    ramp = design['input_ramp_controls']
    original = transform(external)
    adjusted = post_entry_forward_input(external, control, entry[None, :], wait, ramp, command, target)
    np.testing.assert_array_equal(inputs[..., 48:51][forward], original[forward])
    np.testing.assert_array_equal(adjusted[..., 1:], original[..., 1:])
    np.testing.assert_array_equal(adjusted[~forward], original[~forward])
    changed = adjusted[..., 0] != original[..., 0]
    first = changed.argmax(axis=0)
    np.testing.assert_array_equal(first[enters], entry[enters]+wait+1)
    assert not changed[:, ~enters].any()
    safe = plan['initial_stand_controls']+plan['landing_stand_controls']+wait
    assert not changed[:safe+1].any()
    return dict(entering_worlds=int(enters.sum()), first_forward_control_range=[int(entry[enters].min()), int(entry[enters].max())],
                first_altered_control_range=[int(first[enters].min()), int(first[enters].max())],
                common_fork_control_safe_from_frozen_phase_rules=safe, changed_forward_controls=int(changed.sum()),
                nonforward_and_other_command_slots_exact=True,
                source_sha256={item.relative_to(ROOT).as_posix():sha(item) for item in (path, manifest_path, command_path, raw)})


def direct_replay(name, mark, design, candidate):
    score_path = ROOT/mark['path']
    assert sha(score_path) == mark['sha256']
    score = read(score_path)
    folder = ROOT/'records/evaluations'/score['run_id']
    manifest_path = folder/'guarded-evaluation.json'
    assert sha(manifest_path) == score['collection_sha256']
    manifest = read(manifest_path)
    locations = (folder/'shared/commands.npz', folder/'clips'/candidate/'commands.npz')
    digests = (manifest['prefix_sha256']['commands.npz'], manifest['branches'][candidate]['trace_sha256']['commands.npz'])
    entries = []
    for path, digest in zip(locations, digests, strict=True):
        assert sha(path) == digest
        with np.load(path, allow_pickle=False) as data:
            entries.append((data['external_command'], data['actor_input'][..., 48:51]))
    requested, applied = (np.concatenate(pair, axis=0) for pair in zip(*entries, strict=True))
    start = design['standing_controls']
    steps = np.arange(len(requested), dtype=np.float32)[:, None]
    expected = post_entry_forward_input(requested, steps, start, design['shared_controls']-start,
        design['input_ramp_controls'], design['external_forward_command_m_s'], design['internal_forward_input_m_s'][candidate])
    np.testing.assert_array_equal(expected, applied)
    return dict(controls=len(expected), worlds=expected.shape[1], all_command_slots_exact=True,
                independent_score_sha256=mark['sha256'], commands_sha256=dict(zip(
                    (path.relative_to(ROOT).as_posix() for path in locations), digests, strict=True)))


def main():
    project = read(ROOT/'project.json')['goal_execution']
    old = project['roll_friction_development']
    pilot = project['forward_stride_yaw_response']
    result, draft = ROOT/pilot['result'], ROOT/pilot['draft']
    assert old['status'] == 'development-reviewed' and sha(PLANS/'master.json') == old['master_sha256']
    assert pilot['status'] == 'reviewed' and sha(result) == pilot['result_sha256'] and sha(draft) == pilot['draft_sha256']
    master, design, candidate = read(PLANS/'master.json'), read(draft), read(result)['selected_for_full_20_seconds']
    cases = {key:measure(key, master, design, candidate) for key in master['order'] if key.endswith('-low-friction')}
    assert len(cases) == 4 and {case['common_fork_control_safe_from_frozen_phase_rules'] for case in cases.values()} == {300}
    full = project['forward_input_full_route']
    assert full['selected'] == candidate and full['pilot_sha256'] == sha(result)
    replays = {key:direct_replay(key, mark, design, candidate) for key, mark in full['scored'].items()}
    assert {'s1247-friction-low','s1248-friction-low'} <= replays.keys()
    spec = read(PLANS/f'{next(iter(cases))}.json')['walking']['commands']
    names = list(spec)
    commands = np.asarray(list(spec.values()), dtype=np.float32)
    original = transform(commands)
    changed = post_entry_forward_input(commands, 300, 0, 200, design['input_ramp_controls'],
                                      design['external_forward_command_m_s'], design['internal_forward_input_m_s'][candidate])
    changed_names = [name for name, before, after in zip(names, original, changed, strict=True) if not np.array_equal(before, after)]
    assert len(names) == 7 and changed_names == ['forward']
    write(REPORT/'archived-roll-entry-timing.json', dict(cases=cases, selected=candidate, source_pilot_sha256=sha(result),
           roll_master_sha256=sha(PLANS/'master.json'), verified_direct_command_replays=replays,
           isolated_named_commands=names, changed_named_commands=changed_names,
           mapping_source_sha256=sha(ROOT/'scripts/validation/forward_input_timing.py'),
           archived_only=True, physical_steps=0, learning_updates=0,
           heldout_used=False, hardware_io=False))
    for name, case in cases.items():
        print('ARCHIVED_ROLL_FORWARD_TIMING', name, case['entering_worlds'], case['first_altered_control_range'], flush=True)
    print('POST_ENTRY_DIRECT_INPUT_REPLAY', sorted(replays), flush=True)


if __name__ == '__main__':
    main()
