"""Independently read back the intervention and score three original roll sequences."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import numpy as np


PLAN = Path(__file__).resolve().parents[2] / 'configs/experiments/continuous_roll_early_lateral_push.json'


def observe():
    import verify_routed_forward_entry_ramp as windows

    with patch.object(windows, 'PLAN', PLAN):
        windows.observe(windows.read(PLAN))


def verify_velocity(folder, prefix, worlds, shared, heading_control, magnitude, arrays, read):
    a, m = arrays(folder/'injection.npz'), read(folder/'injection.json')
    assert m['control'] == shared and m['heading_control'] == heading_control and m['magnitude_m_s'] == magnitude
    assert m['changed_fields'] == (['warp.qvel'] if magnitude else [])
    np.testing.assert_array_equal(a['before'], prefix['qvel'][-1])
    np.testing.assert_array_equal(a['world_ids'], worlds)
    w, x, y, z = prefix['qpos'][heading_control, worlds, 3:7].astype(np.float64).T
    heading = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    lateral = np.stack((-np.sin(heading), np.cos(heading)), axis=1)
    expected = a['before'].copy()
    target = (lateral*magnitude).astype(expected.dtype)
    np.testing.assert_allclose(a['target_world_xy'], target, rtol=0, atol=1e-8)
    expected[worlds,:2] += target
    np.testing.assert_array_equal(a['after'], expected)
    np.testing.assert_array_equal(a['first_substep_qvel'], expected)
    np.testing.assert_array_equal(a['measured_world_xy'], (expected-a['before'])[worlds,:2])
    np.testing.assert_allclose(np.linalg.norm(a['measured_world_xy'], axis=1), abs(magnitude), rtol=0, atol=1e-7)
    assert a['substep_motor'].shape[:2] == (5, len(a['before']))
    return a['substep_motor'][0], lateral


def verify():
    import torch

    import run_roll_active_lateral as study
    from check_continuous_forward_fork_state import controller_snapshot
    from getup_release_fork_state import digest, signature
    from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence, score_forward
    from verify_continuous_forward_yaw_pair import joined, validate_media
    from verify_continuous_lateral_push import check_endpoint
    from verify_continuous_roll_projected_entry import count, roll_score
    from verify_continuous_roll_stand_walk import arrays, check_policy

    read, sha, write = study.read, study.sha, study.write
    p = study.checked()
    target = study.OUT/'analysis.json'
    assert not target.exists()
    fork_plan = p['perturbation']
    folder = study.ROOT / p['record']
    runtime = read(folder/'runtime-contract.json')
    observer = read(folder/'windows-view-verification.json')
    observed = observer['status'] == 'passed' and observer['running_snapshots_with_original_frame'] >= 3
    assert not runtime['capacity_reached'] and not runtime['extra_observation_or_render_steps']
    assert runtime['interface_checks'] == 1 and sha(folder/'protocol.json') == sha(PLAN)
    horizon, shared, heading = p['maximum_controls'], fork_plan['shared'], fork_plan['heading_control']
    prefix_file, prefix_cmd_file = (folder/name for name in ('shared-trace.npz', 'shared-commands.npz'))
    prefix, prefix_commands = arrays(prefix_file), arrays(prefix_cmd_file)
    startup = arrays(folder/'startup.npz')
    paths = [PLAN, folder/'protocol.json', prefix_file, prefix_cmd_file, folder/'startup.npz',
        folder/'runtime-contract.json', folder/'windows-view-verification.json']

    def eligible(seq):
        return np.flatnonzero((seq.phase == ROLL) & prefix['active'][-1] & ~prefix['head_contact'][heading:].any(axis=0))

    def evidence(**fields):
        assert not torch.cuda.is_initialized()
        write(target, dict(protocol_sha256=sha(PLAN), record=p['record'], heldout_used=False, learning_updates=0,
            physical_resets=1, observer_verified=observed, source_sha256={study.relative(path):sha(path) for path in paths},
            **fields))

    if (folder/'shared-stop.json').is_file():
        early = read(folder/'shared-stop.json')
        replay, stop = count(p, prefix, prefix_commands)
        n = len(prefix_commands['phase_used'])
        if stop is None:
            assert n == shared
            coverage = len(eligible(replay))
            assert coverage < fork_plan['minimum_eligible']
            stop = dict(reason='pre-head-support-coverage', control=shared, eligible=coverage, required=fork_plan['minimum_eligible'])
        assert stop == early['stop'] and stop['control'] == n == runtime['control_steps'] <= shared
        assert runtime['control_transitions'] == n*p['worlds'] and runtime['physics_substeps'] == n*5
        check_endpoint(folder, prefix)
        check_policy(prefix_commands, startup, p, prefix_commands['phase_used'])
        clips = {role:validate_media(folder, clip, prefix, n, horizon) for role,clip in early['clips'].items()}
        paths += [folder/'shared-stop.json', folder/'endpoint-contact.npz']
        evidence(status='verified-shared-stop', stop=stop, clips=clips, passed_development_gate=False,
            controls=n, transitions=n*p['worlds'], physics_substeps=n*5)
        print('ROLL_ACTIVE_LATERAL_SHARED_STOP', stop, flush=True)
        return

    run = read(folder/'evaluation.json')
    assert run['status'] == 'collected-review-required' and run['protocol_sha256'] == sha(PLAN)
    assert not run['learning_updates'] and not run['heldout_used'] and run['physical_resets'] == 1
    assert run['shared'] == shared and len(prefix['active']) == shared+1 and len(prefix_commands['phase_used']) == shared
    seq = Sequence(p)
    for k in range(shared+1):
        seq.update(k, {key:prefix[key][k] for key in STATE_FIELDS})
        np.testing.assert_array_equal(seq.phase, prefix['phase_next'][k])
    fork = read(folder/'fork.json')
    frozen = arrays(folder/'fork-controller.npz')
    assert fork['physics']['counters'] == [shared, shared*5] and fork['heading_control'] == heading
    assert fork['physics']['fields'] == len(fork['field_names']) == len(set(fork['field_names']))
    assert {'warp.qvel', 'rng', 'controller.previous', 'controller.history', 'controller.previous_features'} <= set(fork['field_names'])
    assert fork['physics']['warp_bytes'] <= p['maximum_warp_snapshot_bytes']
    worlds = eligible(seq)
    assert fork['eligible_worlds'] == worlds.tolist() and len(worlds) >= fork_plan['minimum_eligible']
    assert digest(signature(controller_snapshot(seq))) == digest(signature(frozen)) == fork['controller_sha256']
    paths += [folder/'fork.json', folder/'fork-controller.npz', folder/'evaluation.json']
    first, results, lateral_windows = {}, {}, {}
    for role, magnitude in fork_plan['roles'].items():
        branch = folder/'clips'/role
        raw = run['branches'][role]
        last = raw['endpoint']
        assert last == shared+raw['controls'] and shared < last <= horizon
        assert read(branch/'restored-fork.json') == {key:fork[key] for key in ('physics','controller_sha256')}
        suffix, after = arrays(branch/'trace.npz'), arrays(branch/'commands.npz')
        assert len(suffix['active']) == raw['controls']+1 and len(after['phase_used']) == raw['controls']
        for key, value in prefix.items():
            np.testing.assert_array_equal(value[-1], suffix[key][0], err_msg=key)
        motor, axis = verify_velocity(branch, prefix, worlds, shared, heading, magnitude, arrays, read)
        first[role] = tuple(after[key][0] for key in ('observed','actor_input','raw_policy_actions','feedback_yaw','applied_targets')) + (motor,)
        trace, commands = joined(prefix, suffix, 1), joined(prefix_commands, after, 0)
        np.testing.assert_array_equal(trace['active'][1:], np.logical_and.accumulate(~(commands['terminated']|commands['truncated']), axis=0))
        np.testing.assert_array_equal(commands['phase_used'], trace['phase_next'][:-1])
        check_endpoint(branch, trace)
        replay, stop = count(p, trace, commands)
        assert stop == raw['stop'] and (last == horizon or stop is not None and stop['control'] == last)
        policy = check_policy(commands, startup, p, commands['phase_used'])
        roll = roll_score(p, trace, replay)
        roll_ok = np.array([row['success'] for row in roll['episodes']])
        began = replay.forward_at >= 0
        full_window = began & (replay.forward_at+p['forward_controls'] <= last)
        scores = score_forward(trace, np.where(full_window, replay.forward_at, -1), p)
        full = roll_ok & np.array([row['success'] for row in scores])
        first_failed = np.select((~roll_ok, ~began, ~full_window, ~full),
            ('roll', 'landing-stand', 'forward-window', 'forward-score'), default='passed')
        passed = bool(last == horizon and roll_ok.sum() >= p['minimum_roll_completion'] and full.sum() >= p['minimum_continuous_successes'])
        lateral_windows[role] = (trace['qvel'][shared+1:min(shared+11,last+1),worlds,:2]*axis).sum(axis=2)
        results[role] = dict(controls=raw['controls'], endpoint=last, stop=stop, perturbation_m_s=magnitude,
            original_successes=int(full.sum()), passed_gate=passed, roll_completed=int(roll_ok.sum()),
            forward_completed=int(full_window.sum()), original_forward_outcomes=scores, success_by_world=full.tolist(),
            first_failed_stage=first_failed.tolist(), roll_outcomes=roll['episodes'], policy=policy,
            video=validate_media(folder, raw['video'], trace, last, horizon))
        paths.extend(branch/name for name in ('trace.npz','commands.npz','endpoint-contact.npz','restored-fork.json','injection.npz','injection.json'))
    reference = np.asarray(results['reference']['success_by_world'])
    paired = {}
    for role in ('left-push', 'right-push'):
        for value, expected in zip(first[role], first['reference'], strict=True):
            np.testing.assert_array_equal(value, expected)
        case = np.asarray(results[role]['success_by_world'])
        lost = np.flatnonzero(reference & ~case)
        n = min(len(lateral_windows[role]), len(lateral_windows['reference']))
        delta = lateral_windows[role][:n]-lateral_windows['reference'][:n]
        paired[role] = dict(gained_worlds=np.flatnonzero(case & ~reference).tolist(), lost_worlds=lost.tolist(),
            first_failed_stage_for_lost={int(i):results[role]['first_failed_stage'][i] for i in lost},
            first_window_controls=n, mean_world_lateral_delta_m_s=float(delta.mean()))
    total = shared + sum(row['controls'] for row in results.values())
    assert run['controls'] == runtime['control_steps'] == total <= fork_plan['maximum_unique_controls']
    assert run['transitions'] == runtime['control_transitions'] == total*p['worlds']
    assert run['substeps'] == runtime['physics_substeps'] == total*5
    passed = observed and all(row['passed_gate'] for row in results.values())
    evidence(status='passed-independent-verification', shared_controls=shared, heading_control=heading, eligible_worlds=len(worlds),
        restored_physics_fields=fork['physics']['fields'], same_first_observation_input_action_target_and_delayed_motor=True,
        first_physics_qvel_matches_injection=True, controls=total, transitions=run['transitions'], physics_substeps=run['substeps'],
        branches=results, paired=paired, passed_development_gate=passed)
    print('ROLL_ACTIVE_LATERAL_VERIFIED', {role:(row['roll_completed'],row['original_successes']) for role,row in results.items()}, passed, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()
