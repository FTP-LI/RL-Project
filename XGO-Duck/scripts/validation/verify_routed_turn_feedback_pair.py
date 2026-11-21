"""Independently score both saved suffixes with the original seven-command criteria."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/continuous_routed_turn_feedback_pair.json'
ROLES = ('static-turn','restored-gyro')
TURNS = ('turn-left','turn-right')
PHYSICAL = ('qpos','mode','terminated','velocity','angular_velocity','foot_contact','foot_positions','foot_speed')


def observe():
    import verify_routed_forward_entry_ramp as windows
    import verify_walk_live_page as live

    main = live.main

    def extend_deadline():
        with patch.object(sys, 'argv', [*sys.argv[:-1], '1100']):
            main()

    with patch.object(windows, 'PLAN', PLAN), patch.object(live, 'main', extend_deadline):
        windows.observe(windows.read(PLAN))


def trace(folder, chunks, first, last, names, sha):
    buffers = {name:[] for name in names}
    for item in chunks:
        path = folder/item['file']
        assert item['first'] == first and item['last'] >= first and sha(path) == item['sha256']
        with np.load(path, allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['control'], np.arange(first,item['last']+1))
            for name in names:
                buffers[name].append(stored[name])
        first = item['last']+1
    assert first == last+1
    return {name:np.concatenate(values) for name, values in buffers.items()}


def verify_turn_inputs(spec, role, suffix, turn_ids):
    from command_input_mapping import transform
    from walking_command_controller import create_controller

    seven = spec['seven_command']
    requests = np.empty((len(turn_ids),3), np.float32)
    for name in TURNS:
        requests[np.isin(turn_ids, seven['groups'][name])] = seven['commands'][name]
    entered = suffix['mode'][0, turn_ids] == 2
    np.testing.assert_array_equal(suffix['mode'][:,turn_ids] == 2, np.broadcast_to(entered, (1000,len(turn_ids))))
    controller = create_controller(len(turn_ids), .02)
    observed = np.zeros((len(turn_ids),61), np.float32)
    static = transform(requests)[:,2]
    actual = suffix['turn_yaw_input']
    adjusted = 0
    for k, gyro in enumerate(suffix['turn_gyro']):
        observed[:,2] = gyro
        if role == ROLES[1]:
            expected = controller.map(requests, observed)[:,2]
        else:
            expected = static
        np.testing.assert_array_equal(actual[k], np.where(entered,expected,0.))
        adjusted += int(np.count_nonzero((actual[k] != static) & entered))
    assert (adjusted > 0) == (role == ROLES[1])
    return dict(turn_input_samples=int(actual.size), changed_reached_turn_samples=adjusted)


def verify_first(spec, record, state, fork_input, sha):
    from routed_getup_sequence import FORWARD

    rows = {}
    for role in ROLES:
        path = record/role/'first-control.npz'
        assert sha(path) == state['branches'][role]['first_control_sha256']
        with np.load(path, allow_pickle=False) as stored:
            rows[role] = dict(stored)
    a, b = rows.values()
    n = len(a['mode'])
    turn = np.isin(np.arange(n), state['turn_worlds']) & (a['mode'] == FORWARD)
    for data in rows.values():
        np.testing.assert_array_equal(data['sensed'], fork_input['sensed'])
        np.testing.assert_array_equal(data['target'], data['motor_joint_target'])
        active = data['mode'] == FORWARD
        assert abs(data['target'][active]-fork_input['previous'][active]).max() <= .0400002
        assert data['delayed_motor'].shape == (5,n,14) and np.isfinite(data['delayed_motor']).all()
    for name in ('mode','bridge'):
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(a['actor_input'][:,:50],b['actor_input'][:,:50])
    np.testing.assert_array_equal(a['actor_input'][:,51:],b['actor_input'][:,51:])
    np.testing.assert_array_equal(a['actor_input'][~turn,50],b['actor_input'][~turn,50])
    assert (a['actor_input'][turn,50] != b['actor_input'][turn,50]).any()
    for name in ('raw','target'):
        np.testing.assert_array_equal(a[name][~turn],b[name][~turn])
    np.testing.assert_array_equal(a['delayed_motor'][0],b['delayed_motor'][0])
    return dict(exact_fork_sensed=True, exact_nonturn_first_actions=True, first_turns=int(turn.sum()),
                first_actuator_substep_equal=True)


def verify():
    import imageio_ffmpeg
    import run_routed_turn_feedback_pair as run
    from getup_release_fork_state import digest, signature
    from routed_seven_command_score import score

    run.phase(PLAN)
    read, sha, write = run.read, run.sha, run.write
    spec = read(PLAN)
    run.base.shared.assert_sources(spec)
    record = ROOT/spec['record']
    state, view = read(record/'evaluation.json'), read(record/'windows-view-verification.json')
    checksum = sha(PLAN)
    d, n = spec['turn_pair'], spec['environment']['num_envs']
    end, total = spec['per_branch_horizon'], spec['maximum_unique_controls']
    first = d['shared_controls']
    assert state['status'] == 'completed' and state['actor_weights_unchanged']
    assert not state['heldout_used'] and not state['hardware_io'] and not state['learning_updates']
    assert state['protocol_sha256'] == sha(record/'protocol.json') == checksum
    assert state['actual'] == dict(controls=total, attempted_controls=total, substeps=5*total,
                                  attempted_substeps=5*total, resets=1)
    assert state['transitions'] == total*n and state['contact_peak'] < state['contact_capacity']
    assert view['status'] == 'passed' and view['running_snapshots_with_original_frame'] >= 3
    fork = read(record/'fork.json')
    assert sha(record/'fork.json') == state['fork_sha256']
    assert fork['plant']['counters'] == [first,5*first]
    assert len(fork['fields']) == fork['plant']['fields'] and digest(fork['fields']) == fork['plant']['state_sha256']
    for name,key in (('fork-controller.npz','controller_file_sha256'),('fork-input.npz','input_sha256')):
        assert sha(record/name) == fork[key]
    with np.load(record/'fork-controller.npz', allow_pickle=False) as saved:
        assert digest(signature(dict(saved))) == fork['controller_sha256']
        assert not saved['feedback_previous'].any() and not saved['feedback_history'].any()
    with np.load(record/'fork-input.npz', allow_pickle=False) as saved:
        fork_input = dict(saved)
    first_check = verify_first(spec, record, state, fork_input, sha)
    with np.load(record/'initial.npz', allow_pickle=False) as saved:
        initial = dict(saved)
    assert sha(record/'release-poses.npz') == state['release_poses_sha256']
    shared = trace(record, state['shared_chunks'], 1, first, PHYSICAL, sha)
    np.testing.assert_array_equal(shared['qpos'][-1], fork_input['qpos'])
    turn_ids = sorted(sum((spec['seven_command']['groups'][name] for name in TURNS), []))
    assert turn_ids == state['turn_worlds']
    scores, policy, media = {}, {}, {}
    for role in ROLES:
        folder, branch = record/role, state['branches'][role]
        path = folder/'restored-fork.json'
        assert sha(path) == branch['restore_sha256']
        assert read(path) == dict(plant=fork['plant'], controller_sha256=fork['controller_sha256'])
        suffix = trace(folder, branch['chunks'], first+1, end, (*PHYSICAL,'turn_gyro','turn_yaw_input'), sha)
        policy[role] = verify_turn_inputs(spec, role, suffix, turn_ids)
        full = {name:np.concatenate((shared[name],suffix[name])) for name in PHYSICAL}
        assert not full['terminated'].any()
        scores[role] = score(spec, state, record, full, initial)
        assert read(folder/'admission.json') == {name:row['entered'] for name,row in scores[role]['commands'].items()}
        clip = branch['video']
        video = record/clip['path']
        assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps']
        assert clip['common_steps'] == list(range(0,end+1,2))
        q = full['qpos'][1::2,0]
        hashes = [sha256(initial['qpos'][0].tobytes()).hexdigest(), *(sha256(row.tobytes()).hexdigest() for row in q)]
        assert clip['qpos_sha256_by_frame'] == hashes and sha(video) == clip['sha256']
        frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
        meta = next(frames)
        assert meta['fps'] == 25 and meta['size'] == (640,480)
        assert sum(1 for _ in frames) == len(hashes) == 801
        media[role] = dict(path=run.base.rel(video),sha256=clip['sha256'],seconds=32.,frames=801,world=0,
            assigned_command='turn-left',entered=0 not in scores[role]['commands']['turn-left']['missed_handoff'])
        del full,suffix
    comparison = {}
    for name in spec['seven_command']['groups']:
        a,b = (scores[role]['commands'][name] for role in ROLES)
        x,y = ({row['world']:row for row in arm['walking']['episode_scores']} for arm in (a,b))
        assert x.keys() == y.keys() and a['assigned'] == b['assigned']
        gains = np.asarray([x[i]['yaw_error']-y[i]['yaw_error'] for i in x])
        old_pass = a['complete_gate'] and a['walking']['functional_development_pass']
        net = b['complete']-a['complete']
        comparison[name] = dict(complete_delta=net, paired_mean_yaw_error_reduction_rad_s=float(gains.mean()),
            paired_yaw_improved=int((gains > 0).sum()), control_passed=old_pass,
            restored_passed=b['complete_gate'] and b['walking']['functional_development_pass'],
            turn_effect_gate=bool(gains.mean() > 0 and net >= int(not old_pass)) if name in TURNS else None)
    selected = bool(scores[ROLES[1]]['passed_development_gate'] and
                    all(comparison[name]['turn_effect_gate'] for name in TURNS))
    output = run.OUT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='passed-independent-verification',protocol_sha256=checksum,record=spec['record'],
        physical_resets=1,transitions=state['transitions'],physical_substeps=5*total,
        fork_fields=len(fork['fields']),first_boundary=first_check,turn_policy=policy,
        scores=scores,paired=comparison,videos=media,restored_feedback_admitted=selected,
        hardware_io=False,learning_updates=0,heldout_used=False))
    print('ROUTED_TURN_PAIR_VERIFIED', {role:s['complete'] for role,s in scores.items()},
          {name:comparison[name] for name in TURNS}, selected, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('observe','verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()
