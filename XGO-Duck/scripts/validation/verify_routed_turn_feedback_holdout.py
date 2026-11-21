"""Audit single-use provenance, gyro inputs and the original seven-command score."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np

import verify_routed_turn_feedback_pair as pair


PLAN = Path(__file__).resolve().parents[2]/'configs/experiments/continuous_routed_turn_feedback_holdout.json'
ARM = 'restored-gyro'


def observe():
    with patch.object(pair,'PLAN',PLAN):
        pair.observe()


def verify():
    import imageio_ffmpeg
    import run_routed_turn_feedback_holdout as run
    from getup_release_fork_state import digest, signature
    from routed_getup_sequence import FORWARD
    from routed_seven_command_score import score

    run.phase(PLAN)
    read,sha,write = run.read,run.sha,run.write
    spec = read(PLAN)
    run.base.shared.assert_sources(spec)
    assert spec['reserved_holdout'] and spec['data_scope'] == 'reserved-holdout'
    record = run.ROOT/spec['record']
    state = read(record/'evaluation.json')
    receipt = record/'holdout-consumption.json'
    claim = read(receipt)
    checksum = sha(PLAN)
    seed,n = spec['environment']['seed'],spec['environment']['num_envs']
    first,end = spec['turn_pair']['shared_controls'],spec['per_branch_horizon']
    assert seed == claim['seed'] == 2175 and end == spec['maximum_unique_controls'] == 1600
    assert claim['status'] == 'claimed-before-first-physics' and claim['physical_substeps_before_claim'] == 0
    assert not claim['automatic_retry'] and not claim['heldout_used_for_tuning']
    assert claim['protocol_sha256'] == state['protocol_sha256'] == sha(record/'protocol.json') == checksum
    for name,path in (('windows_observer',record/'windows-observer-ready.json'),
                      ('independent_zero',run.OUT/'independent-zero/proof.json'),
                      ('native_zero',run.OUT/'native-zero.json')):
        assert sha(path) == claim[name+'_sha256']
    assert state['heldout_used'] and state['holdout_consumption_sha256'] == sha(receipt)
    assert state['status'] == 'completed' and state['actor_weights_unchanged']
    assert not state['learning_updates'] and not state['hardware_io']
    assert state['actual'] == dict(controls=end,attempted_controls=end,substeps=end*5,attempted_substeps=end*5,resets=1)
    assert state['transitions'] == end*n and state['contact_peak'] < state['contact_capacity']
    assert list(state['branches']) == [ARM] and not (record/'static-turn').exists()
    view_path = record/'windows-view-verification.json'
    view = read(view_path)
    assert view['status'] == 'passed' and view['running_snapshots_with_original_frame'] >= 3
    fork = read(record/'fork.json')
    assert sha(record/'fork.json') == state['fork_sha256'] and fork['plant']['counters'] == [first,5*first]
    assert len(fork['fields']) == fork['plant']['fields'] and digest(fork['fields']) == fork['plant']['state_sha256']
    for name,key in (('fork-controller.npz','controller_file_sha256'),('fork-input.npz','input_sha256')):
        assert sha(record/name) == fork[key]
    with np.load(record/'fork-controller.npz',allow_pickle=False) as saved:
        assert digest(signature(dict(saved))) == fork['controller_sha256']
        assert not saved['feedback_previous'].any() and not saved['feedback_history'].any()
    with np.load(record/'fork-input.npz',allow_pickle=False) as saved:
        fork_input = dict(saved)
    branch,folder = state['branches'][ARM],record/ARM
    assert sha(folder/'restored-fork.json') == branch['restore_sha256']
    assert read(folder/'restored-fork.json') == dict(plant=fork['plant'],controller_sha256=fork['controller_sha256'])
    assert sha(folder/'first-control.npz') == branch['first_control_sha256']
    with np.load(folder/'first-control.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sensed'],fork_input['sensed'])
        np.testing.assert_array_equal(saved['target'],saved['motor_joint_target'])
        reached = saved['mode'] == FORWARD
        assert abs(saved['target'][reached]-fork_input['previous'][reached]).max() <= .0400002
        assert saved['delayed_motor'].shape == (5,n,14) and np.isfinite(saved['delayed_motor']).all()
    with np.load(record/'initial.npz',allow_pickle=False) as saved:
        initial = dict(saved)
    assert sha(record/'release-poses.npz') == state['release_poses_sha256']
    shared = pair.trace(record,state['shared_chunks'],1,first,pair.PHYSICAL,sha)
    np.testing.assert_array_equal(shared['qpos'][-1],fork_input['qpos'])
    turn_ids = sorted(sum((spec['seven_command']['groups'][name] for name in pair.TURNS),[]))
    assert turn_ids == state['turn_worlds']
    suffix = pair.trace(folder,branch['chunks'],first+1,end,(*pair.PHYSICAL,'turn_gyro','turn_yaw_input'),sha)
    inputs = pair.verify_turn_inputs(spec,ARM,suffix,turn_ids)
    full = {name:np.concatenate((shared[name],suffix[name])) for name in pair.PHYSICAL}
    assert not full['terminated'].any()
    scores = score(spec,state,record,full,initial)
    assert read(folder/'admission.json') == {name:row['entered'] for name,row in scores['commands'].items()}
    clip = branch['video']
    video = record/clip['path']
    assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps']
    assert clip['common_steps'] == list(range(0,end+1,2))
    poses = (initial['qpos'][0],*full['qpos'][1::2,0])
    hashes = [sha256(row.tobytes()).hexdigest() for row in poses]
    assert clip['qpos_sha256_by_frame'] == hashes and sha(video) == clip['sha256']
    frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    meta = next(frames)
    assert meta['fps'] == 25 and meta['size'] == (640,480)
    assert sum(1 for _ in frames) == len(hashes) == 801
    output = run.OUT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='passed-independent-verification',protocol_sha256=checksum,record=spec['record'],
        holdout_seed=seed,holdout_consumption_sha256=sha(receipt),observer_sha256=sha(view_path),
        physical_resets=1,transitions=state['transitions'],physical_substeps=end*5,
        turn_policy=inputs,score=scores,final_holdout_passed=scores['passed_development_gate'],
        video=dict(path=run.base.rel(video),sha256=clip['sha256'],seconds=32.,frames=801,world=0,
                   assigned_command='turn-left',entered=0 not in scores['commands']['turn-left']['missed_handoff']),
        hardware_io=False,learning_updates=0,heldout_used=True,holdout_eligible_for_tuning=False))
    print('ROUTED_TURN_HOLDOUT_VERIFIED',scores['complete'],
          {name:row['complete'] for name,row in scores['commands'].items()},scores['passed_development_gate'],flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('observe','verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()
