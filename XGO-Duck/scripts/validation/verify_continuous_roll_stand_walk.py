"""Verify saved continuous control, first-episode scoring, and original media."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from command_input_mapping import transform
from record_io import write
from roll_stand_walk_sequence import FAILED, FORWARD, ROLL, STAND_LANDING, STATE_FIELDS, Sequence, score_forward
from run_continuous_roll_stand_walk import OUT, PAGE, PLAN, TZ, checked, onnx_session
from walking_actuator_yaw_feedback import ActuatorYawFeedback


def arrays(path):
    with np.load(path) as data:
        return {key:data[key] for key in data.files}


def check_policy(commands, scene, p, phase):
    home = np.where((phase==ROLL)[...,None],scene['roll_home'],scene['walk_home'])
    actual = commands['actor_input']
    base = commands['observed']
    shape = phase.shape
    eligible = phase!=FAILED
    expected = base.copy()
    expected[...,6:20] += scene['simulation_home'] - home
    expected[0,:,34:48] = 0
    expected[1:,:,34:48] = commands['raw_policy_actions'][:-1]
    expected[...,34:48][commands['history_reset']] = 0
    expected[...,48:51] = transform(commands['external_command'])
    np.testing.assert_array_equal(actual[eligible],expected[eligible])
    np.testing.assert_array_equal(actual[~eligible],0)
    np.testing.assert_allclose(commands['requested_targets'][eligible],
                               (commands['raw_policy_actions']+home)[eligible],atol=0,rtol=0)
    is_forward = phase==FORWARD
    np.testing.assert_array_equal(commands['external_command'][~is_forward],0)
    np.testing.assert_array_equal(commands['external_command'][is_forward],
        np.broadcast_to(np.asarray(p['walking']['commands']['forward'],np.float32),(int(is_forward.sum()),3)))
    hip = [20+p['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left','right')]
    feature = np.stack((base[...,2],base[...,hip].sum(axis=-1)),axis=-1)
    rate = np.diff(feature,axis=0,prepend=feature[:1])
    feedback = ActuatorYawFeedback(read(ROOT/p['feedback_screen']),p['feedback']['maximum_yaw_input_rad_s'])
    yaw = feedback.commands(np.concatenate((feature[is_forward],rate[is_forward]),axis=1)).astype(np.float32)
    np.testing.assert_allclose(commands['feedback_yaw'][is_forward],yaw,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(commands['feedback_yaw'][~is_forward],0.)

    low, high = np.asarray(p['walking']['joint_limits'],np.float32).T
    desired = np.clip(commands['requested_targets'],low,high)
    alternate = np.clip(commands['alternate_action'][is_forward]+scene['walk_home'],low,high)
    difference = alternate-desired[is_forward]
    scale = p['feedback']['maximum_joint_correction_rad']/np.maximum(abs(difference).max(axis=-1,keepdims=True),1e-8)
    desired[is_forward] += difference*np.minimum(1.,scale)
    prev = np.concatenate((scene['simulation_home'][None],commands['applied_targets'][:-1]))
    maximum = .02*p['bridge_slew_rad_s']
    gate = commands['bridge']
    desired[gate] = prev[gate]+np.clip(desired[gate]-prev[gate],-maximum,maximum)
    desired[~eligible] = prev[~eligible]
    np.testing.assert_allclose(commands['applied_targets'],desired,atol=2e-7,rtol=0)
    np.testing.assert_allclose(commands['actual_motor_target'],desired,atol=1e-6,rtol=0)
    tested = 0
    for name, mask in (('roll',phase==ROLL),('walking',eligible & (phase!=ROLL))):
        config = p[name]
        session = onnx_session(ROOT/config['model'])
        indices = np.argwhere(mask)
        indices = indices[np.unique(np.linspace(0,len(indices)-1,min(768,len(indices)),dtype=int))] if len(indices) else indices
        for i,w in indices:
            action = session.run(None,{'obs':actual[i,w][None]})[0][0]
            np.testing.assert_allclose(commands['raw_policy_actions'][i,w],action,rtol=2e-5,atol=2e-5)
            if is_forward[i,w]:
                inp = actual[i,w].copy()
                inp[50] = commands['feedback_yaw'][i,w]
                correction = session.run(None,{'obs':inp[None]})[0][0]
                np.testing.assert_allclose(commands['alternate_action'][i,w],correction,rtol=2e-5,atol=2e-5)
        tested += len(indices)
    assert actual.shape == shape+(61,)
    return dict(actual_inputs_replayed=tested,maximum_bridge_delta_rad=float(abs((desired-prev)[gate]).max(initial=0.)),
                raw_history_exact=True,control_targets_verified=True)


def main():
    p = checked()
    folder = ROOT/'records/evaluations'/p['run_id']
    state = read(folder/'evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    assert state['status'] == 'collected-review-required' and state['protocol_sha256'] == sha(PLAN)
    assert runtime['interface_checks'] == state['physical_resets'] == 1 and not runtime['capacity_reached']
    assert runtime['control_steps'] == p['maximum_controls']
    assert runtime['control_transitions'] == state['actual_control_transitions'] == p['maximum_control_transitions']
    assert runtime['physics_substeps'] == state['actual_physics_substeps'] == p['maximum_physics_substeps']
    trace_path, commands_path = folder/'trace.npz',folder/'commands.npz'
    assert sha(trace_path) == state['trace_sha256'] and sha(commands_path) == state['command_sha256']
    trace, commands = arrays(trace_path), arrays(commands_path)
    endpoint_file = folder/'endpoint-contact.npz'
    meta = read(folder/'endpoint-contact.json')
    assert meta['complete'] and meta['zero_matches_standalone'] and sha(endpoint_file) == meta['sha256']
    endpoint = arrays(endpoint_file)
    contacts = np.concatenate((trace['ground_contact'][...,None],trace['head_contact'][...,None],trace['foot_contact']),axis=-1)
    np.testing.assert_array_equal(contacts,endpoint['fresh'])
    np.testing.assert_array_equal(contacts,endpoint['constraint_points']>0)
    np.testing.assert_array_equal(trace['qpos'][...,3:7],endpoint['quaternion'])
    np.testing.assert_array_equal(trace['active'][1:],np.logical_and.accumulate(~(commands['terminated']|commands['truncated']),axis=0))
    replay = Sequence(p)
    for k in range(len(trace['active'])):
        change = replay.update(k,{field:trace[field][k] for field in STATE_FIELDS})
        np.testing.assert_array_equal(replay.phase,trace['phase_next'][k])
        if k < p['maximum_controls']:
            np.testing.assert_array_equal(change & np.isin(replay.phase,(ROLL,STAND_LANDING)),commands['history_reset'][k])
            bridge = np.isin(replay.phase,(ROLL,STAND_LANDING)) & (k+1-replay.phase_changed<=p['cross_model_bridge_controls'])
            np.testing.assert_array_equal(bridge,commands['bridge'][k])
    np.testing.assert_array_equal(commands['phase_used'],trace['phase_next'][:-1])
    scene = arrays(folder/'startup.npz')
    policy = check_policy(commands,scene,p,commands['phase_used'])
    outcome = score_forward(trace,replay.forward_at,p)
    saved = read(folder/'raw-score.json')
    assert saved['roll_switch_at'] == replay.stand_at.tolist() and saved['forward_at'] == replay.forward_at.tolist()
    assert saved['forward'] == outcome
    assert not torch.cuda.is_initialized()
    clip, = state['video']
    assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps']
    assert clip['common_steps'] == list(range(0,p['maximum_controls']+1,2))
    video = folder/clip['file']
    with_frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    header = next(with_frames)
    assert header['fps'] == 25 and header['size'] == (640,480)
    assert sum(1 for _ in with_frames) == clip['frames'] == len(clip['common_steps'])
    counts = dict(worlds=p['worlds'],roll_completed=int((replay.stand_at>=0).sum()),
                  entered_forward=int((replay.forward_at>=0).sum()),continuous_successes=sum(row['success'] for row in outcome))
    passed = counts['roll_completed'] >= p['minimum_roll_completion'] and counts['continuous_successes'] >= p['minimum_continuous_successes']
    write(OUT/'analysis.json',dict(status='passed-independent-verification',counts=counts,passed_development_gate=passed,
        scope='One development sequence with continuous physics; remaining switch directions and perturbations require separate verification.',
        record=state['record'],policy=policy,outcomes=outcome,roll_switch_at=saved['roll_switch_at'],forward_at=saved['forward_at'],
        video=dict(path=str(video.relative_to(ROOT)),sha256=sha(video),frames=clip['frames'],seconds=p['maximum_controls']*.02),
        protocol_sha256=sha(PLAN),learning_updates=0,heldout_used=False))
    state['status'] = 'completed'
    write(folder/'evaluation.json',state)
    view = read(folder/'state.json')
    view.update(status='completed',updated_at=datetime.now(TZ).isoformat(),latest_clip=clip['file'])
    with patch.object(live,'PAGE',PAGE):
        live.publish(folder,view)
    print('ROLL_STAND_WALK_VERIFIED',counts,passed,flush=True)


if __name__=='__main__':
    main()
