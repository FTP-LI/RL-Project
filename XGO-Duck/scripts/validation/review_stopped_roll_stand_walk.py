"""Audit the first continuous attempt after its roll gate could no longer pass."""
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from roll_metrics import progress, rotation
from roll_stand_walk_sequence import FAILED, ROLL, STATE_FIELDS, Sequence
from run_continuous_roll_stand_walk import OUT, PLAN, checked
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from verify_continuous_roll_stand_walk import arrays, check_policy


def standing_signals(trace, count, criteria):
    head = slice(0,count+1)
    rot = rotation(trace['quaternion'][head])
    parts = dict(
        active=trace['active'][head],
        both_feet=trace['foot_contact'][head].all(axis=-1),
        joints=trace['joint_envelope_ok'][head],
        height=abs(trace['height'][head]-criteria['nominal_height_m'])<=criteria['height_tolerance_m'],
        tilt=rot[...,2,2]>=np.cos(np.deg2rad(criteria['tilt_max_deg'])),
        gyro=np.linalg.norm(trace['angular_velocity'][head],axis=-1)<=criteria['angular_speed_max_rad_s'],
    )
    return parts


def trailing(mask):
    length = np.zeros(mask.shape[1],dtype=np.int32)
    for row in mask:
        length = np.where(row,length+1,0)
    return length


def main():
    p = checked()
    folder = ROOT/'records/evaluations'/p['run_id']
    receipt = read(folder/'state.json')
    runtime = read(folder/'runtime-contract.json')
    last = runtime['control_steps']
    assert p['initial_stand_controls']+p['roll_controls'] <= last < p['maximum_controls']
    assert receipt['status']=='stopped-review-required' and receipt['controls']==last
    assert (OUT/'sample-exit.txt').read_text().strip()=='130'
    assert runtime['interface_checks']==1 and not runtime['capacity_reached']
    assert runtime['control_transitions']==last*p['worlds']
    assert runtime['physics_substeps']==5*last
    trace_path, cmd_path, contact_path = (folder/name for name in ('trace.npz','commands.npz','endpoint-contact.npz'))
    trace, commands, contact = (arrays(path) for path in (trace_path,cmd_path,contact_path))
    assert len(trace['active'])==last+1 and len(commands['phase_used'])==last
    endpoints = read(folder/'endpoint-contact.json')
    assert sha(contact_path)==endpoints['sha256'] and endpoints['controls']==last+1 and not endpoints['complete']
    assert endpoints['zero_matches_standalone'] and not endpoints['first_mismatches']
    np.testing.assert_array_equal(contact['fresh'],contact['constraint_points']>0)
    np.testing.assert_array_equal(trace['foot_contact'],contact['fresh'][...,2:])
    np.testing.assert_array_equal(trace['qpos'][...,3:7],contact['quaternion'])
    replay = Sequence(p)
    for k in range(last+1):
        replay.update(k,{key:trace[key][k] for key in STATE_FIELDS})
        np.testing.assert_array_equal(replay.phase,trace['phase_next'][k])
    assert (replay.phase==FAILED).all() and (replay.stand_at<0).all() and (replay.forward_at<0).all()
    np.testing.assert_array_equal(commands['phase_used'],trace['phase_next'][:-1])
    policy = check_policy(commands,arrays(folder/'startup.npz'),p,commands['phase_used'])
    first = p['initial_stand_controls']
    entered = trace['phase_next'][first]==ROLL
    signals = standing_signals(trace,first,p['roll']['criteria'])
    window = p['standing_dwell_controls']
    without_contact = np.logical_and.reduce([values for key,values in signals.items() if key!='both_feet'])
    joint = np.logical_and.reduce(list(signals.values()))
    assert np.array_equal(entered,trailing(joint)>=window)
    counts = {key:int((trailing(value)>=window).sum()) for key,value in signals.items()}
    comparison = dict(no_contact_qualification=int((trailing(without_contact)>=window).sum()),
        bilateral_contact_at_boundary=int(signals['both_feet'][-1].sum()),
        dwell_and_boundary_bilateral=int(((trailing(without_contact)>=window)&signals['both_feet'][-1]).sum()),
        joint_stable_dwell=int((trailing(joint)>=window).sum()),
        longest_trailing_combined_dwell=int(trailing(joint).max()))
    full_roll = slice(first,first+p['roll_controls']+1)
    rolled = {key:trace[key][full_roll][:,entered] for key in ('quaternion','ground_contact')}
    angles, supported, _ = progress(rolled)
    reached_full_angle = int((angles.max(axis=0)>=replay.roll_latch.low).sum())
    reached_supported_angle = int((supported.max(axis=0)>=replay.roll_latch.low).sum())
    clip, = read(folder/'clips.json')
    assert not clip['complete_clip'] and not clip['full_action'] and not clip['extra_physics_steps']
    video = folder/clip['file']
    reader = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    header = next(reader)
    assert header['fps']==25 and header['size']==(640,480)
    assert sum(1 for _ in reader)==clip['frames']
    assert clip['common_steps']==list(range(0,last+1,2))
    observer = read(folder/'windows-view-verification.json')
    assert observer['running_snapshots']>=3 and observer['running_snapshots_with_original_frame']>=3
    output = OUT/'stopped-review.json'
    assert not output.exists() and not torch.cuda.is_initialized()
    files = (PLAN,Path(__file__),trace_path,cmd_path,contact_path,folder/'clips.json',folder/'runtime-contract.json',
             folder/'windows-view-verification.json',OUT/'sample-execution.log')
    write(output,dict(status='verified-stopped-development-attempt',record=folder.relative_to(ROOT).as_posix(),
        stop='SIGINT after no world could pass the completed roll window',counts=dict(worlds=p['worlds'],
            entered_roll=int(entered.sum()),reached_geometric_angle=reached_full_angle,
            reached_supported_angle=reached_supported_angle,switched_to_stand=0,entered_forward=0),
        per_standing_signal_dwell=counts,standing_gate_ablation=comparison,policy=policy,
        completed_control_steps=last,completed_transitions=runtime['control_transitions'],
        substeps_in_completed_controls=runtime['physics_substeps'],partial_control_may_have_advanced=True,
        video=dict(path=video.relative_to(ROOT).as_posix(),sha256=sha(video),frames=clip['frames'],
                   seconds=clip['frame_times_seconds'][-1],complete=False),
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        learning_updates=0,heldout_used=False,hardware_io=False))
    print('CONTINUOUS_ROLL_STOP_VERIFIED',int(entered.sum()),reached_full_angle,reached_supported_angle,
          counts,comparison,'completed',last,flush=True)


if __name__=='__main__':
    main()
