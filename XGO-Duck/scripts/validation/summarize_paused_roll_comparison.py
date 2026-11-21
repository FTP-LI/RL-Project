"""Audit the already saved part of a user-paused roll comparison on CPU."""

from argparse import ArgumentParser

import imageio_ffmpeg
import numpy as np
import torch

from record_io import write
from roll_stand_walk_sequence import score_forward
from s1_reference_inputs import ROOT, read, sha
from verify_continuous_forward_yaw_pair import joined
from verify_continuous_lateral_push import check_endpoint
from verify_continuous_roll_projected_entry import count, roll_score
from verify_continuous_roll_stand_walk import arrays
from verify_roll_head_support_control import audit


def score_reference(record, spec, trace, commands, shared):
    end = len(commands['phase_used'])
    assert end == spec['maximum_controls']
    sequence, stop = count(spec,trace,commands)
    assert stop is None
    roll = np.array([row['success'] for row in roll_score(spec,trace,sequence)['episodes']])
    starts = sequence.forward_at
    finished = (starts>=0) & (starts+spec['forward_controls']<=end)
    forward = score_forward(trace,np.where(finished,starts,-1),spec)
    full = roll & np.array([row['success'] for row in forward])
    policy = audit(commands,arrays(record/'startup.npz'),spec,shared,False)
    return dict(roll_completed=int(roll.sum()),continuous_completed=int(full.sum()),policy=policy)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('record')
    parser.add_argument('output')
    args = parser.parse_args()
    record, out = (ROOT/name for name in (args.record,args.output))
    spec, runtime, state = (read(record/name) for name in ('protocol.json','runtime-contract.json','state.json'))
    project = read(ROOT/'project.json')
    assert project['goal_execution']['status']=='paused' and project['staged_training_request']['pause_requested']
    assert not project['staged_training_request']['simulation_running'] and not (record/'evaluation.json').exists()
    assert runtime['status']=='closed' and state['error']=="AssertionError('Stop requested')"
    exit_receipt = read(out/'sample-exit.json')
    assert exit_receipt['exit_code']==1 and runtime['control_steps']==state['controls']
    assert runtime['physics_substeps']==5*runtime['control_steps']
    assert runtime['control_transitions']==spec['worlds']*runtime['control_steps']
    prefix, prefix_commands = (arrays(record/name) for name in ('shared-trace.npz','shared-commands.npz'))
    shared = len(prefix_commands['phase_used'])
    assert shared==spec['perturbation']['shared'] and len(prefix['active'])==shared+1
    files = [record/name for name in ('protocol.json','state.json','runtime-contract.json','shared-trace.npz',
                                      'shared-commands.npz','fork.json','fork-controller.npz','windows-view-verification.json')]
    branches = {}
    for role in spec['perturbation']['roles']:
        folder = record/'clips'/role
        video = folder/'clip-step-0.mp4'
        meta, = read(folder/'clips.json')
        frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
        fmt = next(frames)
        frame_count = sum(1 for _ in frames)
        assert fmt['fps']==25 and frame_count==len(meta['common_steps'])
        row = dict(video=video.relative_to(ROOT).as_posix(),frames=frame_count,
                   video_seconds=meta['common_steps'][-1]*.02,video_sha256=sha(video),
                   complete_clip=meta['complete_clip'],branch_controls=0)
        files.extend((video,folder/'clips.json'))
        trace_file, cmd_file = (folder/name for name in ('trace.npz','commands.npz'))
        if trace_file.exists():
            suffix, after = arrays(trace_file), arrays(cmd_file)
            trace, commands = joined(prefix,suffix,1), joined(prefix_commands,after,0)
            row.update(branch_controls=len(after['phase_used']),timeline_controls=len(commands['phase_used']),
                       trace=trace_file.relative_to(ROOT).as_posix(),commands=cmd_file.relative_to(ROOT).as_posix())
            assert len(trace['active'])==row['timeline_controls']+1
            check_endpoint(folder,trace)
            files.extend((trace_file,cmd_file,folder/'endpoint-contact.npz'))
            if row['complete_clip']:
                row.update(score_reference(record,spec,trace,commands,shared))
        branches[role] = row
    recorded = shared+sum(row['branch_controls'] for row in branches.values())
    # The simulator tests the stop flag immediately after the physical step,
    # before that step is returned to the collector and appended to its trace.
    assert runtime['control_steps']-recorded==1
    assert not torch.cuda.is_initialized()
    observer = read(record/'windows-view-verification.json')
    result = out/'pause.json'
    assert not result.exists()
    write(result,dict(status='paused-by-user',closed_at=exit_receipt['finished_at'],protocol_sha256=sha(record/'protocol.json'),
        run=record.relative_to(ROOT).as_posix(),runtime=runtime,shared_controls=shared,recorded_controls=recorded,
        stop_guard_physics_controls=1,branches=branches,windows_original_frames=observer['running_snapshots_with_original_frame'],
        full_comparison_scored=False,additional_physics_substeps=0,learning_updates=0,
        source_sha256={file.relative_to(ROOT).as_posix():sha(file) for file in (*files,out/'sample-exit.json')}))
    print('ROLL_PAUSE_VERIFIED',runtime['control_steps'],recorded,
          {role:(row.get('timeline_controls'),row['frames'],row.get('continuous_completed')) for role,row in branches.items()},
          flush=True)


if __name__=='__main__':
    main()
