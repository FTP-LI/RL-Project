"""Recompute both the common roll prefix and the complete route from saved evidence."""

import argparse
from pathlib import Path

import run_roll_friction_pair as pair


def score(plan, folder):
    import imageio_ffmpeg
    import numpy as np
    import torch
    from roll_stand_walk_sequence import ROLL, STAND_LANDING, STATE_FIELDS, Sequence, score_forward
    from verify_continuous_roll_stand_walk import arrays, check_policy

    read, sha, require = pair.read, pair.sha, pair.require
    torch.set_num_threads(2)
    manifest, runtime = (read(folder/name) for name in ('guarded-evaluation.json', 'runtime-contract.json'))
    saved_plan = read(folder/'protocol.json')
    require(saved_plan == plan and folder.name == plan['run_id']
        and not manifest['learning_updates'] and not manifest['hardware_io']
        and not manifest['artifact_errors'] and runtime['interface_checks'] == 1
        and runtime['status'] == 'closed' and not runtime['capacity_reached']
        and not runtime['extra_observation_or_render_steps'],
        'Saved route identity or evidence is incomplete')
    controls, worlds = manifest['actual_control_steps'], plan['worlds']
    require(0 < controls <= plan['maximum_controls'] and controls == runtime['control_steps'] == manifest['saved_controls']
        and manifest['saved_states'] == controls+1 and runtime['physics_substeps'] == manifest['actual_physics_substeps'] == controls*5
        and runtime['control_transitions'] == manifest['recorded_control_transitions'] == controls*worlds,
        'Route counters disagree')

    trace_path, command_path = folder/'trace.npz', folder/'commands.npz'
    require(sha(trace_path)==manifest['trace_sha256'] and sha(command_path)==manifest['command_sha256'], 'Route trace hash changed')
    trace, commands = arrays(trace_path), arrays(command_path)
    endpoint_path = folder/'endpoint-contact.npz'
    endpoint_meta = read(folder/'endpoint-contact.json')
    require(sha(endpoint_path)==endpoint_meta['sha256'] and endpoint_meta['zero_matches_standalone'], 'Route contact evidence differs')
    endpoint = arrays(endpoint_path)
    contact = np.concatenate((trace['ground_contact'][..., None], trace['head_contact'][..., None], trace['foot_contact']), axis=-1)
    np.testing.assert_array_equal(contact, endpoint['fresh'])
    np.testing.assert_array_equal(contact, endpoint['constraint_points'] > 0)
    np.testing.assert_array_equal(trace['qpos'][..., 3:7], endpoint['quaternion'])
    np.testing.assert_array_equal(trace['active'][1:], np.logical_and.accumulate(~(commands['terminated'] | commands['truncated']),axis=0))
    np.testing.assert_array_equal(commands['phase_used'], trace['phase_next'][:-1])

    complete = manifest['status'] == 'collected-review-required'
    prefix_steps = plan['initial_stand_controls']+plan['roll_controls']
    replay = Sequence(plan)
    prefix = None
    for step, valid in enumerate(trace['guard_phase_valid']):
        if not bool(valid):
            require(step == controls and not complete, 'Missing route phase in the recorded prefix')
            break
        changed = replay.update(step, {field: trace[field][step] for field in STATE_FIELDS})
        np.testing.assert_array_equal(replay.phase, trace['phase_next'][step])
        if step == prefix_steps:
            prefix = (replay.stand_at >= 0).tolist()
        if step < controls:
            switching = np.isin(replay.phase, (ROLL, STAND_LANDING))
            np.testing.assert_array_equal(changed & switching, commands['history_reset'][step])
            np.testing.assert_array_equal(switching & (step+1-replay.phase_changed <= plan['cross_model_bridge_controls']),
                                          commands['bridge'][step])

    policy = check_policy(commands, arrays(folder/'startup.npz'), plan, commands['phase_used'])
    active = trace['active']
    safety = dict(head_over_limit_worlds=int(np.any(active & (trace['head_penetration_m'] > plan['roll']['criteria']['head_penetration_max_m']), axis=0).sum()),
                  joint_over_limit_worlds=int(np.any(active & ~trace['joint_envelope_ok'], axis=0).sum()))
    require(all(manifest['safety_summary'][key]==count for key,count in safety.items()), 'Route safety count changed')
    safety['nonfinite_samples'] = sum(int(np.count_nonzero(~np.isfinite(block))) for block in trace.values() if block.dtype.kind in 'fc')
    require(safety['nonfinite_samples']==manifest['safety_summary']['nonfinite_samples']
            and not torch.cuda.is_initialized(), 'Safety evidence or CPU-only replay differs')

    route = None
    if complete:
        require(controls==plan['maximum_controls'] and endpoint_meta['complete'] and prefix is not None,
                'Complete route has a short trace')
        outcome = score_forward(trace, replay.forward_at, plan)
        saved = read(folder/'raw-score.json')
        require(saved['roll_switch_at']==replay.stand_at.tolist() and saved['forward_at']==replay.forward_at.tolist()
                and saved['forward']==outcome, 'Route result disagrees with independent replay')
        route = [row['success'] for row in outcome]

    clips = []
    for clip in manifest['video']:
        path = folder / clip['file']
        decoder = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        metadata = next(decoder)
        frames = sum(1 for _ in decoder)
        require(frames==clip['frames']==len(clip['common_steps']) and metadata['fps']==25
                and not clip['extra_physics_steps'], 'Route clip is invalid')
        if complete:
            require(clip['complete_clip'] and clip['full_action'] and clip['common_steps']==list(range(0,controls+1,2)),
                    'Complete route has a partial planned recording')
        clips.append(dict(path=pair.rel(path), sha256=sha(path), frames=frames, seconds=frames/metadata['fps'],
                          complete=bool(clip['full_action'])))
    require(clips and (not complete or any(row['complete'] for row in clips)), 'No viewable route recording')
    pass_gate = (complete and sum(prefix)>=plan['minimum_roll_completion']
                and sum(route)>=plan['minimum_continuous_successes'] and not any(safety.values()))
    return dict(run_id=plan['run_id'], controls=controls, worlds=worlds, full_route=complete, passed_gate=pass_gate,
        roll_completion_at_common_prefix=None if prefix is None else sum(prefix), prefix_controls=prefix_steps,
        route_success=None if route is None else sum(route), roll_world_pass=prefix, route_world_pass=route, safety=safety,
        policy_readback=policy, stop_reason=manifest['stop_reason'], clips=clips, holdout_used=False,
        hardware_io=False, learning_updates=0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path)
    args = parser.parse_args()
    plan = pair.read(args.protocol)
    row = score(plan, pair.ROOT/'records/evaluations'/plan['run_id'])
    print({key:row[key] for key in ('run_id','controls','roll_completion_at_common_prefix','route_success','safety')})
