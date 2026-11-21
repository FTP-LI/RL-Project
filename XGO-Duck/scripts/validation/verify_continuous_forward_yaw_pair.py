"""Independently score the original motion gates from both saved branches."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from check_continuous_forward_fork_state import controller_snapshot
from getup_release_fork_state import digest, signature
from record_io import write
from roll_stand_walk_sequence import FORWARD, STATE_FIELDS, Sequence, score_forward
from run_continuous_forward_yaw_pair import OUT, PAGE, PLAN, ROLES, baseline, checked
from s1_reference_inputs import ROOT, read, sha
from verify_continuous_roll_projected_entry import count, roll_score
from verify_continuous_roll_stand_walk import arrays, check_policy
from verify_roll_entry_isolation import video


def joined(prefix,suffix,skip):
    assert set(prefix)==set(suffix)
    return {key:np.concatenate((value,suffix[key][skip:]),axis=0) for key,value in prefix.items()}


def validate_media(folder,metadata,trace,endpoint,horizon):
    # Recording ends at the real terminal state of camera world 0 if it falls.
    dead = np.flatnonzero(~trace['active'][:,0])
    end = min(endpoint,int(dead[0])) if len(dead) else endpoint
    return video(folder,metadata,0,end,horizon)


def paired_interval(delta):
    if not len(delta):
        return None
    rng = np.random.default_rng(1095)
    boot = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return np.quantile(boot,[.025,.975]).tolist()


def main():
    p = checked()
    folder = ROOT/'records/evaluations'/p['run_id']
    run = read(folder/'evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    horizon,shared = p['maximum_controls'],run['shared']
    assert run['status']=='collected-review-required' and run['protocol_sha256']==sha(PLAN)
    assert run['physical_resets']==runtime['interface_checks']==1 and not runtime['capacity_reached']
    assert not run['heldout_used'] and not run['learning_updates'] and not runtime['extra_observation_or_render_steps']
    shared_trace_path,shared_commands_path = folder/'shared-trace.npz',folder/'shared-commands.npz'
    prefix,prefix_commands = arrays(shared_trace_path),arrays(shared_commands_path)
    assert len(prefix['active'])==shared+1 and len(prefix_commands['phase_used'])==shared
    seq = Sequence(p)
    for k in range(shared+1):
        seq.update(k,{key:prefix[key][k] for key in STATE_FIELDS})
        assert bool((seq.phase==FORWARD).any())==(k==shared)
    fork = read(folder/'fork.json')
    frozen = arrays(folder/'fork-controller.npz')
    assert fork['physics']['counters']==[shared,shared*5]
    assert fork['physics']['fields']==len(fork['field_names'])==len(set(fork['field_names']))
    assert {'rng','controller.previous','controller.history','controller.previous_features'}<=set(fork['field_names'])
    assert fork['physics']['warp_bytes']<=p['maximum_warp_snapshot_bytes']
    assert fork['forward_worlds']==np.flatnonzero(seq.phase==FORWARD).tolist()
    assert digest(signature(controller_snapshot(seq)))==digest(signature(frozen))==fork['controller_sha256']
    paths = [PLAN,folder/'protocol.json',shared_trace_path,shared_commands_path,folder/'fork.json',
             folder/'fork-controller.npz',folder/'startup.npz',folder/'runtime-contract.json']
    assert sha(folder/'protocol.json')==sha(PLAN)
    first,results = {},{}
    startup = arrays(folder/'startup.npz')
    for role in ROLES:
        branch = folder/'clips'/role
        raw = run['branches'][role]
        local,last = raw['controls'],raw['endpoint']
        assert last==shared+local and 0<local<=horizon-shared
        assert read(branch/'restored-fork.json')=={key:fork[key] for key in ('physics','controller_sha256')}
        files = [branch/name for name in ('trace.npz','commands.npz','endpoint-contact.npz','first-motor.npz')]
        suffix,commands_after,contact,motor = map(arrays,files)
        assert len(suffix['active'])==local+1 and len(commands_after['phase_used'])==local
        for name,value in prefix.items():
            np.testing.assert_array_equal(value[-1],suffix[name][0],err_msg=name)
        trace,commands = joined(prefix,suffix,1),joined(prefix_commands,commands_after,0)
        np.testing.assert_array_equal(trace['active'][1:],np.logical_and.accumulate(
            ~(commands['terminated']|commands['truncated']),axis=0))
        np.testing.assert_array_equal(commands['phase_used'],trace['phase_next'][:-1])
        original = read(branch/'endpoint-contact.json')
        assert original['sha256']==sha(files[2]) and original['complete']==(last==horizon)
        assert original['zero_matches_standalone'] and not original['first_mismatches']
        np.testing.assert_array_equal(contact['control'],np.arange(last+1))
        np.testing.assert_array_equal(contact['substep'],5*np.arange(last+1))
        np.testing.assert_array_equal(contact['fresh'],contact['constraint_points']>0)
        np.testing.assert_array_equal(contact['fresh'][...,0],trace['ground_contact'])
        np.testing.assert_array_equal(contact['fresh'][...,1],trace['head_contact'])
        np.testing.assert_array_equal(contact['fresh'][...,2:],trace['foot_contact'])
        np.testing.assert_array_equal(contact['quaternion'],trace['qpos'][...,3:7])
        assert motor['substeps'].shape[:2]==(5,p['worlds'])
        first[role] = {key:commands_after[key][0] for key in ('observed','actor_input','raw_policy_actions','feedback_yaw')}
        first[role]['delayed_motor'] = motor['substeps'][0]
        role_plan = dict(p,feedback=dict(p['feedback'],maximum_yaw_input_rad_s=p['pair']['roles'][role]))
        replay,stop = count(role_plan,trace,commands)
        assert stop==raw['stop'] and (last==horizon or stop is not None and stop['control']==last)
        policy = check_policy(commands,startup,role_plan,commands['phase_used'])
        roll = roll_score(role_plan,trace,replay)
        full_forward = np.where((replay.forward_at>=0)&(replay.forward_at+p['forward_controls']<=last),replay.forward_at,-1)
        forward = score_forward(trace,full_forward,role_plan)
        count_pass = sum(row['success'] for row in forward)
        results[role] = dict(controls=local,endpoint=last,stop=stop,feedback_cap_rad_s=p['pair']['roles'][role],
            roll_completed=roll['successes'],entered_forward=int((replay.forward_at>=0).sum()),
            finished_forward_window=int((full_forward>=0).sum()),continuous_successes=count_pass,
            original_gate=bool(last==horizon and roll['successes']>=p['minimum_roll_completion']
                               and count_pass>=p['minimum_continuous_successes']),
            video=validate_media(folder,raw['video'],trace,last,horizon),policy=policy,forward=forward,
            roll_switch_at=replay.stand_at.tolist(),forward_at=replay.forward_at.tolist())
        paths += files+[branch/'restored-fork.json',branch/'endpoint-contact.json']
    for name in ('observed','actor_input','raw_policy_actions','delayed_motor'):
        np.testing.assert_array_equal(first[ROLES[0]][name],first[ROLES[1]][name],err_msg=name)
    assert np.any(first[ROLES[0]]['feedback_yaw']!=first[ROLES[1]]['feedback_yaw'])
    reference,wider = (results[role]['forward'] for role in ROLES)
    gained,lost,new_safety_failures,yaw_delta = [],[],[],[]
    for world,(before,after) in enumerate(zip(reference,wider,strict=True)):
        if after['success'] and not before['success']:
            gained.append(world)
        if before['success'] and not after['success']:
            lost.append(world)
            if set(after['failed_checks'])&{'physical_joints','survived','no-forward-entry'}:
                new_safety_failures.append(world)
        if before['entered_forward'] and after['entered_forward']:
            yaw_delta.append(after['yaw_error_rad_s']-before['yaw_error_rad_s'])
    delta = np.asarray(yaw_delta)
    net = len(gained)-len(lost)
    paired = dict(gained_worlds=gained,lost_worlds=lost,net_success_gain=net,new_safety_failures=new_safety_failures,
        yaw_comparable_worlds=len(delta),mean_absolute_yaw_error_change_rad_s=float(delta.mean()) if len(delta) else None,
        descriptive_paired_bootstrap_95_interval_rad_s=paired_interval(delta))
    admitted = results['wider-input']['original_gate'] and not new_safety_failures and net>=p['pair']['gate']['minimum_paired_net_gain']
    total = shared+sum(result['controls'] for result in results.values())
    assert run['controls']==runtime['control_steps']==total<=p['pair']['maximum_unique_controls']
    assert run['transitions']==runtime['control_transitions']==total*p['worlds']
    assert run['substeps']==runtime['physics_substeps']==total*5
    assert not torch.cuda.is_initialized()
    analysis = dict(status='passed-independent-verification',protocol_sha256=sha(PLAN),record=folder.relative_to(ROOT).as_posix(),
        shared_controls=shared,restored_physics_fields=fork['physics']['fields'],restored_controller_fields=len(frozen),
        same_first_observation_policy_input_raw_and_delayed_motor=True,controls=total,transitions=run['transitions'],
        physics_substeps=run['substeps'],branches=results,paired=paired,passed_development_gate=admitted,
        heldout_used=False,learning_updates=0,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    write(OUT/'analysis.json',analysis)
    state = read(folder/'state.json')
    state.update(status='completed',purpose='偏航对照已独立核验，原片见本轮文档',updated_at=datetime.now(baseline.TZ).isoformat())
    with patch.object(baseline.live,'PAGE',PAGE):
        baseline.live.publish(folder,state)
    print('CONTINUOUS_FORWARD_YAW_VERIFIED',
          {role:(row['continuous_successes'],row['original_gate']) for role,row in results.items()},paired,admitted,flush=True)


if __name__=='__main__':
    main()
