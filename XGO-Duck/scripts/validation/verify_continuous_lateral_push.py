"""Independently recover original and perturbation scores from saved states."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from check_continuous_forward_fork_state import controller_snapshot
from getup_release_fork_state import digest, signature
from record_io import write
from roll_stand_walk_sequence import FORWARD, STATE_FIELDS, Sequence, score_forward
from run_continuous_lateral_push import OUT, PAGE, PLAN, baseline, checked
from verify_continuous_forward_yaw_pair import joined, paired_interval, validate_media
from verify_continuous_roll_projected_entry import count, roll_score
from verify_continuous_roll_stand_walk import arrays, check_policy


def check_endpoint(folder,trace):
    path = folder/'endpoint-contact.npz'
    c,m = arrays(path),read(folder/'endpoint-contact.json')
    n = len(trace['active'])
    assert m['sha256']==sha(path) and not m['first_mismatches'] and m['zero_matches_standalone']
    assert m['complete']==(n==1551)
    np.testing.assert_array_equal(c['control'],np.arange(n))
    np.testing.assert_array_equal(c['substep'],5*np.arange(n))
    all_contacts = np.concatenate((trace['ground_contact'][...,None],trace['head_contact'][...,None],trace['foot_contact']),axis=-1)
    np.testing.assert_array_equal(all_contacts,c['fresh'])
    np.testing.assert_array_equal(all_contacts,c['constraint_points']>0)
    np.testing.assert_array_equal(trace['qpos'][...,3:7],c['quaternion'])


def verify_velocity(folder,prefix,worlds,shared,magnitude):
    a,m = arrays(folder/'injection.npz'),read(folder/'injection.json')
    assert m['control']==shared and m['magnitude_m_s']==magnitude
    assert m['changed_fields']==(['warp.qvel'] if magnitude else [])
    np.testing.assert_array_equal(a['before'],prefix['qvel'][-1])
    np.testing.assert_array_equal(a['world_ids'],worlds)
    # Rotate the body Y axis into the world plane using the saved free-joint
    # quaternion. Preserve all other qvel components by exact equality.
    w,x,y,z = prefix['qpos'][-1,worlds,3:7].astype(np.float64).T
    heading = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    xy = (np.stack((-np.sin(heading),np.cos(heading)),axis=1)*magnitude).astype(a['before'].dtype)
    np.testing.assert_allclose(a['target_world_xy'],xy,rtol=0,atol=1e-8)
    expected = a['before'].copy()
    expected[worlds,:2] += a['target_world_xy']
    np.testing.assert_array_equal(a['after'],expected)
    np.testing.assert_array_equal(a['first_substep_qvel'],expected)
    np.testing.assert_array_equal(a['measured_world_xy'],(expected-a['before'])[worlds,:2])
    np.testing.assert_allclose(np.linalg.norm(a['measured_world_xy'],axis=1),abs(magnitude),rtol=0,atol=5e-8)
    assert a['substep_motor'].shape[:2]==(5,len(a['before']))
    return a['substep_motor'][0]


def verify_recovery(trace,shared,p,worlds):
    mask = np.zeros(p['worlds'],dtype=bool)
    if len(trace['active'])<=shared+100:
        return mask
    whole = slice(shared+1,shared+101)
    second = slice(shared+51,shared+101)
    v = trace['velocity'][second,:, :2].mean(axis=0)
    limit = p['walking']['criteria']
    error = np.hypot(v[:,0]-p['walking']['commands']['forward'][0],v[:,1])
    passed = (v[:,0]>=limit['signed_speed']['forward'])&(error<=limit['linear_error']['forward'])
    mask[worlds] = passed[worlds]&trace['active'][whole,worlds].all(axis=0)&trace['joint_envelope_ok'][whole,worlds].all(axis=0)
    return mask


def publish(folder,status,message):
    view = read(folder/'state.json')
    view.update(status=status,purpose=message,updated_at=datetime.now(baseline.TZ).isoformat())
    with patch.object(baseline.live,'PAGE',PAGE):
        baseline.live.publish(folder,view)


def main():
    p = checked()
    study = p['perturbation']
    folder = ROOT/'records/evaluations'/p['run_id']
    runtime = read(folder/'runtime-contract.json')
    assert not runtime['capacity_reached'] and not runtime['extra_observation_or_render_steps']
    assert runtime['interface_checks']==1 and sha(folder/'protocol.json')==sha(PLAN)
    horizon,shared = p['maximum_controls'],study['shared']
    prefix_file,prefix_cmd_file = (folder/name for name in ('shared-trace.npz','shared-commands.npz'))
    prefix,prefix_commands = arrays(prefix_file),arrays(prefix_cmd_file)
    startup = arrays(folder/'startup.npz')
    paths = [PLAN,folder/'protocol.json',prefix_file,prefix_cmd_file,folder/'startup.npz',folder/'runtime-contract.json']
    if (folder/'shared-stop.json').exists():
        early = read(folder/'shared-stop.json')
        _,stop = count(p,prefix,prefix_commands)
        n = len(prefix_commands['phase_used'])
        assert stop==early['stop'] and stop['control']==n==runtime['control_steps']<=shared
        assert runtime['control_transitions']==n*p['worlds'] and runtime['physics_substeps']==n*5
        check_endpoint(folder,prefix)
        check_policy(prefix_commands,startup,p,prefix_commands['phase_used'])
        clips = {role:validate_media(folder,clip,prefix,n,horizon) for role,clip in early['clips'].items()}
        paths += [folder/'shared-stop.json',folder/'endpoint-contact.npz']
        assert not torch.cuda.is_initialized()
        write(OUT/'analysis.json',dict(status='verified-shared-stop',protocol_sha256=sha(PLAN),stop=stop,
            record=folder.relative_to(ROOT).as_posix(),clips=clips,passed_development_gate=False,
            controls=n,transitions=n*p['worlds'],physics_substeps=n*5,heldout_used=False,learning_updates=0,
            source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths}))
        publish(folder,'stopped-review-required','共同前段已按原门槛停止，部分原片见本轮文档')
        print('CONTINUOUS_LATERAL_PUSH_SHARED_STOP',stop,flush=True)
        return
    run = read(folder/'evaluation.json')
    assert run['status']=='collected-review-required' and run['protocol_sha256']==sha(PLAN)
    assert not run['learning_updates'] and not run['heldout_used'] and run['physical_resets']==1
    assert run['shared']==shared and len(prefix['active'])==shared+1 and len(prefix_commands['phase_used'])==shared
    seq = Sequence(p)
    for k in range(shared+1):
        seq.update(k,{key:prefix[key][k] for key in STATE_FIELDS})
    fork = read(folder/'fork.json')
    frozen = arrays(folder/'fork-controller.npz')
    assert fork['physics']['counters']==[shared,shared*5]
    assert fork['physics']['fields']==len(fork['field_names'])==len(set(fork['field_names']))
    assert {'warp.qvel','rng','controller.previous','controller.history','controller.previous_features'}<=set(fork['field_names'])
    assert fork['physics']['warp_bytes']<=p['maximum_warp_snapshot_bytes']
    worlds = np.flatnonzero((seq.phase==FORWARD)&prefix['active'][-1])
    assert fork['forward_worlds']==worlds.tolist() and len(worlds)>=p['minimum_continuous_successes']
    assert digest(signature(controller_snapshot(seq)))==digest(signature(frozen))==fork['controller_sha256']
    paths += [folder/'fork.json',folder/'fork-controller.npz']
    first,results,body_lateral = {},{},{}
    for role,magnitude in study['roles'].items():
        branch = folder/'clips'/role
        raw = run['branches'][role]
        last = raw['endpoint']
        assert last==shared+raw['controls'] and shared<last<=horizon
        assert read(branch/'restored-fork.json')=={key:fork[key] for key in ('physics','controller_sha256')}
        suffix,after = arrays(branch/'trace.npz'),arrays(branch/'commands.npz')
        assert len(suffix['active'])==raw['controls']+1 and len(after['phase_used'])==raw['controls']
        for key,value in prefix.items():
            np.testing.assert_array_equal(value[-1],suffix[key][0],err_msg=key)
        motor = verify_velocity(branch,prefix,worlds,shared,magnitude)
        first[role] = tuple(after[key][0] for key in ('observed','actor_input','raw_policy_actions','feedback_yaw','applied_targets'))+(motor,)
        trace,commands = joined(prefix,suffix,1),joined(prefix_commands,after,0)
        np.testing.assert_array_equal(trace['active'][1:],np.logical_and.accumulate(~(commands['terminated']|commands['truncated']),axis=0))
        np.testing.assert_array_equal(commands['phase_used'],trace['phase_next'][:-1])
        check_endpoint(branch,trace)
        replay,stop = count(p,trace,commands)
        recovery = verify_recovery(trace,shared,p,worlds)
        rec_count = int(recovery.sum())
        if last>=shared+100 and (stop is None or stop['control']>shared+100) and rec_count<p['minimum_continuous_successes']:
            stop = dict(reason='two-second-recovery-threshold-impossible',control=shared+100,possible=rec_count,
                        required=p['minimum_continuous_successes'])
        assert stop==raw['stop'] and (last==horizon or stop is not None and stop['control']==last)
        policy = check_policy(commands,startup,p,commands['phase_used'])
        roll = roll_score(p,trace,replay)
        full_forward = np.where((replay.forward_at>=0)&(replay.forward_at+p['forward_controls']<=last),replay.forward_at,-1)
        scores = score_forward(trace,full_forward,p)
        full = np.asarray([row['success'] for row in scores])
        combined = full&recovery
        passed = bool(last==horizon and roll['successes']>=p['minimum_roll_completion'] and combined.sum()>=p['minimum_continuous_successes'])
        body_lateral[role] = trace['velocity'][shared+1:min(shared+11,last+1),worlds,1]
        results[role] = dict(controls=raw['controls'],endpoint=last,stop=stop,perturbation_m_s=magnitude,original_successes=int(full.sum()),
            recovery_successes=rec_count,full_and_recovery=int(combined.sum()),passed_gate=passed,roll_completed=roll['successes'],
            forward_completed=int((full_forward>=0).sum()),original_forward_outcomes=scores,combined_by_world=combined.tolist(),
            recovery_by_world=recovery.tolist(),policy=policy,video=validate_media(folder,raw['video'],trace,last,horizon))
        paths.extend(branch/name for name in ('trace.npz','commands.npz','endpoint-contact.npz','restored-fork.json','injection.npz','injection.json'))
    reference = np.asarray(results['reference']['combined_by_world'])
    paired = {}
    for role in ('left-push','right-push'):
        for value,expected in zip(first[role],first['reference'],strict=True):
            np.testing.assert_array_equal(value,expected)
        this = np.asarray(results[role]['combined_by_world'])
        n = min(len(body_lateral[role]),len(body_lateral['reference']))
        delta = (body_lateral[role][:n]-body_lateral['reference'][:n]).mean(axis=0)
        paired[role] = dict(gained_worlds=np.flatnonzero(this&~reference).tolist(),lost_worlds=np.flatnonzero(~this&reference).tolist(),
            first_window_controls=n,mean_lateral_delta_m_s=float(delta.mean()),paired_bootstrap_95_interval_m_s=paired_interval(delta))
    total = shared+sum(result['controls'] for result in results.values())
    assert run['controls']==runtime['control_steps']==total<=study['maximum_unique_controls']
    assert run['transitions']==runtime['control_transitions']==total*p['worlds']
    assert run['substeps']==runtime['physics_substeps']==total*5
    assert not torch.cuda.is_initialized()
    verdict = all(result['passed_gate'] for result in results.values())
    write(OUT/'analysis.json',dict(status='passed-independent-verification',protocol_sha256=sha(PLAN),
        record=folder.relative_to(ROOT).as_posix(),shared_controls=shared,eligible_worlds=len(worlds),
        restored_physics_fields=fork['physics']['fields'],restored_controller_fields=len(frozen),
        same_first_observation_input_action_target_and_delayed_motor=True,first_physics_qvel_matches_injection=True,
        controls=total,transitions=run['transitions'],physics_substeps=run['substeps'],branches=results,
        paired=paired,passed_development_gate=verdict,heldout_used=False,learning_updates=0,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths}))
    publish(folder,'completed','前进横向扰动已独立核验，原片和逐场景评分见本轮文档')
    print('CONTINUOUS_LATERAL_PUSH_VERIFIED', {role:(row['original_successes'],row['full_and_recovery'],row['stop'])
          for role,row in results.items()},verdict,paired,flush=True)


if __name__=='__main__':
    main()
