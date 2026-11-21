"""Recompute every branch, including planned early stops, without physics."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from record_io import write
from roll_entry_impossibility import maximum_possible
from roll_landing_latch import RollLandingLatch
from roll_metrics import summarize
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from run_continuous_roll_stand_walk import onnx_session
from run_roll_entry_isolation import OUT, PAGE, PLAN, ROLES, TZ, checked
from verify_continuous_roll_stand_walk import arrays


def check_policy(cmd, init, prev, eligible, trace, role, p):
    raw = cmd['raw_policy_actions']
    expected = cmd['observed'].copy()
    expected[...,6:20] += init['simulation_home']-init['roll_home']
    expected[0,:,34:48] = 0.
    expected[1:,:,34:48] = raw[:-1]
    np.testing.assert_array_equal(expected,cmd['actor_input'])
    np.testing.assert_array_equal(expected[...,48:],0.)
    np.testing.assert_array_equal(raw+init['roll_home'],cmd['requested_targets'])
    low,high = np.asarray(p['walking']['joint_limits'],np.float32).T
    target = cmd['requested_targets'] if role==ROLES[0] else np.clip(cmd['requested_targets'],low,high)
    history = np.concatenate((prev[None],cmd['applied_targets'][:-1]))
    if role==ROLES[2]:
        controls = p['cross_model_bridge_controls']
        maximum = .02*p['bridge_slew_rad_s']
        target[:controls] = history[:controls]+np.clip(target[:controls]-history[:controls],-maximum,maximum)
    mask = trace['active'][:-1]&eligible[None]
    target = np.where(mask[...,None],target,history)
    np.testing.assert_allclose(target,cmd['applied_targets'],rtol=0,atol=2e-7)
    np.testing.assert_allclose(target,cmd['actual_motor_target'],rtol=0,atol=1e-6)
    session = onnx_session(ROOT/p['roll']['model'])
    inputs = expected.reshape(-1,expected.shape[-1])
    actions = raw.reshape(-1,raw.shape[-1])
    selected = np.unique(np.linspace(0,len(inputs)-1,min(640,len(inputs)),dtype=int))
    worst = 0.
    for i in selected:
        actual = session.run(None,{'obs':inputs[i:i+1]})[0][0]
        np.testing.assert_allclose(actual,actions[i],rtol=2e-5,atol=2e-5)
        worst = max(worst,float(np.max(np.abs(actual-actions[i]))))
    return dict(onnx_samples=len(selected),maximum_error=worst)


def score(trace, eligible, plan, receipt):
    p = plan['plant']
    horizon = plan['maximum_roll_controls_per_branch']
    gate = plan['minimum_roll_completion']
    latch = RollLandingLatch(p['roll']['criteria'],plan['worlds'])
    trace = dict(trace,active=trace['active']&eligible[None])
    timeline, deaths = [], []
    first_over = np.full(plan['worlds'],-1,dtype=int)
    for k in range(len(trace['active'])):
        before = latch.intact.copy()
        latch.advance({field:trace[field][k] for field in STATE_FIELDS})
        fail = before&~latch.intact&(latch.first_ready<0)
        penetrated = trace['active'][k]&(trace['head_penetration_m'][k]>latch.depth_limit)
        first_over[(first_over<0)&penetrated] = k
        for i in np.flatnonzero(fail):
            causes = [name for name,test in (
                ('inactive',not trace['active'][k,i]),
                ('head-penetration',penetrated[i]),
                ('joint-envelope',not trace['joint_envelope_ok'][k,i]),
                ('reverse-rotation',latch.angle[i]<-latch.reverse),
                ('excess-rotation',latch.angle[i]>latch.high),
            ) if test]
            if not causes and k==0:
                causes = ['initial-posture']
            assert causes
            deaths.append(dict(world=int(i),roll_control=k,causes=causes,depth_m=float(trace['head_penetration_m'][k,i])))
        timeline.append(list(maximum_possible(latch,horizon)))
    assert timeline==receipt['timeline'] and latch.first_ready.tolist()==receipt['ready_steps']
    assert all(row[0]>=gate for row in timeline[:-1])
    assert timeline[-1]==[receipt['possible'],receipt['certified']]
    assert receipt['controls']==len(timeline)-1
    assert receipt['reason']==('full-roll-window' if receipt['controls']==horizon else 'original-threshold-impossible')
    if receipt['controls']<horizon:
        assert timeline[-1][0]<gate
    # Inactive padding cannot complete a hold or improve an original score.
    padding = horizon+1-len(timeline)
    oracle = {k:np.concatenate((trace[k],np.repeat(trace[k][-1:],padding,axis=0))) for k in STATE_FIELDS}
    oracle['active'][len(timeline):] = False
    oracle['time'] = np.arange(horizon+1)*p['roll']['criteria']['step_dt']
    verified = summarize(oracle,p['roll']['criteria'])
    np.testing.assert_array_equal([row['success'] for row in verified['episodes']],latch.first_ready>=0)
    return dict(certified=verified['successes'],meets_development_gate=verified['successes']>=gate,
        possible_at_stop=timeline[-1][0],first_irrevocable_failures=deaths,first_penetration_steps=first_over.tolist(),
        peak_depth_mm=float(trace['head_penetration_m'].max()*1000),
        outcomes=verified['episodes'])


def video(folder, metadata, warm, controls, horizon):
    path = folder/metadata['path']
    assert sha(path)==metadata['sha256'] and not metadata['extra_physics_steps']
    actual, complete = warm+controls, controls==horizon
    assert metadata['complete_clip']==complete and metadata['full_action']==complete
    assert metadata['common_steps']==list(range(0,actual+1,2))+([actual] if actual%2 else [])
    assert metadata['frame_times_seconds']==[step*.02 for step in metadata['common_steps']]
    reader = imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    header = next(reader)
    assert header['fps']==25 and header['size']==(640,480)
    assert sum(1 for _ in reader)==metadata['frames']==len(metadata['common_steps'])
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),frames=metadata['frames'],
                seconds=actual*.02,complete=complete)


def main():
    import torch

    plan = checked()
    p = plan['plant']
    folder = ROOT/'records/evaluations'/plan['run_id']
    run = read(folder/'evaluation.json')
    assert run['status']=='collected-review-required' and run['protocol_sha256']==sha(PLAN)
    assert run['physical_resets']==1 and not run['heldout_used'] and not run['learning_updates']
    warm,horizon = plan['shared_stand_controls'],plan['maximum_roll_controls_per_branch']
    init, prefix = arrays(folder/'startup.npz'),arrays(folder/'standing-trace.npz')
    standing = arrays(folder/'standing-commands.npz')
    assert len(standing['applied_targets'])==warm and len(prefix['active'])==warm+1
    seq = Sequence(p)
    for k in range(warm+1):
        seq.update(k,{field:prefix[field][k] for field in STATE_FIELDS})
    eligible = seq.phase==ROLL
    fork = read(folder/'fork.json')
    assert int(eligible.sum())==fork['eligible'] and fork['captured']['counters']==[warm,warm*5]
    first, results = {}, {}
    paths = [PLAN,folder/'protocol.json',folder/'standing-trace.npz',folder/'standing-commands.npz',folder/'fork.json']
    for role in ROLES:
        path = folder/'clips'/role
        receipt = run['branches'][role]
        assert read(path/'restored-fork.json')==fork['captured']
        trace,cmd,contacts,motor = (arrays(path/name) for name in
            ('trace.npz','commands.npz','endpoint-contact.npz','first-motor.npz'))
        assert len(cmd['applied_targets'])==receipt['controls'] and len(trace['active'])==receipt['controls']+1
        for key,row in trace.items():
            np.testing.assert_array_equal(row[0],prefix[key][-1],err_msg=key)
        np.testing.assert_array_equal(contacts['fresh'],contacts['constraint_points']>0)
        after = contacts['fresh'][warm:]
        np.testing.assert_array_equal(after[...,0],trace['ground_contact'])
        np.testing.assert_array_equal(after[...,1],trace['head_contact'])
        np.testing.assert_array_equal(after[...,2:],trace['foot_contact'])
        np.testing.assert_array_equal(contacts['quaternion'][warm:],trace['qpos'][...,3:7])
        np.testing.assert_array_equal(trace['active'][1:],trace['active'][:1]&np.logical_and.accumulate(
            ~(cmd['terminated']|cmd['truncated']),axis=0))
        item = (cmd['actor_input'][0],cmd['raw_policy_actions'][0],motor['substeps'][0])
        if first:
            for actual,expected in zip(item,first[ROLES[0]],strict=True):
                np.testing.assert_array_equal(actual,expected)
        first[role] = item
        results[role] = dict(score(trace,eligible,plan,receipt),controls=receipt['controls'],stop=receipt['reason'],
            policy=check_policy(cmd,init,standing['applied_targets'][-1],eligible,trace,role,p),
            video=video(folder,receipt['video'],warm,receipt['controls'],horizon))
        paths += [path/name for name in ('trace.npz','commands.npz','endpoint-contact.npz','first-motor.npz','restored-fork.json')]
    total = warm+sum(row['controls'] for row in results.values())
    assert run['controls']==total<=plan['maximum_controls_all_branches']
    assert run['transitions']==total*plan['worlds'] and run['substeps']==total*5
    assert not torch.cuda.is_initialized()
    write(OUT/'analysis.json',dict(status='passed-independent-verification',record=run['record'],protocol_sha256=sha(PLAN),
        worlds=plan['worlds'],eligible=int(eligible.sum()),common_stand_controls=warm,controls=total,
        transitions=run['transitions'],substeps=run['substeps'],heldout_used=False,learning_updates=0,
        restored_fork_fields=fork['captured']['fields'],same_first_input_raw_and_delayed_motor=True,
        branches=results,source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths}))
    run['status'] = 'completed'
    write(folder/'evaluation.json',run)
    state = read(folder/'state.json')
    state.update(status='completed',purpose='三组已独立核验，原片见本轮文档',updated_at=datetime.now(TZ).isoformat())
    if any(row['video']['complete'] for row in results.values()):
        state['latest_clip'] = next(run['branches'][role]['video']['path'] for role in ROLES if results[role]['video']['complete'])
    with patch.object(live,'PAGE',PAGE):
        live.publish(folder,state)
    print('ROLL_ENTRY_ISOLATION_VERIFIED',{role:(row['controls'],row['certified'],row['possible_at_stop'])
          for role,row in results.items()},flush=True)


if __name__=='__main__':
    main()
