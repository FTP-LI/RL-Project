"""Read back the independent continuous trial, including an early-stop case."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from record_io import write
from roll_entry_impossibility import maximum_possible
from roll_metrics import summarize
from roll_stand_walk_sequence import FAILED, ROLL, STAND_LANDING, STATE_FIELDS, Sequence, score_forward
from run_continuous_roll_projected_entry import OUT, PAGE, PLAN, baseline, checked
from verify_continuous_roll_stand_walk import arrays, check_policy
from verify_roll_entry_isolation import video


def count(p,trace,commands):
    seq = Sequence(p)
    last = len(commands['phase_used'])
    first_stop = None
    for k in range(last+1):
        changed = seq.update(k,{key:trace[key][k] for key in STATE_FIELDS})
        np.testing.assert_array_equal(seq.phase,trace['phase_next'][k])
        if k<last:
            np.testing.assert_array_equal(changed & np.isin(seq.phase,(ROLL,STAND_LANDING)),commands['history_reset'][k])
            bridge = (seq.phase==STAND_LANDING)&(k+1-seq.phase_changed<=p['cross_model_bridge_controls'])
            np.testing.assert_array_equal(bridge,commands['bridge'][k])
        if first_stop is None and seq.roll_begin<=k<=seq.roll_end:
            possible,certified = maximum_possible(seq.roll_latch,p['roll_controls'])
            if possible<p['minimum_roll_completion']:
                first_stop = dict(reason='original-roll-threshold-impossible',control=k,possible=possible,
                    certified=certified,required=p['minimum_roll_completion'])
        potential = int(((seq.phase!=FAILED)|((seq.forward_at>=0)&(k>seq.forward_at+p['forward_controls']))).sum())
        if first_stop is None and potential<p['minimum_continuous_successes']:
            first_stop = dict(reason='sequence-threshold-impossible',control=k,possible=potential,
                required=p['minimum_continuous_successes'])
    return seq,first_stop


def roll_score(p,trace,seq):
    start,horizon = seq.roll_begin,p['roll_controls']
    if len(trace['active'])<=start:
        assert (seq.stand_at<0).all()
        return dict(attempts=p['worlds'],successes=0,episodes=[
            dict(env_index=i,success=False,reasons=['standing-prefix-ended']) for i in range(p['worlds'])])
    observed = min(len(trace['active'])-start,horizon+1)
    end = np.where(seq.stand_at>=0,seq.stand_at,start+observed-1)
    eligible = trace['phase_next'][start]==ROLL
    part = {key:trace[key][start:start+observed].copy() for key in STATE_FIELDS}
    part['active'] &= eligible[None]&(np.arange(start,start+observed)[:,None]<=end[None])
    padded = {key:np.concatenate((value,np.repeat(value[-1:],horizon+1-observed,axis=0)))
              for key,value in part.items()}
    padded['active'][observed:] = False
    padded['time'] = np.arange(horizon+1)*.02
    result = summarize(padded,p['roll']['criteria'])
    np.testing.assert_array_equal([row['success'] for row in result['episodes']],seq.stand_at>=0)
    return result


def main():
    p = checked()
    folder = ROOT/'records/evaluations'/p['run_id']
    runtime = read(folder/'runtime-contract.json')
    assert runtime['interface_checks']==1 and not runtime['capacity_reached']
    last = runtime['control_steps']
    assert runtime['control_transitions']==last*p['worlds'] and runtime['physics_substeps']==last*5
    trace_path,commands_path = folder/'trace.npz',folder/'commands.npz'
    trace,commands = arrays(trace_path),arrays(commands_path)
    assert len(trace['active'])==last+1 and len(commands['phase_used'])==last
    np.testing.assert_array_equal(commands['phase_used'],trace['phase_next'][:-1])
    np.testing.assert_array_equal(trace['active'][1:],np.logical_and.accumulate(
        ~(commands['terminated']|commands['truncated']),axis=0))
    endpoint_file = folder/'endpoint-contact.npz'
    endpoint,meta = arrays(endpoint_file),read(folder/'endpoint-contact.json')
    assert sha(endpoint_file)==meta['sha256'] and meta['zero_matches_standalone'] and not meta['first_mismatches']
    fresh = endpoint['fresh']
    np.testing.assert_array_equal(fresh,endpoint['constraint_points']>0)
    np.testing.assert_array_equal(fresh[...,0],trace['ground_contact'])
    np.testing.assert_array_equal(fresh[...,1],trace['head_contact'])
    np.testing.assert_array_equal(fresh[...,2:],trace['foot_contact'])
    np.testing.assert_array_equal(endpoint['quaternion'],trace['qpos'][...,3:7])
    replay,stop = count(p,trace,commands)
    policy = check_policy(commands,arrays(folder/'startup.npz'),p,commands['phase_used'])
    roll = roll_score(p,trace,replay)
    verified_forward = np.where((replay.forward_at>=0)&(replay.forward_at+p['forward_controls']<=last),replay.forward_at,-1)
    forward = score_forward(trace,verified_forward,p)
    if last<p['maximum_controls']:
        early = read(folder/'planned-auto-stop.json')
        assert stop is not None and stop['control']==last
        assert all(early[key]==value for key,value in stop.items())
        assert early['completed_controls']==last and early['physical_substeps']==last*5
        assert not meta['complete']
    else:
        run = read(folder/'evaluation.json')
        assert run['status']=='collected-review-required' and run['protocol_sha256']==sha(PLAN)
        assert run['trace_sha256']==sha(trace_path) and run['command_sha256']==sha(commands_path)
        # A full-horizon parameter screen deliberately keeps collecting after
        # the original success threshold becomes impossible.  In that mode
        # ``stop`` is a diagnostic of the first impossible control, not an
        # instruction that the rollout stopped.  Only the early-stop branch
        # above requires the stop marker to coincide with the final control.
        assert stop is None or 0 <= stop['control'] <= last
        assert meta['complete'] and read(folder/'raw-score.json')['forward']==forward
    clip, = read(folder/'clips.json')
    clip_path = folder/clip['file']
    media = video(folder,dict(clip,path=clip['file'],sha256=sha(clip_path)),0,last,p['maximum_controls'])
    total = sum(row['success'] for row in forward)
    assert not torch.cuda.is_initialized()
    counts = dict(worlds=p['worlds'],roll_completed=roll['successes'],entered_forward=int((replay.forward_at>=0).sum()),
                  finished_forward_window=int((verified_forward>=0).sum()),continuous_successes=total)
    gate = bool(last==p['maximum_controls'] and roll['successes']>=p['minimum_roll_completion']
                and total>=p['minimum_continuous_successes'])
    write(OUT/'analysis.json',dict(status='passed-independent-verification',record=folder.relative_to(ROOT).as_posix(),
        protocol_sha256=sha(PLAN),counts=counts,passed_development_gate=gate,early_stop=stop,
        completed_controls=last,transitions=last*p['worlds'],physical_substeps=last*5,
        roll=roll,forward=forward,policy=policy,roll_switch_at=replay.stand_at.tolist(),
        forward_at=replay.forward_at.tolist(),video=media,heldout_used=False,learning_updates=0,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in
                       (PLAN,trace_path,commands_path,endpoint_file,folder/'runtime-contract.json')}))
    state = read(folder/'state.json')
    state.update(status='completed',purpose='连续试验已独立核验，原片见本轮文档',updated_at=datetime.now(baseline.TZ).isoformat())
    if media['complete']:
        state['latest_clip'] = clip['file']
    with (folder/'README.md').open('a',encoding='utf-8') as out:
        out.write(f'\n[{"完整" if media["complete"] else "部分"}原片]({clip["file"]})\n')
    with patch.object(live,'PAGE',PAGE):
        live.publish(folder,state)
    print('ROLL_PROJECTED_ENTRY_VERIFIED',counts,gate,stop,flush=True)


if __name__=='__main__':
    main()
