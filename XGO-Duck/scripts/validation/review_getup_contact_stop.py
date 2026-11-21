"""Close the infeasible promotion gate without relabeling an unfinished arm.

Keep original thresholds, both completed evaluations, all consumed samples and
the candidate's partial optimizer checkpoint. It has no scheduled final model
or fixed evaluation, so no reward-effect conclusion is admissible.
"""
from pathlib import Path
import json
import numpy as np
import torch
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from standup_metrics import summarize

OUT=ROOT/'records/reports/getup-late-contact'


def main():
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    master=read(ROOT/'configs/experiments/getup_late_contact_pair.json');stop=read(OUT/'ceiling-stop.json')
    assert stop['required_candidate_supine']==33 and stop['maximum_candidate_supine']==32
    counts={};evaluations={}
    for role in ('original','control'):
        review=read(OUT/f'eval-{role}/analysis.json');assert review['status']=='passed'
        record=Path(read(OUT/f'eval-{role}/progress.json')['path']);p=read(record/'protocol.json');counts[role]={}
        for pose in p['poses']:
            with np.load(record/f'{pose}-trace.npz') as f:data={k:f[k] for k in f.files}
            result=summarize(data,p['criteria']);counts[role][pose]=result['successes']
            assert result['successes']==next(c for c in review['conditions'] if c['condition']==pose)['successes']
        evaluations[role]=str(record.relative_to(ROOT))
    assert counts=={'original':{'prone':32,'supine':31},'control':{'prone':32,'supine':31}}
    p=master['arms']['candidate'];record=ROOT/'records/experiments'/p['run_id']
    state=read(record/'state.json');runtime=read(record/'runtime.json')
    assert state['status']=='stopped-review-required' and 'KeyboardInterrupt' in state['error']
    assert state['completed_updates']==42 and runtime['control_steps']==1032
    assert not (record/'model-final.pt').exists() and not (OUT/'eval-candidate').exists()
    final=torch.load(record/'stopped-partial.pt',map_location='cpu',weights_only=True)
    initial=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    for role in ('actor','critic'):
        for key,v in final[role+'_state_dict'].items():
            assert torch.isfinite(v).all()
            if not key.startswith('mlp.'):torch.testing.assert_close(v,initial[role+'_state_dict'][key],atol=0,rtol=0)
    assert final['control_steps']==1032 and final['completed_updates']==42
    with np.load(record/'training-trace.npz') as f:
        assert f['qpos'].shape==(1032,32,21) and np.isfinite(f['qpos']).all()
        assert f['step'].tolist()==list(range(1,1033))
    # The interruption stack is backward(), after the43rd rollout collection;
    # no physics integration was in progress when the signal was handled.
    log=(record/'execution.log').read_text();assert 'value_loss.backward()' in log and 'KeyboardInterrupt' in log
    clips=[]
    for c in state['clips']:
        video=record/c['file'];reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);n=sum(1 for _ in reader)
        assert n==c['frames']==len(c['common_steps']) and meta['fps']==25.
        clips.append(dict(path=str(video.relative_to(ROOT)),frames=n,seconds=c['frame_times_seconds'][-1],complete=c['complete_clip'],sha256=sha(video)))
    assert len(clips)==2 and clips[0]['complete'] and not clips[1]['complete']
    assert read(record/'matched-start.json')['status']=='passed'
    result=dict(status='passed',protocol_status='closed-infeasible-promotion-gate',counts=counts,
        candidate_evaluation='not-run; candidate stopped before scheduled50update checkpoint',candidate_continue=False,
        common_adaptation_lead=False,candidate_reward_effect_established=False,
        candidate_complete_updates=42,candidate_collected_rollouts=43,
        candidate_actor_optimizer_steps=final['actor_steps_this_run'],candidate_critic_optimizer_steps=final['critic_steps_this_run'],
        candidate_control_transitions=runtime['control_transitions'],actual_learning_transitions=1228800+runtime['control_transitions'],
        actual_evaluation_transitions=70400,unconsumed_learning_transitions=2457600-1228800-runtime['control_transitions'],
        unconsumed_evaluation_transitions=35200,candidate_partial_checkpoint_sha256=sha(record/'stopped-partial.pt'),
        candidate_clips=clips,evaluation_records=evaluations,thresholds_changed=False,budgets_extended=False,
        matched_start_verified=True,source_sha256=sha(Path(__file__)),heldout_used=False,hardware_io=False,
        scope='Completed control preserved; incomplete candidate has no final evaluation. Gate infeasibility is not evidence that the contact penalty failed.')
    write(OUT/'analysis.json',result);write(record/'post-stop-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
