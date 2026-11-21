"""Independently verify a completed, observed roll learner profile from disk."""

import csv
from hashlib import file_digest
import json
import math
from pathlib import Path
import statistics

import imageio_ffmpeg
import numpy as np
import torch

from verify_roll_exposure_parent import finite

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok,issue):
    if not ok:
        raise RuntimeError(issue)


def independent(plan,plan_path,arm,report):
    torch.set_num_threads(2)
    ident=arm['run_id']
    record=ROOT/'records/experiments'/ident
    receipt, entry, view, ack = (read(report/name) for name in (
        'supervisor-result.json','entry-result.json','final-view.json','viewing-confirmation.json'))
    zero, result, observed, state, critic = (read(record/name) for name in (
        'zero-step.json','online-result.json','observation.json','state.json','critic-only.json'))
    recipe=plan['recipe']
    n=arm['environments']
    total=arm['updates']
    warm=recipe['critic_only_updates']
    rollout=recipe['rollout_controls']
    controls=total*rollout
    require(receipt['status']=='child-exited' and receipt['exit_code']==0 and receipt['lock_acquired']
            and receipt['lock_passed_to_child'] and receipt['protocol_sha256']==entry['plan_sha256']==ack['protocol_sha256']==sha(plan_path)
            and receipt['log_sha256']==sha(report/'execution.log') and ack['screenshot_sha256']==sha(ROOT/ack['screenshot'])
            and view['status']=='saved' and view['source_sha256']==sha(record/'observation.json')
            and entry['status']==result['status']=='collected-review-required'
            and entry['online_result_sha256']==sha(record/'online-result.json')
            and entry['authorization_monitor']['failure'] is None, 'Run, observer or supervisor receipt is not complete')
    require(all(x['run_id']==ident for x in (receipt,entry,ack,view,result,state,observed)),'Run identity mismatch')
    require(zero['status']=='passed-learner-reset-only' and zero['controls']==zero['physics_substeps']==zero['learning_updates']==0
            and zero['matched_actor_critic_and_adam'] and sha(record/'zero-step.json')==result['zero_step_sha256']
            and sha(record/'startup-inputs.npz')==zero['startup_inputs_sha256'], 'Reset-only evidence mismatch')
    with np.load(record/'startup-inputs.npz',allow_pickle=False) as p:
        mu=p['sliding_mu']
        require(mu.shape==(n,) and zero['sliding_mu_below_0p85']==int((mu<.85).sum()),'摩擦数组与分组统计不一致')
        require(p['actor'].shape==(32,61) and p['policy'].shape==(32,14)
                and p['qpos'].shape==(n,21) and p['qvel'].shape==(n,20),'Actual learner dimensions changed')
    require(critic['status']=='actor-byte-unchanged' and critic['updates']==warm and critic['controls']==warm*rollout,
            'Critic warmup proof not available')
    require(result['transitions']==arm['transitions']==controls*n
            and observed['controls']==state['controls']==result['controls']==controls
            and result['physics_substeps']==state['physical_substeps']==5*controls
            and observed['completed_updates']==state['completed_updates']==result['completed_updates']==total
            and result['critic_only_updates']==warm and result['actor_updates']==total-warm, 'Actual work differs from freeze')
    metrics=record/'learning-metrics.jsonl'
    require(sha(metrics)==result['metrics_sha256'],'Metric file digest mismatch')
    data=[json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines()]
    require(observed['rows']==data and len(data)==total, 'Observer metrics do not match per-update evidence')
    for i,row in enumerate(data,1):
        require(row['update']==i and row['control']==i*rollout and row['phase']==('critic-only' if i<=warm else 'full-ppo')
                and row['measured_batch_wall_seconds']>0 and row['loss'] and all(math.isfinite(x) for x in row['loss'].values())
                and math.isclose(row['net_transitions_per_second'],n*rollout/row['measured_batch_wall_seconds'],abs_tol=1e-6),
                'Training work or timing sample differs')
    with (report/'final-view/native-metrics.csv').open(encoding='utf-8',newline='') as csv_file:
        shown=list(csv.DictReader(csv_file))
    require(len(shown)==total and (report/'final-view/native-curve.png').stat().st_size>0,'Final viewer was incomplete')
    parent_path=ROOT/recipe['parent_checkpoint']
    ckpt=record/result['checkpoint']
    require(ckpt.parent==record and sha(ckpt)==result['checkpoint_sha256']
            and sha(parent_path)==result['parent_sha256']==recipe['parent_sha256'],'Checkpoint hash mismatch')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    require(finite(saved) and saved['iter']==total-1 and saved['infos']['env_state']['common_step_counter']==controls,
            'Checkpoint cannot restore the declared final update')
    for net in ('actor_state_dict','critic_state_dict'):
        require(set(saved[net])==set(parent[net]),'Network layout changed')
        require(any(not torch.equal(saved[net][k],v) for k,v in parent[net].items() if k.startswith('mlp.')),
                'Learner did not update '+net)
    require(all(torch.equal(saved['actor_state_dict'][k],v) for k,v in parent['actor_state_dict'].items()
                if k.startswith('obs_normalizer.')), 'Frozen actor observation normalization moved')
    adam=saved['optimizer_state_dict']['state']
    require({int(item['step']) for item in adam.values()}=={20*(total-warm),20*total},
            'Optimizer counts do not match critic-only and full-PPO budget')
    clips=[]
    for item in read(record/'clips.json'):
        mp4=record/item['file']
        reader=imageio_ffmpeg.read_frames(str(mp4),pix_fmt='rgb24')
        meta=next(reader)
        frames=sum(1 for _ in reader)
        require(frames==item['frames'] and meta['fps']==item['fps']==25 and not item['extra_physics_steps'],'Video does not decode')
        if item['full_action']:
            require(item['complete_clip'] and item['terminal_before_reset']
                    and item['common_steps']==list(range(item['common_steps'][0],item['common_steps'][0]+351,2))
                    and frames==176,'Complete episode interval differs')
        clips.append(dict(path=mp4.relative_to(ROOT).as_posix(),sha256=sha(mp4),frames=frames,complete_episode=item['full_action']))
    require(any(v['complete_episode'] for v in clips),'Missing complete originally recorded episode')
    mem=result['device']['rows']
    peak=max(v['used_mib'] for v in mem)
    limit=min(v['total_mib']*recipe['hard_gpu_fraction'] for v in mem)
    headroom=limit-peak
    net=result['transitions']/result['measured_wall_seconds']
    require(peak==result['device']['peak_used_mib'] and headroom>=0 and math.isclose(net,result['measured_transitions_per_second'])
            and not torch.cuda.is_initialized(),'Performance telemetry or CPU-only verification violated')
    return dict(status='passed-independent-ppo-profile',run_id=ident,environments=n,updates=total,optimizer_full_updates=total-warm,
        controls=controls,transitions=controls*n,substeps=5*controls,net_transitions_per_second=net,
        last_actor_updates_median_rate=statistics.median(row['net_transitions_per_second'] for row in data[warm:]),
        peak_gpu_used_mib=peak,headroom_to_limit_mib=headroom,median_gpu_utilization=result['device']['median_utilization'],
        wall_seconds=result['measured_wall_seconds'],cpu_seconds=result['process_cpu_seconds'],
        component_wall_seconds=result['component_wall_seconds'],clips=clips,checkpoint_sha256=sha(ckpt),
        model_promotion_allowed=False,new_learning_updates=0,new_physical_steps=0,hardware_io=False)
