"""Verify both trained arms, native observation and export before action scoring."""

import fcntl
import json
from pathlib import Path
from statistics import median

import torch

from observed_getup_entry import finalize_view
from run_nominal_low_friction_support_ppo import KEY, OUT, PLAN, PROJECT, ROLES, ROOT, STAGE, original, parent
from run_nominal_low_friction_support_ppo import read, rel, selected, sha, write
from verify_getup_supine_training import verify_clips, verify_trace
from verify_nominal_supine_upright_band_training import (
    checkpoint, check_checkpoints, check_draws, check_export, compare_training_release,
)


def check_arm(role, plan, node, warm, zero):
    run = plan['runs'][role]
    report = OUT/role
    record = ROOT/'records/experiments'/run
    limit = plan['limit']
    result, state = read(report/'result.json'),read(record/'state.json')
    product = node['results'][role]
    assert result['complete'] and product['complete'] and state['status']==result['status']=='collected-review-required'
    assert state['protocol_sha256']==node['protocol_sha256'] and result['run_id']==state['run_id']==run
    assert state['controls']==state['attempted_controls']==limit['controls']
    assert state['substeps']==state['attempted_substeps']==result['physics_substeps']==limit['substeps']
    assert state['terminal_resets']==1 and state['paired_zero_start_verified']
    assert sha(record/'matched-state.json')==zero['proof'][role]['matched_state_sha256']
    assert sha(record/'learning-metrics.jsonl')==result['metrics_sha256']
    assert sha(record/'model-final.pt')==result['checkpoint_sha256']
    lines = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
    assert len(lines)==plan['updates_per_arm']==result['updates']==state['completed_updates']
    assert [row['controls'] for row in lines]==list(range(24,limit['controls']+1,24))
    assert result['transitions']==limit['transitions']==plan['environments']*limit['controls']
    init = read(record/'support-credit-startup.json')
    assert init['applied_in_all_worlds'] and init['foot_sliding_friction']==plan['foot_sliding_friction']==.8
    assert init['actual_reset_controls']==0 and init['enabled']==(role=='candidate')
    source, saved = check_checkpoints(record,plan,warm,lines)
    check_draws(plan,source,saved,lines)
    trace,rollout = verify_trace(record,plan['environment'],limit['controls'],lines)
    clips = verify_clips(record,trace,state['clips'])
    assert len(clips)==1
    reward_path = report/'reward-audits.json'
    reward = read(reward_path)
    rows = reward['rows']
    assert [row['before_control_age'] for row in rows]==list(range(123,151))
    assert reward['weight']==plan['upright_credit_weight'] and all(row['active_reward']==(role=='candidate') for row in rows)
    assert rows[0]['actuation_intervals']==rows[-1]['actuation_intervals']==0
    assert [row['actuation_intervals'] for row in rows[1:-1]]==[5]*26
    assert all(0<row['positive']<=row['supine'] for row in rows[1:-1])
    assert all(row['rewarded']==(row['positive'] if role=='candidate' else 0) for row in rows)
    telemetry = read(report/'gpu-telemetry.json')['samples']
    assert telemetry and all(row['used_mib']/row['total_mib']<=plan['gpu_memory_fraction_limit'] for row in telemetry)
    auth = read(report/'authorization-monitor.json')
    assert auth['full_checks']>0 and auth['failure'] is None
    live = read(report/'live-readback.json')
    assert live['health']['run_id']==run and live['health']['source_status']=='training'
    assert live['health']['metric_rows']>0 and live['image_sha256']==live['health']['frame_source_sha256']
    observation = read(record/'observation.json')
    assert observation['completed_updates']==plan['updates_per_arm'] and observation['latest_clip']==state['clips'][0]['file']
    assert observation['checkpoint']=='model-final.pt' and len(observation['rows'])==len(lines)
    view=read(report/'final-view.json') if (report/'final-view.json').is_file() else finalize_view(report,run,result)
    assert view['status']=='saved' and view['source_sha256']==sha(record/'observation.json')
    exported=check_export(ROOT/'artifacts'/run/'policy.onnx',record,plan,saved)
    return dict(record=rel(record),report=rel(report),run_id=run,controls=state['controls'],update_count=len(lines),
        training_transitions=result['transitions'],batched_physical_substeps=state['substeps'],
        actor_gradient_steps=saved['actor_steps_this_run'],critic_gradient_steps=saved['critic_steps_this_run'],
        onnx=exported,rollout=rollout,videos=clips,final_view=view,live_readback_sha256=sha(report/'live-readback.json'),
        reward_sha256=sha(reward_path),gpu_peak_mib=max(row['used_mib'] for row in telemetry),
        median_gpu_busy_percent=median(row['gpu_percent'] for row in telemetry),
        net_transitions_per_second=result['throughput_transitions_per_second'],
        end_to_end_transitions_per_second=result['transitions']/state['elapsed_seconds']),lines


def main():
    torch.set_num_threads(4)
    project=read(PROJECT)
    task,node=project['staged_training_request'],project['goal_execution'][KEY]
    assert parent.allowed(project,STAGE) and not task['simulation_running'] and node['status']=='paired-evaluation-pending'
    plan=read(PLAN,node['protocol_sha256'])
    original.sources_checked(plan)
    matched=read(ROOT/node['zero'],node['zero_sha256'])
    assert matched['protocol_sha256']==node['protocol_sha256']
    warm=checkpoint(ROOT/plan['models']['checkpoint'])
    arms,lines={},{}
    for role in ROLES:
        arms[role],lines[role]=check_arm(role,plan,node,warm,matched)
    for control,candidate in zip(*(lines[role] for role in ROLES),strict=True):
        for name in ('anchor_batch_sha256','retention_batch_sha256'):
            assert candidate[name]==control[name]
    difference=compare_training_release([ROOT/arms[role]['record'] for role in ROLES])
    output=OUT/'training-verification.json'
    assert not output.exists() and not torch.cuda.is_initialized()
    write(output,dict(status='passed-matched-training-artifacts',protocol_sha256=sha(PLAN),arms=arms,
                       matched_cuda_reset=True,actual_training_release_difference=difference,matched_replay_draws=True,
                       physical_substeps_in_verifier=0,learning_updates_in_verifier=0,heldout_used=False))
    node.update(status='training-verified-action-comparison-pending',verification=rel(output),verification_sha256=sha(output))
    selected(project,STAGE,PLAN,STAGE+'-verified',output,
             'Compare the parent, reward control and transfer reward from the same actual release under the original gates.')
    write(PROJECT,project)
    print('SUPPORT_PPO_TRAINING_VERIFIED',sha(output),flush=True)


if __name__=='__main__':
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
