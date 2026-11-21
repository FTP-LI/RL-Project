"""Verify the paired training artifacts and compare their actions from one release."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

from observed_getup_entry import finalize_view
import evaluate_nominal_low_friction_support_ppo as comparison
import run_nominal_low_friction_loaded_upright_ppo as training
from verify_getup_supine_training import verify_clips, verify_trace
from verify_nominal_supine_upright_band_training import checkpoint, check_checkpoints, check_draws, check_export, compare_training_release


ROOT,PROJECT,parent=training.ROOT,training.PROJECT,training.parent
read,rel,sha,write=training.read,training.rel,training.sha,training.write
KEY=training.KEY+'_action'
STAGE='nominal-low-friction-loaded-upright-action'
PLAN=training.PLAN.with_stem(KEY)
REPORT=training.OUT/'action-comparison'
ROLES=('parent-policy','original-reward','loaded-upright-reward')
MODEL_ROLES=dict(zip(ROLES,('getup-supine','getup-supine-control','getup-supine-candidate'),strict=True))
LABELS=dict(zip(('shared',*ROLES),('共同跌倒与释放','父策略','原奖励对照','承重后转正候选'),strict=True))


def verify_arm(role,plan,node,warm,zero):
    run=plan['runs'][role]
    report=training.OUT/role
    record=ROOT/'records/experiments'/run
    lim=plan['limit']
    status,state=read(report/'result.json'),read(record/'state.json')
    assert status['complete'] and node['results'][role]['complete']
    assert state['status']==status['status']=='collected-review-required'
    assert state['protocol_sha256']==node['protocol_sha256'] and state['run_id']==status['run_id']==run
    assert state['controls']==state['attempted_controls']==lim['controls']
    assert state['substeps']==state['attempted_substeps']==lim['substeps']==status['physics_substeps']
    assert state['terminal_resets']==1 and state['paired_zero_start_verified']
    assert sha(record/'matched-state.json')==zero['proof'][role]['matched_state_sha256']
    assert sha(record/'learning-metrics.jsonl')==status['metrics_sha256']
    assert sha(record/'model-final.pt')==status['checkpoint_sha256']
    rows=[json.loads(row) for row in (record/'learning-metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['controls'] for row in rows]==list(range(24,lim['controls']+1,24))
    assert len(rows)==plan['updates_per_arm']==status['updates']==state['completed_updates']
    assert status['transitions']==lim['transitions']==plan['environments']*lim['controls']
    init=read(record/'loaded-upright-startup.json')
    enabled=role=='candidate'
    assert init['initial_clock_empty'] and init['friction_matched_all_worlds'] and init['foot_sliding_friction']==.8
    assert init['controls']==0 and init['role']==role and init['enabled']==enabled
    source, trained=check_checkpoints(record,plan,warm,rows)
    check_draws(plan,source,trained,rows)
    trace,rollout=verify_trace(record,plan['environment'],lim['controls'],rows)
    clips=verify_clips(record,trace,state['clips'])
    assert len(clips)==1
    reward_path=report/'reward-audits.json'
    rewards=read(reward_path)
    audits=rewards['rows']
    first,last=[plan['term'][key] for key in ('first','last')]
    assert [a['before_control_age'] for a in audits]==list(range(first-1,last+1))
    assert rewards['weight']==plan['upright_credit_weight'] and all(a['active_reward']==enabled for a in audits)
    assert [a['actuation_intervals'] for a in audits]==[0]+[5]*(last-first)+[0]
    assert any(a['positive']>0 for a in audits) and any(a['negative']>0 for a in audits)
    assert all(a['rewarded']==(a['positive']+a['negative'] if enabled else 0) for a in audits)
    telemetry=read(report/'gpu-telemetry.json')['samples']
    assert telemetry and all(r['used_mib']/r['total_mib']<=plan['gpu_memory_fraction_limit'] for r in telemetry)
    guard=read(report/'authorization-monitor.json')
    assert guard['full_checks']>0 and guard['failure'] is None
    live=read(report/'live-readback.json')
    assert live['health']['run_id']==run and live['health']['source_status']=='training'
    assert live['health']['metric_rows']>0 and live['image_sha256']==live['health']['frame_source_sha256']
    observation=read(record/'observation.json')
    assert len(observation['rows'])==len(rows) and observation['completed_updates']==plan['updates_per_arm']
    assert observation['latest_clip']==state['clips'][0]['file'] and observation['checkpoint']=='model-final.pt'
    output=report/'final-view.json'
    view=read(output) if output.is_file() else finalize_view(report,run,status)
    assert view['status']=='saved' and view['source_sha256']==sha(record/'observation.json')
    model=check_export(ROOT/'artifacts'/run/'policy.onnx',record,plan,trained)
    return dict(run_id=run,record=rel(record),report=rel(report),controls=state['controls'],update_count=len(rows),
        training_transitions=status['transitions'],batched_physical_substeps=state['substeps'],onnx=model,rollout=rollout,
        videos=clips,final_view=view,live_readback_sha256=sha(report/'live-readback.json'),reward_sha256=sha(reward_path),
        actor_gradient_steps=trained['actor_steps_this_run'],critic_gradient_steps=trained['critic_steps_this_run'],
        gpu_peak_mib=max(r['used_mib'] for r in telemetry),median_gpu_busy_percent=median(r['gpu_percent'] for r in telemetry),
        net_transitions_per_second=status['throughput_transitions_per_second'],
        end_to_end_transitions_per_second=status['transitions']/state['elapsed_seconds']),rows


def verify_training():
    project=read(PROJECT)
    node=project['goal_execution'][training.KEY]
    assert parent.allowed(project,training.STAGE) and not project['staged_training_request']['simulation_running']
    assert node['status']=='paired-evaluation-pending'
    plan=read(training.PLAN,node['protocol_sha256'])
    training.original.sources_checked(plan)
    zero=read(ROOT/node['zero'],node['zero_sha256'])
    assert zero['protocol_sha256']==node['protocol_sha256'] and zero['physics_steps']==zero['learning_updates']==0
    torch.set_num_threads(4)
    warm=checkpoint(ROOT/plan['models']['checkpoint'])
    arms,lines={},{}
    for role in training.ROLES:
        arms[role],lines[role]=verify_arm(role,plan,node,warm,zero)
    for a,b in zip(*(lines[role] for role in training.ROLES),strict=True):
        for key in ('anchor_batch_sha256','retention_batch_sha256'):
            assert a[key]==b[key]
    release=compare_training_release([ROOT/arms[role]['record'] for role in training.ROLES])
    output=training.OUT/'training-verification.json'
    assert not output.exists() and not torch.cuda.is_initialized()
    write(output,dict(status='passed-matched-training-artifacts',protocol_sha256=node['protocol_sha256'],
        verification_script_sha256=sha(Path(__file__)),arms=arms,matched_cuda_reset=True,matched_replay_draws=True,
        actual_training_release_difference=release,physical_substeps_in_verifier=0,learning_updates_in_verifier=0,heldout_used=False))
    node.update(status='training-verified-action-comparison-pending',verification=rel(output),verification_sha256=sha(output))
    training.earlier.selected(project,training.STAGE,training.PLAN,training.STAGE+'-verified',output,
                             'Freeze the original scored low-friction shared release and test both arms and their parent.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_TRAINING_VERIFIED',sha(output),{r:arms[r]['training_transitions'] for r in training.ROLES},flush=True)


def freeze_action():
    project=read(PROJECT)
    task,goal=project['staged_training_request'],project['goal_execution']
    assert parent.allowed(project,training.STAGE) and not task['simulation_running'] and not torch.cuda.is_initialized()
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    pair=goal[training.KEY]
    product=ROOT/pair['verification']
    verified=read(product,pair['verification_sha256'])
    assert pair['status']=='training-verified-action-comparison-pending' and verified['status']=='passed-matched-training-artifacts'
    assert verified['matched_cuda_reset'] and verified['matched_replay_draws'] and not verified['heldout_used']
    assert len({row['training_transitions'] for row in verified['arms'].values()})==1
    reference=goal['nominal_low_friction_support_ppo_action']
    original_plan=ROOT/reference['protocol']
    plan=read(original_plan,reference['protocol_sha256'])
    d,p=plan['composition'],plan['environment']
    assert (p['seed'],d['worlds'],d['foot_friction'],d['release_control'],d['roll_start_control'])==(1257,896,.8,100,600)
    assert d['existing_low_friction_reserved_seed']==2246 and plan['roll_route']['standing_dwell_controls']==25
    assert plan['roll_route']['roll']['criteria']['tilt_max_deg']==15 and set(plan['seven_command']['minimum_by_command'].values())=={122}
    assert plan['low_friction_trial']['minimum_forward_gain']==2 and plan['low_friction_trial']['require_left_turn_nondecrease']
    paths=[Path(__file__),original_plan,product,training.PLAN,Path(comparison.__file__),Path(comparison.measured.__file__),
           Path(comparison.collector.__file__),Path(comparison.verifier.__file__)]
    for role,target in zip(training.ROLES,ROLES[1:],strict=True):
        artifact=verified['arms'][role]['onnx']
        model,save=(ROOT/artifact[k] for k in ('model','checkpoint'))
        assert sha(model)==artifact['model_sha256'] and sha(save)==artifact['checkpoint_sha256']
        plan['models'][MODEL_ROLES[target]]=dict(model=rel(model),model_sha256=sha(model),seed=p['seed'])
        plan['policy_home'][MODEL_ROLES[target]]=plan['policy_home'][MODEL_ROLES[ROLES[0]]]
        paths.extend((model,save))
    plan['references'].update({rel(path):sha(path) for path in paths})
    plan['upright_band_comparison'].update(roles=list(ROLES),model_by_role=MODEL_ROLES)
    plan['low_friction_trial']['training_verification']=rel(product)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run=f'duck-getup-loaded-upright-action-friction080-s1257-{stamp}'
    p['run_id']=run
    plan.update(stage=STAGE,record=f'records/evaluations/{run}')
    parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    observer=parent.preflight()
    write(PLAN,plan)
    proof=REPORT/'preflight.json'
    write(proof,dict(protocol_sha256=sha(PLAN),training_verification_sha256=pair['verification_sha256'],observer=observer,
        original_release=100,stop_control=600,maximum_transitions=plan['maximum_control_transitions'],
        reserved_low_friction_seed=2246,physical_steps=0,learning_updates=0,hardware_io=False))
    goal[KEY]=dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,preflight=rel(proof),hardware_io=False)
    training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',proof,
                             'Compare parent and both trained arms from the same release with the original standing gates.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_ACTION_FROZEN',sha(PLAN),run,flush=True)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('verify','freeze','sample','score'))
    action=cli.parse_args().action
    driver=comparison.driver
    registry=SimpleNamespace(KEY=training.KEY,selected=training.earlier.selected)
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(comparison,training=registry,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES,LABELS=LABELS),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=comparison.measured.collect),
          patch.multiple(parent,config=comparison.measured.friction.configured,PLAN=PLAN),
          patch.multiple(comparison.collector,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES,LABELS=LABELS),
          patch.multiple(comparison.verifier,KEY=KEY,PLAN=PLAN,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES),
          patch.object(comparison.collector.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'verify':verify_training,'freeze':freeze_action,'sample':driver.sample,'score':comparison.score}[action]()


if __name__=='__main__':
    main()
