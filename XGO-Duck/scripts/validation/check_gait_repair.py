"""CPU preflight for the matched repair pilot, with the actual loader/reward.

Verify identical parent state, declared configuration diff, reward values and
nonmutation on saved trajectories, scoped pause/budget gates, and task wiring.
No environment construction, inference, optimizer step or CUDA initialization.
"""
import ast
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
import gait_repair
from gait_rewards import action_target_envelope_cost
from profile_components import read,sha,write
from resume_state import prepare_loaded_runner
from inspect_resume_checkpoint import assert_same

def main():
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.envs.mdp.actions.actions import JointPositionAction
    torch.set_num_threads(1)
    cfg,protocol,plan,evidence=gait_repair.prepare('control','gait_repair_control_check_s42')
    candidate,_,other,other_evidence=gait_repair.prepare('target_envelope','gait_repair_target_envelope_check_s42')
    assert plan==other and evidence['source_sha256']==other_evidence['source_sha256']
    assert candidate.env.rewards['action_target_envelope'].func is action_target_envelope_cost
    assert candidate.env.rewards['action_target_envelope'].weight==-0.1
    assert cfg.video_interval==candidate.video_interval==1200
    assert cfg.agent.save_interval==candidate.agent.save_interval==50
    checkpoint=Path(plan['checkpoint']);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    agent=yaml.load((checkpoint.parent/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
    obs=TensorDict({key:torch.zeros(1,saved[key+'_state_dict']['mlp.0.weight'].shape[1]) for key in ('actor','critic')},batch_size=[1])
    models={}
    for key,outputs in (('actor',14),('critic',1)):
        c=agent[key];distribution=copy.deepcopy(c['distribution_cfg']) if key=='actor' else None
        if distribution:distribution['init_std']=float(distribution['init_std'])
        models[key]=MLPModel(obs,agent['obs_groups'],key,outputs,hidden_dims=[int(n) for n in c['hidden_dims']],
            activation=c['activation'],obs_normalization=True,distribution_cfg=distribution)
    algorithm=PPO(models['actor'],models['critic'],storage=None,learning_rate=float(agent['algorithm']['learning_rate']),device='cpu')
    base=load_runner_cls(plan['task']);runner=base.__new__(base);runner.alg=algorithm
    runner.cfg={'num_steps_per_env':24};runner.current_learning_iteration=0
    runner.env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    runner.load(str(checkpoint),map_location='cpu');correction=prepare_loaded_runner(runner,plan)
    assert runner.current_learning_iteration==5000 and cfg.agent.max_iterations==250
    assert runner.current_learning_iteration+cfg.agent.max_iterations==5250
    assert algorithm.learning_rate==plan['restored_learning_rate']
    assert {int(v['step'].item()) for v in algorithm.optimizer.state.values()}=={100000}
    for key in ('actor','critic'):assert_same(getattr(algorithm,key).state_dict(),saved[key+'_state_dict'])
    assert_same(algorithm.optimizer.state_dict(),saved['optimizer_state_dict'])
    reference=ROOT/'records/evaluations/custom_onnx_reference_20261101T144436_s1001'
    report=read(reference/'evaluation.json')
    meta=read(ROOT/'records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostics/20261101T090019523741Z/live-status.json')
    names=report['model_metadata']['joint_names'].split(',');assert names==meta['joint_names']
    limits=np.asarray(meta['joint_ranges_rad'],dtype=np.float32)
    lower=limits[:,0]-0.3;upper=limits[:,1]+0.3;roll=[names.index(s+'_hip_roll') for s in ('left','right')]
    lower[roll]=-np.deg2rad(10);upper[roll]=np.deg2rad(10)
    term=object.__new__(JointPositionAction);term._scale=1.0;term._target_ids=torch.arange(14);term._target_names=names
    hard=torch.tensor(limits)[None];original_limits=hard.clone()
    fixture=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name:term),scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos_limits=hard))})
    checks=[]
    for item in report['conditions']:
        with np.load(reference/(item['condition']+'-trace.npz')) as data:
            active=data['active'];targets=data['requested_joint_targets'][active]
            homes=np.broadcast_to(np.asarray(report['simulation_home'],dtype=np.float32),data['requested_joint_targets'].shape)[active].copy()
        term._offset=torch.tensor(homes);term._raw_actions=torch.tensor(targets-homes);before=term._raw_actions.clone()
        actual=action_target_envelope_cost(fixture).numpy()
        expected=(np.maximum(targets-upper,0)+np.maximum(lower-targets,0)).sum(axis=-1)
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=3e-6)
        assert torch.equal(term._raw_actions,before) and torch.equal(hard,original_limits)
        checks.append(dict(condition=item['condition'],frames=len(actual),mean_cost_rad=float(actual.mean()),
            projected_weighted_cost_before_dt=float(-0.1*actual.mean()),max_difference=float(np.abs(actual-expected).max())))
    original_read=gait_repair.read
    try:
        def paused(path):
            value=original_read(path)
            if path==ROOT/'project.json':value['formal_training']['status']='paused-by-user'
            return value
        gait_repair.read=paused
        try:gait_repair.require_pilot_enabled();raise AssertionError('Pause did not block pilot')
        except SystemExit:pass
        def wrong_budget(path):
            value=original_read(path)
            if path==gait_repair.PROTOCOL:value['additional_updates_per_arm']=2500
            return value
        gait_repair.read=wrong_budget
        rejected=False
        try:gait_repair.require_pilot_enabled()
        except AssertionError:rejected=True
        assert rejected,'Pilot entry accepted a long-training budget'
    finally:gait_repair.read=original_read
    task_config=read(ROOT/'.vscode/tasks.json');task=next(t for t in task_config['tasks'] if t['label'].startswith('XGO-Duck: 28 '))
    assert task_config['presentation']['panel']=='shared' and task_config['presentation']['reveal']=='always' and task['runOptions']['instanceLimit']==1
    subprocess.run(['bash','-n',str(ROOT/'scripts/training/experiments/gait_repair.sh')],check=True)
    for name,path in evidence['source_paths'].items():
        if str(path).endswith('.py'):ast.parse(Path(path).read_text())
    viewer=read(ROOT/'records/reports/live-view-entry-check/gui-check.json')
    assert viewer['status']=='passed' and viewer['observer_sha256']==sha(ROOT/'scripts/training/live_observer.py')
    assert not torch.cuda.is_initialized() and sha(checkpoint)==plan['checkpoint_sha256']
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(gait_repair.PROTOCOL),
        plan=plan,source_paths=evidence['source_paths'],source_sha256=evidence['source_sha256'],
        arm_changes={'control':evidence['configuration_changes'],'target_envelope':other_evidence['configuration_changes']},
        unchanged_loaded_actor_critic_normalizers_and_adam=True,correction=correction,
        reward_checks=checks,reward_does_not_mutate_actions_or_limits=True,
        user_pause_and_excess_budget_rejected=True,shared_visible_terminal_verified=True,
        simulation_created=False,cuda_initialized=False,training_started=False,optimizer_steps_executed=0)
    gait_repair.PREFLIGHT.parent.mkdir(parents=True,exist_ok=True);write(gait_repair.PREFLIGHT,result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_paths','source_sha256')},indent=2))

if __name__=='__main__':main()
