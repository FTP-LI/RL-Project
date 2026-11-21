"""Validate the exact live-view recovery inputs with the real CPU loader.

No environment, simulation integration, inference or optimizer step is used.
Checks preserve weights/Adam/normalizers and the remaining cumulative budget;
actual rendering and gait require separate evidence from the visible run.
"""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from live_resume import prepare
from profile_components import read, sha, write
from resume_state import prepare_loaded_runner
from inspect_resume_checkpoint import assert_same


def main():
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.rl.runner import MjlabOnPolicyRunner
    torch.set_num_threads(1)
    cfg,protocol,plan,evidence=prepare('baseline_live_check_s42')
    checkpoint=Path(plan['checkpoint'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    agent=yaml.load((checkpoint.parent/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
    observations=TensorDict({k:torch.zeros(1,saved[k+'_state_dict']['mlp.0.weight'].shape[1])
                             for k in ('actor','critic')},batch_size=[1])
    models={}
    for key,outputs in (('actor',14),('critic',1)):
        c=agent[key]
        distribution=copy.deepcopy(c['distribution_cfg']) if key=='actor' else None
        if distribution: distribution['init_std']=float(distribution['init_std'])
        models[key]=MLPModel(observations,agent['obs_groups'],key,outputs,
            hidden_dims=[int(n) for n in c['hidden_dims']],activation=c['activation'],
            obs_normalization=True,distribution_cfg=distribution)
    algorithm=PPO(models['actor'],models['critic'],storage=None,
                  learning_rate=float(agent['algorithm']['learning_rate']),device='cpu')
    base=load_runner_cls(plan['task'])
    assert base.load is MjlabOnPolicyRunner.load
    runner=base.__new__(base)
    runner.alg=algorithm
    runner.cfg={'num_steps_per_env':24}
    runner.current_learning_iteration=0
    runner.env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    runner.load(str(checkpoint),map_location='cpu')
    correction=prepare_loaded_runner(runner,plan)
    assert runner.current_learning_iteration==plan['next_iteration_label']
    assert runner.current_learning_iteration+cfg.agent.max_iterations==7500
    assert algorithm.learning_rate==plan['restored_learning_rate']
    assert runner.env.unwrapped.common_step_counter==plan['common_step_counter']
    assert {int(v['step'].item()) for v in algorithm.optimizer.state.values()}=={plan['completed_updates']*20}
    for key in ('actor','critic'):
        assert_same(getattr(algorithm,key).state_dict(),saved[key+'_state_dict'])
    assert_same(algorithm.optimizer.state_dict(),saved['optimizer_state_dict'])
    assert plan['parent_observation_count']+plan['remaining_updates']*24*4096==368640000
    viewer_check=read(ROOT/'records/reports/live-view-entry-check/gui-check.json')
    assert viewer_check['status']=='passed' and viewer_check['gui_open_and_sync_verified']
    assert viewer_check['observer_sha256']==sha(ROOT/'scripts/training/live_observer.py')
    assert (ROOT/'records/reports/live-view-entry-check/gui-probe-exit-code.txt').read_text(encoding='utf-8-sig').strip()=='0'
    tasks=read(ROOT/'.vscode/tasks.json')
    task=next(t for t in tasks['tasks'] if t['label'].startswith('XGO-Duck: 25 '))
    assert tasks['presentation']['panel']=='shared' and tasks['presentation']['reveal']=='always'
    assert task['runOptions']['instanceLimit']==1
    wrapper=ROOT/'scripts/training/experiments/resume_with_live_view.sh'
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    for path in ('training/visible_training.py','training/live_resume.py','training/live_observer.py',
                 'validation/finish_baseline_extension.py','validation/verify_resumed_progress.py'):
        ast.parse((ROOT/'scripts'/path).read_text())
    assert not torch.cuda.is_initialized() and sha(checkpoint)==plan['checkpoint_sha256']
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        plan=plan,correction=correction,source_sha256=evidence['source_sha256'],
        configuration_matches=evidence['configuration_matches'],
        unchanged_actor_critic_normalizers_and_adam=True,shared_terminal_verified=True,
        cuda_initialized=False,simulation_created=False,optimizer_steps_executed=0,
        training_started=False,actual_reset_and_rollout_verified=False)
    write(ROOT/'records/reports/live-view-entry-check/resume-check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
