"""Bounded PPO comparison: instantaneous versus temporally averaged planar reward.

Reuse the verified official bootstrap checkpoint and observer/save/stop loop.
Both arms compute the same world-frame EMA and reset it in the same event;
only the scalar returned to the linear-velocity reward changes. The actor's
61 inputs, inference graph, raw vertical penalty and all other terms stay the
same. Reliability checks: separate CPU reward tests, frozen config comparison,
live matched-start packet, exact parent tensors, checkpoint/export verification.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import torch
from training_watch import ROOT,CONFIG,read,run_name
from guarded_training import guarded_class,sha
from official_bootstrap import prepare as prepare_bootstrap
from display_preferences import save_model_metadata
from record_io import write
from planar_velocity_reward import reward,reset_statistic,STATE

PAIR=ROOT/'configs/experiments/planar_filter_pair.json'
OUT=ROOT/'records/reports/planar-filter'


def prepare(plan):
    from mjlab.managers import EventTermCfg
    assert plan['status']=='frozen' and plan['additional_updates']==50 and not plan['automatic_extension']
    assert plan['use_filtered_xy']==(plan['arm']=='candidate') and plan['tau_seconds']==.2
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert saved['iter']==49 and saved['infos']['env_state']['common_step_counter']==1200
    assert int(saved['actor_state_dict']['obs_normalizer.count'])==0
    cfg,display,_=prepare_bootstrap(read(ROOT/'configs/experiments/official_actor_bootstrap.json'))
    term=cfg.env.rewards['track_linear_velocity']
    assert term.weight==3. and term.params['std']==.25 and term.params['command_name']=='twist'
    term.func=reward
    term.params.update(tau_seconds=plan['tau_seconds'],use_filtered_xy=plan['use_filtered_xy'])
    assert 'reset_planar_velocity_statistic' not in cfg.env.events
    cfg.env.events['reset_planar_velocity_statistic']=EventTermCfg(func=reset_statistic,mode='reset')
    cfg.agent.run_name=run_name('walk','planar-filter',plan['arm'],42,plan['stamp'])
    cfg.agent.max_iterations=50
    resume=dict(saved_iteration_label=49,next_iteration_label=50,common_step_counter=1200,
        num_steps_per_env=24,restored_learning_rate=.0001,remaining_updates=50,total_updates=100,final_iteration_label=99)
    assert not torch.cuda.is_initialized()
    return cfg,display,resume


def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('control','candidate'));args=p.parse_args()
    assert os.isatty(1),'Use the existing visible shared VS Code terminal'
    pair=read(PAIR);plan_path=ROOT/pair['arms'][args.arm];plan=read(plan_path)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PAIR)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S2a-planar-filter-pair'
    cfg,display,resume=prepare(plan)
    import training_watch,guarded_training
    training_watch.write=guarded_training.write=write
    record=ROOT/'records/experiments'/cfg.agent.run_name;record.mkdir(exist_ok=False)
    shutil.copy2(plan_path,record/'protocol.json')
    for name,value in [('resume-plan.json',resume),('display-state.json',display),('watch.json',read(CONFIG))]:write(record/name,value)
    sources=record/'sources';sources.mkdir()
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,sources/Path(name).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(plan_path),protocol_sha256=sha(plan_path)))
    class Tee:
        def __init__(self,stream,file):self.stream,self.file=stream,file
        def __getattr__(self,name):return getattr(self.stream,name)
        def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush();return len(text)
        def flush(self):self.stream.flush();self.file.flush()
    log=(record/'training.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    from mjlab.scripts import train
    from mjlab.utils.torch import configure_torch_backends
    base=train.load_runner_cls(plan['task'])
    observed=guarded_class(base,dict(change='planar-filter',parent_checkpoint=plan['checkpoint'],visuals_enabled=True),
                           read(CONFIG),resume,record)

    class FilterRunner(observed):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            configure_torch_backends(allow_tf32=False);self.alg.actor.obs_normalizer.until=0
            parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
            for role in ('actor','critic'):
                actual=getattr(self.alg,role).state_dict()
                assert all(torch.equal(v,actual[k].detach().cpu()) for k,v in parent[role+'_state_dict'].items())
            save_model_metadata(self.env,record)
            write(record/'loaded-tensors-check.json',dict(status='passed',actor_and_critic_exact=True,actor_normalizer_frozen=True))

        def save(self,path,infos=None):
            statistic=getattr(self.env.unwrapped,STATE,None)
            super().save(path,{**(infos or {}),'bootstrap':dict(actor_normalizer_frozen=True,critic_only_updates=10),
                'planar_filter':dict(protocol_sha256=sha(plan_path),tau_seconds=.2,use_filtered_xy=plan['use_filtered_xy'],
                    filter_updates=statistic.updates if statistic is not None else 0,
                    state_reinitialized_on_episode_reset=True)})

        def learn(self,*a,**kw):
            from head_command_start import capture
            from hip_limit_repair import capture_start
            original_act=self.alg.act;original_log=self.logger.log;first=True;samples=[]
            def act(obs):
                nonlocal first
                packet=capture(self,obs)[0] if first else None
                action=original_act(obs)
                if first:capture_start(self,obs,action,packet,record,plan);first=False
                return action
            def log(*args,**kwargs):
                statistic=getattr(self.env.unwrapped,STATE)
                row=dict(iteration=kwargs['it'],common_step=int(self.env.unwrapped.common_step_counter),
                    filter_updates=statistic.updates,reset_calls=statistic.reset_calls,
                    raw_reward_mean=float(statistic.last_raw_reward.mean()),
                    filtered_reward_mean=float(statistic.last_filtered_reward.mean()),
                    raw_velocity_mean=statistic.last_raw_velocity.mean(dim=0).cpu().tolist(),
                    filtered_velocity_mean=statistic.last_filtered_velocity.mean(dim=0).cpu().tolist(),
                    scope='Last reward-time batch of this update; not an episode or full-update performance average')
                samples.append(row);write(record/'reward-statistic-snapshots.json',samples)
                original_log(*args,**kwargs)
            self.alg.act,self.logger.log=act,log
            try:super().learn(*a,**kw)
            finally:self.alg.act,self.logger.log=original_act,original_log
            if read(record/'state.json')['status']!='budget-complete':return
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
            artifact=ROOT/'artifacts'/record.name;artifact.mkdir(exist_ok=False)
            self.export_policy_to_onnx(str(artifact),'policy.onnx')
            attach_metadata_to_onnx(str(artifact/'policy.onnx'),get_base_metadata(self.env.unwrapped,record.name))
            last=read(record/'last-checkpoint.json');saved=Path(last['checkpoint'])
            recovery=ROOT/'artifacts/recovery'/record.name;recovery.mkdir(exist_ok=False)
            shutil.copy2(saved,recovery/saved.name);assert sha(recovery/saved.name)==last['checkpoint_sha256']
            write(artifact/'manifest.json',dict(run_id=record.name,checkpoint=str(saved),checkpoint_sha256=last['checkpoint_sha256'],
                model_sha256=sha(artifact/'policy.onnx'),protocol_sha256=sha(plan_path),additional_updates=50,
                status='exported-awaiting-independent-checks',hardware_deployment_ready=False,
                inference_requires_velocity_filter=False))

    train.load_runner_cls=lambda task:FilterRunner
    train.launch_training(plan['task'],cfg)
    state=read(record/'state.json');print('PLANAR_FILTER_TRAINING_EXIT:',state['status'],flush=True)
    if state['status']!='budget-complete':raise SystemExit(42)


if __name__=='__main__':main()
