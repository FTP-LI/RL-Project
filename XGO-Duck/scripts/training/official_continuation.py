"""Reviewed continuation of a verified official-actor initialization.

Preserve actor, critic, Adam, frozen actor statistics and environment counter.
The first bootstrap had 10 critic-only updates, so optimizer ages differ.
This adapter handles that declared history instead of pretending it is a
uniform-age legacy checkpoint. Every run has a new fixed budget and review.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import shutil
import sys
import torch

from training_watch import ROOT,CONFIG,read,write,run_name
from guarded_training import guarded_class,sha
from official_bootstrap import prepare as prepare_bootstrap
from display_preferences import save_model_metadata

PLAN=ROOT/'configs/experiments/official_continuation.json'


def prepare(plan):
    assert plan['status']=='frozen' and 1<=plan['additional_updates']<=250
    assert plan['automatic_extension'] is False
    review=read(ROOT/plan['gate_record'])
    assert sha(ROOT/plan['gate_record'])==plan['gate_sha256'] and review['reviewed_continuation_allowed']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    parent=Path(plan['checkpoint']);assert sha(parent)==plan['checkpoint_sha256']
    saved=torch.load(parent,map_location='cpu',weights_only=True)
    label=int(saved['iter']);completed=label+1;steps=int(saved['infos']['env_state']['common_step_counter'])
    assert steps==completed*24 and saved['infos']['bootstrap']['actor_normalizer_frozen']
    assert int(saved['actor_state_dict']['obs_normalizer.count'])==0
    assert int(saved['critic_state_dict']['obs_normalizer.count'])==completed*24*4096
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}=={completed*20,(completed-10)*20}
    assert all(float(x['lr'])==.0001 for x in saved['optimizer_state_dict']['param_groups'])
    cfg,display,_=prepare_bootstrap(read(ROOT/'configs/experiments/official_actor_bootstrap.json'))
    cfg.agent.run_name=run_name('walk','official-adaptation','candidate',42,plan['stamp'])
    cfg.agent.max_iterations=plan['additional_updates']
    resume=dict(saved_iteration_label=label,next_iteration_label=completed,common_step_counter=steps,
        num_steps_per_env=24,restored_learning_rate=.0001,remaining_updates=plan['additional_updates'],
        total_updates=completed+plan['additional_updates'],final_iteration_label=label+plan['additional_updates'])
    assert not torch.cuda.is_initialized()
    return cfg,display,resume


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    plan=read(PLAN);cfg,display,resume=prepare(plan)
    if args.preflight:
        output=dict(status='passed',protocol_sha256=sha(PLAN),run_id=cfg.agent.run_name,resume=resume,
                    cuda_initialized=False)
        write(ROOT/'records/reports/official-actor-bootstrap/continuation-preflight.json',output)
        print(json.dumps(output,indent=2));return
    assert os.isatty(1),'Use the existing visible VS Code terminal'
    stage=read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']=='S2a-official-continuation' and not stage.get('pause_requested',False)
    assert read(ROOT/'records/reports/official-actor-bootstrap/continuation-preflight.json')['protocol_sha256']==sha(PLAN)
    record=ROOT/'records/experiments'/cfg.agent.run_name;record.mkdir(exist_ok=False)
    shutil.copy2(PLAN,record/'protocol.json');write(record/'resume-plan.json',resume)
    write(record/'display-state.json',display);write(record/'watch.json',read(CONFIG))
    source=record/'sources';source.mkdir()
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,source/Path(name).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(PLAN),protocol_sha256=sha(PLAN)))
    class Tee:
        def __init__(self,stream,file):self.stream,self.file=stream,file
        def __getattr__(self,name):return getattr(self.stream,name)
        def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush();return len(text)
        def flush(self):self.stream.flush();self.file.flush()
    output=(record/'training.log').open('a',buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,output),Tee(sys.stderr,output)
    from mjlab.scripts import train
    from mjlab.utils.torch import configure_torch_backends
    base=train.load_runner_cls(plan['task'])
    observed=guarded_class(base,dict(change='official-adaptation',parent_checkpoint=plan['checkpoint'],visuals_enabled=True),
                           read(CONFIG),resume,record)
    class Continuation(observed):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            configure_torch_backends(allow_tf32=False)
            self.alg.actor.obs_normalizer.until=0
            saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
            for role in ('actor','critic'):
                current=getattr(self.alg,role).state_dict()
                assert all(torch.equal(v,current[k].detach().cpu()) for k,v in saved[role+'_state_dict'].items())
            save_model_metadata(self.env,record)
            write(record/'loaded-tensors-check.json',dict(status='passed',actor_and_critic_exact=True,
                actor_normalizer_frozen=True,matmul_precision=torch.backends.cuda.matmul.fp32_precision))
        def save(self,path,infos=None):
            super().save(path,{**(infos or {}),'bootstrap':dict(actor_normalizer_frozen=True,
                critic_only_updates=10,protocol_sha256=sha(PLAN))})
        def learn(self,*a,**kw):
            super().learn(*a,**kw)
            state=read(record/'state.json')
            if state['status']!='budget-complete':return
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
            artifact=ROOT/'artifacts'/record.name;artifact.mkdir(exist_ok=False)
            self.export_policy_to_onnx(str(artifact),'policy.onnx')
            attach_metadata_to_onnx(str(artifact/'policy.onnx'),get_base_metadata(self.env.unwrapped,record.name))
            last=read(record/'last-checkpoint.json');saved=Path(last['checkpoint'])
            recovery=ROOT/'artifacts/recovery'/record.name;recovery.mkdir(exist_ok=False)
            shutil.copy2(saved,recovery/saved.name);assert sha(recovery/saved.name)==sha(saved)
            write(artifact/'manifest.json',dict(run_id=record.name,model_sha256=sha(artifact/'policy.onnx'),
                checkpoint=str(saved),checkpoint_sha256=sha(saved),status='exported-awaiting-independent-checks',
                protocol_sha256=sha(PLAN),additional_updates=plan['additional_updates'],hardware_deployment_ready=False))
    train.load_runner_cls=lambda task:Continuation
    train.launch_training(plan['task'],cfg)
    state=read(record/'state.json')
    print('OFFICIAL_ADAPTATION_COMPLETE:',state['status'],'Review before any extension.',flush=True)
    if state['status']!='budget-complete':raise SystemExit(42)


if __name__=='__main__':main()
