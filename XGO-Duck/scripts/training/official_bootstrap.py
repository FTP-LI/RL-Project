"""Visible, bounded initialization from the verified official actor.

This engineering pilot replaces the failed learned starting policy. It is
not a matched reward ablation or a continuation of official PPO state. Reuse
the verified observer/save/stop loop; initialize a fresh critic and Adam,
fit only the critic for 10 updates, then allow 40 cautious PPO updates.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from training_watch import ROOT, CONFIG, read, write, run_name, UpdateBoundary
from guarded_training import guarded_class, sha
from official_actor import import_official_actor, align_home, OFFICIAL_SHA
from display_preferences import prepare_display, save_model_metadata

WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PLAN = ROOT/'configs/experiments/official_actor_bootstrap.json'
OUT = ROOT/'records/reports/official-actor-bootstrap'


def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    import onnx
    from head_command_bounds import prepare_reachable_head_commands
    assert plan['status']=='frozen' and plan['updates']==50 and plan['critic_only_updates']==10
    assert plan['num_envs']==4096 and plan['seed']==42
    assert plan['initial_std']==.05 and plan['learning_rate']==1e-4
    assert plan['automatic_extension'] is False
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    model=ROOT/plan['official_model'];assert sha(model)==OFFICIAL_SHA
    checked=read(OUT/'actor-import-check.json')
    assert checked['status']=='passed' and sha(OUT/'actor-import-check.json')==plan['import_check_sha256']
    assert checked['importer_sha256']==sha(ROOT/'scripts/training/official_actor.py')
    assert sha(WORK/'uv.lock')==plan['lock_sha256']
    assert sha(WORK/'src/mjlab_microduck/tasks/__init__.py')==plan['task_source_sha256']
    metadata={m.key:m.value for m in onnx.load(model).metadata_props}
    cfg=TrainConfig.from_task(plan['task'])
    cfg=replace(cfg,env=align_home(cfg.env,metadata))
    cfg=replace(cfg,env=prepare_reachable_head_commands(cfg.env)[0])
    cfg.env.scene.num_envs=cfg.env.scene.terrain.num_envs=plan['num_envs']
    cfg.env.seed=cfg.agent.seed=plan['seed']
    cfg.agent.run_name=run_name('walk','official-actor-bootstrap','candidate',plan['seed'],plan['stamp'])
    cfg.agent.max_iterations=plan['updates'];cfg.agent.save_interval=10**9
    cfg.agent.logger='tensorboard';cfg.agent.upload_model=False;cfg.agent.resume=False
    cfg.agent.actor.distribution_cfg['init_std']=plan['initial_std']
    cfg.agent.algorithm.learning_rate=plan['learning_rate']
    cfg.agent.algorithm.schedule='fixed';cfg.agent.algorithm.entropy_coef=0.
    cfg,display,_=prepare_display(cfg,sparse_snapshots=True)
    display.update(mode='periodic-actual-state-video',video_enabled=True,video_interval_wall_seconds=180,
                   video_duration_simulation_seconds=3,recording_playback_fps=25)
    assert cfg.agent.clip_actions is None and cfg.env.actions['joint_pos'].clip is None
    assert cfg.agent.num_steps_per_env==24 and cfg.env.decimation*cfg.env.sim.mujoco.timestep==.02
    assert not torch.cuda.is_initialized()
    return cfg,display,metadata


def bootstrap_class(base,plan,cfg,record):
    # Only the monitoring methods are reused. The parent's loading constructor
    # is deliberately bypassed: no checkpoint or optimizer history is invented.
    monitor_plan=dict(change='official-actor-bootstrap',visuals_enabled=True,
                      parent_checkpoint=str(ROOT/plan['official_model']))
    resume=dict(next_iteration_label=0)
    watch=read(CONFIG)
    observed=guarded_class(base,monitor_plan,watch,resume,record)

    class Bootstrap(observed):
        def __init__(self,*args,**kwargs):
            base.__init__(self,*args,**kwargs)
            imported=import_official_actor(self.alg.actor,ROOT/plan['official_model'])
            assert not self.alg.optimizer.state
            assert self.env.unwrapped.common_step_counter==0
            self.guard_boundary=UpdateBoundary(watch,record)
            self.initial_actor={k:v.detach().cpu().clone() for k,v in self.alg.actor.state_dict().items()}
            for parameter in self.alg.actor.parameters():parameter.requires_grad_(False)
            self.current_learning_iteration=-1
            initial=Path(self.logger.log_dir).resolve()/'model_init.pt'
            self.save(str(initial),dict(initialization='official-actor-fresh-critic-and-optimizer'))
            self.current_learning_iteration=0
            monitor_plan['parent_checkpoint']=str(initial)
            save_model_metadata(self.env,record)
            write(record/'initialization.json',dict(**imported,checkpoint=str(initial),checkpoint_sha256=sha(initial),
                initial_std=plan['initial_std'],critic_only_updates=plan['critic_only_updates'],
                resumed_official_training=False,initialization_complete=True))
            write(record/'runner.json',dict(log_dir=str(Path(self.logger.log_dir).resolve())))
            # Check the actual freshly built environment's coordinates and
            # inference before any physical step or optimizer update.
            names=imported['metadata']['joint_names'].split(',')
            assert list(self.env.unwrapped.action_manager.get_term('joint_pos').target_names)==names
            expected=np.array(imported['metadata']['default_joint_pos'].split(','),dtype=np.float32)
            home=self.env.unwrapped.scene['robot'].data.default_joint_pos.detach().cpu().numpy()
            assert np.allclose(home,expected,atol=1e-7,rtol=0)
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=1
            session=ort.InferenceSession(str(ROOT/plan['official_model']),sess_options=options,providers=['CPUExecutionProvider'])
            obs=self.env.get_observations();sample=obs[:32]
            precision_before=torch.backends.cuda.matmul.fp32_precision
            with torch.no_grad(): fast=self.alg.actor(sample).detach().cpu().numpy()
            expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in sample['actor'].cpu().numpy()])
            # The framework enables TF32 by default. Keep the strict imported
            # model check and measure both paths on the SAME observations.
            # Full FP32 remains enabled for this explicitly revised pilot.
            from mjlab.utils.torch import configure_torch_backends
            configure_torch_backends(allow_tf32=False)
            with torch.no_grad(): actual=self.alg.actor(sample).detach().cpu().numpy()
            np.savez_compressed(record/'live-import-parity.npz',observations=sample['actor'].cpu().numpy(),
                default_precision_actions=fast,ieee_actions=actual,onnx_actions=expected)
            passed=bool(np.allclose(actual,expected,atol=2e-5,rtol=2e-5))
            write(record/'live-import-parity.json',dict(status='passed' if passed else 'failed',observations=32,
                max_absolute_error=float(np.abs(actual-expected).max()),actual_home_matches=True,
                precision_before=precision_before,training_matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                default_precision_max_error=float(np.abs(fast-expected).max()),
                default_precision_passed=bool(np.allclose(fast,expected,atol=2e-5,rtol=2e-5)),
                tolerance=dict(absolute=2e-5,relative=2e-5),
                physical_steps_before_check=int(self.env.unwrapped.common_step_counter)))
            assert passed,'Imported actor mismatch remains in full FP32; no learning allowed'

        def save(self,path,infos=None):
            # until is an attribute, not a state_dict buffer. Persist this
            # contract explicitly so a later reviewed resume can restore it.
            infos={**(infos or {}),'bootstrap':dict(actor_normalizer_frozen=True,
                critic_only_updates=plan['critic_only_updates'],protocol_sha256=sha(PLAN))}
            super().save(path,infos)

        def learn(self,*args,**kwargs):
            original=self.logger.log
            def progress(*a,**kw):
                completed=kw['it']+1
                if completed==plan['critic_only_updates']:
                    assert all(torch.equal(value,self.alg.actor.state_dict()[key].detach().cpu())
                               for key,value in self.initial_actor.items())
                    write(record/'critic-warmup-check.json',dict(status='passed',completed_updates=completed,
                        actor_unchanged=True,actor_normalizer_count=int(self.alg.actor.obs_normalizer.count)))
                    for parameter in self.alg.actor.parameters():parameter.requires_grad_(True)
                    print('BOOTSTRAP: critic warmup complete; actor updates now enabled',flush=True)
                original(*a,**kw)
            self.logger.log=progress
            try:super().learn(*args,**kwargs)
            finally:self.logger.log=original
            state=read(record/'state.json')
            if state['status']=='budget-complete':
                from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
                artifact=ROOT/'artifacts'/record.name;artifact.mkdir(exist_ok=False)
                self.export_policy_to_onnx(str(artifact),'policy.onnx')
                attach_metadata_to_onnx(str(artifact/'policy.onnx'),get_base_metadata(self.env.unwrapped,record.name))
                last=read(record/'last-checkpoint.json')
                recovery=ROOT/'artifacts/recovery'/record.name;recovery.mkdir(exist_ok=False)
                saved=Path(last['checkpoint']);shutil.copy2(saved,recovery/saved.name)
                assert sha(recovery/saved.name)==sha(saved)
                write(artifact/'manifest.json',dict(run_id=record.name,checkpoint=str(saved),checkpoint_sha256=sha(saved),
                    model_sha256=sha(artifact/'policy.onnx'),status='exported-awaiting-independent-parity-and-gait',
                    protocol_sha256=sha(PLAN),actual_updates=50,critic_only_updates=10,actor_updates=40,
                    hardware_deployment_ready=False))
    return Bootstrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args()
    plan=read(PLAN);cfg,display,metadata=prepare(plan)
    if args.preflight:
        result=dict(status='passed',protocol_sha256=sha(PLAN),run_id=cfg.agent.run_name,
            actual_training_home=cfg.env.scene.entities['robot'].init_state.joint_pos,
            actor_normalizer='frozen imported official statistics',critic='fresh',optimizer='fresh',
            total_updates=50,critic_only_updates=10,actor_updates=40,cuda_initialized=False)
        write(OUT/'preflight.json',result);print(json.dumps(result,indent=2));return
    assert os.isatty(1),'Use the single visible VS Code task terminal'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S2a-official-actor-bootstrap'
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    record=ROOT/'records/experiments'/cfg.agent.run_name;record.mkdir(exist_ok=False)
    write(record/'protocol.json',plan);write(record/'display-state.json',display)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(PLAN),protocol_sha256=sha(PLAN)))
    source=record/'sources';source.mkdir()
    for relative in plan['source_sha256']:shutil.copy2(ROOT/relative,source/Path(relative).name)
    class Tee:
        def __init__(self,stream,file):self.stream,self.file=stream,file
        def __getattr__(self,name):return getattr(self.stream,name)
        def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush();return len(text)
        def flush(self):self.stream.flush();self.file.flush()
    output=(record/'training.log').open('a',buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,output),Tee(sys.stderr,output)
    write(OUT/'progress.json',dict(status='training',run_id=record.name,record=str(record),protocol_sha256=sha(PLAN)))
    from mjlab.scripts import train
    base=train.load_runner_cls(plan['task'])
    train.load_runner_cls=lambda task:bootstrap_class(base,plan,cfg,record)
    try:
        train.launch_training(plan['task'],cfg)
        state=read(record/'state.json')
        write(OUT/'progress.json',dict(status=state['status'],run_id=record.name,record=str(record),
            completed_updates=state['completed_updates_this_run'],checkpoint=read(record/'last-checkpoint.json'),
            protocol_sha256=sha(PLAN),ended_at=datetime.now(timezone.utc).isoformat()))
        print('OFFICIAL_BOOTSTRAP_COMPLETE:',state['status'],'Review required; no extension.',flush=True)
        if state['status']!='budget-complete':raise SystemExit(42)
    except BaseException as error:
        progress=read(OUT/'progress.json');progress.update(error=repr(error))
        if progress['status']=='training':progress['status']='failed'
        write(OUT/'progress.json',progress);raise


if __name__=='__main__':main()
