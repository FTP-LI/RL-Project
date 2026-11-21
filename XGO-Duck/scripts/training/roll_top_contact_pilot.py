"""Bounded visible roll adaptation; source reset is distinct from model HOME.

Both arms import the same official actor, freeze unknown normalization history,
fit a fresh critic for14updates, then cautiously update actor for36. Only the
non-top head-pivot reward differs. Sparse actual clips never advance physics.
Roll quality checks occur late in the episode, not during the intended flip.
"""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import shutil
import sys
import time
import numpy as np
import torch
from training_watch import ROOT,read,Snapshot,SparseVideo
from record_io import write
from guarded_training import sha
from metadata_actor_import import import_actor
from roll_top_contact import adapt

WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
sys.path.insert(0,str(ROOT/'scripts/validation'))


def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    import onnx
    from display_preferences import prepare_display
    assert plan['status']=='frozen' and plan['num_envs']==1024 and plan['updates_each']==50
    assert plan['critic_only_updates']==14 and plan['learning_rate']==2e-5 and plan['initial_std']==.05
    model=ROOT/plan['model'];assert sha(model)==plan['model_sha256']
    assert sha(ROOT/'records/reports/roll-top-contact/input-checks.json')==plan['input_check_sha256']
    metadata={x.key:x.value for x in onnx.load(model).metadata_props}
    cfg=TrainConfig.from_task(plan['task']);cfg=replace(cfg,env=adapt(cfg.env,metadata,plan['arm']=='candidate'))
    cfg.env.scene.num_envs=cfg.env.scene.terrain.num_envs=plan['num_envs'];cfg.env.seed=cfg.agent.seed=plan['seed']
    cfg.agent.run_name=plan['run_id'];cfg.agent.max_iterations=50;cfg.agent.save_interval=10**9
    cfg.agent.logger='tensorboard';cfg.agent.upload_model=False;cfg.agent.resume=False
    cfg.agent.actor.distribution_cfg['init_std']=.05
    cfg.agent.algorithm.learning_rate=2e-5;cfg.agent.algorithm.schedule='fixed';cfg.agent.algorithm.entropy_coef=0.
    cfg,display,_=prepare_display(cfg,sparse_snapshots=True)
    display.update(mode='periodic-actual-roll-video',video_interval_wall_seconds=180,video_duration_simulation_seconds=3)
    assert cfg.env.decimation*cfg.env.sim.mujoco.timestep==.02 and cfg.agent.num_steps_per_env==24
    assert not torch.cuda.is_initialized()
    return cfg,display,metadata


def roll_score(env,nominal_height):
    """Late-episode tripwire only; independent9s evaluation remains mandatory."""
    robot=env.scene['robot'];age=env.episode_length_buf*env.step_dt
    mask=(age>=6.)&(age<=6.9);count=int(mask.sum().item())
    if count<32:return dict(eligible=count,qualified_stable_fraction=None)
    q=robot.data.root_link_quat_w;tilt=torch.acos(torch.clamp(1-2*(q[:,1]**2+q[:,2]**2),-1,1))
    feet=env.scene['feet_ground_contact'].data.found.reshape(env.num_envs,2,-1).any(dim=2).all(dim=1)
    qualified=(env._roulade_max>=np.deg2rad(330.))&env._roulade_head_latch
    stable=(abs(robot.data.root_link_pos_w[:,2]-nominal_height)<=.04)&(tilt<=np.deg2rad(20.))&feet
    stable&=torch.linalg.vector_norm(robot.data.root_link_ang_vel_b,dim=1)<=1.
    return dict(eligible=count,qualified_fraction=float(qualified[mask].float().mean().item()),
        qualified_stable_fraction=float((qualified&stable)[mask].float().mean().item()),
        mean_episode_age=float(age[mask].mean().item()))


def runner_class(base,plan,record,metadata):
    class RollPilot(base):
        def __init__(self,*args,**kwargs):
            base.__init__(self,*args,**kwargs)
            from mjlab.utils.torch import configure_torch_backends
            configure_torch_backends(allow_tf32=False)
            imported=import_actor(self.alg.actor,ROOT/plan['model'],plan['model_sha256'])
            assert not self.alg.optimizer.state and self.env.unwrapped.common_step_counter==0
            self.record=record;self.plan=plan;self.completed=0;self.started=time.perf_counter();self.anchor=None
            write(record/'runner.json',dict(log_dir=str(Path(self.logger.log_dir).resolve())))
            self.initial_actor={k:v.detach().cpu().clone() for k,v in self.alg.actor.state_dict().items()}
            self._live_check(imported)
            for parameter in self.alg.actor.parameters():parameter.requires_grad_(False)
            self.current_learning_iteration=-1
            self.save(str(Path(self.logger.log_dir)/'model_init.pt'))
            self.current_learning_iteration=0
            self.snapshot=Snapshot(self.env,record);self.snapshot.cam.distance=1.2
            self.video=SparseVideo(self.snapshot,dict(video_fps=25,video_seconds=3.,visual_interval_seconds=180),record)
            self.old_step=self.env.step;self.step_calls=0
            def observed_step(actions):
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite policy action; stop roll pilot')
                result=self.old_step(actions);self.step_calls+=1
                if self.step_calls==1:
                    mask=~result[2].bool()
                    torch.testing.assert_close(result[0]['actor'][mask,34:48],actions[mask],atol=1e-6,rtol=0)
                    np.savez_compressed(record/'first-actions.npz',actions=actions.detach().cpu().numpy(),
                        next_observations=result[0]['actor'].detach().cpu().numpy())
                    if plan['arm']=='candidate':
                        with np.load(ROOT/'records/experiments'/plan['matching_control_run']/'first-actions.npz') as prior:
                            np.testing.assert_allclose(actions.detach().cpu().numpy(),prior['actions'],atol=1e-7,rtol=0)
                        write(record/'matched-first-actions.json',dict(status='passed',sampled_actions_match=True,physical_steps=1))
                    write(record/'first-step-check.json',dict(status='passed',raw_action_history_matches=True,control_steps=1))
                if self.step_calls%24==0:
                    if not torch.isfinite(result[0]['actor']).all():raise RuntimeError('Nonfinite roll observation')
                    if not torch.isfinite(result[1]).all():raise RuntimeError('Nonfinite roll reward')
                self.video.sample();return result
            self.env.step=observed_step;self.video.sample()
            self._publish('training')

        def _live_check(self,imported):
            import onnxruntime as ort
            env=self.env.unwrapped;names=metadata['joint_names'].split(',')
            action=env.action_manager.get_term('joint_pos');assert list(action.target_names)==names
            offset=action._offset.detach().cpu().numpy();home=np.asarray(metadata['default_joint_pos'].split(','),np.float32)
            np.testing.assert_allclose(offset,np.broadcast_to(home,offset.shape),atol=1e-7,rtol=0)
            obs=self.env.get_observations()[:32]
            np.testing.assert_allclose(obs['actor'][:,34:48].cpu().numpy(),0.,atol=1e-7,rtol=0)
            joint_ids,_=env.scene['robot'].find_joints(names,preserve_order=True)
            default=env.scene['robot'].data.default_joint_pos[:,joint_ids].cpu().numpy()
            np.testing.assert_allclose(default,np.broadcast_to(plan['simulation_home'],default.shape),atol=1e-7,rtol=0)
            with torch.no_grad():actual=self.alg.actor(obs).detach().cpu().numpy()
            options=ort.SessionOptions();options.intra_op_num_threads=1
            session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
            expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs['actor'].cpu().numpy()])
            np.savez_compressed(record/'live-import-parity.npz',observations=obs['actor'].cpu().numpy(),torch_actions=actual,onnx_actions=expected,
                initial_qpos=env.sim.data.qpos.detach().cpu().numpy(),initial_qvel=env.sim.data.qvel.detach().cpu().numpy())
            passed=bool(np.allclose(actual,expected,atol=2e-5,rtol=2e-5))
            write(record/'live-import-parity.json',dict(status='passed' if passed else 'failed',maximum_error=float(abs(actual-expected).max()),
                physical_steps=int(env.common_step_counter),action_home_matches=True,source_reset_home_matches=True,
                initial_raw_history_zero=True,imported=imported))
            assert passed and env.common_step_counter==0,'No physics/learning allowed before live actor parity'
            from tensordict import TensorDict
            parent=read(ROOT/'records/reports/roll-action-memory/startup-recovery/control/progress.json')
            source=Path(parent['path'])/'forward-roll-trace.npz'
            assert sha(source)==plan['drift_probe_trace_sha256']
            with np.load(source) as saved:
                indices=[1,16,31,46,61,76,151,301,450]
                probe=saved['observations'][indices].reshape(-1,61).copy()
                target=saved['raw_policy_actions'][indices].reshape(-1,14).copy()
            self.probe=TensorDict({'actor':torch.as_tensor(probe,device=env.device)},batch_size=[len(probe)])
            self.teacher_probe=torch.as_tensor(target,device=env.device)
            with torch.no_grad():initial_probe=self.alg.actor(self.probe)
            torch.testing.assert_close(initial_probe,self.teacher_probe,atol=2e-5,rtol=2e-5)
            if plan['arm']=='candidate':
                other=ROOT/'records/experiments'/plan['matching_control_run']
                with np.load(other/'live-import-parity.npz') as prior:
                    np.testing.assert_allclose(obs['actor'].cpu().numpy(),prior['observations'],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(env.sim.data.qpos.cpu().numpy(),prior['initial_qpos'],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(env.sim.data.qvel.cpu().numpy(),prior['initial_qvel'],atol=1e-7,rtol=0)
                old_directory=Path(read(other/'runner.json')['log_dir'])
                saved=torch.load(old_directory/'model_init.pt',map_location='cpu',weights_only=True)
                for role,network in (('actor',self.alg.actor),('critic',self.alg.critic)):
                    assert all(torch.equal(v.detach().cpu(),saved[role+'_state_dict'][k]) for k,v in network.state_dict().items())
                write(record/'matched-start.json',dict(status='passed',control_run=plan['matching_control_run'],
                    initial_actor_critic_exact=True,initial_observations_and_physical_state_match=True,physical_steps=0))

        def save(self,path,infos=None):
            from mjlab.rl.runner import MjlabOnPolicyRunner
            infos={**(infos or {}),'roll_pilot':dict(actor_normalizer_frozen=True,model_home=metadata['default_joint_pos'],
                source_reset_preserved=True,completed_updates=getattr(self,'completed',0),critic_only_updates=14)}
            target=Path(path);temporary=target.with_suffix('.partial')
            # Retain complete state but avoid the task runner's per-save ONNX.
            MjlabOnPolicyRunner.save(self,str(temporary),infos)
            saved=torch.load(temporary,map_location='cpu',weights_only=True)
            def finite(value):
                if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
                if isinstance(value,dict):return all(finite(v) for v in value.values())
                if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
                return True
            assert finite(saved),'Invalid checkpoint retained as .partial; last complete checkpoint unchanged'
            temporary.replace(target)
            write(record/'last-checkpoint.json',dict(checkpoint=str(Path(path).resolve()),sha256=sha(path),
                complete_update=True,completed_updates=getattr(self,'completed',0),common_step_counter=int(self.env.unwrapped.common_step_counter)))

        def _publish(self,status,error=None):
            write(record/'state.json',dict(status=status,run_id=plan['run_id'],skill='roll',completed_updates=self.completed,
                completed_updates_this_run=self.completed,planned_updates=50,num_envs=1024,step_count=int(self.env.unwrapped.common_step_counter),
                learning_seconds=time.perf_counter()-self.started,clips=self.video.clips if hasattr(self,'video') else [],
                images=[self.video.image] if hasattr(self,'video') and self.video.image else [],error=error,
                phase_monitor_baseline=self.anchor,automatic_retry=False,hardware_io=False))

        def learn(self,*args,**kwargs):
            original=self.logger.log
            def log(*a,**kw):
                self.completed=kw['it']+1
                def stop_after_complete(message):
                    prior=self.current_learning_iteration;self.current_learning_iteration=kw['it']
                    self.save(str(Path(self.logger.log_dir)/f'model_{kw["it"]}.pt'),dict(stop_reason=message))
                    self.current_learning_iteration=prior
                    raise RuntimeError(message)
                for net in (self.alg.actor,self.alg.critic):
                    assert all(torch.isfinite(v).all() for v in net.state_dict().values()),'Nonfinite learned state'
                with torch.no_grad():deviation=self.alg.actor(self.probe)-self.teacher_probe
                drift=dict(completed_updates=self.completed,rms=float(deviation.square().mean().sqrt().item()),maximum=float(deviation.abs().max().item()))
                with (record/'policy-drift.jsonl').open('a') as stream:stream.write(json.dumps(drift)+'\n')
                if drift['rms']>.15 or drift['maximum']>.6:stop_after_complete('Policy drift exceeds fixed roll-pilot limits')
                if self.completed in (14,28,42):
                    score=roll_score(self.env.unwrapped,plan['online_nominal_height_m']);write(record/f'phase-watch-{self.completed}.json',score)
                    if self.completed==14:
                        assert all(torch.equal(v,self.alg.actor.state_dict()[k].detach().cpu()) for k,v in self.initial_actor.items())
                        if score['qualified_stable_fraction'] is None or score['qualified_stable_fraction']<.3:
                            stop_after_complete('Fixed teacher insufficient in training conditions')
                        self.anchor=score['qualified_stable_fraction']
                        write(record/'critic-warmup-check.json',dict(status='passed',completed_updates=14,actor_unchanged=True,score=score))
                    elif score['qualified_stable_fraction'] is not None:
                        if score['qualified_stable_fraction']<max(.1,self.anchor-.25):
                            stop_after_complete('Roll completion/landing degraded: stop and review')
                saved_iteration=self.current_learning_iteration;self.current_learning_iteration=kw['it']
                if self.completed in (14,50) or read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False):
                    self.save(str(Path(self.logger.log_dir)/f'model_{kw["it"]}.pt'))
                self.current_learning_iteration=saved_iteration
                self._publish('training')
                original(*a,**kw)
                if self.completed==14:
                    for parameter in self.alg.actor.parameters():parameter.requires_grad_(True)
                    print('ROLL_PILOT: critic-only14complete; actor enabled, fixed small learning rate',flush=True)
                assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False),'Stop requested at complete update'
            self.logger.log=log
            kwargs['init_at_random_ep_len']=False
            try:
                base.learn(self,*args,**kwargs)
                assert self.completed==50
                self._export()
                status='budget-complete';error=None
            except BaseException as exc:status='stopped-review-required';error=repr(exc);raise
            finally:
                self.logger.log=original;self.env.step=self.old_step;self.video.finish(complete=False);self.snapshot.close()
                self._publish(status,error)

        def _export(self):
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
            artifact=ROOT/'artifacts'/plan['run_id'];artifact.mkdir(exist_ok=False)
            self.export_policy_to_onnx(str(artifact),'policy.onnx')
            values=get_base_metadata(self.env.unwrapped,plan['run_id'])
            values['default_joint_pos']=metadata['default_joint_pos'];values['joint_names']=metadata['joint_names'];values['action_scale']='1.0'
            attach_metadata_to_onnx(str(artifact/'policy.onnx'),values)
            last=read(record/'last-checkpoint.json')
            write(artifact/'manifest.json',dict(status='exported-awaiting-independent-check',run_id=plan['run_id'],
                model_sha256=sha(artifact/'policy.onnx'),model_home_from_metadata=True,source_reset_preserved=True,
                checkpoint=last['checkpoint'],checkpoint_sha256=last['sha256'],
                completed_updates=50,critic_only_updates=14,actor_updates=36,hardware_deployment_ready=False))
    return RollPilot


def run(planfile):
    plan=read(Path(planfile));assert os.isatty(1),'Use one visible VS Code terminal'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage'] in (
        'S4-top-contact-pilot', 'S6-roll-airborne-support-pilot')
    for source,digest in plan['source_sha256'].items():assert sha(ROOT/source)==digest,source
    for source,digest in plan['training_source_sha256'].items():assert sha(WORK/source)==digest,source
    for source,digest in plan['environment_source_sha256'].items():assert sha(Path(source))==digest,source
    cfg,display,metadata=prepare(plan)
    record=ROOT/'records/experiments'/plan['run_id'];record.mkdir(exist_ok=False)
    write(record/'protocol.json',plan);write(record/'display-state.json',display)
    source_dir=record/'sources';source_dir.mkdir()
    for source in plan['source_sha256']:shutil.copy2(ROOT/source,source_dir/Path(source).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(planfile)))
    write(record/'run-event.json',dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        foreground_tty=os.isatty(1),protocol=str(planfile),protocol_sha256=sha(planfile)))
    write(record/'state.json',dict(status='starting',run_id=plan['run_id'],completed_updates=0,clips=[]))
    from evaluate_standup_reference import Tee
    with (record/'training.log').open('a',buffering=1) as output:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,output),Tee(olderr,output)
        try:
            from mjlab.scripts import train
            base=train.load_runner_cls(plan['task']);train.load_runner_cls=lambda task:runner_class(base,plan,record,metadata)
            train.launch_training(plan['task'],cfg)
        except BaseException as error:
            state=read(record/'state.json');state.update(status='stopped-review-required',error=repr(error))
            write(record/'state.json',state);raise
        finally:sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('protocol');args=parser.parse_args();run(args.protocol)
