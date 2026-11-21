"""One bounded collision-rule diagnosis; fixed official policy, no learning.

The shared full-contact plant can trap the folded head near the feet in saved
supine failures. Compare it to the source getup collision masks while keeping
the same shared robot, three CAD exclusions, actuator, IMU and reset recipe.
This relaxed mask arm is a causal diagnostic, never a deployment candidate.

Validation: compile both CPU scenes and allow only two mask fields to differ;
check both actual GPU zero-step scenes and paired reset/RNG/IMU snapshots before
any integration; retain original getup hold criteria, replay policy inputs and
decode full videos. Exactly32supine trials/arm,11s total, seed1010. No retry.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import numpy as np
import torch
import yaml
from mjlab.scene import Scene
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from common_plant_screen import prepare as common_prepare
from common_scene_contract import model_arrays,compare_arrays,interface_check
import evaluate_standup_reference as collector

PLAN=ROOT/'configs/experiments/getup_contact_isolation.json'
OUT=ROOT/'records/reports/getup-contact-isolation'
ARMS=('full','source-masks')
MASKS={'geom_contype','geom_conaffinity'}

def config(p):
    cfg=common_prepare('getup',p)
    if p['role']=='source-masks':
        old=collector_original_config(p)
        cfg.scene.entities['robot'].collisions=deepcopy(old.scene.entities['robot'].collisions)
    return cfg

collector_original_config=collector.config

def arrayfile(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}

def freeze():
    assert os.isatty(1) and not PLAN.exists() and not OUT.exists()
    prior=ROOT/'records/reports/common-plant-screen';assert read(prior/'progress.json')['status']=='completed-review-required'
    assert read(prior/'failure-review.json')['status']=='passed' and read(prior/'contact-geometry-review.json')['status']=='passed'
    old=read(ROOT/'configs/experiments/common_plant/getup.json')
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in old['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    sources=dict(old['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    plans={};models={};configs={}
    for arm in ARMS:
        p=deepcopy(old);p.update(role=arm,seed=1010,poses=['supine'],source_sha256=sources,
            stage='S5-getup-contact-isolation',scope='Supine collision-mask causal diagnosis; relaxed masks are not an accepted robot or deployment candidate')
        cfg=config(p);models[arm]=model_arrays(Scene(cfg.scene,device='cpu').compile())
        configs[arm]=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader);plans[arm]=p
    a,b=models.values();assert {k for k in a if not np.array_equal(a[k],b[k])}==MASKS
    assert len(a['exclude_signature'])==len(b['exclude_signature'])==3
    assert not torch.cuda.is_initialized()
    OUT.mkdir();
    for arm in ARMS:
        (OUT/arm/'startup').mkdir(parents=True)
        np.savez_compressed(OUT/arm/'cpu-scene-fields.npz',**models[arm])
    master=dict(status='frozen-before-simulation',stage='S5-getup-contact-isolation',
        frozen_at=datetime.now(timezone.utc).isoformat(),arms=plans,configs=configs,source_sha256=sources,
        source_review_sha256=sha(prior/'failure-review.json'),contact_geometry_review_sha256=sha(prior/'contact-geometry-review.json'),
        allowed_static_field_differences=sorted(MASKS),maximum_episodes=64,maximum_control_transitions=35200,
        learning_updates=0,heldout_used=False,automatic_retry=False,automatic_extension=False,hardware_io=False,
        gate='Diagnosis only: paired successes/failures with unchanged95percent getup gate; do not accept relaxed collisions as a solution',
        required_before_physics='Both zero-step scenes match CPU and all paired reset, IMU, expanded fields, observations and RNG snapshots match')
    write(PLAN,master);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        only_two_masks_differ=True,cad_exclusions_unchanged=True,cuda_initialized=False))
    print('CONTACT_ISOLATION_FROZEN',sha(PLAN),'64episodes35200controls0updates',flush=True)

def checked(arm):
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    p=master['arms'][arm]
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(ROOT/p['model'])==p['model_sha256']
    return p

def install(p,folder,startup):
    import mjlab.envs
    import mujoco
    base=mjlab.envs.ManagerBasedRlEnv
    cpu=lambda x:x.detach().cpu().numpy().copy()
    class Audit(base):
        def __init__(self,*a,**kw):
            self.ready=False;self.steps=0;self.peak=0;self.rows=[];self.resets=0
            super().__init__(*a,**kw);self.ready=True
            fields=model_arrays(self.sim.mj_model);compare_arrays(fields,arrayfile(OUT/p['role']/'cpu-scene-fields.npz'))
            np.savez_compressed(folder/'scene-static-fields.npz',**fields)
            assert self.common_step_counter==0 and self.physics_dt==.004 and self.step_dt==.02
            self.slots=int(self.sim.wp_data.contact.dist.shape[0])
            self.geom_names=[mujoco.mj_id2name(self.sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(self.sim.mj_model.ngeom)]
        def reset(self,*a,**kw):
            if self.ready:assert kw.get('env_ids') is None and self.resets==0,'No resampling or second episode'
            value=super().reset(*a,**kw)
            if not self.ready:return value
            self.resets+=1
            proof=interface_check(self,value[0],p,folder,'reset-0')
            assert proof['rng_unchanged'] and self.common_step_counter==0
            state=dict(qpos=cpu(self.sim.data.qpos),qvel=cpu(self.sim.data.qvel),actor=cpu(value[0]['actor']),
                imu_misalignment=cpu(self._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),cuda_rng=cpu(torch.cuda.get_rng_state(self.device)))
            state.update({'expanded__'+k:cpu(getattr(self.sim.model,k)) for k in sorted(self.sim.expanded_fields)})
            np.savez_compressed(folder/'paired-start-state.npz',**state)
            if not startup:compare_arrays(state,arrayfile(OUT/p['role']/'startup/paired-start-state.npz'))
            return value
        def step(self,actions):
            assert not startup and self.steps<550 and torch.isfinite(actions).all()
            value=super().step(actions);self.steps+=1
            contact=self.sim.wp_data.contact;count=int(self.sim.wp_data.nacon.numpy()[0]);self.peak=max(self.peak,count)
            assert 0<=count<self.slots,'Contact capacity exhausted'
            geom=contact.geom.numpy()[:count];world=contact.worldid.numpy()[:count]
            valid=(contact.dist.numpy()[:count]<0)&((contact.type.numpy()[:count]&1)!=0)&(geom>=0).all(axis=1)
            pairs=np.zeros((self.num_envs,3),np.int16)
            for (i,j),w in zip(geom[valid],world[valid]):
                a,b=self.geom_names[int(i)],self.geom_names[int(j)]
                if a!='robot/head_collision' and b!='robot/head_collision':continue
                other=b if a=='robot/head_collision' else a
                col=0 if other in ('robot/left_foot_collision','robot/right_foot_collision') else 1 if other in ('robot/left_hip_yaw_collision','robot/right_hip_yaw_collision') else 2 if not other.startswith('robot/') else None
                if col is not None:pairs[int(w),col]+=1
            self.rows.append(dict(qpos=cpu(self.sim.data.qpos),qvel=cpu(self.sim.data.qvel),head_contact_pair_counts=pairs))
            if self.steps%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
            return value
        def close(self):
            if self.ready:
                if self.rows:np.savez_compressed(folder/'physical-contact-trace.npz',**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
                write(folder/'runtime-contract.json',dict(status='closed',startup=startup,control_steps=self.steps,
                    control_transitions=self.steps*self.num_envs,contact_peak=self.peak,contact_slots=self.slots,
                    extra_render_steps=0,physics_substeps=self.steps*5,
                    contact_columns=['head-foot','head-hip-yaw','head-floor'],
                    contact_semantics='Negative-distance constraint contacts from the final physics substep; counts, not forces'))
            super().close()
    mjlab.envs.ManagerBasedRlEnv=Audit
    return Audit

def stage(arm,startup):
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    p=checked(arm)
    if startup:
        folder=OUT/arm/'startup';assert not (folder/'runtime-contract.json').exists()
        cls=install(p,folder,True);env=cls(cfg=config(p),device='cuda:0',render_mode=None)
        try:env.seed(p['seed']);env.reset();assert env.common_step_counter==0
        finally:env.close()
        return
    assert not (OUT/arm/'progress.json').exists()
    run=f'duck_getup_contact-isolation_{arm}_s1010_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run;folder.mkdir();(folder/'videos').mkdir()
    for name in p['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(folder/'protocol.json',p)
    state=dict(status='running',run_id=run,path=str(folder),pid=os.getpid(),conditions=[],
        protocol_sha256=sha(folder/'protocol.json'),model_sha256=p['model_sha256'],learning_updates=0,heldout_used=False,scope=p['scope'])
    def save():write(folder/'evaluation.json',state);write(OUT/arm/'progress.json',state)
    save();install(p,folder,False);collector.config=config;collector.OUT=OUT/arm
    with (folder/'execution.log').open('a',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=collector.Tee(stdout,log),collector.Tee(stderr,log)
        try:
            collector.evaluate(p,folder,state);state['status']='completed';save()
        except BaseException as e:state.update(status='failed',error=repr(e));save();raise
        finally:sys.stdout,sys.stderr=stdout,stderr
    import verify_standup_reference as verifier
    verifier.OUT=OUT/arm;verifier.main()

def analyze():
    import onnxruntime as ort
    results={};raw={};replayed=0
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/checked('full')['model']),sess_options=options,providers=['CPUExecutionProvider'])
    for arm in ARMS:
        state=read(OUT/arm/'progress.json');assert state['status']=='completed'
        folder=Path(state['path']);t=arrayfile(folder/'supine-trace.npz');extra=arrayfile(folder/'physical-contact-trace.npz')
        r=read(folder/'runtime-contract.json');assert r['control_steps']==550 and r['control_transitions']==17600
        assert len(extra['qpos'])==550
        assert (t['initial_qpos']==arrayfile(OUT/arm/'startup/paired-start-state.npz')['qpos']).all()
        for k in range(100,550,30):
            for i in range(32):
                action=session.run(None,{'obs':t['observations'][k,i:i+1]})[0][0]
                np.testing.assert_allclose(action,t['raw_policy_actions'][k,i],atol=2e-5,rtol=2e-5);replayed+=1
        raw[arm]=t;item=read(OUT/arm/'analysis.json')['conditions'][0]
        results[arm]=dict(successes=item['successes'],episodes=32,wilson95=item['wilson_95'],
            original_gate_passed=item['success_fraction']>=.95,record=str(folder.relative_to(ROOT)),
            failed_environment_indices=[e['env_index'] for e in item['episodes'] if not e['success']],
            late_head_contact_counts=extra['head_contact_pair_counts'][450:].sum(axis=0).tolist(),
            trace_sha256=sha(folder/'supine-trace.npz'),physical_contact_trace_sha256=sha(folder/'physical-contact-trace.npz'))
    # Pairing is established before physics, never by a favorable final outcome.
    np.testing.assert_array_equal(raw['full']['initial_qpos'],raw['source-masks']['initial_qpos'])
    a=np.array([e['success'] for e in read(OUT/'full/analysis.json')['conditions'][0]['episodes']])
    b=np.array([e['success'] for e in read(OUT/'source-masks/analysis.json')['conditions'][0]['episodes']])
    result=dict(status='passed',arms=results,paired_source_mask_wins=int((~a&b).sum()),paired_source_mask_losses=int((a&~b).sum()),
        protocol_sha256=sha(PLAN),replayed_inputs=replayed,learning_updates=0,actual_control_transitions=35200,heldout_used=False,
        interpretation='Collision-mask intervention only; any success difference supports a contact-rule effect in this single supine development screen. Relaxed collisions remain diagnostic, not deployable.',
        automatic_extension=False)
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2),flush=True)

def run():
    assert os.isatty(1) and read(ROOT/'project.json')['staged_training_request']['current_stage']=='S5-getup-contact-isolation'
    freeze();state=dict(status='running',pid=os.getpid(),completed=[],protocol_sha256=sha(PLAN));write(OUT/'progress.json',state)
    try:
        for arm in ARMS:
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--startup',arm],cwd=WORK,check=True)
            state['completed'].append('startup-'+arm);write(OUT/'progress.json',state)
        compare_arrays(arrayfile(OUT/'full/startup/paired-start-state.npz'),arrayfile(OUT/'source-masks/startup/paired-start-state.npz'))
        write(OUT/'paired-startup.json',dict(status='passed',all_snapshot_arrays_exact=True,physics_steps=0,
            only_static_mask_difference=sorted(MASKS),snapshot_sha256={a:sha(OUT/a/'startup/paired-start-state.npz') for a in ARMS}))
        for arm in ARMS:
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--evaluate',arm],cwd=WORK,check=True)
            state['completed'].append(arm);write(OUT/'progress.json',state)
        analyze();state['status']='completed-review-required'
    except BaseException as e:state.update(status='stopped-review-required',error=repr(e));raise
    finally:write(OUT/'progress.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--startup',choices=ARMS);parser.add_argument('--evaluate',choices=ARMS);parser.add_argument('--analyze',action='store_true');a=parser.parse_args()
    logs=ROOT/'records/reports/common-plant';path=logs/f'getup-contact-{datetime.now().strftime("%Y%m%dT%H%M%S")}-{os.getpid()}.log'
    event=dict(argv=sys.argv,started_at=datetime.now(timezone.utc).isoformat(),foreground_tty=os.isatty(1),log=str(path.relative_to(ROOT)))
    with path.open('x',buffering=1) as file:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=collector.Tee(stdout,file),collector.Tee(stderr,file)
        try:
            print('VISIBLE_GETUP_CONTACT_COMMAND',json.dumps(event),flush=True)
            if a.startup:stage(a.startup,True)
            elif a.evaluate:stage(a.evaluate,False)
            elif a.analyze:analyze()
            else:run()
            event['exit_code']=0
        except BaseException as e:event.update(exit_code=1,error=repr(e));traceback.print_exc();raise
        finally:
            event['ended_at']=datetime.now(timezone.utc).isoformat()
            with (logs/'getup-contact-commands.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            sys.stdout,sys.stderr=stdout,stderr
