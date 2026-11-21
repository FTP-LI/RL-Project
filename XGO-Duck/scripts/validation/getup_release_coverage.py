"""Fixed-policy release coverage with complete eleven-second recordings.

Change: one shuffled batch covers all four requested falls, with extra side
starts because many settle into supine before policy release. No optimizer,
retries, rejection sampling or relaxed collisions. Preserve the old strict
requested/release score beside the prospectively frozen functional score.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK, read, sha
from record_io import write
from getup_coverage_metrics import POSES, evaluate, group_rows

PLAN = ROOT/'configs/experiments/getup_release_coverage.json'
OUT = ROOT/'records/reports/getup-release-coverage/fixed-evaluation'


def actor_model(p, device='cpu'):
    from rsl_rl.models import MLPModel
    from getup_value_bootstrap import td
    from metadata_actor_import import import_actor
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(p['seed'])
        actor = MLPModel(td('actor', torch.zeros(1, 61)), {'actor':['actor']}, 'actor', 14,
            hidden_dims=(512,256,128), activation='elu', obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution', init_std=.05, std_type='scalar'))
        import_actor(actor, ROOT/p['model'], p['model_sha256'])
    actor.eval().requires_grad_(False)
    return actor.to(device)


def prepare(p):
    from getup_contact_training import prepare as base
    cfg, display, metadata = base(p)
    cfg.env.episode_length_s = 11.
    cfg.env.observations['critic'].terms['remaining_time'].params['horizon_seconds'] = 11.
    assert not cfg.env.auto_reset and cfg.env.is_finite_horizon
    assert not cfg.env.rewards['late_head_leg_contact'].params['enabled']
    display.update(recorder='Four requested-pose representatives, complete11s, same sampled states',
                   native_live_window=False)
    return cfg, display, metadata


def cpu_check(plan):
    import onnxruntime as ort
    from getup_value_bootstrap import td
    # The old data still fail the new requirements: this is not an old pass.
    old = read(ROOT/'records/reports/getup-release-coverage/diagnosis.json')
    rows = []
    for pose, value in old['observed'].items():
        for row in value['rows']:
            rows.append(dict(requested_pose=row['requested_pose'], release_pose=pose,
                hold_start_seconds=2. if row['timely_hold'] else None,
                success=row['original_success'], release_pose_valid=row['requested_pose']==pose))
    check = group_rows(rows)
    assert check['overall']['timely_holds']==119 and check['strict_requested_successes']==89
    assert not check['functional_coverage_gate']
    # Independent threshold/coverage/unknown-class boundary fixtures.
    fixture = [dict(requested_pose=pose, release_pose=pose, hold_start_seconds=2.,
                    success=True, release_pose_valid=True) for pose in POSES for _ in range(32)]
    assert group_rows(fixture)['functional_coverage_gate']
    fixture[0]['hold_start_seconds']=None
    assert group_rows(fixture)['functional_coverage_gate']
    fixture[1]['hold_start_seconds']=None
    assert not group_rows(fixture)['functional_coverage_gate']
    assert not group_rows(fixture[2:])['functional_coverage_gate']
    fixture[1].update(hold_start_seconds=2., release_pose='other')
    assert not group_rows(fixture)['functional_coverage_gate']
    source = ROOT/'records/experiments/duck_getup_value-bootstrap_reference_s42_20261103T053111/value-data.npz'
    with np.load(source) as f:
        inputs = f['actor_obs_sample'][[0,99,100,125,200,300,399]].reshape(-1,61).copy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    errors={}
    for role, p in plan['roles'].items():
        actor=actor_model(p)
        with torch.no_grad(): actual=actor(td('actor',torch.from_numpy(inputs))).numpy()
        session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
        expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in inputs])
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        errors[role]=float(abs(actual-expected).max())
    assert not torch.cuda.is_initialized()
    return dict(status='passed', old_data_still_fail=True, old_timely_holds=119,
        threshold_and_coverage_boundaries_passed=True, imported_inputs_per_model=len(inputs),
        maximum_import_error=errors, cuda_initialized=False, physics_steps=0)


def freeze():
    import yaml
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays, compare_arrays
    assert not PLAN.exists() and not OUT.exists(), 'Keep existing frozen protocol'
    prior=read(ROOT/'configs/experiments/getup_late_contact_pair.json')
    base=deepcopy(prior['arms']['control'])
    codes=np.random.default_rng(1012).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    groups={pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(POSES)}
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    base.update(seed=1012,num_envs=320,split=dict(pose_codes=codes.tolist(),groups=groups),
        episode_seconds=11.,maximum_control_steps=550,maximum_control_transitions=176000,
        success_rate_min=.95,actual_release_minimum=32,criteria=prior['evaluation_template']['criteria'],
        applied_joint_limit_tolerance_deg=2.,learning_updates=0,
        video_worlds={pose:groups[pose][0] for pose in POSES})
    roles={}
    for role in ('original','control'):
        p=deepcopy(base)
        if role=='control':
            model=ROOT/'artifacts'/prior['arms']['control']['run_id']/'policy.onnx'
            assert sha(model)==read(model.parent/'manifest.json')['model_sha256']
            p.update(model=str(model.relative_to(ROOT)),model_sha256=sha(model))
        p.update(role=role,run_id=f'duck_getup_release-coverage_{role}_s1012_{stamp}')
        roles[role]=p
    sources=deepcopy(prior['source_sha256'])
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name in ('scripts/validation/getup_release_coverage.py','scripts/validation/getup_coverage_metrics.py',
                 'scripts/validation/verify_getup_release_coverage.py','scripts/training/live_observer.py'):
        sources[name]=sha(ROOT/name)
    for name,digest in base['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    plan=dict(status='frozen-before-data',roles=roles,source_sha256=sources,
        frozen_at=datetime.now(timezone.utc).isoformat(),maximum_evaluation_transitions=352000,
        learning_updates=0,heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        rationale='Mandatory2s freeze changes some side falls to supine; require every requested group95% timely recovery AND each actual release class>=32 and95%, overall95%, zero unknown releases. Old strict requested-release failures remain separate.',
        selection='Prefer simpler official if both satisfy functional coverage; this single new scene seed cannot establish training improvement or robustness',
        release_policy='All320first episodes retained, no rejection/resampling/reset or replacement; inadequate coverage fails',
        scope='Prospective independent development coverage; continuous transitions, multi-seed reliability and hardware remain outstanding')
    proof=cpu_check(plan)
    cfg,display,_=prepare(roles['original']);model=Scene(cfg.env.scene,device='cpu').compile()
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:
        compare_arrays(model_arrays(model),{k:f[k] for k in f.files})
    other,_,_=prepare(roles['control']);compare_arrays(model_arrays(Scene(other.env.scene,device='cpu').compile()),model_arrays(model))
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True);write(PLAN,plan)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),fixture=proof,
        cpu_physical_fields_exact=True,config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
        display=display,requested_counts={k:len(v) for k,v in groups.items()},
        video_worlds=base['video_worlds'],maximum_evaluation_transitions=352000,cuda_initialized=False))
    print('GETUP_COVERAGE_FROZEN',sha(PLAN),flush=True)


def checked():
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['roles'].values():
        assert sha(ROOT/p['model'])==p['model_sha256']
        for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
        for name,digest in p['library_source_sha256'].items():assert sha(Path(name))==digest,name
    return m


def render_index(snapshot, index):
    """Render this sampled world; forward geometry only, never integration."""
    from live_observer import VISUAL_FIELDS
    sim=snapshot.env.sim
    for name in sim.expanded_fields & VISUAL_FIELDS:
        dest=getattr(snapshot.model,name)
        dest[:]=getattr(sim.model,name)[index].detach().cpu().numpy().reshape(dest.shape)
    for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','xfrc_applied'):
        dest=getattr(snapshot.data,name)
        if dest.size:dest[:]=getattr(sim.data,name)[index].detach().cpu().numpy().reshape(dest.shape)
    snapshot.mj.mj_forward(snapshot.model,snapshot.data)
    snapshot.cam.lookat[:]=snapshot.data.xpos[snapshot.root_body]
    snapshot.renderer.update_scene(snapshot.data,camera=snapshot.cam)
    return snapshot.renderer.render().copy()


def run(role):
    from contextlib import ExitStack
    import mediapy
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from getup_value_bootstrap import td
    from recover_getup_value_startup import startup
    from training_watch import Snapshot
    from common_scene_contract import compare_arrays
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original single visible VS Code terminal'
    master=checked();p=master['roles'][role]
    progress=OUT/f'{role}-progress.json';assert not progress.exists(),'No repeat or extension'
    record=ROOT/'records/evaluations'/p['run_id'];record.mkdir();(record/'videos').mkdir()
    write(record/'protocol.json',p)
    for name in master['source_sha256']:shutil.copy2(ROOT/name,record/Path(name).name)
    state=dict(status='running',run_id=p['run_id'],path=str(record),pid=os.getpid(),conditions=[],
        model=str(ROOT/p['model']),model_sha256=p['model_sha256'],learning_updates=0,scope=master['scope'],
        started_at=datetime.now(timezone.utc).isoformat(),control_steps=0)
    env=snapshot=None;trace=[];peak=0;frame_times=[];video_hashes={pose:[] for pose in POSES}
    def publish():
        if env is not None:state['control_steps']=int(env.common_step_counter)
        write(record/'evaluation.json',state);write(progress,state)
    publish();write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(record),run_id=p['run_id'],pid=os.getpid()))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,meta=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            actor=actor_model(p,env.device);before={k:v.cpu().clone() for k,v in actor.state_dict().items()}
            startup(env,obs,actor,p,record)
            cpu=lambda x:x.detach().cpu().numpy().copy()
            initial=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),actor_obs=cpu(obs['actor']),
                critic_obs=cpu(obs['critic']),imu=cpu(env._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),
                cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
            np.savez_compressed(record/'paired-start.npz',**initial)
            if role=='control':
                original=Path(read(OUT/'original-progress.json')['path'])
                with np.load(original/'paired-start.npz') as f:compare_arrays(initial,{k:f[k] for k in f.files})
                write(record/'matched-start.json',dict(status='passed',physics_steps=0,physical_observations_imu_rng_exact=True))
            robot=env.scene['robot'];ids,names=robot.find_joints(meta['joint_names'].split(','),preserve_order=True)
            import mujoco
            jids=[mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in names]
            ranges=env.sim.mj_model.jnt_range[jids].copy();tolerance=np.deg2rad(2.)
            snapshot=Snapshot(env,record);slots=int(env.sim.wp_data.contact.dist.shape[0]);start=time.monotonic()
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe());parity=[]
            with ExitStack() as stack:
                writers={pose:stack.enter_context(mediapy.VideoWriter(str(record/'videos'/f'{pose}_env{i}.mp4'),shape=(480,640),fps=25,qp=22)) for pose,i in p['video_worlds'].items()}
                def frame(step):
                    for pose,index in p['video_worlds'].items():
                        im=Image.fromarray(render_index(snapshot,index));draw=ImageDraw.Draw(im)
                        draw.rectangle((0,0,640,56),fill='#10202d')
                        draw.text((8,6),f'GETUP COVERAGE {role} | initial {pose} | env {index} | t={step*.02:.2f}s',fill='white')
                        draw.text((8,28),'Actual sampled state; freeze2s, recover by8s, hold3s; no extra physics',fill='white')
                        writers[pose].add_image(np.asarray(im));video_hashes[pose].append(__import__('hashlib').sha256(cpu(env.sim.data.qpos[index]).tobytes()).hexdigest())
                        if step in (0,100,150,250,400,550):im.save(record/'videos'/f'{pose}_t{step*.02:g}.jpg')
                    frame_times.append(step*.02)
                frame(0)
                with torch.no_grad():
                    for step in range(1,551):
                        raw=torch.zeros((320,14),device=env.device) if step<=100 else actor(obs)
                        sent=raw.clone();observed=cpu(obs['actor'])
                        if step in (1,100,101,150,250,400,550):
                            parity.append(dict(step=step,obs=observed[:32],actions=cpu(actor(td('actor',obs['actor'][:32])))))
                        obs,reward,term,timeout,_=env.step(raw)
                        torch.testing.assert_close(raw,sent,atol=0,rtol=0)
                        torch.testing.assert_close(obs['actor'][:,34:48],sent,atol=0,rtol=0)
                        positions=cpu(robot.data.joint_pos[:,ids]);qpos=cpu(env.sim.data.qpos);qvel=cpu(env.sim.data.qvel)
                        valid=cpu(~term & (~timeout | (step==550)))
                        bounded=((positions>=ranges[:,0]-tolerance)&(positions<=ranges[:,1]+tolerance)).all(1)
                        trace.append(dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                            quaternion=cpu(robot.data.root_link_quat_w),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(320,2,-1).any(2),
                            joint_envelope_ok=bounded,joint_positions=positions,observations=observed,raw_policy_actions=cpu(sent),
                            applied_targets=cpu(env.action_manager.get_term('joint_pos')._processed_actions),qpos=qpos,qvel=qvel,
                            episode_ages=cpu(env.episode_length_buf),terminated=cpu(term),truncated=cpu(timeout)))
                        assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and torch.isfinite(reward).all()
                        assert not term.any() and (not timeout.any() if step<550 else bool(timeout.all())),'Unexpected episode boundary; no reset'
                        assert bool((env.episode_length_buf==step).all()),'Unexpected reset'
                        count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                        if step%2==0:frame(step)
                        if step%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                        if step%100==0 or step==550:
                            publish();print('GETUP_COVERAGE',role,step,'/550; 320 first episodes; wall',round(time.monotonic()-start,1),flush=True)
            arrays={key:np.asarray([r[key] for r in trace]) for key in trace[0]}
            arrays.update(requested_pose=np.array([POSES[i] for i in p['split']['pose_codes']]),initial_qpos=initial['qpos'],initial_qvel=initial['qvel'])
            result=evaluate(arrays,p);rows=result['episodes']
            np.savez_compressed(record/'coverage-trace.npz',**arrays)
            np.savez_compressed(record/'inference-parity.npz',observations=np.concatenate([v['obs'] for v in parity]),actions=np.concatenate([v['actions'] for v in parity]))
            assert all(torch.equal(v.cpu(),before[k]) for k,v in actor.state_dict().items())
            write(record/'actor-integrity.json',dict(status='passed',actor_and_normalizer_unchanged=True,optimizer_steps=0))
            for pose,index in p['video_worlds'].items():
                group=[r for r in rows if r['requested_pose']==pose];score=result['requested'][pose]
                state['conditions'].append(dict(condition=pose,episodes=group,successes=score['timely_holds'],
                    numeric_gate_passed=score['passed'],strict_requested_successes=sum(r['success'] for r in group),
                    video=dict(path=f'videos/{pose}_env{index}.mp4',sha256=sha(record/'videos'/f'{pose}_env{index}.mp4'),
                        frame_times_seconds=frame_times,environment_index=index,fps=25,qpos_sha256_by_frame=video_hashes[pose],
                        complete_clip=True,source='actual-sampled-GPU-state-render',extra_physics_steps=0)))
            write(record/'coverage.json',result)
            state.update(status='completed',all_conditions_passed=result['functional_coverage_gate'],elapsed_seconds=time.monotonic()-start,
                trace_sha256=sha(record/'coverage-trace.npz'))
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));traceback.print_exc();raise
        finally:
            if trace and state['status']!='completed':np.savez_compressed(record/'partial-trace.npz',**{k:np.asarray([v[k] for v in trace]) for k in trace[0]})
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),control_transitions=int(env.common_step_counter)*320,
                    physics_substeps=int(env.common_step_counter)*5,contact_peak=peak,contact_slots=slots if 'slots' in locals() else None,
                    resets_after_initialization=0,optimizer_steps=0,extra_render_physics_steps=0))
                if hasattr(env,'_getup_contact_audits'):write(record/'contact-audits.json',env._getup_contact_audits)
                env.close()
            state['ended_at']=datetime.now(timezone.utc).isoformat();publish();sys.stdout,sys.stderr=stdout,stderr


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--role',choices=('original','control'));args=parser.parse_args()
    if args.freeze:freeze();return
    assert os.isatty(1),'Use original visible VS Code task terminal'
    if args.role:run(args.role);return
    checked();assert not (OUT/'coordinator.json').exists(),'No repeat of an existing attempt'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    try:
        for role in ('original','control'):
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--role',role],cwd=WORK,check=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_release_coverage.py'),'--role',role],cwd=WORK,check=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_release_coverage.py')],cwd=WORK,check=True)
        write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':main()
