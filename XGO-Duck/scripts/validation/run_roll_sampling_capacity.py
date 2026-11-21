"""Measure observed roll sampling at three CUDA scales; no optimizer or holdout."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from native_roll_observation_bounded import NativeRollObservationBounded
from observed_getup_entry import viewing_confirmed
from record_io import write
from recorded_process import run_recorded

ENTRY = Path(__file__).resolve()
PLAN = ROOT/'configs/experiments/roll_friction_sampling_capacity.json'
PREVIOUS = ROOT/'configs/experiments/roll_friction_exposure_preflight.json'
PROOF = ROOT/'records/reports/roll-friction-exposure-preflight/cuda-repeat-zero.json'
REPORT = ROOT/'records/reports/roll-friction-sampling-capacity'
STAGE = 'roll-sampling-capacity'
WORLD_ORDER = (2048, 3072, 3584, 3584, 3072, 2048)
SEED = 1245
WARM = 48
CONTROLS = 384
CHUNK = 48
CAP = .85
LEARNER_RESERVE_MIB = 768


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def put(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def require(flag, reason):
    if not flag:
        raise ValueError(reason)


def active(project):
    state, goal = project['staged_training_request'], project['goal_execution']
    return (goal['status']=='active' and not goal['pause_requested_by_user'] and
            not any(state[k] for k in ('learning_running','pause_requested','deployment_authorized_now')))


def freeze():
    project = read(ROOT/'project.json')
    st, goal = project['staged_training_request'], project['goal_execution']
    proof = read(PROOF)
    require(active(project) and st['current_stage']==STAGE+'-preflight' and not st['simulation_running']
            and goal['roll_friction_exposure']['status']=='cpu-and-cuda-zero-verified'
            and not goal['roll_friction_exposure']['learning_authorized']
            and proof['status']=='passed-repeated-cuda-reset-only' and not proof['violations'], 'Zero-step proof not ready')
    parents = read(PREVIOUS)
    sources = dict(parents['source_sha256'])
    extras = [ENTRY, PROOF, PREVIOUS,
              ROOT/'configs/experiments/roll_friction_exposure_repeat_zero.json',
              ROOT/'configs/environments/action_recording.json']
    extras += [ROOT/'scripts/training'/name for name in (
        'native_roll_observation.py','native_roll_observation_bounded.py','native_observer_guard.py',
        'native_run_observer.py','native_run_observer_bounded.py','observed_getup_entry.py',
        'recorded_process.py','parent_exit_exec.py','metadata_actor_import.py','roll_top_contact.py',
        'record_io.py','training_watch.py','action_video.py','getup_value_bootstrap.py')]
    extras += [ROOT/'scripts/validation'/name for name in (
        'run_local_getup_ppo.py','local_getup_scale.py','getup_release_coverage.py')]
    sources.update({rel(f):sha(f) for f in extras})
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_roll_sampling-e2048-3072-3584_s{SEED}_{stamp}'
    spec = dict(status='frozen-before-sampling-only', run_id=run_id, stage=STAGE, order=WORLD_ORDER,
        seed=SEED, warmup_controls=WARM, timed_controls=CONTROLS, publish_every=CHUNK,
        maximum_batched_control_steps=len(WORLD_ORDER)*(WARM+CONTROLS),
        max_world_transitions=sum(WORLD_ORDER)*(WARM+CONTROLS), max_wall_seconds=1400,
        max_gpu_fraction=CAP, learner_reserve_mib=LEARNER_RESERVE_MIB, minimum_paired_gain=.03,
        learner_updates=0, hardware_io=False, holdout_used=False, automatic_retry=False,
        source_sha256=sources, installed_sha256=parents['local_training_sha256'])
    require(all(sha(ROOT/f)==h for f,h in sources.items()),'Preflight source changed before freeze')
    require(all(sha(Path(f))==h for f,h in spec['installed_sha256'].items()),'Training installation changed')
    REPORT.mkdir()
    put(PLAN,spec)
    st.update(current_stage=STAGE+'-ready',current_protocol=rel(PLAN),current_protocol_frozen=True,
              next_gate='Observe the bounded sampling trial in VS Code before CUDA stepping.')
    goal['roll_sampling_capacity'] = dict(status='frozen-ready', plan=rel(PLAN), sha256=sha(PLAN), run_id=run_id,
        learning_updates=0, hardware_io=False)
    write(ROOT/'project.json',project)
    print('ROLL_SAMPLING_FROZEN',run_id,sha(PLAN),flush=True)


def check_plan():
    p = read(ROOT/'project.json')
    spec = read(PLAN)
    r = p['goal_execution']['roll_sampling_capacity']
    require(active(p) and r['sha256']==sha(PLAN) and r['run_id']==spec['run_id']
            and spec['status']=='frozen-before-sampling-only' and spec['order']==list(WORLD_ORDER)
            and not any(spec[k] for k in ('learner_updates','hardware_io','holdout_used','automatic_retry')), 'Plan not admitted')
    for root, key in ((ROOT,'source_sha256'),(Path('/'),'installed_sha256')):
        require(all(sha(root/f)==h for f,h in spec[key].items()),'Frozen sampling sources changed')
    return p,spec


def folder_for(spec):
    return ROOT/'records/benchmarks'/spec['run_id']


def worker():
    project,spec = check_plan()
    folder, ident = folder_for(spec), spec['run_id']
    start_sha = sha(PLAN)
    task, goal = project['staged_training_request'], project['goal_execution']
    run = goal['roll_sampling_capacity']
    log = (REPORT/'execution.log').stat()
    require((log.st_dev,log.st_ino)==(os.fstat(1).st_dev,os.fstat(1).st_ino)
            and (log.st_dev,log.st_ino)==(os.fstat(2).st_dev,os.fstat(2).st_ino), 'Use the recorded supervisor')
    require(task['current_stage']==STAGE+'-ready' and not task['simulation_running'] and run['status']=='awaiting-view',
            'Sampling group is not ready')
    running = False
    owner = os.getpid()
    result = dict(status='stopped-review-required', reason='not-started', learner_updates=0, hardware_io=False, plan_sha256=start_sha)

    def allowed():
        p = read(ROOT/'project.json')
        t=p['staged_training_request']
        r=p['goal_execution']['roll_sampling_capacity']
        return (active(p) and sha(PLAN)==start_sha==r['sha256'] and t['current_protocol'] == rel(PLAN)
                and t['current_protocol_frozen'] and t['simulation_running'] is running
                and (r.get('owner_pid')==owner and t['current_stage']==STAGE+'-running'
                     if running else t['current_stage']==STAGE+'-ready' and r['status']=='awaiting-view'))

    bridge = NativeRollObservationBounded(ident,REPORT,allowed)
    # Keep one benchmark directory. The observer reads real progress and media there.
    bridge.run_folder, bridge.source = folder, folder/'native-state.json'
    bridge.state.update(purpose='滚翻训练场景采样短测，等待VS Code确认，学习更新为0',
                        maximum_controls=len(WORLD_ORDER)*CONTROLS)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Recorded supervisor stopped')))
    try:
        bridge.start()
        print('WAITING_FOR_VS_CODE',bridge.view,flush=True)
        deadline=time.monotonic()+210
        while not viewing_confirmed(REPORT/'viewing-confirmation.json',bridge,start_sha):
            require(bridge.check(force=True) and time.monotonic()<deadline,'Current VS Code observer missing')
            time.sleep(.2)
        require(allowed(),'Current budget changed before starting')
        project=read(ROOT/'project.json')
        project['staged_training_request'].update(current_stage=STAGE+'-running', simulation_running=True)
        project['goal_execution']['roll_sampling_capacity'].update(owner_pid=owner,status='running')
        write(ROOT/'project.json',project)
        running=True
        bridge.last_check=0
        collect(spec,folder,bridge)
        result.update(status='completed-awaiting-selection',reason=None,summary=rel(folder/'trials.json'))
    except BaseException as exc:
        result['reason']=f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        try:
            bridge.close(result)
        finally:
            if running:
                p=read(ROOT/'project.json')
                r=p['goal_execution']['roll_sampling_capacity']
                require(r['owner_pid']==owner,'Benchmark owner changed')
                r.update(status=result['status'],result=rel(REPORT/'result.json'))
                p['staged_training_request'].update(current_stage=STAGE+'-review',simulation_running=False)
                write(ROOT/'project.json',p)
            put(REPORT/'result.json',result)
    return 0 if result['status']=='completed-awaiting-selection' else 2


def collect(spec,folder,bridge):
    import resource
    import numpy as np
    import onnxruntime as ort
    import torch
    from action_video import ActionVideo
    from check_roll_friction_exposure import ManagerBasedRlEnv, make, ROLES
    from getup_release_coverage import actor_model
    from getup_value_bootstrap import td
    from mjlab.utils.torch import configure_torch_backends
    from roll_top_contact import adapt
    from run_local_getup_ppo import MemoryWatch
    from training_watch import Snapshot

    model_path = 'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx'
    model = ROOT/model_path
    opts = ort.SessionOptions()
    opts.intra_op_num_threads=1
    session = ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    actor = actor_model(dict(seed=SEED,model=model_path,model_sha256=spec['source_sha256'][model_path]),'cuda:0')
    rows, done_world_steps, done_controls, updates = [], 0, 0, []
    bridge.begin_collect()

    def trial(i,n):
        directory = folder/f'pass-{i:02d}-e{n}'
        directory.mkdir()
        cfg = make(ROLES[0],n,SEED)
        # Keep training rewards intact. Only the model-HOME coordinate adapter is needed.
        reward=cfg.env.rewards['roulade_head_pivot']
        cfg.env=adapt(cfg.env,metadata,False)
        cfg.env.rewards['roulade_head_pivot']=reward
        require(cfg.env.auto_reset and cfg.env.episode_length_s==7 and cfg.env.decimation==5, 'Training scene changed')
        env = None
        started=time.perf_counter()
        m=MemoryWatch(CAP)
        video, snapshot = None,None
        def gate():
            m.check()
            require(bridge.check(),'Observer or project guard stopped sampling: '+str(bridge.refusal))

        def publish(k, speed, image=None, latest=None):
            global_count = done_controls+k
            captured = dict(run_id=spec['run_id'],status='running',purpose=f'滚翻采样短测 {i}/{len(WORLD_ORDER)} · {n}个环境 · 学习更新为0',
                controls=global_count,maximum_controls=len(WORLD_ORDER)*CONTROLS,transitions=done_world_steps+n*k,
                learning_updates=0,updates=list(updates))
            if speed is not None:
                updates.append(dict(control=global_count,net_transitions_per_second=float(speed)))
                captured['updates']=list(updates)
            if image:
                captured['frame']=directory.name+'/'+image['file']
            if latest:
                captured['latest_clip']=directory.name+'/'+latest['file']
            bridge.publish(folder,captured)

        try:
            with m.active(),torch.inference_mode():
                gate()
                env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
                obs,_=env.reset(seed=SEED)
                require(env.common_step_counter==env._sim_step_counter==0 and obs['actor'].shape==(n,61),'Zero-step body or input changed')
                require(tuple(metadata['joint_names'].split(','))==tuple(env.action_manager.get_term('joint_pos').target_names),
                        'Joint order mismatch')
                y=obs['actor'][:16]
                np.testing.assert_allclose(actor(td('actor',y)).cpu().numpy(),
                    np.concatenate([session.run(None,{'obs':x[None]})[0] for x in y.cpu().numpy()]),atol=2e-5,rtol=2e-5)
                m.sample()
                zero=time.perf_counter()-started
                warm_start=time.perf_counter()
                for _ in range(WARM):
                    gate()
                    obs,*_=env.step(actor(td('actor',obs['actor'])))
                torch.cuda.synchronize()
                warm=time.perf_counter()-warm_start
                obs,_=env.reset(seed=SEED)
                require(int(env.episode_length_buf[0])==0 and env.common_step_counter==WARM,'Measured start not at episode boundary')
                snapshot=Snapshot(env,directory)
                video=ActionVideo(snapshot,'roll',directory)
                publish(0,None)
                t=time.perf_counter()
                cpu0=time.process_time()
                infer=sim=pub=0.
                video.sample()
                chunk=t
                latest=None
                for k in range(1,CONTROLS+1):
                    gate()
                    s=time.perf_counter()
                    control=actor(td('actor',obs['actor']))
                    after=time.perf_counter()
                    obs,reward,_,_,_=env.step(control)
                    infer+=after-s
                    sim+=time.perf_counter()-after
                    video.sample()
                    if k%CHUNK==0:
                        torch.cuda.synchronize()
                        require(torch.isfinite(obs['actor']).all().item() and torch.isfinite(reward).all().item(),
                                'Invalid observation or reward')
                        m.check()
                        start_io=time.perf_counter()
                        if video.clips:
                            latest=video.clips[-1]
                        publish(k,n*CHUNK/(start_io-chunk), image=snapshot.capture(), latest=latest)
                        pub+=time.perf_counter()-start_io
                        chunk=time.perf_counter()
                video.close()
                elapsed=time.perf_counter()-t
                cpu=time.process_time()-cpu0
                m.sample()
                require(env.common_step_counter==WARM+CONTROLS and env._sim_step_counter==5*(WARM+CONTROLS),'Physical counters differ')
                require(video.clips and video.clips[0]['full_action'] and video.clips[0]['frames']==176,
                        'Measured episode has no complete original clip')
                return dict(trial=i,envs=n,status='completed',measured_transitions=n*CONTROLS,warmup_transitions=n*WARM,
                    measured_seconds=elapsed,net_transitions_per_second=n*CONTROLS/elapsed,
                    initialization_seconds=zero,warmup_seconds=warm,cpu_process_seconds=cpu,
                    max_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    actor_enqueue_wall_seconds=infer,environment_step_wall_seconds=sim,
                    media_phase_wall_seconds=video.elapsed_seconds,status_and_snapshot_wall_seconds=pub,
                    exact_physics_substeps=5*(WARM+CONTROLS),gpu_rows=m.rows,complete_clip=video.clips[0],
                    video_path=rel(directory/video.clips[0]['file']),video_sha256=sha(directory/video.clips[0]['file']))
        except BaseException as exc:
            put(directory/'stop.json',dict(status='stopped-review-required',reason=f'{type(exc).__name__}: {exc}',
                controls=None if env is None else env.common_step_counter,gpu_rows=m.rows,learning_updates=0))
            raise
        finally:
            if video is not None: video.close()
            if snapshot is not None: snapshot.close()
            if env is not None: env.close()

    for i,n in enumerate(WORLD_ORDER,1):
        r=trial(i,n)
        put(folder/f'pass-{i:02d}-e{n}'/'result.json',r)
        rows.append(r)
        done_world_steps+=n*CONTROLS
        done_controls+=CONTROLS
        torch.cuda.empty_cache()
        print('ROLL_CAPACITY',i,n,round(r['net_transitions_per_second'],1),
              max(s['used_mib'] for s in r['gpu_rows']),flush=True)
    put(folder/'trials.json', dict(trials=rows, timings='Observed rollout plus media, status and telemetry; hardware-only utilization.',
        original_physics_steps=sum(r['exact_physics_substeps'] for r in rows), new_learning_updates=0))


def confirm(path):
    import datetime as dt
    p=read(ROOT/'project.json')
    spec=read(PLAN)
    health=read(REPORT/'native-view/native-health.json')
    image=Path(path).resolve()
    source=folder_for(spec)/'native-state.json'
    stamp=dt.datetime.fromisoformat(health['heartbeat_at'].replace('Z','+00:00'))
    age=(dt.datetime.now(dt.timezone.utc)-stamp).total_seconds()
    require(active(p) and health['health']=='fresh' and health['run_id']==spec['run_id']
            and type(health['observer_pid']) is int and -5<=age<=15
            and (ROOT/health['source']).resolve()==source.resolve()
            and image.parent==REPORT.resolve() and image.read_bytes()[:3] == b'\xff\xd8\xff',
            'Current native viewer or screenshot invalid')
    put(REPORT/'viewing-confirmation.json',dict(status='viewed-current-run',run_id=spec['run_id'],
        protocol_sha256=sha(PLAN),observer_pid=health['observer_pid'],source=rel(source),
        screenshot=rel(image),screenshot_sha256=sha(image),checked_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('LIVE_VIEW_CONFIRMED',spec['run_id'],flush=True)


def run():
    p,spec=check_plan()
    t=p['staged_training_request']
    f=folder_for(spec)
    require(t['current_stage']==STAGE+'-ready' and not t['simulation_running'] and not f.exists()
            and not (REPORT/'execution.log').exists(),'Run already attempted or not ready')
    f.mkdir()
    p['goal_execution']['roll_sampling_capacity']['status']='awaiting-view'
    write(ROOT/'project.json',p)
    cmd=[sys.executable,'-B',str(ROOT/'scripts/training/parent_exit_exec.py'),str(os.getpid()),'--',
         sys.executable,'-B','-u',str(ENTRY),'child']
    receipt=run_recorded(cmd,cwd=ROOT,log_path=REPORT/'execution.log',receipt_path=REPORT/'execution-receipt.json',
                         max_wall_seconds=spec['max_wall_seconds'])
    print('ROLL_CAPACITY_RECEIPT',receipt['status'],receipt['exit_code'],flush=True)
    return 0 if receipt['status']=='completed' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','child','confirm'))
    parser.add_argument('--screenshot')
    args=parser.parse_args()
    if args.action=='confirm':
        return confirm(args.screenshot)
    if args.action=='child':
        return worker()
    with Path('/home/atlas/XGO-Duck/training.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='freeze':
            return freeze()
        os.environ.update(MUJOCO_GL='egl',GALLIUM_DRIVER='d3d12',MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
        return run()


if __name__=='__main__':
    sys.exit(main())
