"""Capture the full runtime state of actual backward stalls, then restore it from disk."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from unittest.mock import patch

from mjlab.scene import Scene
import numpy as np
import onnxruntime as ort
import torch
import warp as wp

from s1_reference_inputs import ROOT, read, sha
import collect_walk_backward_onset as onset
from common_scene_contract import compare_arrays, model_arrays
from common_walk_forward_phase_snapshot import WalkingSnapshot
from getup_release_fork_state import digest, signature
from record_io import write
from run_local_getup_ppo import MemoryWatch
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_backward_onset import score
from verify_walk_backward_start_pulse import verify_inputs
import walk_full_state_archive as disk
import training_live_page as live


STAGE = 'S2-backward-stall-capture-capacity'
PREVIOUS = ROOT/'configs/experiments/common_walk_backward_stall_capture.json'
RECOVERY = PREVIOUS.with_stem(PREVIOUS.stem+'_recovery')
PLAN = PREVIOUS.with_stem(PREVIOUS.stem+'_capacity')
DRAFT = PREVIOUS.with_stem(PREVIOUS.stem+'_draft')
HISTORY_OUT = ROOT/'records/reports/common-walk-backward-stall-capture'
CAPACITY_STOP = HISTORY_OUT/'recovery/capacity-stop.json'
OUT = HISTORY_OUT/'capacity'
PROJECT = ROOT/'project.json'
SCENE = onset.OUT/'cpu-scene-fields.npz'
ROLES = ('backward','reference-repeat')
HISTORY = ('_action','_prev_action','_prev_prev_action')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def page(seed):
    return ROOT/f'records/walk-backward-stall-capture-s{seed}-capacity-live.html'


def exercise_archive(path):
    @dataclass
    class Probe:
        array: object

    values = dict(warp=wp.array(np.arange(12,dtype=np.float32).reshape(4,3),dtype=wp.vec3,device='cpu'),
                  empty=wp.empty((0,),dtype=wp.vec3,device='cpu'),
                  nested=Probe([np.asarray([1.,-0.],dtype=np.float32),torch.tensor([4,5]),np.int64(2)]),
                  tuple=(None,False,2.25,'seed'))
    expected = signature(values)
    disk.save(path,values)
    try:
        values['warp'].zero_()
        values['nested'].array[1].zero_()
        assert signature(values)!=expected
        assert signature(disk.load(path,values))==expected
    finally:
        path.unlink(missing_ok=True)


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    project = read(PROJECT)['staged_training_request']
    draft = read(DRAFT)
    original = read(PREVIOUS)
    stop_path = HISTORY_OUT/'zero-step-stop.json'
    stop = read(stop_path)
    recovery = read(RECOVERY)
    capacity = read(CAPACITY_STOP)
    assert project['current_stage']==recovery['stage'] and ROOT/project['current_protocol']==RECOVERY
    assert project['current_protocol_frozen'] and not project['simulation_running']
    assert stop['status']=='stopped-before-environment' and stop['protocol_sha256']==sha(PREVIOUS)
    assert not any(stop[key] for key in ('physical_substeps','learning_updates','seed_consumed'))
    assert sha(HISTORY_OUT/'sampling.log')==stop['sampling_log_sha256']
    assert not (ROOT/'records/evaluations'/stop['original_run']).exists()
    old_source = HISTORY_OUT/'stopped-source'
    for name,digest_ in stop['archived_sources'].items():
        assert sha(old_source/name)==digest_==original['source_sha256'][f'scripts/validation/{name}']
    assert capacity['status']=='stopped-zero-on-snapshot-capacity' and capacity['protocol_sha256']==sha(RECOVERY)
    assert capacity['actual']['initial_resets']==1 and capacity['actual']['attempted_substeps']==0
    old_record = ROOT/capacity['record']
    assert sha(old_record/'protocol.json')==sha(RECOVERY)
    assert read(old_record/'evaluation.json')['actual']==capacity['actual']
    telemetry = old_record/'gpu-telemetry.json'
    assert sha(telemetry)==capacity['memory_telemetry_sha256']
    recovery_source = CAPACITY_STOP.parent/'stopped-source'
    for name,digest_ in capacity['archived_sources'].items():
        assert sha(recovery_source/name)==digest_==recovery['source_sha256'][f'scripts/validation/{name}']
    inputs = {}
    for key in ('prefix_audit','reference_plant','retention'):
        path,digest_ = draft[key]
        assert sha(ROOT/path)==digest_
        inputs[key] = ROOT/path
    prefix = read(inputs['prefix_audit'])
    assert prefix['full_state_capture_trial_admitted'] and not prefix['heldout_used']
    assert read(inputs['retention'])['total_samples']==3072
    spec = deepcopy(read(inputs['reference_plant']))
    onset.prior.check_sources(spec)
    seed,h,k = capacity['next_seed'],draft['episode_control_horizon'],draft['fork_control']
    limit = draft['maximum_warp_snapshot_bytes']
    prior_worlds,multiple = capacity['environment_count'],capacity['world_multiple']
    assert capacity['observed_warp_bytes']>limit==capacity['snapshot_limit_bytes']
    limit_worlds = limit*(1-capacity['capacity_margin_fraction'])*prior_worlds/capacity['observed_warp_bytes']
    n = int(limit_worlds//multiple)*multiple
    assert n==896 and n<prior_worlds==draft['proposed_parallel_worlds']
    total = k+len(ROLES)*(h-k)
    assert n*total<=draft['maximum_control_transitions'] and 5*total==draft['maximum_batched_physical_substeps']
    assert draft['reference_continuations']==len(ROLES)
    assert not any(f'_s{seed}_' in entry.name for entry in (ROOT/'records/evaluations').iterdir())
    assert not page(seed).exists()
    run_id = f'duck_walk_backward-stall-capture_capacity_s{seed}_{datetime.now(onset.TZ):%Y%m%dT%H%M%S}'
    p = spec['environment']
    p.update(seed=seed,stage=STAGE,parallel_worlds=n,episodes_per_command=n,
             groups={'backward':list(range(n))},maximum_control_steps=h,maximum_transitions=n*total)
    assert p['criteria']['block_steps']*2==k and spec['block_seconds']==draft['prefix']['block_seconds']
    assert spec['near_static_max_absolute_block_speed_m_s']==draft['prefix']['maximum_absolute_block_speed_m_s']
    pair = read(ROOT/'records/reports/common-walk-backward-start-enrichment/recovery/decision.json')
    witness = ROOT/pair['records']['control']/'windows-view-verification.json'
    assert read(witness)['status']=='passed'
    spec.update(stage=STAGE,runs={str(seed):run_id},phase={'warmup_controls':k},roles=list(ROLES),
                maximum_control_transitions=n*total,maximum_physics_substeps=5*total,full_budget=n*total,
                maximum_warp_snapshot_bytes=draft['maximum_warp_snapshot_bytes'],capture=draft,windows_reference=rel(witness),
                scope='Original fixed policy; actual stalled-state capture and disk restore. No training.',
                sizing=dict(previous_worlds=prior_worlds,previous_warp_bytes=capacity['observed_warp_bytes'],
                            estimated_warp_bytes=int(capacity['observed_warp_bytes']*n/prior_worlds),selected_worlds=n))
    for path in (DRAFT,PREVIOUS,RECOVERY,stop_path,CAPACITY_STOP,telemetry,*old_source.iterdir(),
                 *recovery_source.iterdir(),*inputs.values(),witness):
        spec['reference_sha256'][rel(path)] = sha(path)
    for path in (Path(__file__),Path(__file__).with_suffix('.sh'),Path(disk.__file__),
                 Path(__file__).with_name('verify_walk_live_page.py'),Path(__file__).with_name('run_local_getup_ppo.py')):
        spec['source_sha256'][rel(path)] = sha(path)
    cfg,_ = onset.config(onset.environment(spec,seed))
    assert cfg.scene.num_envs==n and cfg.decimation==5
    with np.load(SCENE) as saved:
        compare_arrays(model_arrays(Scene(cfg.scene,device='cpu').compile()),{key:saved[key] for key in saved.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    exercise_archive(OUT/'codec-fixture.npz')
    scene = OUT/'cpu-scene-fields.npz'
    shutil.copy2(SCENE,scene)
    spec['reference_sha256'][rel(scene)] = sha(scene)
    onset.prior.check_sources(spec)
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),
          native_scene_exact=True,codec_cpu_round_trip=True,physical_substeps=0,learning_updates=0))
    page(seed).write_text(f'<!doctype html><meta charset="utf-8"><title>XGO-Duck</title>'
                          f'<h1>后退停滞状态</h1><p>等待采集 · {run_id}</p>',encoding='utf-8')
    print('BACKWARD_STALL_CAPTURE_FROZEN',sha(PLAN),run_id,flush=True)


def selected_prefix(velocities, active, live_at_fork, block, maximum):
    n = active.shape[1]
    assert velocities.shape==active.shape==(2*block,n)
    speed = -velocities.astype(float).reshape(2,block,n).mean(axis=1)
    return np.flatnonzero(active.all(axis=0)&live_at_fork&(abs(speed).max(axis=0)<=maximum)),speed


def sample():
    with patch.multiple(onset,PLAN=PLAN,OUT=OUT):
        spec = onset.checked()
    seed, = map(int,spec['runs'])
    entry = read(OUT/'windows-entry.json')
    assert entry['status']=='passed' and entry['protocol_sha256']==sha(PLAN) and entry['page']==rel(page(seed))
    record = ROOT/'records/evaluations'/spec['runs'][str(seed)]
    watch = MemoryWatch(spec['capture']['gpu_memory_fraction_limit'])
    start,original_collect = onset.startup,onset.prior.collect

    class GuardSnapshot(WalkingSnapshot):
        def __init__(self, env, limit):
            super().__init__(env,limit)
            watch.sample()
            watch.check()

    def startup(env,obs,actor,p,desired,options,folder):
        with patch.object(onset,'WalkingSnapshot',GuardSnapshot):
            start(env,obs,actor,p,desired,options,folder)
        assert env.common_step_counter==env._sim_step_counter==0
        write(folder/'cuda-zero-memory.json',dict(status='passed',physical_substeps=0,**watch.summary()))
        (folder/'README.md').write_text('# 后退停滞状态\n\n'
            f'[运行观察](../../{page(seed).name}) · [冻结方案](../../../configs/experiments/{PLAN.name})'
            ' · [采集日志](execution.log)\n',encoding='utf-8')

    def collect(env,actor,options,folder,state,desired):
        real_step = env.step
        host = onset.prior.cpu
        k = spec['phase']['warmup_controls']
        robot = env.scene['robot']
        alive = np.ones(env.num_envs,bool)
        velocities,active_rows,controllers = [],[],[]

        class Controller(onset.BackwardController):
            def __init__(self,p,settings,commands):
                super().__init__(p,settings,commands)
                controllers.append(self)

        class ArchivedSnapshot(GuardSnapshot):
            def __init__(self,env,limit):
                super().__init__(env,limit)
                assert env.common_step_counter==k and len(controllers)==1
                chosen,speed = selected_prefix(np.stack(velocities),np.stack(active_rows),alive,
                    spec['environment']['criteria']['block_steps'],spec['near_static_max_absolute_block_speed_m_s'])
                self.external = dict(previous=host(env.action_manager._action),active=alive.copy(),
                                     controller=controllers[0].snapshot(),command=desired.copy())
                assert not self.external['previous'][~alive].any()
                self.archive = folder/'fork-runtime-state.npz'
                payload = self.payload()
                count = disk.save(self.archive,payload)
                self.archive_sha256 = sha(self.archive)
                self.receipt = self.verify()
                capture = folder/'fork-selected-states.npz'
                np.savez_compressed(capture,ids=chosen,prefix_speed_m_s=speed[:,chosen],
                    qpos=host(env.sim.data.qpos)[chosen],qvel=host(env.sim.data.qvel)[chosen],
                    ctrl=host(env.sim.data.ctrl)[chosen],actor=host(env.obs_buf['actor'])[chosen],
                    critic=host(env.obs_buf['critic'])[chosen],**{name:host(getattr(env.action_manager,name))[chosen] for name in HISTORY})
                write(folder/'fork-archive.json',dict(status='captured',control=k,eligible=chosen.tolist(),
                    runtime_sha256=self.archive_sha256,runtime_bytes=self.archive.stat().st_size,array_entries=count,
                    selected_sha256=sha(capture),physical=self.receipt,external_sha256=digest(signature(self.external))))
                print('BACKWARD_STALL_RUNTIME_CAPTURE',len(chosen),self.receipt['fields'],self.archive.stat().st_size,flush=True)

            def payload(self):
                return dict(values=self.values,rng=self.rng,external=self.external)

            def restore(self):
                assert sha(self.archive)==self.archive_sha256
                saved = disk.load(self.archive,self.payload())
                assert digest(signature(saved['external']))==digest(signature(self.external))
                self.values,self.rng = saved['values'],saved['rng']
                self.synchronize()
                assert self.signatures(self.values,self.rng)==self.expected
                proof = super().restore()
                assert proof==self.receipt
                watch.sample()
                write(folder/'fork-disk-restore.json',dict(status='passed',archive_sha256=self.archive_sha256,
                      physical=proof,external_sha256=digest(signature(self.external)),all_inventory_fields_from_disk=True))
                print('BACKWARD_STALL_DISK_RESTORED',proof['fields'],proof['state_sha256'],flush=True)
                return proof

        def step(action):
            watch.check()
            out = real_step(action)
            if len(velocities)<k:
                active_rows.append(alive.copy())
                velocities.append(host(robot.data.root_link_lin_vel_b[:,0]))
                _,_,terminated,truncated,_ = out
                terminal = host(terminated|truncated)
                for name in ('fell_over','nan_state','out_of_terrain_bounds'):
                    terminal |= host(env.termination_manager.get_term(name))
                alive[:] &= ~terminal
            return out

        with patch.object(env,'step',step),patch.multiple(onset.prior,ROLES=ROLES,ForwardPhase=Controller,
                                                        WalkingSnapshot=ArchivedSnapshot):
            original_collect(env,actor,options,folder,state,desired)

    try:
        with (
            watch.active(),
            patch.multiple(onset,PLAN=PLAN,OUT=OUT,STAGE=STAGE,page=page,startup=startup),
            patch.object(onset.prior,'collect',collect),
        ):
            onset.sample(seed)
            watch.check()
    finally:
        if record.is_dir():
            write(record/'gpu-telemetry.json',dict(samples=watch.rows,summary=watch.summary(),
                                                   error=repr(watch.error) if watch.error else None))


def verify():
    with patch.multiple(onset,PLAN=PLAN,OUT=OUT):
        spec = onset.checked()
    seed, = map(int,spec['runs'])
    p = onset.environment(spec,seed)
    record = ROOT/'records/evaluations'/p['run_id']
    state = read(record/'evaluation.json')
    assert state['status']=='collected-review-required' and sha(record/'protocol.json')==state['protocol_sha256']==sha(PLAN)
    assert state['actor_unchanged'] and not state['heldout_used'] and not state['learning_updates'] and not torch.cuda.is_initialized()
    controls = spec['maximum_control_transitions']//p['parallel_worlds']
    assert state['actual']==dict(attempted_controls=controls,completed_controls=controls,
          attempted_substeps=controls*5,completed_substeps=controls*5,initial_resets=1)
    assert 0<=state['contact_peak']<state['contact_slots']
    start = read(record/'startup.json')
    assert start['status']=='passed' and start['physics_substeps']==0 and start['scene_exact'] and start['actual_processor_restored']
    zero = read(record/'cuda-zero-memory.json')
    telemetry = read(record/'gpu-telemetry.json')
    assert zero['status']=='passed' and zero['physical_substeps']==0 and not telemetry['error']
    assert all(row['used_mib']/row['total_mib']<=spec['capture']['gpu_memory_fraction_limit'] for row in telemetry['samples'])
    captured,restore = [read(record/name) for name in ('fork-archive.json','fork-disk-restore.json')]
    fork = read(record/'fork-readback.json')
    k = spec['phase']['warmup_controls']
    assert restore['status']=='passed' and restore['all_inventory_fields_from_disk']
    assert captured['runtime_sha256']==restore['archive_sha256']==sha(record/'fork-runtime-state.npz')
    assert captured['physical']==restore['physical']==fork[ROLES[0]]['physical']==fork[ROLES[1]]['physical']
    assert captured['physical']['counters']==[k,k*5] and captured['external_sha256']==restore['external_sha256']
    assert fork[ROLES[0]]['controller_sha256']==fork[ROLES[1]]['controller_sha256']
    assert digest(read(record/'fork-state-fields.json'))==captured['physical']['state_sha256']
    source = record/'trace-shared-prefix.npz'
    prefix = arrays(source)
    terminal = np.logical_or.reduce([prefix[field][-1] for field in ('terminated','truncated','fell_over','nan_state','out_of_bounds')])
    chosen,speed = selected_prefix(prefix['velocity'][...,0],prefix['active'],~terminal,
                  p['criteria']['block_steps'],spec['near_static_max_absolute_block_speed_m_s'])
    assert chosen.tolist()==captured['eligible']
    sample_path = record/'fork-selected-states.npz'
    assert sha(sample_path)==captured['selected_sha256']
    with np.load(sample_path) as saved:
        np.testing.assert_array_equal(saved['ids'],chosen)
        np.testing.assert_array_equal(saved['prefix_speed_m_s'],speed[:,chosen])
        np.testing.assert_array_equal(saved['qpos'],prefix['qpos'][-1,chosen])
        np.testing.assert_array_equal(saved['_action'],prefix['actions'][-1,chosen])
        assert np.isfinite(saved['qvel']).all() and np.isfinite(saved['actor']).all()
    initial_path = record/'physical-initial.npz'
    assert sha(initial_path)==state['initial_state_sha256']
    initial = arrays(initial_path)
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']),sess_options=ort_options,providers=['CPUExecutionProvider'])
    scores,videos,hashes,onnx_inputs,first = {},{},{},{},{}
    lo,hi = np.asarray(p['joint_limits'],np.float32).T
    home = np.asarray(p['model_home'],np.float32)
    motors = arrays(record/'first-motor-commands.npz')
    np.testing.assert_array_equal(motors[ROLES[0]],motors[ROLES[1]])
    verify_resets(arrays(record/'resets-shared-prefix.npz'),prefix)
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        post = arrays(path)
        trace = {key:np.concatenate((prefix[key],post[key])) for key in prefix}
        assert trace['active'].shape==(p['maximum_control_steps'],p['parallel_worlds'])
        first[role] = {key:post[key][0] for key in ('qpos','actions','observations')}
        expected = np.clip(trace['actions']+home,lo,hi)
        onnx_inputs[role] = verify_inputs(trace,p,session,expected)
        verify_resets(arrays(record/f'resets-{role}.npz'),trace)
        scores[role] = score(trace,p,spec)
        clip = next(item for item in state['conditions'] if item['role']==role)['video']
        videos[role] = verify_movie(record,clip,trace,initial)
        hashes[role] = sha(path)
    differences = {key:float(abs(first[ROLES[0]][key]-first[ROLES[1]][key]).max()) for key in first[ROLES[0]]}
    remain = set(chosen.tolist()).intersection(*(set(scores[role]['near_static_worlds']) for role in ROLES))
    ended = {role:{r['world'] for r in scores[role]['worlds'] if not r['survived']} for role in ROLES}
    extra_ends = sorted((ended[ROLES[1]]-ended[ROLES[0]])&set(chosen.tolist()))
    gates = spec['capture']['gate']
    admission = dict(prefix_count=len(chosen)>=gates['minimum_captured_prefix_cases'],
                     persistent_count=len(remain)>=gates['minimum_still_near_static_after_six_seconds'],
                     first_action=differences['actions']<=gates['maximum_first_continued_action_difference_rad'],
                     first_qpos=differences['qpos']<=gates['maximum_first_continued_qpos_difference'],
                     no_additional_selected_ends=not extra_ends)
    result = OUT/'analysis.json'
    write(result,dict(status='passed-stall-capture-integrity',protocol_sha256=sha(PLAN),seed=seed,record=state['record'],
          transitions=spec['maximum_control_transitions'],physical_substeps=spec['maximum_physics_substeps'],
          eligible=chosen.tolist(),common_persistent=sorted(remain),extra_selected_ends=extra_ends,
          first_continuation_max_abs_difference=differences,gate=admission,capture_admitted=all(admission.values()),
          runtime_archive_sha256=captured['runtime_sha256'],runtime_archive_bytes=captured['runtime_bytes'],
          fork=restore['physical'],selected_state_sha256=captured['selected_sha256'],
          memory=telemetry['summary'],scores=scores,videos=videos,onnx_inputs=onnx_inputs,trace_sha256=hashes,
          shared_prefix_sha256=sha(source),learning_updates=0,heldout_used=False))
    state.update(status='completed',branch='closed',analysis_sha256=sha(result))
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    clip = videos[ROLES[1]]
    view.update(status='completed',updated_at=datetime.now(onset.TZ).isoformat(),purpose='后退停滞状态 · 已核验',
                latest_clip=(ROOT/clip['path']).relative_to(record).as_posix() if clip['complete_window'] else view['latest_clip'])
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('BACKWARD_STALL_CAPTURE_VERIFIED',len(chosen),len(remain),differences,admission,flush=True)


if __name__=='__main__':
    cli = ArgumentParser()
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--sample',action='store_true')
    mode.add_argument('--verify',action='store_true')
    args = cli.parse_args()
    (freeze if args.freeze else sample if args.sample else verify)()
