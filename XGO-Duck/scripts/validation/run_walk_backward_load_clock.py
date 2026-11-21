"""Observe solved foot loads in the unchanged backward rollout."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from inspect import getsourcefile
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

import collect_walk_backward_onset as onset
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_support_transfer import first_support
from roll_contact_clock_capture import ContactClock, LABELS
from s1_reference_inputs import ROOT, read, sha
import verify_walk_backward_onset as verifier


STAGE = 'S2-backward-load-clock-development'
PLAN = ROOT/'configs/experiments/common_walk_backward_load_clock.json'
DRAFT = PLAN.with_stem(PLAN.stem+'_draft')
OUT = ROOT/'records/reports/common-walk-backward-load-clock'
PROJECT = ROOT/'project.json'
PARENT = onset.PLAN
PARENT_SCENE = onset.OUT/'cpu-scene-fields.npz'
ROLL_DECISION = ROOT/'records/reports/common-walk-backward-roll-pair/seed1103-decision.json'


def page(seed):
    return ROOT/f'records/walk-backward-load-clock-s{seed}-live.html'


def relative(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    import mujoco_warp
    from mjlab.scene import Scene

    assert not PLAN.exists() and not torch.cuda.is_initialized()
    project = read(PROJECT)['staged_training_request']
    assert ROOT/project['current_protocol']==DRAFT and not project['simulation_running']
    draft,static = read(DRAFT),read(OUT/'preflight.json')
    assert draft['stage']==project['current_stage'] and not project['current_protocol_frozen']
    assert static['status']=='passed-cpu-load-channel-zero-step' and static['draft_sha256']==sha(DRAFT)
    assert static['script_sha256']==sha(Path(__file__).with_name('check_walk_backward_load_clock.py'))
    for path,digest in draft['sources'].values():
        assert sha(ROOT/path)==digest
    spec = deepcopy(read(PARENT))
    onset.prior.check_sources(spec)
    design = draft['proposed_development_probe']
    seed = design['seed']
    assert not any(f'_s{seed}_' in f.name for folder in ('records/evaluations','records/training-archive/evaluations')
                   for f in (ROOT/folder).iterdir())
    p = spec['environment']
    n,h = design['parallel_worlds'],design['controls']
    assert p['maximum_control_steps']==h and n*h==design['maximum_control_transitions']
    assert spec['maximum_physics_substeps']==design['maximum_batched_substeps']
    assert spec['phase']['warmup_controls']==0 and not spec['heldout_used'] and not spec['learning_updates']
    stamp = datetime.now(onset.TZ).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_walk_backward-load-clock_development_s{seed}_{stamp}'
    p.update(seed=seed,stage=STAGE,parallel_worlds=n,episodes_per_command=n,
             groups={'backward':list(range(n))},maximum_transitions=n*h)
    witness = ROOT/read(ROLL_DECISION)['record']/'windows-view-verification.json'
    assert read(witness)['status']=='passed'
    spec.update(stage=STAGE,runs={str(seed):run_id},load_clock=design,
                maximum_control_transitions=n*h,full_budget=n*h,
                maximum_sampling_wall_seconds=design['maximum_sampling_wall_seconds'],
                force_witness_controls=[81,100,150,200,250,300],
                matched_window_controls=[-8,-design['minimum_precontact_lead_controls']],
                knee_description=dict(excursion_rad=.03,persistence_controls=3,response_controls=3,
                                      swing_load_share=.35),
                scope='Observe solved sole loads before integration. Original actor, scene, command, raw history and joint processing are unchanged.',
                windows_reference=relative(witness))
    for path in (PARENT,DRAFT,OUT/'preflight.json',ROLL_DECISION,witness):
        spec['reference_sha256'][relative(path)] = sha(path)
    for path in (Path(__file__),Path(__file__).with_suffix('.sh'),
                 Path(__file__).with_name('roll_contact_clock_capture.py'),
                 Path(__file__).with_name('verify_walk_live_page.py'),
                 Path(__file__).with_name('review_walk_backward_support_transfer.py')):
        spec['source_sha256'][relative(path)] = sha(path)
    native = Path(getsourcefile(mujoco_warp.contact_force))
    spec['library_source_sha256'][str(native)] = sha(native)
    cfg,_ = onset.config(onset.environment(spec,seed))
    assert cfg.decimation==5 and cfg.sim.mujoco.timestep==design['physics_dt']
    fields = model_arrays(Scene(cfg.scene,device='cpu').compile())
    with np.load(PARENT_SCENE) as archive:
        compare_arrays(fields,{name:archive[name] for name in archive.files})
    scene = OUT/'cpu-scene-fields.npz'
    shutil.copy2(PARENT_SCENE,scene)
    spec['reference_sha256'][relative(scene)] = sha(scene)
    onset.prior.check_sources(spec)
    assert not torch.cuda.is_initialized()
    write(PLAN,spec)
    write(OUT/'frozen.json',dict(status='passed-cpu-freeze',protocol_sha256=sha(PLAN),record=run_id,
                                same_physical_fields=True,physical_substeps=0,learning_updates=0))
    assert not page(seed).exists()
    page(seed).write_text(f'<!doctype html><meta charset="utf-8"><title>XGO-Duck</title>'
                         f'<h1>后退脚底受力</h1><p>等待采集 · {run_id}</p>'
                         f'<p><a href="reports/common-walk-backward-load-clock/README.md">本轮方案</a></p>',
                         encoding='utf-8')
    print('BACKWARD_LOAD_CLOCK_FROZEN',sha(PLAN),run_id,flush=True)


def checked():
    spec = read(PLAN)
    assert read(OUT/'frozen.json')['protocol_sha256']==sha(PLAN)
    onset.prior.check_sources(spec)
    return spec


class WalkingLoadClock(ContactClock):
    def __init__(self, env, spec):
        design = spec['load_clock']
        super().__init__(env,dict(capture_seconds=(0.,design['controls']*design['control_dt']),
                                 maximum_batched_substeps=spec['maximum_physics_substeps'],
                                 load_threshold_nominal_weight_fraction=design['minimum_combined_sole_load_fraction_body_weight']))
        assert env.physics_dt==design['physics_dt'] and env.step_dt==design['control_dt']
        self.begin,self.end = design['capture_controls']
        self.witness_controls = spec['force_witness_controls']
        self.witnesses = {}

    def record(self, original_step):
        n = self.env._sim_step_counter
        assert n==self.completed+1 and n<=self.maximum
        control,phase = divmod(n,self.env.cfg.decimation)
        take = phase==0 and self.begin<=control<=self.end
        sim = self.env.sim
        if take:
            before = onset.prior.cpu(sim.data.qpos)
        original_step()
        self.completed = n
        if take:
            points,loads = self.read_contacts()
            after = onset.prior.cpu(sim.data.qpos)
            self.rows.append(dict(control=control,substep=n,before=before,after=after,points=points,load_n=loads))
            if control in self.witness_controls:
                raw = sim.wp_data
                count = int(raw.nacon.numpy()[0])
                for key,array in dict(geom=raw.contact.geom,world=raw.contact.worldid,force=self.forces).items():
                    self.witnesses[f'{key}_{control}'] = array.numpy()[:count].copy()

    def preserve(self, folder):
        if not self.rows:
            return
        archive = folder/'load-clock.npz'
        witnesses = folder/'load-clock-witnesses.npz'
        np.savez_compressed(archive,**{k:np.asarray([row[k] for row in self.rows]) for k in self.rows[0]})
        np.savez_compressed(witnesses,**self.witnesses)
        write(folder/'load-clock.json',dict(labels=LABELS,nominal_weight_n=self.weight,floor=self.floor,
                    kinds=self.kind.tolist(),slots=self.slots,completed_substeps=self.completed,
                    recorded_controls=len(self.rows),complete=len(self.rows)==self.end-self.begin+1,
                    archive_sha256=sha(archive),witnesses_sha256=sha(witnesses)))


def sample():
    spec = checked()
    seed, = map(int,spec['runs'])
    entry = read(OUT/'windows-entry.json')
    assert entry['status']=='passed' and entry['protocol_sha256']==sha(PLAN)
    assert entry['page']==relative(page(seed)) and page(seed).is_file()
    start = onset.startup
    collect = onset.prior.collect
    clock = None

    def zero(env, obs, actor, p, desired, options, record):
        nonlocal clock
        (record/'README.md').write_text(
            '# 后退脚底受力\n\n[冻结方案](../../../configs/experiments/common_walk_backward_load_clock.json)'
            ' · [运行观察](../../walk-backward-load-clock-s1111-live.html) · [执行日志](execution.log)\n',
            encoding='utf-8')
        start(env,obs,actor,p,desired,options,record)
        clock = WalkingLoadClock(env,spec)
        cpu = onset.prior.cpu
        sim = env.sim.data
        before = {key:cpu(getattr(sim,key)) for key in ('qpos','qvel','ctrl')}
        rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state(env.device)
        points,loads = clock.read_contacts()
        for key,value in before.items():
            np.testing.assert_array_equal(cpu(getattr(sim,key)),value)
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(env.device))
        assert env.common_step_counter==env._sim_step_counter==0
        assert points.shape==loads.shape==(env.num_envs,len(LABELS)) and np.isfinite(loads).all()
        write(record/'load-clock-zero.json',dict(status='passed',physical_substeps=0,clock_shape=list(loads.shape),
                    physical_state_exact=True,rng_exact=True,nominal_weight_n=clock.weight))
        print('BACKWARD_LOAD_CLOCK_CUDA_ZERO',clock.slots,flush=True)

    def observed(env, actor, options, record, state, desired):
        assert clock is not None
        original = env.sim.step
        with patch.object(env.sim,'step',lambda:clock.record(original)):
            try:
                collect(env,actor,options,record,state,desired)
            finally:
                clock.preserve(record)

    with patch.multiple(onset,PLAN=PLAN,OUT=OUT,STAGE=STAGE,page=page,checked=checked,startup=zero), \
         patch.object(onset.prior,'collect',observed):
        onset.sample(seed)


def verify_witnesses(record, status, archive, spec):
    path = record/'load-clock-witnesses.npz'
    assert sha(path)==status['witnesses_sha256']
    kinds,floor = status['kinds'],status['floor']
    assert floor>=0 and kinds[floor]==-1
    with np.load(path) as raw:
        for control in spec['force_witness_controls']:
            location = np.flatnonzero(archive['control']==control).item()
            counts = np.zeros_like(archive['points'][location])
            loads = np.zeros_like(archive['load_n'][location])
            for pair,world,wrench in zip(raw[f'geom_{control}'],raw[f'world_{control}'],raw[f'force_{control}'],strict=True):
                if (pair==floor).sum()!=1:
                    continue
                label = kinds[pair[pair!=floor].item()]
                if label<0:
                    continue
                assert 0<=world<len(counts) and wrench[0]>=-1e-5
                for channel in (0,label) if label else (0,):
                    counts[world,channel] += 1
                    loads[world,channel] += max(0.,float(wrench[0]))
            np.testing.assert_array_equal(counts,archive['points'][location])
            np.testing.assert_array_equal(loads,archive['load_n'][location])


def load_result(trace,score,archive,status,spec):
    design = spec['load_clock']
    data = archive['load_n'][...,2:]
    total = data.sum(axis=-1,keepdims=True)
    load = (total>=status['nominal_weight_n']*design['minimum_combined_sole_load_fraction_body_weight'])
    shares = np.divide(data,total,out=np.zeros_like(data),where=total>0.)
    loaded = load & (shares>=design['minimum_support_share'])
    persistence = design['persistence_controls']
    start = design['capture_controls'][0]
    stall = score['near_static_worlds']
    matched = []
    left,right = spec['matched_window_controls']
    knee = spec['knee_description']
    names = spec['environment']['joint_names']

    def find(world,contact,side):
        indices = np.arange(contact+left,contact+right+1)-start
        assert indices.min()>=0 and indices.max()<len(loaded)
        flags = loaded[indices,world,:]
        runs = np.lib.stride_tricks.sliding_window_view(flags,persistence,axis=0).all(axis=-1)
        choices = np.argwhere(runs if side is None else runs[:,side:side+1])
        return int(left+choices[0,0]+persistence-1) if len(choices) else None

    def knee_response(world,contact,support_side):
        swing = 1-support_side
        joint = names.index(('left','right')[swing]+'_knee')
        first,last = contact+left-1,contact+right-1
        steps = slice(first,last+1)
        if not trace['active'][steps,world].all():
            return None
        target = trace['applied_joint_targets'][steps,world,joint]
        flags = abs(target-target[0])>=knee['excursion_rad']
        persistent = knee['persistence_controls']
        candidates = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(flags,persistent).all(axis=-1))
        if not len(candidates):
            return None
        confirmed = first+int(candidates[0])+persistent-1
        after = confirmed+knee['response_controls']
        if after>=len(trace['active']) or not trace['active'][confirmed:after+1,world].all():
            return None
        sample = confirmed+1-start
        measured = trace['joint_positions'][confirmed:after+1,world,joint]
        initial_angle = trace['joint_positions'][first,world,joint]
        return dict(target_confirm_control=confirmed+1,swing_load_share=float(shares[sample,world,swing]),
                    swing_loaded=bool(load[sample,world,0] and shares[sample,world,swing]>=knee['swing_load_share']),
                    measured_motion=bool(np.max(abs(measured-initial_angle))>=knee['excursion_rad']))

    for item in score['worlds']:
        onset_block = item['onset_block_end_seconds'] or 0
        if onset_block<=2 or not item['survived']:
            continue
        world = item['world']
        result = first_support(trace,world,100,onset_block*50)
        if result is None:
            continue
        index,side = result
        control = index+1
        matched.append(dict(world=world,contact_control=control,stance=('left','right')[side],
                            before_contact_lead_controls=find(world,control,side),
                            knee=knee_response(world,control,side),
                            stalls=[dict(world=i,lead_controls=find(i,control,None),
                                         knee=knee_response(i,control,side)) for i in stall]))
    n,m = len(matched),len(stall)
    starters_marked = sum(row['before_contact_lead_controls'] is not None for row in matched)
    stalls_marked = sum(case['lead_controls'] is not None for row in matched for case in row['stalls'])
    a,b = starters_marked/max(n,1),stalls_marked/max(n*m,1)
    gate = dict(coverage=n>=design['coverage_minimum_late_starts'] and m>=design['coverage_minimum_near_static_worlds'],
                starter_rate=a>=design['minimum_starter_rate'],
                separation=a-b>=design['minimum_starter_minus_stall_rate'])
    knee_groups = ([row['knee'] for row in matched],[case['knee'] for row in matched for case in row['stalls']])
    knee_summary = {}
    for name,group in zip(('starters','stall_windows'),knee_groups,strict=True):
        samples = [item for item in group if item is not None]
        knee_summary[name] = dict(target_windows=len(samples),swing_loaded=sum(item['swing_loaded'] for item in samples),
                                  measured_motion=sum(item['measured_motion'] for item in samples),
                                  loaded_without_motion=sum(item['swing_loaded'] and not item['measured_motion']
                                                            for item in samples))
    return dict(eligible_late_starts=n,near_static_worlds=m,starters_marked=starters_marked,
                stall_windows=n*m,stall_windows_marked=stalls_marked,starter_rate=a,stall_rate=b,
                gate=gate,replication_admitted=all(gate.values()),descriptive_knee=knee_summary,rows=matched)


def verify():
    spec = checked()
    seed, = map(int,spec['runs'])
    p = onset.environment(spec,seed)
    record = ROOT/'records/evaluations'/p['run_id']
    zero,status = (read(record/name) for name in ('load-clock-zero.json','load-clock.json'))
    assert zero['status']=='passed' and zero['physical_substeps']==0 and status['complete']
    assert zero['nominal_weight_n']==status['nominal_weight_n'] and status['labels']==list(LABELS)
    path = record/'load-clock.npz'
    assert sha(path)==status['archive_sha256'] and not torch.cuda.is_initialized()
    trace_path = record/'trace-backward.npz'
    trace = verifier.arrays(trace_path)
    with np.load(path) as source:
        archive = {key:source[key] for key in source.files}
    design = spec['load_clock']
    first,last = design['capture_controls']
    np.testing.assert_array_equal(archive['control'],np.arange(first,last+1))
    np.testing.assert_array_equal(archive['substep'],archive['control']*5)
    np.testing.assert_array_equal(archive['after'],trace['qpos'][first-1:last])
    assert archive['before'].shape==archive['after'].shape and np.isfinite(archive['before']).all()
    assert status['completed_substeps']==design['maximum_batched_substeps']
    points,loads = archive['points'],archive['load_n']
    assert points.shape==loads.shape==(last-first+1,p['parallel_worlds'],len(LABELS))
    assert np.isfinite(loads).all() and (loads>=0).all() and (points>=0).all()
    assert (loads[points==0]==0).all() and (points[...,0]>=points[...,1:].sum(axis=-1)).all()
    verify_witnesses(record,status,archive,spec)
    result = load_result(trace,verifier.score(trace,p,spec),archive,status,spec)
    with patch.multiple(verifier,PLAN=PLAN,OUT=OUT,page=page,checked=checked):
        verifier.verify(seed)
    base = OUT/f'seed{seed}-analysis.json'
    assert read(base)['source_sha256']==sha(trace_path)
    write(OUT/f'seed{seed}-load-analysis.json',dict(status='passed-load-clock-integrity',seed=seed,
           protocol_sha256=sha(PLAN),source_sha256=sha(base),trace_sha256=sha(trace_path),
           archive_sha256=sha(path),independent_raw_force_witnesses=spec['force_witness_controls'],
           endpoint_qpos_exact=True,clock_offset_seconds=-design['physics_dt'],
           physical_substeps=status['completed_substeps'],learning_updates=0,heldout_used=False,**result))
    print('BACKWARD_LOAD_CLOCK_VERIFIED',seed,{key:value for key,value in result.items() if key!='rows'},flush=True)


if __name__=='__main__':
    cli = ArgumentParser()
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--sample',action='store_true')
    mode.add_argument('--verify',action='store_true')
    args = cli.parse_args()
    (freeze if args.freeze else sample if args.sample else verify)()
