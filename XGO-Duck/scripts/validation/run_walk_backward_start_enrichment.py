"""Run a fixed-policy pair that changes only the declared backward reset slots."""
from argparse import ArgumentParser
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import shutil
from unittest.mock import patch

from mjlab.scene import Scene
import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import collect_walk_backward_onset as onset
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
import verify_walk_backward_onset as verifier
from walk_backward_start_reset import DifficultBackwardReleases, add_difficult_backward_event


PREVIOUS = ROOT/'configs/experiments/common_walk_backward_start_enrichment.json'
PLAN = PREVIOUS.with_stem(PREVIOUS.stem+'_recovery')
DRAFT = PREVIOUS.with_stem(PREVIOUS.stem+'_draft')
HISTORY_OUT = ROOT/'records/reports/common-walk-backward-start-enrichment'
OUT = HISTORY_OUT/'recovery'
PROJECT = ROOT/'project.json'
SCENE = onset.OUT/'cpu-scene-fields.npz'
STAGE = 'S2-backward-start-enrichment-recovery'
ARMS = ('control','candidate')
HISTORY = ('_action','_prev_action','_prev_prev_action')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def page(arm):
    return ROOT/f'records/walk-backward-enrichment-s1129-{arm}-recovery-live.html'


def environment(spec, arm, seed):
    assert seed==spec['development_seed']
    return dict(spec['environment'],seed=seed,run_id=spec['paired_runs'][arm])


def record_for(spec, arm):
    return ROOT/'records/evaluations'/spec['paired_runs'][arm]


def load_states(spec):
    path = ROOT/spec['reset_readback_archive']
    assert sha(path)==spec['reference_sha256'][rel(path)]
    with np.load(path) as data:
        return data['source_local_qpos'],data['source_qvel']


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    stage = read(PROJECT)['staged_training_request']
    draft = read(DRAFT)
    original = read(PREVIOUS)
    stop_path = HISTORY_OUT/'zero-step-stop.json'
    stop = read(stop_path)
    assert stage['current_stage']==original['stage'] and ROOT/stage['current_protocol']==PREVIOUS
    assert stage['current_protocol_frozen'] and not stage['simulation_running']
    assert stop['status']=='stopped-before-env-construction' and stop['protocol_sha256']==sha(PREVIOUS)
    assert not any(stop['actual'].values()) and not stop['prior_seed_consumed']
    stopped = ROOT/stop['record']
    assert stopped==record_for(original,'control') and sha(stopped/'protocol.json')==sha(PREVIOUS)
    assert not any(read(stopped/'evaluation.json')['actual'].values())
    archived = HISTORY_OUT/'stopped-source'
    for name,digest in stop['original_sources'].items():
        assert sha(archived/name)==digest==original['source_sha256'][f'scripts/validation/{name}']
    keys = ('cpu_reset_readback','cuda_reset_readback','retention','reference_plant','initial_cases')
    paths = {key:ROOT/draft[key][0] for key in keys}
    for key,path in paths.items():
        assert sha(path)==draft[key][1]
    cpu,cuda = [read(paths[f'{device}_reset_readback']) for device in ('cpu','cuda')]
    for proof in (cpu,cuda):
        assert proof['status']=='passed-zero-step-reset-readback' and proof['unselected_state_and_histories_exact']
        assert proof['source_case_count']==draft['difficult_slots']
        assert not any(proof[k] for k in ('physical_substeps','learning_updates','heldout_used'))
    reset_code = ROOT/'scripts/training/walk_backward_start_reset.py'
    assert sha(reset_code)==cuda['reset_code_sha256']==cpu['reset_code_sha256']
    retained = read(paths['retention'])
    assert retained['total_samples']==3072 and not retained['heldout_used']
    assert read(paths['initial_cases'])['reset_readback_admitted']
    saved = paths['cuda_reset_readback'].with_name('cuda-readback.npz')
    assert sha(saved)==cuda['readback_sha256']
    spec = deepcopy(read(paths['reference_plant']))
    onset.prior.check_sources(spec)
    seed,n,h = (draft[key] for key in ('first_development_seed','parallel_worlds_per_arm','controls_per_arm'))
    assert n==128 and h==300 and draft['difficult_slots']==63
    assert draft['maximum_transitions_per_pair']==len(ARMS)*n*h
    assert draft['maximum_batched_physics_substeps_per_pair']==len(ARMS)*5*h
    attempts = [entry for folder in ('records/evaluations','records/training-archive/evaluations')
                for entry in (ROOT/folder).iterdir() if f'_s{seed}_' in entry.name]
    assert attempts and all(entry.samefile(stopped) for entry in attempts)
    assert not any(page(arm).exists() for arm in ARMS)
    stamp = datetime.now(onset.TZ).strftime('%Y%m%dT%H%M%S')
    runs = {arm:f'duck_walk_backward-enrichment_{arm}-recovery_s{seed}_{stamp}' for arm in ARMS}
    p = spec['environment']
    assert p['parallel_worlds']==n and p['maximum_control_steps']==h and spec['phase']['warmup_controls']==0
    p.update(seed=seed,stage=STAGE)
    old = read(ROOT/'records/reports/common-walk-backward-load-clock/seed1111-analysis.json')
    witness = ROOT/old['record']/'windows-view-verification.json'
    assert read(witness)['status']=='passed'
    spec.update(stage=STAGE,development_seed=seed,paired_runs=runs,maximum_control_transitions=n*h,
                full_budget=len(ARMS)*n*h,maximum_physics_substeps=5*h,difficult_slots=list(range(draft['difficult_slots'])),
                reset_readback_archive=rel(saved),windows_reference=rel(witness),pair_gate=draft['gate'],
                state_sample_control=100,unselected_divergence_threshold=1e-6)
    for path in (DRAFT,PREVIOUS,stop_path,*paths.values(),saved,witness,*(archived.iterdir())):
        spec['reference_sha256'][rel(path)] = sha(path)
    for path in (Path(__file__),Path(__file__).with_suffix('.sh'),reset_code,
                 Path(__file__).with_name('verify_walk_live_page.py')):
        spec['source_sha256'][rel(path)] = sha(path)
    cfg,_ = onset.config(environment(spec,'control',seed))
    with np.load(SCENE) as archive:
        compare_arrays(model_arrays(Scene(cfg.scene,device='cpu').compile()),{key:archive[key] for key in archive.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    for arm in ARMS:
        (OUT/arm).mkdir()
        page(arm).write_text(f'<!doctype html><meta charset="utf-8"><title>XGO-Duck</title>'
                             f'<h1>后退初态复验</h1><p>等待采集 · {runs[arm]}</p>',encoding='utf-8')
    scene = OUT/'cpu-scene-fields.npz'
    shutil.copy2(SCENE,scene)
    spec['reference_sha256'][rel(scene)] = sha(scene)
    onset.prior.check_sources(spec)
    write(PLAN,spec)
    write(OUT/'frozen.json',dict(status='passed-cpu-freeze',protocol_sha256=sha(PLAN),runs=runs,
                                physical_substeps=0,learning_updates=0))
    print('BACKWARD_ENRICHMENT_FROZEN',sha(PLAN),runs,flush=True)


def checked():
    spec = read(PLAN)
    assert read(OUT/'frozen.json')['protocol_sha256']==sha(PLAN)
    onset.prior.check_sources(spec)
    return spec


def rng(device):
    host = onset.prior.cpu
    return {key:sha256(host(value).tobytes()).hexdigest() for key,value in
            (('cpu',torch.get_rng_state()),('cuda',torch.cuda.get_rng_state(device)))}


def initial_archive(env, obs, desired):
    host = onset.prior.cpu
    return dict(qpos=host(env.sim.data.qpos),qvel=host(env.sim.data.qvel),ctrl=host(env.sim.data.ctrl),
                origin=host(env.scene.env_origins),actor=host(obs['actor']),command=desired.copy(),
                **{key:host(getattr(env.action_manager,key)) for key in HISTORY})


def sample(arm):
    from mjlab import envs

    spec = checked()
    seed = spec['development_seed']
    entry = read(OUT/'windows-entry.json')
    assert entry['status']=='passed' and entry['protocol_sha256']==sha(PLAN)
    assert entry['pages'][arm]==rel(page(arm))
    hard = np.asarray(spec['difficult_slots'])
    if arm=='candidate':
        control = record_for(spec,'control')
        control_zero = read(control/'reset-startup.json')
        control_proof = read(OUT/'control'/f'seed{seed}-analysis.json')
        assert control_zero['status']=='passed' and control_proof['status']=='passed-integrity'
        assert control_proof['protocol_sha256']==sha(PLAN)
    local_qpos,qvel = load_states(spec)
    original_env,original_config = envs.ManagerBasedRlEnv,onset.config
    original_start,original_collect = onset.startup,onset.prior.collect

    def config(p):
        cfg,display = original_config(p)
        if arm=='candidate':
            add_difficult_backward_event(cfg)
        return cfg,display

    class CandidateEnv(original_env):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.difficult_backward_releases = DifficultBackwardReleases(self,hard,local_qpos,qvel)

    def startup(env, obs, actor, p, desired, options, record):
        origin_rng = rng(env.device)
        initial = initial_archive(env,obs,desired)
        assert not any(initial[key].any() for key in HISTORY) and not initial['actor'][:,34:48].any()
        if arm=='candidate':
            ref = control/'reset-startup.npz'
            assert sha(ref)==control_zero['archive_sha256'] and origin_rng==control_zero['rng']
            regular = np.setdiff1d(np.arange(env.num_envs),hard)
            with np.load(ref) as data:
                for key in ('qpos','qvel','ctrl','actor'):
                    np.testing.assert_array_equal(initial[key][regular],data[key][regular],err_msg=key)
                for key in ('command','origin',*HISTORY):
                    np.testing.assert_array_equal(initial[key],data[key],err_msg=key)
            target = local_qpos.copy()
            target[:,:3] += initial['origin'][hard]
            np.testing.assert_array_equal(initial['qpos'][hard],target)
            np.testing.assert_array_equal(initial['qvel'][hard],qvel)
        original_start(env,obs,actor,p,desired,options,record)
        assert rng(env.device)==origin_rng and env.common_step_counter==env._sim_step_counter==0
        path = record/'reset-startup.npz'
        np.savez_compressed(path,**initial)
        write(record/'reset-startup.json',dict(status='passed',rng=origin_rng,arm=arm,archive_sha256=sha(path),
              unselected_matched_to_control=arm=='candidate',physical_substeps=0))
        (record/'README.md').write_text('# 后退初态复验\n\n'
            f'[运行观察](../../{page(arm).name}) · [冻结方案](../../../configs/experiments/common_walk_backward_start_enrichment_recovery.json)'
            ' · [采集日志](execution.log)\n',encoding='utf-8')
        print('BACKWARD_ENRICHMENT_CUDA_ZERO',arm,flush=True)

    def collect(env, actor, options, record, state, desired):
        real_step,real_reset = env.step,env.reset
        resets = []

        def step(action):
            out = real_step(action)
            if env.common_step_counter==spec['state_sample_control']:
                values = initial_archive(env,out[0],desired)
                values['critic'] = onset.prior.cpu(out[0]['critic'])
                path = record/'two-second-state.npz'
                np.savez_compressed(path,**values)
                write(record/'two-second-state.json',dict(control=env.common_step_counter,sha256=sha(path),
                      raw_history=True,full_solver_snapshot=False))
            return out

        def reset(**kwargs):
            prior_rng = rng(env.device)
            out = real_reset(**kwargs)
            resets.append(dict(control=env.common_step_counter,worlds=onset.prior.cpu(kwargs['env_ids']).tolist(),
                               rng_before=prior_rng,rng_after=rng(env.device)))
            return out

        with patch.object(env,'step',step),patch.object(env,'reset',reset):
            try:
                original_collect(env,actor,options,record,state,desired)
            finally:
                write(record/'selective-reset-rng.json',dict(resets=resets,final_rng=rng(env.device)))

    # Load the native wrapper before binding the candidate class. The control
    # arm retains the exact original environment class.
    original_config(environment(spec,arm,seed))
    wrapped = patch.object(envs,'ManagerBasedRlEnv',CandidateEnv) if arm=='candidate' else nullcontext()
    with wrapped,patch.multiple(
         onset,PLAN=PLAN,OUT=OUT,STAGE=STAGE,checked=checked,config=config,startup=startup,
         environment=lambda options,key:environment(options,arm,key),page=lambda _:page(arm)), \
         patch.object(onset.prior,'collect',collect):
        onset.sample(seed)


def verify(arm):
    spec = checked()
    seed = spec['development_seed']
    record = record_for(spec,arm)
    snapshot = read(record/'two-second-state.json')
    path = record/'two-second-state.npz'
    assert snapshot['sha256']==sha(path) and snapshot['control']==spec['state_sample_control']
    with np.load(path) as data,np.load(record/'trace-backward.npz') as trace:
        k = snapshot['control']-1
        np.testing.assert_array_equal(data['qpos'],trace['qpos'][k])
        np.testing.assert_array_equal(data['_action'],trace['actions'][k])
        assert np.isfinite(data['qvel']).all() and data['actor'].shape==(128,61)
    with patch.multiple(verifier,PLAN=PLAN,OUT=OUT/arm,page=lambda _:page(arm),checked=checked,
                        environment=lambda options,key:environment(options,arm,key)):
        verifier.verify(seed)


def compare():
    spec = checked()
    seed = spec['development_seed']
    path = OUT/'decision.json'
    assert not path.exists() and not torch.cuda.is_initialized()
    records = {arm:record_for(spec,arm) for arm in ARMS}
    proofs = {arm:read(OUT/arm/f'seed{seed}-analysis.json') for arm in ARMS}
    for arm,proof in proofs.items():
        assert proof['status']=='passed-integrity' and proof['protocol_sha256']==sha(PLAN)
        assert not proof['heldout_used'] and not proof['learning_updates']
        assert proof['source_sha256']==sha(records[arm]/'trace-backward.npz')
        view = read(records[arm]/'windows-view-verification.json')
        assert view['status']=='passed' and view['observer_host']=='Windows'
    zero = {arm:read(record/'reset-startup.json') for arm,record in records.items()}
    assert zero['candidate']['unselected_matched_to_control'] and zero['control']['rng']==zero['candidate']['rng']
    hard = set(spec['difficult_slots'])
    stalled = {arm:sorted(hard&set(proof['analysis']['near_static_worlds'])) for arm,proof in proofs.items()}
    ended = {arm:{row['world'] for row in proof['analysis']['worlds'] if not row['survived']} for arm,proof in proofs.items()}
    unsafe = sorted((ended['candidate']-ended['control'])&hard)
    added = len(stalled['candidate'])-len(stalled['control'])
    gate = spec['pair_gate']
    passed = (added>=gate['minimum_net_additional_near_static_in_difficult_slots'] and
              len(unsafe)<=gate['maximum_candidate_only_early_endings_in_difficult_slots'])
    regular = sorted(set(range(spec['environment']['parallel_worlds']))-hard)
    traces = {arm:verifier.arrays(record/'trace-backward.npz') for arm,record in records.items()}
    differences = {}
    for field in ('qpos','observations','actions'):
        a,b = (traces[arm][field][:,regular] for arm in ARMS)
        diff = np.max(np.abs(a-b),axis=(1,2))
        different = np.flatnonzero(diff>spec['unselected_divergence_threshold'])
        exact = np.flatnonzero(diff!=0.)
        differences[field] = dict(first_control_over_tolerance=int(different[0]+1) if len(different) else None,
                                  first_nonexact_control=int(exact[0]+1) if len(exact) else None,
                                  maximum_abs_difference=float(diff.max()))
    for field in (slice(34,48),slice(48,51)):
        np.testing.assert_array_equal(traces['control']['observations'][0,regular,field],
                                      traces['candidate']['observations'][0,regular,field])
    reset = {arm:read(record/'selective-reset-rng.json') for arm,record in records.items()}
    write(path,dict(status='passed-reset-pair-integrity',protocol_sha256=sha(PLAN),seed=seed,
          records={arm:rel(record) for arm,record in records.items()},
          score_sha256={arm:sha(OUT/arm/f'seed{seed}-analysis.json') for arm in ARMS},
          actual_transitions=sum(p['transitions'] for p in proofs.values()),
          actual_physical_substeps=sum(p['physical_substeps'] for p in proofs.values()),
          difficult_slots=sorted(hard),near_static_in_difficult_slots=stalled,net_additional_near_static=added,
          candidate_only_early_endings=unsafe,unselected_differences=differences,
          selective_resets=reset,final_rng_equal=reset['control']['final_rng']==reset['candidate']['final_rng'],
          replication_admitted=passed,learning_updates=0,heldout_used=False))
    print('BACKWARD_ENRICHMENT_DECISION', {arm:len(values) for arm,values in stalled.items()},
          'net',added,'candidate-only early endings',unsafe,'replication',passed,flush=True)


if __name__=='__main__':
    cli = ArgumentParser()
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--sample',choices=ARMS)
    mode.add_argument('--verify',choices=ARMS)
    mode.add_argument('--compare',action='store_true')
    args = cli.parse_args()
    if args.freeze:
        freeze()
    elif args.sample:
        sample(args.sample)
    elif args.verify:
        verify(args.verify)
    else:
        compare()
