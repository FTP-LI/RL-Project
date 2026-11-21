"""Separate first-episode collection repair after the original seed1041 stop.

Honor the installed mjlab reset API only for ended, permanently unscored
worlds. Preserve the old46-source run and every old score/failure. New seeds,
an independent source freeze and explicit reset evidence keep this recovery
separate from any policy, target, criterion or learned-weight improvement.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
import common_walk_confirmation as base
import collect_common_walk
import collect_common_walk_resets
from ended_world_resets import cpu_check
from record_io import write
from s1_reference_inputs import WORK,read,sha
from common_walk_targets import prepare,processor_fixture
from common_walk_startup import make_actor,td

PARENT=ROOT/'configs/experiments/common_walk_confirmation.json'
OLD=ROOT/'records/reports/common-walk-confirmation'
PLAN=ROOT/'configs/experiments/common_walk_reset_recovery.json'
OUT=OLD/'reset-recovery'
STAGE='S5-common-walking-reset-recovery'
base.PLAN,base.OUT,base.STAGE=PLAN,OUT,STAGE
collect_common_walk.collect=collect_common_walk_resets.collect


def parent_evidence():
    m=read(PARENT)
    assert sha(PARENT)=='a1be791e28c82a4f5eea203543e164a54dfc06fc57236fa4e482237d424ff1f3'
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    suite=read(OLD/'suite-progress.json')
    assert suite['status']=='stopped-integrity-review-required' and suite['completed_seeds']==[1040]
    assert not (OLD/'heldout-consumption.json').exists()
    first=read(OLD/'s1040/analysis.json')
    assert first['functional_development_passed'] and sum(r['successes'] for r in first['rows'])==223
    p=read(ROOT/m['protocols']['1041']);folder=ROOT/'records/evaluations'/p['run_id']
    runtime=read(folder/'runtime-contract.json');report=read(folder/'evaluation.json')
    assert runtime['control_steps']==79 and runtime['control_transitions']==17696
    assert report['status']=='failed' and 'must be reset' in report['error']
    terminated=[]
    for name,ids in p['groups'].items():
        item=next(r for r in report['conditions'] if r['condition']==name)
        path=folder/f'{name}-trace.npz';assert sha(path)==item['trace_sha256']
        with np.load(path) as f:
            assert f['active'].shape==(79,32)
            terminated.extend((name,ids[int(i)]) for i in np.flatnonzero(f['terminated'][-1]|f['truncated'][-1]))
    assert terminated==[('backward',98)],terminated
    return dict(status='passed-cpu-parent-failure-audit',parent_sha256=sha(PARENT),
        source_count=len(m['source_sha256']),original_first_seed_successes=223,
        original_control_transitions=241696,partial_controls=79,failed_world=98,
        failed_condition='backward',heldout_consumed=False,learning_updates=0,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in
            (OLD/'suite-progress.json',OLD/'s1040/analysis.json',folder/'evaluation.json',folder/'runtime-contract.json')})


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    assert not PLAN.exists() and not (OUT/'preflight.json').exists()
    proof=parent_evidence();fixture=cpu_check();parent=read(PARENT)
    sources=deepcopy(parent['source_sha256'])
    for name in ('ended_world_resets.py','collect_common_walk_resets.py',
                 'verify_common_walk_resets.py','recover_common_walk_resets.py'):
        path=ROOT/'scripts/validation'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    seeds=[1042,1043,2040];stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    for directory in ('records/experiments','records/evaluations'):
        assert not any(any(f'_s{seed}_' in f.name for seed in seeds) for f in (ROOT/directory).iterdir())
    OUT.mkdir();write(OUT/'parent-failure-audit.json',proof);write(OUT/'cpu-reset-check.json',fixture)
    for name in ('cpu-scene-fields.npz','cpu-import-inputs.npz'):
        shutil.copy2(OLD/name,OUT/name);assert sha(OLD/name)==sha(OUT/name)
    old=read(ROOT/parent['protocols']['1040']);directory=ROOT/'configs/experiments/common_walk_resets'
    assert not directory.exists();directory.mkdir();paths={};preflights={}
    for seed in seeds:
        p=deepcopy(old);p.update(stage=STAGE,seed=seed,assignment_seed=70000+seed,
            source_sha256=sources,heldout_used=seed==2040,output_root=str(OUT),
            run_id=f'duck_walk_terminal-reset_reference_s{seed}_{stamp}',
            reset_rule='Only mandatory ended unscored worlds; first-episode masks never reopen; terminal captured before reset',
            parent_protocol=str(PARENT.relative_to(ROOT)),parent_actual_transitions=241696)
        permutation=np.random.default_rng(p['assignment_seed']).permutation(224)
        p['groups']={name:permutation[i*32:(i+1)*32].tolist() for i,name in enumerate(p['commands'])}
        cfg=prepare(p);processor=processor_fixture(p)
        assert not cfg.auto_reset and cfg.scene.num_envs==224 and cfg.episode_length_s==20.
        arrays=model_arrays(Scene(cfg.scene,device='cpu').compile())
        with np.load(OUT/'cpu-scene-fields.npz') as f:compare_arrays(arrays,{k:f[k] for k in f.files})
        actor=make_actor(p,'cpu')
        with np.load(OUT/'cpu-import-inputs.npz') as f:x=f['observations'];expected=f['onnx_actions']
        with torch.no_grad():actual=actor(td(torch.from_numpy(x))).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        preflights[seed]=dict(status='passed-cpu-only',config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
            full_scene_exact=True,processor=processor,reset_fixture=fixture,
            cpu_batch_inputs=len(x),maximum_cpu_batch_error=float(abs(actual-expected).max()),
            cuda_initialized=False,physical_steps=0)
        path=directory/f's{seed}.json';write(path,p);paths[str(seed)]=str(path.relative_to(ROOT))
        destination=OUT/f's{seed}';destination.mkdir()
        write(destination/'preflight.json',dict(preflights[seed],protocol_sha256=sha(path)))
    assert not torch.cuda.is_initialized()
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name,digest in old['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for name,digest in parent['library_source_sha256'].items():assert sha(Path(name))==digest,name
    references=deepcopy(parent['reference_sha256'])
    for path in (PARENT,OUT/'parent-failure-audit.json',OUT/'cpu-reset-check.json',
                 OUT/'cpu-scene-fields.npz',OUT/'cpu-import-inputs.npz'):
        references[str(path.relative_to(ROOT))]=sha(path)
    m=dict(status='frozen-before-dynamics',stage=STAGE,stamp=stamp,ordered_seeds=seeds,
        development_seeds=seeds[:2],heldout_seed=2040,protocols=paths,
        protocol_sha256={k:sha(ROOT/name) for k,name in paths.items()},source_sha256=sources,
        reference_sha256=references,library_source_sha256=parent['library_source_sha256'],
        parent_protocol_sha256=sha(PARENT),parent_actual_transitions=241696,
        maximum_transitions=672000,maximum_first_episodes=672,learning_updates=0,
        automatic_retry=False,automatic_extension=False,hardware_io=False,
        stop_rule=parent['stop_rule'],success_rule=parent['success_rule'],target_rule=parent['target_rule'],
        reset_rule='Only already ended unscored worlds; log terminal and postreset qpos/qvel; active physics/action histories exact; masks never reopen',
        scope='Separate collection repair, unchanged model/targets/physics/criteria; isolated seven commands only; whole S2-S6 remains open')
    write(PLAN,m);write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),
        source_count=len(sources),seeds=seeds,max_transitions=672000,original_actual_transitions=241696,
        cuda_initialized=False,physical_steps=0))
    print('COMMON_WALK_RESET_FROZEN',sha(PLAN),len(sources),'sources',seeds,'0learning',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--evaluate',type=int);parser.add_argument('--verify',type=int);a=parser.parse_args()
    if a.freeze:freeze();return
    if a.evaluate:
        assert os.isatty(1);base.evaluate(a.evaluate);return
    if a.verify:
        from verify_common_walk_resets import verify
        _,p,_=base.checked(a.verify);verify(ROOT/'records/evaluations'/p['run_id']);return
    assert os.isatty(1) and read(ROOT/'project.json')['staged_training_request']['current_stage']==STAGE
    m=read(PLAN);assert not (OUT/'suite-progress.json').exists(),'No repeated suite'
    progress=dict(status='running',pid=os.getpid(),completed_seeds=[],heldout_used=False,learning_updates=0,
        parent_actual_transitions=241696)
    write(OUT/'suite-progress.json',progress)
    project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    try:
        for seed in m['ordered_seeds']:
            for action in ('--evaluate','--verify'):
                subprocess.run([sys.executable,'-B',str(Path(__file__)),action,str(seed)],cwd=WORK,check=True)
            result=read(OUT/f's{seed}/analysis.json');progress['completed_seeds'].append(seed)
            progress['heldout_used']|=seed==2040
            if not result['functional_development_passed']:
                progress.update(status='closed-functional-failed',failed_seed=seed);break
            write(OUT/'suite-progress.json',progress)
        else:progress['status']='isolated-seven-command-confirmed'
    except BaseException as error:
        progress.update(status='stopped-integrity-review-required',error=repr(error));raise
    finally:
        progress['finished_at']=datetime.now(timezone.utc).isoformat();write(OUT/'suite-progress.json',progress)
        project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)
    print('COMMON_WALK_RESET_SUITE_CLOSED',json.dumps(progress),flush=True)


if __name__=='__main__':main()
