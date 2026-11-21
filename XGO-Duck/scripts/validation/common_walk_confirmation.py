"""Frozen common walking regression; one visible shared terminal, no learning.

Noise is already common. Preserve the fixed controller and motion criteria,
project modelHOME+raw destinations, and batch official inference on GPU after
CPU/live parity checks. Two fresh development seeds gate the final heldout.
Any integrity or functional failure closes this protocol without resampling.
This confirms isolated walking, not untested cross-model entry or sequences.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,inspect,json,os,shutil,subprocess,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import WORK,read,sha
from record_io import write
from common_walk_targets import prepare,processor_fixture,JointPositionAction
from common_walk_startup import make_actor,td,MLPModel

PLAN=ROOT/'configs/experiments/common_walk_confirmation.json'
OUT=ROOT/'records/reports/common-walk-confirmation'
STAGE='S5-common-walking-confirmation'


def freeze():
    from mjlab.scene import Scene
    from mjlab.envs import ManagerBasedRlEnv
    from common_scene_contract import model_arrays,compare_arrays
    import collect_common_walk,verify_common_walk
    assert not PLAN.exists() and not (OUT/'preflight.json').exists()
    audit=read(OUT/'interface-audit.json');decision=read(OUT/'target-decision.json')
    assert audit['status']=='passed-cpu-saved-data' and audit['old_noise_already_common']
    assert decision['status']=='passed-cpu-only'
    for name,digest in decision['source_sha256'].items():assert sha(ROOT/name)==digest,name
    base=read(ROOT/'configs/experiments/common_plant/walk.json')
    seeds=[1040,1041,2040];stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    for directory in ('records/experiments','records/evaluations'):
        assert not any(any(f'_s{seed}_' in f.name for seed in seeds) for f in (ROOT/directory).iterdir())
    sources=deepcopy(base['source_sha256'])
    # Capture the actual imported project helpers, including the noise adapter
    # and its source dependencies, rather than silently editing frozen parents.
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        path=Path(value).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT/'scripts'):
            sources[str(path.relative_to(ROOT))]=sha(path)
    for name in ('configs/environments/action_recording.json','configs/environments/training_display.json',
        'scripts/validation/inspect_common_walk_interface.py','scripts/validation/check_common_walk_targets.py'):
        sources[name]=sha(ROOT/name)
    plans={};preflights={};library={}
    for seed in seeds:
        p=deepcopy(base);p.update(stage=STAGE,seed=seed,assignment_seed=70000+seed,
            model_home=audit['model_home'],source_sha256=sources,heldout_used=seed==seeds[-1],
            video_seconds=4.,video_fps=25,output_root=str(OUT),learning_updates=0,
            run_id=f'duck_walk_common-projection_reference_s{seed}_{stamp}',
            scope='Isolated fixed-controller walking in common fullcontact plant with legal destinations; no continuous-entry or hardware claim')
        permutation=np.random.default_rng(p['assignment_seed']).permutation(224)
        p['groups']={name:permutation[i*32:(i+1)*32].tolist() for i,name in enumerate(p['commands'])}
        cfg=prepare(p);fixture=processor_fixture(p)
        assert not cfg.auto_reset and cfg.scene.num_envs==224 and cfg.episode_length_s==20.
        arrays=model_arrays(Scene(cfg.scene,device='cpu').compile())
        with np.load(OUT/'cpu-scene-fields.npz') as f:compare_arrays(arrays,{k:f[k] for k in f.files})
        actor=make_actor(p,'cpu')
        with np.load(OUT/'cpu-import-inputs.npz') as f:x=f['observations'];expected=f['onnx_actions']
        with torch.no_grad():actual=actor(td(torch.from_numpy(x))).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        for cls in (JointPositionAction,ManagerBasedRlEnv,MLPModel,type(actor.obs_normalizer)):
            path=Path(inspect.getfile(cls));library[str(path)]=sha(path)
        preflights[seed]=dict(status='passed-cpu-only',config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
            full_scene_exact=True,processor=fixture,cpu_batch_inputs=len(x),maximum_cpu_batch_error=float(abs(actual-expected).max()),
            cuda_initialized=False,physical_steps=0)
        plans[seed]=p
    assert not torch.cuda.is_initialized()
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name,digest in base['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    references={str(path.relative_to(ROOT)):sha(path) for path in
        (OUT/'interface-audit.json',OUT/'target-decision.json',OUT/'cpu-scene-fields.npz',OUT/'cpu-import-inputs.npz')}
    directory=ROOT/'configs/experiments/common_walk';assert not directory.exists();directory.mkdir()
    paths={}
    for seed,p in plans.items():
        p['library_source_sha256']=library
        path=directory/f's{seed}.json';write(path,p);paths[str(seed)]=str(path.relative_to(ROOT))
        destination=OUT/f's{seed}';destination.mkdir()
        write(destination/'preflight.json',dict(preflights[seed],protocol_sha256=sha(path)))
    m=dict(status='frozen-before-dynamics',stage=STAGE,stamp=stamp,ordered_seeds=seeds,
        development_seeds=seeds[:2],heldout_seed=seeds[-1],protocols=paths,
        protocol_sha256={k:sha(ROOT/name) for k,name in paths.items()},source_sha256=sources,reference_sha256=references,
        library_source_sha256=library,maximum_transitions=672000,maximum_first_episodes=672,
        learning_updates=0,automatic_retry=False,automatic_extension=False,hardware_io=False,
        stop_rule='Stop after any integrity or functional failure. Heldout only after both development seeds pass every unchanged command gate.',
        success_rule='Original percommand firstepisode functional criteria and>=95percent successes, no target or posture threshold relaxation',
        target_rule='Always project absolute modelHOME+raw; raw history unchanged; no sustained recovery2rad/s cap',
        scope='Common-physics isolated seven-command regression only; fullS2-S6 remains open')
    write(PLAN,m);write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),
        source_count=len(sources),scene_sha256=references[str((OUT/'cpu-scene-fields.npz').relative_to(ROOT))],
        seeds=seeds,max_transitions=672000,cuda_initialized=False))
    print('COMMON_WALK_FROZEN',sha(PLAN),len(sources),'sources',seeds,'0learning',flush=True)


def checked(seed):
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in m['reference_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in m['library_source_sha256'].items():assert sha(Path(name))==digest,name
    file=ROOT/m['protocols'][str(seed)];assert sha(file)==m['protocol_sha256'][str(seed)]
    p=read(file);assert sha(ROOT/p['model'])==p['model_sha256']
    for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    return m,p,file


def evaluate(seed):
    from collect_common_walk import collect
    from evaluate_standup_reference import Tee
    m,p,file=checked(seed);out=OUT/f's{seed}';assert not (out/'progress.json').exists(),'No retry or extension'
    if p['heldout_used']:
        assert all(read(OUT/f's{x}/analysis.json')['functional_development_passed'] for x in m['development_seeds'])
    folder=ROOT/'records/evaluations'/p['run_id'];folder.mkdir();(folder/'videos').mkdir()
    shutil.copy2(file,folder/'protocol.json')
    for name in p['source_sha256']:
        destination=folder/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    state=dict(status='running',run_id=p['run_id'],path=str(folder),pid=os.getpid(),seed=seed,
        num_envs=32,parallel_worlds=224,episode_seconds=20,conditions=[],model=str(ROOT/p['model']),
        model_sha256=p['model_sha256'],protocol_sha256=sha(file),learning_updates=0,heldout_used=p['heldout_used'],
        started_at=datetime.now(timezone.utc).isoformat(),hardware_io=False,formal_acceptance_eligible=True,scope=p['scope'])
    def save():write(folder/'evaluation.json',state);write(out/'progress.json',state)
    save();write(folder/'command.json',dict(argv=sys.argv,pid=os.getpid(),foreground_tty=os.isatty(1)))
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=p['run_id'],pid=os.getpid()))
    if p['heldout_used']:write(OUT/'heldout-consumption.json',dict(seed=seed,run_id=p['run_id'],retry_allowed=False,started_at=state['started_at']))
    groups={k:np.asarray(v) for k,v in p['groups'].items()};desired=np.zeros((224,3),np.float32)
    for name,ids in groups.items():desired[ids]=p['commands'][name]
    with (folder/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:collect(p,read(out/'preflight.json'),folder,state,save,desired,groups,224,1000)
        finally:sys.stdout,sys.stderr=stdout,stderr


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--evaluate',type=int);parser.add_argument('--verify',type=int);a=parser.parse_args()
    if a.freeze:freeze();return
    if a.evaluate:
        assert os.isatty(1);evaluate(a.evaluate);return
    if a.verify:
        from verify_common_walk import verify
        m,p,_=checked(a.verify);verify(ROOT/'records/evaluations'/p['run_id']);return
    assert os.isatty(1) and read(ROOT/'project.json')['staged_training_request']['current_stage']==STAGE
    m=read(PLAN);assert not (OUT/'suite-progress.json').exists(),'No repeated suite'
    progress=dict(status='running',pid=os.getpid(),completed_seeds=[],heldout_used=False,learning_updates=0)
    write(OUT/'suite-progress.json',progress)
    project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    try:
        for seed in m['ordered_seeds']:
            for action in ('--evaluate','--verify'):
                subprocess.run([sys.executable,'-B',str(Path(__file__)),action,str(seed)],cwd=WORK,check=True)
            result=read(OUT/f's{seed}/analysis.json');progress['completed_seeds'].append(seed)
            progress['heldout_used']|=seed==m['heldout_seed']
            if not result['functional_development_passed']:
                progress.update(status='closed-functional-failed',failed_seed=seed);break
            write(OUT/'suite-progress.json',progress)
        else:progress['status']='isolated-seven-command-confirmed'
    except BaseException as error:
        progress.update(status='stopped-integrity-review-required',error=repr(error));raise
    finally:
        progress['finished_at']=datetime.now(timezone.utc).isoformat();write(OUT/'suite-progress.json',progress)
        project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)
    print('COMMON_WALK_SUITE_CLOSED',json.dumps(progress),flush=True)


if __name__=='__main__':main()
