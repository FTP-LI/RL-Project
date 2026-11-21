"""Exercise the real entry, supervisor and observer around a synthetic worker.

All project flags and zero-step receipts in the fixture are synthetic. The real
project and historical training files are never changed; no Torch is imported.
Evidence distinguishes these process checks from training/physics acceptance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import observed_getup_entry as entry

OUT = ROOT / "records/reports/native-ppo-observation-preflight/observed-entry-check"
WORKER = '''"""Synthetic fixture worker: no Torch, simulator, model or optimizer."""
import json, time
from pathlib import Path
from getup_ppo_live import publish
from record_io import write
ROOT = Path(__file__).resolve().parents[2]
(ROOT/'trainer-imported.txt').write_text('synthetic module imported')
PLAN = None
def check_project():
    raise RuntimeError('Entry must replace this callback')
class MemoryWatch:
    def check(self):
        check_project()
def worker(plan, role):
    assert PLAN.is_file()
    arm=plan['arms'][role]; record=ROOT/'records/experiments'/arm['run_id']
    record.mkdir(); watch=MemoryWatch(); rows=[]
    state=dict(run_id=arm['run_id'], arm=role, status='gpu-zero-step', controls=0,
               completed_updates=0, images=[], clips=[])
    write(record/'state.json',state)
    publish(record,state,rows,4,2,{'peak_mib':0})
    for update in (1,2):
        watch.check()
        state.update(status='training',controls=24*update,completed_updates=update)
        rows.append(dict(update=update,controls=24*update,environment_reward_mean=.1*update,
                         policy_loss=.01,value_loss=.02,net_transitions_per_second=96.0))
        write(record/'state.json',state)
        publish(record,state,rows,4,2,{'peak_mib':0})
        time.sleep(.4)
    if arm['fixture_mode'] in ('pause','stall'):
        while True:
            watch.check()
            time.sleep(.1)
    state.update(status='collected-review-required')
    write(record/'state.json',state)
    publish(record,state,rows,4,2,{'peak_mib':0})
    write(record/'online-result.json',dict(status='collected-review-required',actual_updates=2,
          actual_transitions=192,fixture_only=True))
'''


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    options=parser.parse_args()
    OUT=options.output.resolve()
    if not OUT.is_relative_to((ROOT/'records/reports/native-ppo-observation-preflight').resolve()):
        raise ValueError('Check output must be in the observation-preflight report directory')
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Use the installed WSL Python')
    OUT.mkdir(parents=True,exist_ok=True)
    result_path=OUT/'verification.json'
    if result_path.exists():
        raise FileExistsError('Keep existing check evidence')
    real_project_sha=sha(ROOT/'project.json')
    sources=sorted(entry.REQUIRED_SOURCES|{'scripts/validation/figure_style.py'})
    source_hashes={name:sha(ROOT/name) for name in sources}
    cases={}; receipts={}; errors=[]
    fixture=Path(tempfile.mkdtemp(prefix='synthetic-entry-',dir=OUT)).resolve()
    assert fixture.parent==OUT.resolve()
    children=[]
    try:
        with ZipFile(OUT/'fixture-logs.zip','x',ZIP_DEFLATED) as archive:
            for mode in ('normal','missing_view','wrong_view','pause','stall'):
                root=fixture/mode
                for name in sources:
                    dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(ROOT/name,dest)
                (root/'scripts/validation/run_local_getup_ppo.py').write_text(WORKER,encoding='utf-8')
                (root/'scripts/training/training_live_page.py').write_text(
                    "from pathlib import Path\nROOT=Path(__file__).resolve().parents[2]\nPAGE=ROOT/'records/training-live.html'\n",encoding='utf-8')
                config=root/'configs/experiments';config.mkdir(parents=True)
                records=root/'records';(records/'experiments').mkdir(parents=True)
                (root/'home/XGO-Duck').mkdir(parents=True)
                environment=dict(os.environ,HOME=str(root/'home'),MPLCONFIGDIR=str(root/'home/mpl-cache'))
                report=records/'reports/process-fixture/probe';report.mkdir(parents=True)
                zero=records/'zero/probe';zero.mkdir(parents=True)
                dependency=root/'home/XGO-Duck/workspace/xgoduck_rl/fixture.py'
                dependency.parent.mkdir(parents=True);dependency.write_text('# synthetic dependency\n')
                inherited=config/'inherited.txt';inherited.write_text('frozen synthetic inherited source')
                parent=config/'parent.json';write(parent,{'training':{'candidate':{
                    'source_sha256':{'configs/experiments/inherited.txt':sha(inherited)},
                    'training_source_sha256':{'fixture.py':sha(dependency)},
                    'library_source_sha256':{str(dependency):sha(dependency)}}}})
                (zero/'paired-start.npz').write_bytes(b'fixture no physical state')
                (zero/'model-init.pt').write_bytes(b'fixture no model')
                plan_path=config/'observed.json'
                script=root/'scripts/training/observed_getup_entry.py'
                run_id='synthetic_observed_'+mode
                source_manifest={name:sha(root/name) for name in sources}
                source_manifest['configs/experiments/parent.json']=sha(parent)
                source_manifest['configs/experiments/inherited.txt']=sha(inherited)
                plan=dict(status='frozen-observed-getup-ppo',stage='synthetic-observed-entry',
                    purpose='合成接线检查（非训练）',heldout_used=False,automatic_retry=False,automatic_extension=False,
                    seed=1,parent_protocol='configs/experiments/parent.json',zero_step_record='records/zero',
                    coordinator='records/reports/process-fixture',adaptive_settings={},limits_rad=[.05,.25],
                    step_scales=[1.0],memory_fraction_limit=.85,complete_episodes_per_arm=0,
                    active_samples_per_arm=0,final_hold_samples_per_arm=0,
                    arms={'probe':dict(run_id=run_id,num_envs=4,training_updates=2,fixture_mode=mode)},
                    limits={'probe':dict(rollout_controls=24,controls=48,transitions=192)},
                    source_sha256=source_manifest,
                    supervisor_sha256=sha(root/'scripts/training/supervised_ppo_process.py'),
                    parent_exit_guard_sha256=sha(root/'scripts/training/parent_exit_exec.py'),
                    supervision=dict(startup_seconds=45,viewing_seconds=4 if mode=='missing_view' else 15,
                        stall_seconds=6,max_wall_seconds=55,heartbeat_seconds=15,graceful_seconds=6,poll_seconds=.1))
                command=[sys.executable,'-B',str(script),'--mode','worker','--protocol',str(plan_path),'--arm','probe']
                plan['supervisor_workers']={run_id:command}
                settings={k:plan[k] for k in ('seed','parent_protocol','adaptive_settings','limits_rad','step_scales',
                    'memory_fraction_limit','complete_episodes_per_arm','active_samples_per_arm','final_hold_samples_per_arm')}
                settings.update(arm='probe',parameters=plan['arms']['probe'],limits=plan['limits']['probe'],parent_protocol_sha256=sha(parent))
                settings_sha=hashlib.sha256(json.dumps(settings,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
                write(zero/'result.json',dict(status='passed',envs=4,physical_substeps=0,learning_updates=0,
                    training_run_id=run_id,training_settings_sha256=settings_sha,training_source_sha256=source_manifest,
                    parent=dict(matched_networks_and_adam=True,physical_rng_unchanged=True,
                        start_archive_sha256=sha(zero/'paired-start.npz'),initial_checkpoint_sha256=sha(zero/'model-init.pt'))))
                plan['zero_result_sha256']={'probe':sha(zero/'result.json')};write(plan_path,plan)
                project=dict(goal_execution=dict(status='active',pause_requested_by_user=False),
                    staged_training_request=dict(pause_requested=False,current_stage=plan['stage'],
                        current_protocol_frozen=True,current_protocol='configs/experiments/observed.json',
                        learning_running=False,simulation_running=False))
                write(root/'project.json',project)

                def check_cli(expected, detail):
                    argv=list(command);argv[argv.index('worker')]='check'
                    completed=subprocess.run(argv,cwd=root,env=environment,capture_output=True,text=True,timeout=15)
                    decision=json.loads(completed.stdout.strip().splitlines()[-1])
                    cases[detail]=(completed.returncode==expected and not (root/'trainer-imported.txt').exists())
                    receipts[detail]=decision

                if mode=='normal':
                    check_cli(0,'admitted_read_only_without_trainer_import')
                    dependency.write_text('# changed synthetic dependency\n')
                    check_cli(2,'changed_wsl_dependency_refused')
                    dependency.write_text('# synthetic dependency\n')
                    frozen_inherited=plan['source_sha256'].pop('configs/experiments/inherited.txt')
                    write(plan_path,plan)
                    check_cli(2,'missing_inherited_source_refused')
                    plan['source_sha256']['configs/experiments/inherited.txt']=frozen_inherited;write(plan_path,plan)
                    project['staged_training_request']['pause_requested']=True;write(root/'project.json',project)
                    check_cli(2,'pause_refuses_before_trainer_import')
                    project['staged_training_request']['pause_requested']=False;write(root/'project.json',project)
                    old=(zero/'result.json').read_bytes()
                    proof=read(zero/'result.json');proof['training_run_id']='previous-run';write(zero/'result.json',proof)
                    plan['zero_result_sha256']['probe']=sha(zero/'result.json');write(plan_path,plan)
                    check_cli(2,'previous_run_zero_evidence_refused')
                    (zero/'result.json').write_bytes(old);plan['zero_result_sha256']['probe']=sha(zero/'result.json');write(plan_path,plan)
                    completed=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True,timeout=15)
                    cases['direct_worker_without_supervisor_refused']=(completed.returncode==2 and not (root/'trainer-imported.txt').exists())
                argv=list(command);argv[argv.index('worker')]='supervise'
                log=root/'launcher.log'
                with log.open('w',encoding='utf-8') as output:
                    child=subprocess.Popen(argv,cwd=root,env=environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(child);deadline=time.monotonic()+60;confirmed=False;paused=False;observer_pid=None
                    health=report/'native-view/native-health.json';state=records/'experiments'/run_id/'state.json'
                    while child.poll() is None and time.monotonic()<deadline:
                        if health.is_file():
                            h=read(health)
                            if h.get('health')=='fresh' and h.get('observer_pid'):observer_pid=h['observer_pid']
                            if h.get('health')=='fresh' and not confirmed and mode!='missing_view':
                                image=report/'synthetic-view.txt';image.write_text('fixture acknowledgement, not a desktop screenshot')
                                write(report/'viewing-confirmation.json',dict(status='viewed-current-run',
                                    run_id=run_id if mode!='wrong_view' else 'another-run',protocol_sha256=sha(plan_path),
                                    observer_pid=h['observer_pid'],source=h['source'],checked_at=datetime.now(timezone.utc).isoformat(),
                                    screenshot=image.relative_to(root).as_posix(),screenshot_sha256=sha(image)))
                                confirmed=True
                        if mode=='pause' and state.is_file() and not paused and read(state).get('completed_updates',0)>0:
                            project['staged_training_request']['pause_requested']=True;write(root/'project.json',project);paused=True
                        time.sleep(.1)
                    if child.poll() is None:
                        os.killpg(child.pid,15);child.wait(timeout=8)
                        raise TimeoutError('Fixture launcher did not exit: '+mode)
                receipt=read(report/'entry-result.json')
                sup=read(report/'supervisor-result.json')
                receipts[mode]={'entry':receipt,'supervisor':sup,'exit_code':child.returncode}
                if mode=='normal':
                    cases['observed_worker_completes_under_supervisor']=(child.returncode==0 and receipt['status']=='collected-review-required' and receipt['worker_called'])
                    cases['new_run_overview_references_current_plan']=('observed.json' in (state.parent/'README.md').read_text() and 'local_getup_ppo.json' not in (state.parent/'README.md').read_text())
                    final_rows=list(csv.DictReader((report/'final-view/native-metrics.csv').open(encoding='utf-8')))
                    cases['final_view_contains_last_update']=len(final_rows)==2 and float(final_rows[-1]['environment_reward_mean'])==.2
                    cases['final_chart_matches_last_worker_snapshot']=((report/'final-view/native-curve.png').is_file()
                        and read(report/'final-view/native-curve-source.json')['source_sha256']==sha(state.parent/'observation.json'))
                    cases['final_view_is_historical']=read(report/'final-view/native-health.json')['mode']=='historical'
                else:
                    cases[mode+'_stops_without_success']=(child.returncode!=0 and receipt['status']=='stopped-review-required')
                    if mode in ('missing_view','wrong_view'):
                        cases[mode+'_does_not_import_trainer']=not (root/'trainer-imported.txt').exists()
                    if mode=='pause':cases['pause_was_injected_after_progress']=paused
                final=read(records/'experiments'/(run_id+'-native-state.json'))
                cases[mode+'_observer_finalized']=final['status']=='finished'
                if observer_pid is None:
                    cases[mode+'_observer_exited']=False
                else:
                    try:os.kill(observer_pid,0);cases[mode+'_observer_exited']=False
                    except ProcessLookupError:cases[mode+'_observer_exited']=True
                for folder in (report,root):
                    files=folder.rglob('*') if folder==report else [log]
                    for f in files:
                        if f.is_file():archive.write(f,str(Path(mode)/f.relative_to(root)))
        cases['real_project_unchanged']=sha(ROOT/'project.json')==real_project_sha
        cases['fixture_imported_no_torch_in_check_process']='torch' not in sys.modules
        cases['entry_source_unchanged_during_checks']=sha(entry.ENTRY)==source_hashes['scripts/training/observed_getup_entry.py']
    except BaseException as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,15);child.wait(timeout=8)
    result=dict(status='passed-synthetic-entry-process-check' if all(cases.values()) and not errors else 'failed',
        checked_at=datetime.now(timezone.utc).isoformat(),cases=cases,errors=errors,receipts=receipts,
        source_sha256=source_hashes,test_sha256=sha(Path(__file__)),physical_steps=0,learning_updates=0,
        fixture_zero_receipts_are_synthetic=True,desktop_viewing_verified=False,real_training_verified=False,
        scope='Actual entry, supervisor, hooks and observer processes around a synthetic worker in an isolated project; no training or simulation.')
    write(result_path,result)
    if result['status']=='passed-synthetic-entry-process-check':
        assert fixture.parent==OUT.resolve() and fixture.name.startswith('synthetic-entry-')
        shutil.rmtree(fixture)
    print(json.dumps({k:v for k,v in result.items() if k!='receipts'},ensure_ascii=False))
    return 0 if result['status']=='passed-synthetic-entry-process-check' else 1


if __name__=='__main__':
    raise SystemExit(main())
