"""Freeze, supervise and independently verify the local roll PPO capacity trials."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
import supervised_ppo_process as supervisor
from record_io import write
from observed_getup_entry import finalize_view, viewing_confirmed
from monitored_ppo_authorization import AuthorizationMonitor
from native_ppo_observation import NativePPOObservation

ENTRY = Path(__file__).resolve()
WORK = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl')
PLAN = ROOT/'configs/experiments/roll_friction_ppo_capacity.json'
REPORT = ROOT/'records/reports/roll-friction-ppo-capacity'
STAGE = 'roll-ppo-capacity'
DRAFT = ROOT/'configs/experiments/roll_friction_exposure_training_draft.json'
SAMPLING = ROOT/'records/reports/roll-friction-sampling-capacity-bound/selection.json'
SECTIONS = [(3584,4096,4096,3584),(4096,4608,4608,4096)]
P = ROOT/'project.json'
TRIAL_BUDGET = 20


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def put(p,data):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def rel(p):
    return p.relative_to(ROOT).as_posix()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def compact_now():
    return datetime.now(timezone.utc).isoformat()


def command(key,action):
    return [sys.executable,'-B','-u',str(ENTRY),action,key]


def allowed(project,stage):
    s,g=project['staged_training_request'],project['goal_execution']
    return (g['status']=='active' and g['pause_requested_by_user'] is False
            and s['pause_requested'] is False and s['deployment_authorized_now'] is False
            and s['current_stage']==stage)


def freeze():
    p, draft, sampling = read(P),read(DRAFT),read(SAMPLING)
    s,g = p['staged_training_request'],p['goal_execution']
    cap = g['roll_ppo_capacity']
    require(allowed(p,STAGE+'-preflight') and not s['simulation_running'] and not s['learning_running']
            and cap['status']=='preflight-only' and sha(SAMPLING)==g['roll_sampling_capacity']['selection_sha256']
            and sampling['chosen_worlds']==3584 and sampling['larger_bounded_sampling_screen_warranted']
            and draft['learning_authorized'] is False and g['roll_friction_exposure']['learning_authorized'] is False,
            'Capacity is not the current approved activity')
    root_names = [
        'scripts/training/roll_friction_ppo.py','scripts/training/roll_friction_exposure.py',
        'scripts/training/roll_top_contact.py','scripts/training/roll_top_contact_pilot.py',
        'scripts/training/native_ppo_observation.py','scripts/training/native_ppo_viewer.py',
        'scripts/training/native_roll_observation.py','scripts/training/native_run_observer.py',
        'scripts/training/native_observer_guard.py','scripts/training/observed_getup_entry.py',
        'scripts/training/monitored_ppo_authorization.py','scripts/training/supervised_ppo_process.py',
        'scripts/training/parent_exit_exec.py','scripts/training/record_io.py','scripts/training/training_watch.py',
        'scripts/training/live_observer.py','scripts/training/action_video.py','scripts/training/native_getup_ppo_hooks.py',
        'scripts/validation/verify_roll_exposure_parent.py','scripts/validation/verify_roll_ppo_capacity.py',
        'scripts/validation/selective_friction_guard.py','scripts/validation/run_local_getup_ppo.py',
        'scripts/validation/local_getup_scale.py','scripts/setup/project_environment.sh',
        'configs/environments/action_recording.json',
    ]
    anchor=read(ROOT/draft['parent']['independent_validation'])
    require(anchor['status']=='verified-common-parent-checkpoint'
            and sha(ROOT/anchor['anchor'])==anchor['anchor_sha256']==draft['parent']['learning_checkpoint_sha256']
            and sha(ROOT/anchor['official_onnx'])==anchor['official_onnx_sha256'], 'PPO parent evidence differs')
    repeat=ROOT/'records/reports/roll-friction-exposure-preflight/cuda-repeat-zero.json'
    zero=read(repeat)
    require(zero['status']=='passed-repeated-cuda-reset-only' and not zero['violations']
            and zero['physics_substeps']==0,'Paired reset preflight missing')
    roots = [ROOT/n for n in root_names] + [ENTRY,DRAFT,SAMPLING,repeat,ROOT/anchor['anchor'],ROOT/anchor['official_onnx'],
                                           ROOT/draft['parent']['independent_validation']]
    training = read(ROOT/'configs/experiments/roll_friction_exposure_preflight.json')['local_training_sha256']
    require(all(sha(p)==h for p,h in training.items()),'Installed roll configuration changed')
    packages = WORK/'.venv/lib/python3.12/site-packages'
    extra_packages = ['rsl_rl/algorithms/ppo.py','rsl_rl/models/mlp_model.py','rsl_rl/runners/on_policy_runner.py',
                      'rsl_rl/utils/logger.py','mjlab/rl/runner.py','mjlab/rl/vecenv_wrapper.py']
    training.update({str(packages/f):sha(packages/f) for f in extra_packages})
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    profiles = {}
    sections=[]
    for section_id,order in enumerate(SECTIONS,1):
        ids=[]
        for offset,n in enumerate(order,1):
            key=f'phase-{section_id}-pass-{offset}-e{n}'
            ids.append(key)
            profiles[key]=dict(run_id=f'duck_roll_ppo-capacity_p{section_id}t{offset}_e{n}_s1249_{stamp}',
                environments=n,updates=TRIAL_BUDGET,transitions=n*TRIAL_BUDGET*24,friction_role='official-distribution')
        sections.append(ids)
    recipe=dict(task=draft['task'],seed=1249,rollout_controls=24,critic_only_updates=15,
        hard_gpu_fraction=.85,reserve_mib=768,min_improvement=.03,
        late_monitor_updates=[14],minimum_late_stability=.30,nominal_height_m=.139158,maximum_late_drop=.25,
        learning_rate=2e-5,initial_std=.05,parent_checkpoint=anchor['anchor'],parent_sha256=anchor['anchor_sha256'],
        official_onnx=anchor['official_onnx'],official_sha256=anchor['official_onnx_sha256'])
    spec=dict(status='frozen-observed-ppo-capacity',stage=STAGE,scope='capacity-profile',
              created_at=compact_now(),sections=sections,profiles=profiles,recipe=recipe,
              authorization='Learning is exclusively for measuring an optimizer workload; never nominate or publish these weights.',
              source_sha256={rel(f):sha(f) for f in roots},
              installed_sha256=training,supervisor_sha256=sha(Path(supervisor.__file__)),
              parent_exit_guard_sha256=sha(ROOT/'scripts/training/parent_exit_exec.py'),
              supervisor_workers={row['run_id']:command(key,'worker') for key,row in profiles.items()},
              supervision=dict(startup_seconds=325,stall_seconds=90,max_wall_seconds=900,
                 heartbeat_seconds=30,graceful_seconds=12,viewing_seconds=210),
              final_holdout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False)
    require(not PLAN.exists() and not REPORT.exists(),'Keep previous frozen capacity plan')
    REPORT.mkdir()
    put(PLAN,spec)
    cap.update(status='frozen-ready',protocol=rel(PLAN),sha256=sha(PLAN),next_trial=sections[0][0],
               active_section=0,result_report=rel(REPORT),reviewed={})
    s.update(current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
             next_gate='Verify first trial is visible in the original VS Code before learning.')
    write(P,p)
    print('ROLL_PPO_CAPACITY_FROZEN',sha(PLAN),sections[0][0],flush=True)


def checked(key):
    p,plan=read(P),read(PLAN)
    s,g = p['staged_training_request'],p['goal_execution']
    cap=g['roll_ppo_capacity']
    require(allowed(p,STAGE) and s['current_protocol_frozen'] and s['current_protocol']==rel(PLAN)
        and cap['sha256']==sha(PLAN) and cap['next_trial']==key and key in plan['profiles']
        and not any(plan[k] for k in ('hardware_io','final_holdout_used','automatic_retry','automatic_extension')),
        'Trial is not admitted')
    for base,section in ((ROOT,'source_sha256'),(Path('/'),'installed_sha256')):
        require(all(sha(base/f)==v for f,v in plan[section].items()),'Pinned source differs: '+section)
    require(plan['supervisor_workers'][plan['profiles'][key]['run_id']]==command(key,'worker'),'Worker command changed')
    return p,plan,plan['profiles'][key]


def source_for(row):
    return ROOT/'records/experiments'/(row['run_id']+'-native-state.json')


def confirmed(key,screenshot):
    p,plan,arm=checked(key)
    report=REPORT/key
    file=Path(screenshot).resolve()
    health=read(report/'native-view/native-health.json')
    fresh=datetime.fromisoformat(health['heartbeat_at'].replace('Z','+00:00'))
    seconds=(datetime.now(timezone.utc)-fresh).total_seconds()
    source=source_for(arm)
    require(health['health']=='fresh' and health['run_id']==arm['run_id'] and type(health['observer_pid']) is int
            and -5<=seconds<=15 and (ROOT/health['source']).resolve()==source.resolve()
            and file.parent==report.resolve() and file.read_bytes()[:3]==b'\xff\xd8\xff','Viewer proof is missing or does not match')
    put(report/'viewing-confirmation.json',dict(status='viewed-current-run',run_id=arm['run_id'],protocol_sha256=sha(PLAN),
        observer_pid=health['observer_pid'],source=rel(source),screenshot=rel(file),
        screenshot_sha256=sha(file),checked_at=compact_now()))
    print('PPO_VIEW_CONFIRMED',arm['run_id'],flush=True)


def worker(key):
    p,plan,arm=checked(key)
    report=REPORT/key
    sha_plan=sha(PLAN)
    expected=command(key,'worker')
    ident=arm['run_id']
    owner=os.getpid()
    args=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    require(args[-4:-1]==[os.fsencode(x) for x in command(key,'supervise')[-3:]],'Use the authorized supervisor')
    log=(report/'execution.log').stat()
    require(all((log.st_dev,log.st_ino)==(os.fstat(fd).st_dev,os.fstat(fd).st_ino) for fd in (1,2)),'Unrecorded process')
    start=time.monotonic()
    result=dict(status='stopped-review-required',reason='not-started',run_id=ident,plan_sha256=sha_plan,hardware_io=False)
    active=False
    monitor=None
    record=ROOT/'records/experiments'/ident

    def authorized():
        p=read(P)
        s,g=p['staged_training_request'],p['goal_execution']['roll_ppo_capacity']
        if (g['next_trial']!=key or not allowed(p,STAGE)
                or s['simulation_running'] is not active or s['learning_running'] is not active):
            return False
        if active and g.get('worker_pid')!=owner:
            return False
        return supervisor._authorized(PLAN,sha_plan,STAGE,ident,expected,before_start=False)

    def live_allowed():
        return authorized() if monitor is None else monitor.check()

    view=NativePPOObservation(ident,report,live_allowed,environments=arm['environments'],planned_updates=arm['updates'],
                              purpose=f'滚翻PPO容量短测 · {arm["environments"]}个环境')
    try:
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Supervisor ended the profile')))
        view.start()
        print('WAITING_FOR_CURRENT_VIEW',view.view,flush=True)
        until=time.monotonic()+plan['supervision']['viewing_seconds']
        while not viewing_confirmed(report/'viewing-confirmation.json',view,sha_plan):
            view.require_active()
            require(time.monotonic()<until,'No confirmed current VS Code view')
            time.sleep(.2)
        require(authorized(),'Approval changed before compute')
        p=read(P)
        p['staged_training_request'].update(simulation_running=True,learning_running=True)
        p['goal_execution']['roll_ppo_capacity'].update(worker_pid=owner,status='running')
        write(P,p)
        active=True
        monitor=AuthorizationMonitor(authorized).start()
        view.require_active()
        record.mkdir()
        put(record/'protocol-arm.json',dict(arm,plan_sha256=sha_plan,trial=key,scope=plan['scope']))
        import roll_friction_ppo
        proof=roll_friction_ppo.collect(plan,arm,record,view)
        result.update(status=proof['status'],reason=None,record=rel(record),
                      online_result_sha256=sha(record/'online-result.json'),elapsed_after_process_start=time.monotonic()-start)
    except BaseException as exc:
        result['reason']=type(exc).__name__+': '+str(exc)
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.close()
            result['authorization_monitor']=monitor.summary()
        try:
            view.close(result)
        finally:
            if active:
                p=read(P)
                g=p['goal_execution']['roll_ppo_capacity']
                require(g.get('worker_pid')==owner,'Worker PID differs from project')
                g.update(status='review-required',last_attempt=key,last_result=rel(report/'entry-result.json'))
                p['staged_training_request'].update(simulation_running=False,learning_running=False)
                write(P,p)
            put(report/'entry-result.json',result)
    return 0 if result['status']=='collected-review-required' else 2


def supervise(key):
    p,plan,arm=checked(key)
    s,g=p['staged_training_request'],p['goal_execution']['roll_ppo_capacity']
    report=REPORT/key
    require(g['status']=='frozen-ready' and not s['simulation_running'] and not s['learning_running'],
            'Another profile is active or needs review')
    report.mkdir()
    os.environ.update(MUJOCO_GL='egl',GALLIUM_DRIVER='d3d12',MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    timing={k:v for k,v in plan['supervision'].items() if k!='viewing_seconds'}
    receipt=supervisor.run_supervised(command(key,'worker'),cwd=ROOT,
        log_path=report/'execution.log',receipt_path=report/'supervisor-result.json',source=source_for(arm),
        run_id=arm['run_id'],protocol=PLAN,protocol_sha256=sha(PLAN),stage=STAGE,**timing)
    terminal=read(report/'entry-result.json') if (report/'entry-result.json').is_file() else {'status':'worker-not-confirmed'}
    if receipt['status']=='child-exited' and receipt['exit_code']==0 and terminal['status']=='collected-review-required':
        final=finalize_view(report,arm['run_id'],terminal)
        print('ROLL_PPO_CAPACITY_PROCESSED',key,terminal['status'],final['status'],flush=True)
        return 0 if final['status']=='saved' else 2
    print('ROLL_PPO_CAPACITY_STOP',key,receipt['status'],receipt['exit_code'],terminal.get('reason'),flush=True)
    return 2


def score_section(plan,reviewed,section):
    rows={n:[] for n in SECTIONS[section]}
    for key in plan['sections'][section]:
        analysis=read(ROOT/reviewed[key])
        proof=plan['profiles'][key]
        require(analysis['status']=='passed-independent-ppo-profile' and analysis['environments']==proof['environments']
                and analysis['run_id']==proof['run_id'],'Missing verified repeat')
        rows[analysis['environments']].append(analysis)
    small,big=sorted(rows)
    low_big=min(r['net_transitions_per_second'] for r in rows[big])
    high_small=max(r['net_transitions_per_second'] for r in rows[small])
    cap=plan['recipe']
    min_reserve={str(n):min(x['headroom_to_limit_mib'] for x in a) for n,a in rows.items()}
    eligible={str(n):r>=cap['reserve_mib'] for n,r in ((int(k),v) for k,v in min_reserve.items())}
    larger_wins=eligible[str(big)] and low_big>=high_small*(1+cap['min_improvement'])
    winner=big if larger_wins else (small if eligible[str(small)] else None)
    return dict(status='reviewed-two-way-profiles',phase=section+1,larger_wins=larger_wins,
                conservative_gain=low_big/high_small-1,eligible=eligible,provisional_worlds=winner,
                minimum_headroom_mib=min_reserve,
                net_rates={str(n):[r['net_transitions_per_second'] for r in a] for n,a in rows.items()})


def verify(key):
    p,plan,arm=checked(key)
    s,g=p['staged_training_request'],p['goal_execution']['roll_ppo_capacity']
    require(g['status']=='review-required' and not s['simulation_running'] and not s['learning_running'],'Profile not ready')
    from verify_roll_ppo_capacity import independent
    report=REPORT/key
    proof=independent(plan,PLAN,arm,report)
    out=report/'independent.json'
    put(out,proof)
    g['reviewed'][key]=rel(out)
    section=g['active_section']
    roles=plan['sections'][section]
    index=roles.index(key)
    if index+1<len(roles):
        g.update(status='frozen-ready',next_trial=roles[index+1])
    else:
        choice=score_section(plan,g['reviewed'],section)
        path=REPORT/f'phase-{section+1}-selection.json'
        put(path,choice)
        if section==0 and choice['larger_wins']:
            g.update(status='frozen-ready',active_section=1,next_trial=plan['sections'][1][0],first_selection=rel(path))
        else:
            g.update(status='verified-selection' if choice['provisional_worlds'] else 'review-required',
                     next_trial=None,chosen_environments=choice['provisional_worlds'],selection=rel(path))
            s.update(current_stage=STAGE+'-complete',current_evidence=rel(path),
                     next_gate='Use selected learner scale for a paired two-arm zero-step admission.')
    write(P,p)
    print('ROLL_PPO_CAPACITY_VERIFIED',key,proof['net_transitions_per_second'],g['next_trial'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','supervise','worker','confirm','verify'))
    parser.add_argument('trial',nargs='?')
    parser.add_argument('--screenshot')
    args=parser.parse_args()
    if args.action=='freeze':
        return freeze()
    if not args.trial:
        parser.error('trial required')
    if args.action=='confirm':
        return confirmed(args.trial,args.screenshot)
    return {'worker':worker,'supervise':supervise,'verify':verify}[args.action](args.trial)


if __name__=='__main__':
    sys.exit(main())
