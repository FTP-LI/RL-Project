"""Profile the actual get-up sampler, then run a bounded low-friction PPO pair."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from functools import partial
import os
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_nominal_low_friction_support_credit import KEY as CREDIT, credit
from getup_support_credit_clock import GetupSupportCredit, support_credit_rate
from local_getup_scale import gpu_sample
from recover_nominal_supine_upright_band_observation import JsonRunObserver
from roll_contact_clock_capture import ContactClock
from run_nominal_supine_upright_band_candidate import RecordedAuthorization
from selective_friction_guard import resolve_foot_geometry_ids
from training_watch import Snapshot
import run_nominal_supine_upright_band as original


ROOT, PROJECT, parent = original.ROOT, original.PROJECT, original.parent
read, rel, sha, write = original.read, original.rel, original.sha, original.write
ENTRY = Path(__file__).resolve()
KEY, PROFILE_KEY = 'nominal_low_friction_support_ppo', 'nominal_low_friction_support_ppo_capacity'
BEFORE = 'nominal-low-friction-support-credit'
PROFILE_STAGE, STAGE = 'nominal-low-friction-support-ppo-capacity', 'nominal-low-friction-support-ppo-training'
PROFILE = original.PLAN.with_name(f'{PROFILE_KEY}.json')
PLAN = original.PLAN.with_name(f'{KEY}.json')
TEMPLATE = original.PLAN
OUT = parent.REPORT/'low-friction-support-effort'/'support-credit-ppo'
CAPACITY = OUT/'capacity'
TERM = 'solved_weight_transfer'
INITIALIZE, CONFIG = original.initialize, original.original_config
SIZES = (3072, 3328, 3584)
ROLES = original.ROLES
TITLES = {'control': '原奖励对照', 'candidate': '承重交接奖励候选'}
INPUT_SEED = 5417


def selected(project, stage, plan_path, execution, evidence, next_gate=None):
    update = dict(current_protocol_execution=execution, current_evidence=rel(evidence))
    project.update(update)
    task = project['staged_training_request']
    task.update(update, current_stage=stage, current_protocol=rel(plan_path), current_protocol_frozen=True)
    if next_gate:
        task['next_gate'] = next_gate


def config(plan):
    cfg, display = CONFIG(plan)
    foot = cfg.env.events['foot_friction']
    assert foot.mode=='startup' and tuple(foot.params['ranges'])==(1.,1.)
    coefficient = plan['foot_sliding_friction']
    foot.params['ranges'] = (coefficient,coefficient)
    return cfg, display


def initialize(plan, role, record, check):
    env, inputs, learner = INITIALIZE(plan,role,record,check)
    try:
        data = env.sim.model.geom_friction
        ids = list(resolve_foot_geometry_ids(env.sim.mj_model))
        torch.testing.assert_close(data[:,ids,0],torch.full_like(data[:,ids,0],plan['foot_sliding_friction']),rtol=0,atol=0)
        env._support_transfer = GetupSupportCredit(env, **plan['term'])
        write(record/'support-credit-startup.json',dict(foot_sliding_friction=plan['foot_sliding_friction'],
            applied_in_all_worlds=True,actual_reset_controls=int(env.common_step_counter),worlds=env.num_envs,
            actuation_intervals_per_control=env.cfg.decimation,term_weight=plan['upright_credit_weight'],enabled=role=='candidate',
            maximum_discounted_addition=plan['upright_credit_weight'] if role=='candidate' else 0.))
        return env, inputs, learner
    except BaseException:
        env.close()
        raise


def audit_reward(env, plan, role):
    manager = env.reward_manager
    assert manager._term_names[-1]==TERM and manager._scale_by_dt
    original_compute, meter = manager.compute, env._support_transfer
    rows = []
    enabled = role=='candidate'
    weight = plan['upright_credit_weight']

    def compute(dt):
        result = original_compute(dt)
        age = meter.age
        if meter.first-1<=age<=meter.last:
            assert dt==env.step_dt and (env.episode_length_buf==age+1).all()
            rate = manager._step_reward[:,-1]
            expected = meter.delta / (meter.normalization*dt) if enabled else meter.zero
            torch.testing.assert_close(rate,expected*weight*dt/dt,rtol=0,atol=2e-7)
            assert not meter.delta[~env._early_support_release.supine].any()
            summary = torch.stack((meter.delta.min(),meter.delta.max(),meter.delta.mean(),(meter.delta>0).sum(),
                env._early_support_release.supine.sum(),(rate>0).sum()))
            lo,hi,average,positive,supine,rewarded = summary.tolist()
            assert 0.<=lo<=hi<=1.+1e-6 and (meter.samples==5 or hi==0.)
            rows.append(dict(control=int(env.common_step_counter),before_control_age=age,actuation_intervals=meter.samples,
                min_credit=lo,max_credit=hi,mean_credit=average,positive=int(positive),supine=int(supine),
                rewarded=int(rewarded),active_reward=enabled))
        return result

    manager.compute = compute
    return rows


def resize(plan, worlds):
    setup = deepcopy(plan)
    env = setup['environment']
    seed = setup['training_seed']
    env.update(num_envs=worlds, seed=seed, maximum_control_transitions=400*worlds,
               anchor_seed=seed*100+4, retention_seed=seed*100+5)
    env['learner']['minibatch_seed'] = seed*100+3
    env['split']['pose_codes'] = [0,1,2,3]*(worlds//4)
    setup['environments'] = worlds
    setup['limit']['transitions'] = worlds*setup['limit']['controls']
    return setup


def freeze_profile():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert parent.allowed(project,BEFORE) and not task['simulation_running'] and not torch.cuda.is_initialized()
    assert PROFILE_KEY not in goal and not (PROFILE.exists() or CAPACITY.exists())
    arch, reduced = (goal[name] for name in (CREDIT,CREDIT+'_device_load'))
    raw = ROOT/arch['protocol']
    result_path = ROOT/arch['result']
    result = read(result_path,arch['result_sha256'])
    device_path = ROOT/reduced['evidence']
    device = read(device_path,reduced['evidence_sha256'])
    spec = read(raw,arch['protocol_sha256'])
    assert arch['admitted'] and result['gpu_readback_preflight_admitted'] and len(device['controls'])==5
    assert spec['release_control']==100 and spec['controls']==[124,150] and spec['discount_per_control']==.99
    previous = read(TEMPLATE)
    original.sources_checked(previous)
    env = previous['environment']
    assert (previous['models']['widths'],previous['limit']['controls'],previous['updates_per_arm'])==({'actor':61,'critic':76},408,17)
    assert previous['release_age']==env['actual_release_control']==spec['release_control']
    assert previous['gpu_memory_fraction_limit']==.85
    assert all(INPUT_SEED not in spec['confirmation_seeds_reserved'] and f'-s{INPUT_SEED}-' not in f.name
               for f in (ROOT/'records/experiments').iterdir())
    assert previous['models']['onnx_sha256']==sha(ROOT/previous['models']['onnx'])
    assert previous['models']['checkpoint_sha256']==sha(ROOT/previous['models']['checkpoint'])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    checked_sources = (ENTRY,TEMPLATE,raw,result_path,device_path,ROOT/'scripts/training/getup_support_credit_device.py',
        ROOT/'scripts/training/getup_support_credit_clock.py',Path(RecordedAuthorization.close.__code__.co_filename),
        Path(JsonRunObserver.publish.__code__.co_filename),ROOT/'scripts/validation/verify_nominal_low_friction_support_ppo.py')
    previous['integrity'].update({rel(path):sha(path) for path in checked_sources})
    previous.update(training_seed=INPUT_SEED, training_stage=STAGE, preparation_stage=PROFILE_STAGE,
        foot_sliding_friction=.8, term=dict(first=124,last=150,per_foot=spec['per_foot_load_fraction'],
                                           total_foot=spec['total_foot_load_fraction'],discount=spec['discount_per_control']),
        model_change='Both arms read five solved loads per eligible control. The candidate alone adds bounded discounted credit.',
        capacity=dict(stage=PROFILE_STAGE, sizes=list(SIZES), stops_at_control=150, timed_windows=[[125,137],[137,149]],
            cpu_reference_before_controls=[124,149], selection_tolerance_fraction=.02, observation_controls=[0,50,100,124,137,150],
            maximum_wall_seconds=300, optimizer_steps=0, runs={str(n):f'duck-getup-support-ppo-profile-n{n}-s{INPUT_SEED}-{stamp}' for n in SIZES},
            gpu_cap=.85, upper_size_requires_projection_within_cap=True), reserved_confirmations=spec['confirmation_seeds_reserved'],
        observer_preflight=parent.preflight())
    previous['environment'].update(stage=STAGE,scope='Matched low-friction get-up PPO with original action acceptance gates')
    CAPACITY.mkdir(parents=True)
    write(PROFILE,previous)
    evidence = CAPACITY/'preflight.json'
    write(evidence,dict(protocol_sha256=sha(PROFILE),best_available_starting_scale=3072,
        sizes=list(SIZES),physics_steps=0,learning_updates=0,contact_readback_sha256=reduced['evidence_sha256'],
        before_first_episode_reset=True,observer=previous['observer_preflight']))
    goal[PROFILE_KEY] = dict(status='profile-ready', protocol=rel(PROFILE), protocol_sha256=sha(PROFILE), preflight=rel(evidence),
        reviewed={}, hardware_io=False)
    selected(project,PROFILE_STAGE,PROFILE,PROFILE_STAGE+'-frozen',evidence,
             'Compare the actual CUDA contact/reward sampler on the predeclared capacity ladder.')
    write(PROJECT,project)
    print('LOW_FRICTION_SUPPORT_PPO_CAPACITY_FROZEN',sha(PROFILE),previous['capacity']['runs'],flush=True)


def profile_context():
    project = read(PROJECT)
    task, node = project['staged_training_request'], project['goal_execution'][PROFILE_KEY]
    assert parent.allowed(project,PROFILE_STAGE) and not task['simulation_running'] and node['status']=='profile-ready'
    assert task['current_protocol']==rel(PROFILE) and task['current_protocol_frozen']
    spec = read(PROFILE,node['protocol_sha256'])
    original.sources_checked(spec)
    return project,node,spec


def profile(worlds):
    from mjlab.utils.torch import configure_torch_backends
    import resource

    project,node,plan = profile_context()
    design = plan['capacity']
    label = str(worlds)
    reviewed = node['reviewed']
    assert list(reviewed)==[str(n) for n in SIZES[:SIZES.index(worlds)]]
    report = CAPACITY/f'n{worlds}'
    report.mkdir()
    result_path = report/'result.json'
    run = design['runs'][label]
    result = dict(status='review-required',run_id=run,environments=worlds,controls=0,physics_substeps=0,
                  learning_updates=0,protocol_sha256=sha(PROFILE))
    if worlds==SIZES[-1]:
        first, second = [read(ROOT/reviewed[str(n)]['path'],reviewed[str(n)]['sha256']) for n in SIZES[:2]]
        projected = second['peak_gpu_mib'] + max(second['peak_gpu_mib']-first['peak_gpu_mib'],0.)
        ceiling = design['gpu_cap']*second['total_gpu_mib']
        if projected>ceiling:
            result.update(status='projected-over-memory-cap',gpu_projected_mib=projected, gpu_limit_mib=ceiling)
            write(result_path,result)
            reviewed[label]=dict(path=rel(result_path),sha256=sha(result_path),status=result['status'])
            write(PROJECT,project)
            print('SUPPORT_PPO_PROFILE_PROJECTED_MEMORY',worlds,projected,ceiling,flush=True)
            return
    owner = os.getpid()
    project['staged_training_request']['simulation_running']=True
    node.update(owner=owner, active_profile=worlds)
    write(PROJECT,project)
    checksum = node['protocol_sha256']

    def authorized():
        p = read(PROJECT)
        t, current = p['staged_training_request'],p['goal_execution'][PROFILE_KEY]
        return (parent.allowed(p,PROFILE_STAGE) and t['simulation_running'] and current.get('owner')==owner
                and t['current_protocol']==rel(PROFILE) and t['current_protocol_frozen'] and sha(PROFILE)==checksum)

    record = ROOT/'records/evaluations'/run
    record.mkdir()
    observer = parent.RecordedObserver(run,report,authorized)
    observer.state.update(purpose=f'仰卧起身吞吐短测｜{worlds}环境',maximum_controls=design['stops_at_control'])
    watch = original.MemoryWatch(design['gpu_cap'])
    env = camera = None
    audit = []
    clock_evidence = []
    metrics = []
    windows = []
    thread_start = process_time()
    began = perf_counter()
    try:
        observer.start()
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        with watch.active(),torch.inference_mode():
            sized = resize(plan,worlds)
            env,obs,learner = initialize(sized,'candidate',record,watch.check)
            watch.sample()
            result['initialized_gpu_mib']=watch.rows[-1]['used_mib']
            assert len(obs['actor'])==worlds and env.common_step_counter==env._sim_step_counter==0
            meter = env._support_transfer
            audit = audit_reward(env,sized,'candidate')
            clock = ContactClock(env,dict(capture_seconds=[0.,.004],maximum_batched_substeps=1,
                                         load_threshold_nominal_weight_fraction=plan['term']['per_foot']))
            options = dict(per_foot_load_fraction=plan['term']['per_foot'],total_foot_load_fraction=plan['term']['total_foot'])
            compare_steps = set(design['cpu_reference_before_controls'])

            def same_force(control, substep):
                if control not in compare_steps:
                    return
                points, normal = clock.read_contacts()
                measured = meter.solved.normal.numpy()
                actual_credit = meter.solved.credit.numpy()
                expected = credit(normal[:,1:],clock.weight,options)
                np.testing.assert_allclose(measured,normal,atol=5e-5,rtol=0)
                np.testing.assert_allclose(actual_credit,expected,atol=1e-5,rtol=0)
                clock_evidence.append(dict(before_control=control,substep=substep,
                    ground_contacts=int(points[:,0].sum()),max_force_error_n=float(abs(measured-normal).max()),
                    max_credit_error=float(abs(actual_credit-expected).max())))

            meter.readback_hook=same_force
            camera = Snapshot(env,record)
            observer.begin_collect()

            def publish():
                k = int(env.common_step_counter)
                frame = camera.capture()['file']
                observer.publish(record,dict(run_id=run,status='running',purpose=f'仰卧起身吞吐短测｜{worlds}环境',
                    controls=k,maximum_controls=design['stops_at_control'],transitions=k*worlds,learning_updates=0,
                    updates=metrics,frame=frame,updated_at=datetime.now(timezone.utc).isoformat()))

            publish()
            active_begin = None
            both_starts={a:b for a,b in design['timed_windows']}
            for control in range(design['stops_at_control']):
                watch.check()
                assert observer.check(),observer.refusal
                assert perf_counter()-began<design['maximum_wall_seconds']
                if control in both_starts:
                    torch.cuda.synchronize()
                    active_begin=control,perf_counter()
                action = learner.actor(obs,stochastic_output=True)
                learner.critic(obs)
                obs,reward,terminated,timeout,_ = env.step(action)
                assert not terminated.any() and not timeout.any() and torch.isfinite(reward).all()
                assert torch.isfinite(obs['actor']).all()
                count = control+1
                result.update(controls=count,physics_substeps=int(env._sim_step_counter))
                if active_begin is not None and count==both_starts[active_begin[0]]:
                    torch.cuda.synchronize()
                    seconds=perf_counter()-active_begin[1]
                    entry = dict(first=active_begin[0],last=count,seconds=seconds,transitions=(count-active_begin[0])*worlds)
                    entry['transitions_per_second']=entry['transitions']/seconds
                    windows.append(entry)
                    metrics.append(dict(control=count,transitions_per_second=entry['transitions_per_second']))
                    active_begin=None
                if count in design['observation_controls']:
                    publish()
                    print('SUPPORT_PPO_PROFILE_PROGRESS',worlds,count,flush=True)
                assert env.common_step_counter==count and env._sim_step_counter==count*5
            watch.sample()
            assert len(clock_evidence)==10 and all(x['ground_contacts']>0 for x in clock_evidence)
            active = [row for row in audit if row['actuation_intervals']==5]
            assert [row['before_control_age'] for row in active]==list(range(124,150))
            assert all(row['positive']==row['rewarded']>0 for row in active)
            result.update(status='completed',windows=windows,
                          conservative_transitions_per_s=min(row['transitions_per_second'] for row in windows))
    except BaseException as exc:
        result['reason']=repr(exc)
        raise
    finally:
        try:
            if camera is not None:
                camera.close()
            if env is not None:
                env.close()
            result.update(peak_gpu_mib=max((x['used_mib'] for x in watch.rows),default=0),
                total_gpu_mib=watch.rows[0]['total_mib'] if watch.rows else None, gpu_samples=watch.rows,
                wall_seconds=perf_counter()-began,cpu_seconds=process_time()-thread_start,
                peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                actual_transitions=worlds*result['controls'],scope='Fixed actor and real reward path, no optimizer.')
            write(report/'native-clock-parity.json',dict(rows=clock_evidence,reward_audit=audit))
            write(result_path,result)
            observer.close(result)
        finally:
            p = read(PROJECT)
            current = p['goal_execution'][PROFILE_KEY]
            if current.get('owner')==owner:
                del current['owner'],current['active_profile']
                current['reviewed'][label]=dict(path=rel(result_path),sha256=sha(result_path),status=result['status'])
                if result['status']!='completed':
                    current['status']='review-required'
                p['staged_training_request']['simulation_running']=False
                selected(p,PROFILE_STAGE,PROFILE,PROFILE_STAGE+'-observed',result_path)
                write(PROJECT,p)
    print('SUPPORT_PPO_PROFILE_SCORE',worlds,result['conservative_transitions_per_s'],result['peak_gpu_mib'],flush=True)


def freeze_training():
    project,node,spec = profile_context()
    assert list(node['reviewed'])==[str(n) for n in SIZES]
    evidence = {int(k):read(ROOT/v['path'],v['sha256']) for k,v in node['reviewed'].items()}
    measured = [row for row in evidence.values() if row['status']=='completed']
    assert len(measured)>=2 and all(row['learning_updates']==0 and row['controls']==150 for row in measured)
    assert all(row['peak_gpu_mib']<=row['total_gpu_mib']*spec['capacity']['gpu_cap'] for row in measured)
    top = max(row['conservative_transitions_per_s'] for row in measured)
    tolerance=spec['capacity']['selection_tolerance_fraction']
    candidate = min((row for row in measured if row['conservative_transitions_per_s']>=top*(1-tolerance)),
                    key=lambda row:row['environments'])
    worlds = candidate['environments']
    plan = resize(spec,worlds)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    runs = {role:f'duck-getup-support-credit-friction080-{role}-s{INPUT_SEED}-{stamp}' for role in ROLES}
    plan.update(runs=runs)
    assert not (PLAN.exists() or KEY in project['goal_execution'])
    files = [ROOT/v['path'] for v in node['reviewed'].values()]
    plan['integrity'].update({rel(path):sha(path) for path in (PROFILE,*files)})
    receipt = CAPACITY/'selection.json'
    write(receipt,dict(environments=worlds,conservative_transitions_per_s=candidate['conservative_transitions_per_s'],
        criterion='Maximize the slower of two measured windows; pick the smaller size within 2 percent of best.',
        observed={str(n):dict(status=row['status'],conservative_transitions_per_s=row.get('conservative_transitions_per_s'),
                              peak_gpu_mib=row.get('peak_gpu_mib'),gpu_projected_mib=row.get('gpu_projected_mib')) for n,row in evidence.items()},
        throughput_scope='Rollout sampling, no PPO update in the benchmark.'))
    plan['integrity'][rel(receipt)]=sha(receipt)
    write(PLAN,plan)
    project['goal_execution'][KEY]=dict(status='cuda-zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),runs=runs,
        selected_capacity=rel(receipt),hardware_io=False)
    node.update(status='reviewed',selection=rel(receipt),selection_sha256=sha(receipt))
    selected(project,STAGE,PLAN,STAGE+'-zero-ready',receipt,
             'Verify identical CUDA initial state, then train the same-parent equal-budget PPO arms.')
    write(PROJECT,project)
    print('SUPPORT_PPO_TRAINING_FROZEN',sha(PLAN),worlds,runs,flush=True)


class SupportObserver(JsonRunObserver):
    def __init__(self, run_id, report, allowed, **settings):
        settings['purpose']=f'低摩擦仰卧起身｜{TITLES[report.name]}'
        super().__init__(run_id,report,allowed,**settings)

    def check(self, *, force=False):
        admitted = super().check(force=force)
        proof = self.report/'live-readback.json'
        if admitted and self.in_collect and self.last_completed>=2 and not proof.exists():
            try:
                info = read(self.view/'native-health.json')
                chart = read(self.view/'native-curve-source.json')
                image = self.view/'native-latest.jpg'
                graph = self.view/'native-curve.png'
                if (info['run_id']==self.run_id and info['health']=='fresh' and info['source_status']=='training'
                        and info['metric_rows']>0 and info['chart_source']==chart
                        and sha(image)==info['frame_source_sha256'] and graph.is_file()):
                    write(proof,dict(health=info,checked_at=datetime.now(timezone.utc).isoformat(),
                        image_sha256=sha(image),curve_sha256=sha(graph)))
            except (OSError, ValueError, KeyError):
                # Publication can briefly be between frame and health writes.
                # The following healthy callback reads the completed snapshot.
                pass
        return admitted


def run_write(path, payload):
    write(path,payload)
    if path.name=='state.json' and payload.get('run_id','').startswith('duck-getup-support-credit-friction080-'):
        note = path.with_name('README.md')
        if not note.exists():
            label = TITLES[payload['arm']]
            note.write_text(f'# 低摩擦仰卧起身｜{label}\n\n[冻结配置](../../../{rel(PLAN)}) · '
                            '[完整指标](observation.json) · [起身原片索引](clips.json)\n',encoding='utf-8')


def select_original(project, stage, execution, evidence):
    if execution.startswith('nominal-supine-upright-band'):
        execution = execution.replace('nominal-supine-upright-band',STAGE,1)
    selected(project,stage,PLAN,execution,evidence)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze-profile','profile','freeze-training','zero','match','train'))
    parser.add_argument('argument',nargs='?')
    cli = parser.parse_args()
    action,arg = cli.action,cli.argument
    args = {'profile':tuple(map(str,SIZES)),'zero':ROLES,'train':ROLES}
    assert arg in args.get(action,(None,))
    authorization=partial(RecordedAuthorization,report=OUT/str(arg)/'authorization-monitor.json')
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,
          patch.multiple(original,KEY=KEY,OUT=OUT,PLAN=PLAN,TERM=TERM,band_rate=support_credit_rate,
                         original_config=config,initialize=initialize,audit_band=audit_reward,RunObserver=SupportObserver,
                         AuthorizationMonitor=authorization,write=run_write,select=select_original)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze-profile':freeze_profile,'profile':lambda:profile(int(arg)),'freeze-training':freeze_training,
         'zero':lambda:original.zero(arg),'match':original.match_zero,'train':lambda:original.train(arg)}[action]()


if __name__=='__main__':
    main()
