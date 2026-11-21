"""Measure the live support reward and run its equal-budget PPO pair."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
from functools import partial
import os
from pathlib import Path
import resource
from time import perf_counter, process_time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_low_friction_support_ppo as earlier
import verify_loaded_upright_progress_device as verified
from getup_loaded_upright_sampler import GetupLoadedUprightSampler, loaded_upright_rate
from review_nominal_low_friction_first_support_effort import tilt
from roll_contact_clock_capture import ContactClock


ROOT,PROJECT,parent,original=earlier.ROOT,earlier.PROJECT,earlier.parent,earlier.original
read,rel,sha,write=earlier.read,earlier.rel,earlier.sha,earlier.write
KEY='nominal_low_friction_loaded_upright_ppo'
PROFILE_KEY=KEY+'_capacity'
PROFILE_STAGE='nominal-low-friction-loaded-upright-sampling'
STAGE='nominal-low-friction-loaded-upright-training'
PROFILE=ROOT/'configs/experiments'/f'{PROFILE_KEY}.json'
PLAN=ROOT/'configs/experiments'/f'{KEY}.json'
OUT=verified.OUT.parent/'loaded-upright-ppo'
CAPACITY=OUT/'capacity'
TERM='loaded_upright_progress'
ROLES=earlier.ROLES
SIZES=(1536,2304,3072)
TRAIN_SEED=5517
INITIALIZE=earlier.INITIALIZE
TITLES={'control':'原奖励对照','candidate':'承重后转正候选'}


def select(project,stage,execution,evidence):
    execution=execution.replace('nominal-supine-upright-band',STAGE,1)
    earlier.selected(project,stage,PLAN,execution,evidence)


def freeze_capacity():
    project=read(PROJECT)
    task,goal=project['staged_training_request'],project['goal_execution']
    assert parent.allowed(project,verified.STAGE) and not task['simulation_running'] and not torch.cuda.is_initialized()
    node=goal[verified.KEY]
    assert node['status']=='reviewed' and node['live_sampler_admitted']
    archive=read(ROOT/node['result'],node['result_sha256'])
    device=read(ROOT/node['protocol'],node['protocol_sha256'])
    assert archive['eligible_for_live_sampler'] and not archive['holdout_used'] and device['first_read_elapsed_ms']==400
    base=goal[earlier.KEY]
    plan=read(ROOT/base['protocol'],base['protocol_sha256'])
    original.sources_checked(plan)
    assert base['status']=='training-and-action-reviewed'
    prior_capacity=ROOT/base['selected_capacity']
    evidence=read(prior_capacity)
    assert evidence['environments']==3072 and evidence['measured']['3328']['peak_gpu_mib']>evidence['gpu_limit_mib']
    assert (plan['foot_sliding_friction'],plan['release_age'],plan['gpu_memory_fraction_limit'])==(.8,100,.85)
    assert plan['limit']['controls']==408 and plan['updates_per_arm']==17 and TRAIN_SEED not in plan['reserved_confirmations']
    assert not any(f'-s{TRAIN_SEED}-' in p.name for p in (ROOT/'records/experiments').iterdir())
    assert not any(path.exists() for path in (OUT,PROFILE,PLAN)) and PROFILE_KEY not in goal and KEY not in goal
    paths=[Path(__file__),ROOT/base['protocol'],prior_capacity,ROOT/node['protocol'],ROOT/node['result'],
           ROOT/'scripts/training/getup_loaded_upright_sampler.py',ROOT/'scripts/training/getup_loaded_upright_progress.py',
           ROOT/'scripts/training/getup_support_credit_device.py',ROOT/'scripts/validation/verify_loaded_upright_progress_device.py']
    plan['integrity'].update({rel(path):sha(path) for path in paths})
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan.update(training_stage=STAGE,preparation_stage=PROFILE_STAGE,training_seed=TRAIN_SEED,
        term=dict(first=120,last_search=156,last=160,shape=device['shape']),
        model_change='Use the same signed, bounded 80 ms loaded-upright clock in both arms; only the candidate adds it to reward.',
        capacity=dict(stage=PROFILE_STAGE,sizes=list(SIZES),stops_at_control=160,timed_windows=[[124,138],[139,153]],
            cpu_compare_controls=[120,155],observation_controls=[0,40,80,120,138,153,160],
            maximum_wall_seconds=300,optimizer_steps=0,gpu_cap=.85,selection_tolerance_fraction=.02,
            runs={str(n):f'duck-getup-loaded-upright-profile-n{n}-s{TRAIN_SEED}-{stamp}' for n in SIZES}),
        observer_preflight=parent.preflight())
    plan['environment'].update(stage=STAGE,scope='Same-parent low-friction upright-progress pair; original recovery and roll gates')
    CAPACITY.mkdir(parents=True)
    write(PROFILE,plan)
    preflight=CAPACITY/'preflight.json'
    write(preflight,dict(protocol_sha256=sha(PROFILE),observer=plan['observer_preflight'],
        device_evidence_sha256=node['result_sha256'],previous_memory_ceiling=rel(prior_capacity),
        executable_profiles=list(SIZES),physical_steps=0,learning_updates=0,hardware_io=False))
    goal[PROFILE_KEY]=dict(status='profile-ready',protocol=rel(PROFILE),protocol_sha256=sha(PROFILE),
                          preflight=rel(preflight),reviewed={},hardware_io=False)
    earlier.selected(project,PROFILE_STAGE,PROFILE,PROFILE_STAGE+'-frozen',preflight,
                     'Verify same-solve live contact, root pose and reward and compare the frozen throughput sizes.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_CAPACITY_FROZEN',sha(PROFILE),plan['capacity']['runs'],flush=True)


def initialize(plan,role,record,check):
    env,obs,learner=INITIALIZE(plan,role,record,check)
    try:
        ids=list(earlier.resolve_foot_geometry_ids(env.sim.mj_model))
        data=env.sim.model.geom_friction[:,ids,0]
        torch.testing.assert_close(data,torch.full_like(data,plan['foot_sliding_friction']),rtol=0,atol=0)
        env._loaded_upright_sampler=GetupLoadedUprightSampler(env,**plan['term'])
        write(record/'loaded-upright-startup.json',dict(foot_sliding_friction=plan['foot_sliding_friction'],
            friction_matched_all_worlds=True,controls=int(env.common_step_counter),role=role,
            enabled=role=='candidate',bounds=[-plan['upright_credit_weight'],plan['upright_credit_weight']],
            initial_clock_empty=bool((env._loaded_upright_sampler.clock.elapsed<0).all())))
        return env,obs,learner
    except BaseException:
        env.close()
        raise


def audit_reward(env,plan,role,full_readback=None):
    meter,manager=env._loaded_upright_sampler,env.reward_manager
    assert manager._term_names[-1]==TERM and manager._scale_by_dt
    compute=manager.compute
    weight=plan['upright_credit_weight']
    enabled=role=='candidate'
    rows=[]

    def patched(dt):
        result=compute(dt)
        k=meter.age
        if meter.first-1<=k<=meter.last:
            assert dt==env.step_dt and (env.episode_length_buf==k+1).all()
            actual=manager._step_reward[:,-1]
            expected=(meter.credit/dt*weight*dt)/dt if enabled else meter.zero
            torch.testing.assert_close(actual,expected,rtol=0,atol=3e-7)
            mask=env._early_support_release.supine
            assert not meter.credit[~mask].any()
            summaries=torch.stack((meter.credit.min(),meter.credit.max(),(meter.credit>0).sum(),
                                   (meter.credit<0).sum(),mask.sum(),(actual!=0).sum()))
            lo,hi,pos,neg,supine,rewarded=summaries.tolist()
            assert -.250001<=lo<=hi<=.250001 and pos+neg<=supine
            rows.append(dict(before_control_age=k,actuation_intervals=meter.samples,min_credit=lo,max_credit=hi,
                positive=int(pos),negative=int(neg),supine=int(supine),rewarded=int(rewarded),active_reward=enabled))
            if full_readback is not None and meter.record:
                full_readback.append(actual.clone())
        return result

    manager.compute=patched
    return rows


def readback_check(block,reward,plan,n,weight,force_checks):
    term=plan['term']
    shape=term['shape']
    count=5*(term['last']-term['first'])
    assert block.shape==(count,n,17) and reward.shape==(count//5,n)
    loads,before,after=block[:,:,:4],block[:,:,4:8],block[:,:,8:12]
    raw,unit,confirmed,active,enabled=block[:,:,12:].transpose(2,0,1)
    np.testing.assert_array_equal(before[1:],after[:-1])
    search=np.arange(count)[:,None]<5*(term['last_search']-term['first'])
    hits=(loads[:,:,3]>weight*shape['foot_fraction']) & (enabled!=0.) & search
    windows=np.lib.stride_tricks.sliding_window_view(hits,shape['settled_reads'],axis=0).all(axis=-1)
    first=np.where(windows.any(axis=0),windows.argmax(axis=0)+shape['settled_reads']-1,-1)
    ticks=np.arange(count)[:,None]
    elapsed=ticks-first
    valid=(first>=0)&(elapsed>0)&(elapsed<=shape['window_reads'])
    np.testing.assert_array_equal(confirmed,ticks==first)
    np.testing.assert_array_equal(active,valid)
    feet=loads[:,:,2:]
    total=feet.sum(axis=2)
    support=(feet>shape['foot_fraction']*weight).all(axis=2)&(total>=shape['both_feet_fraction']*weight)
    raw_reference=np.where(valid&support,(tilt(before)-tilt(after))*np.clip(total/weight,0.,1.),0.)
    raw_error=float(abs(raw_reference-raw).max())
    assert raw_error<=.0002
    expected_unit=(raw_reference/shape['maximum_tick_progress_deg']).clip(-1.,1.)/shape['window_reads']
    np.testing.assert_allclose(unit,expected_unit,rtol=0,atol=.0002/(shape['maximum_tick_progress_deg']*shape['window_reads']))
    summed=np.cumsum(unit.reshape(count//5,5,n),axis=1)[:,-1]
    dt=np.float32(.02)
    from_ticks=(summed/dt*np.float32(plan['upright_credit_weight'])*dt)/dt
    reward_error=float(abs(reward-from_ticks).max())
    assert reward_error<=3e-7 and (np.abs(unit).sum(axis=0)<=1.000001).all()
    assert confirmed.any() and (unit>0.).any() and (unit<0.).any()
    assert len(force_checks)==10 and all(c['ground_contacts']>0 and c['maximum_normal_error_n']<=5e-5 for c in force_checks)
    return dict(status='live-same-solve-clock-and-reward-verified',worlds=n,physical_readouts=count*n,
        before_after_continuous=True,exact_confirmation_and_active_masks=True,confirmed_worlds=int(confirmed.sum()),
        maximum_tick_error_deg=raw_error,maximum_manager_rate_error=reward_error,
        signed_credit_verified=True,independent_contact_solver_reads=force_checks)


def profile_context():
    project=read(PROJECT)
    node=project['goal_execution'][PROFILE_KEY]
    task=project['staged_training_request']
    assert parent.allowed(project,PROFILE_STAGE) and not task['simulation_running'] and node['status']=='profile-ready'
    assert task['current_protocol']==rel(PROFILE) and task['current_protocol_frozen']
    plan=read(PROFILE,node['protocol_sha256'])
    original.sources_checked(plan)
    return project,node,plan


def profile(n):
    from mjlab.utils.torch import configure_torch_backends

    project,node,base=profile_context()
    assert list(node['reviewed'])==[str(i) for i in SIZES[:SIZES.index(n)]]
    assert all(row['status']=='completed' for row in node['reviewed'].values())
    plan=earlier.resize(base,n)
    design=plan['capacity']
    run=design['runs'][str(n)]
    report=CAPACITY/f'n{n}'
    record=ROOT/'records/evaluations'/run
    report.mkdir()
    record.mkdir()
    owner=os.getpid()
    saved=report/'result.json'
    result=dict(status='review-required',worlds=n,run_id=run,controls=0,physics_substeps=0,learning_updates=0,
                protocol_sha256=node['protocol_sha256'],hardware_io=False)
    node.update(owner=owner,active_profile=n)
    project['staged_training_request']['simulation_running']=True
    write(PROJECT,project)

    def allowed():
        current=read(PROJECT)
        t=current['staged_training_request']
        visit=current['goal_execution'][PROFILE_KEY]
        return (parent.allowed(current,PROFILE_STAGE) and t['simulation_running'] and visit.get('owner')==owner
                and t['current_protocol']==rel(PROFILE) and sha(PROFILE)==node['protocol_sha256'])

    observer=parent.RecordedObserver(run,report,allowed)
    label=f'低摩擦承重转正采样｜{n}场景'
    observer.state.update(purpose=label,maximum_controls=design['stops_at_control'])
    watch=original.MemoryWatch(design['gpu_cap'])
    trace,manager_rates,force_checks,updates,windows,audit=[],[],[],[],[],[]
    env=camera=None
    began,cpu_start=perf_counter(),process_time()
    try:
        observer.start()
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        with watch.active(),torch.inference_mode():
            env,obs,learner=initialize(plan,'candidate',record,watch.check)
            watch.sample()
            result['initialized_gpu_mib']=watch.rows[-1]['used_mib']
            meter=env._loaded_upright_sampler
            independent=ContactClock(env,dict(capture_seconds=[0.,.004],maximum_batched_substeps=1,
                                               load_threshold_nominal_weight_fraction=plan['term']['shape']['foot_fraction']))

            def read_samples(k,s,loads,before,after,raw,unit,confirmed,active,enabled):
                trace.append(torch.cat((loads,before,after,torch.stack((raw,unit,confirmed,active,enabled),dim=1)),dim=1))
                if k in design['cpu_compare_controls']:
                    counts,cpu_load=independent.read_contacts()
                    actual=loads.cpu().numpy()
                    force_checks.append(dict(control_before=k,solver_tick=s,ground_contacts=int(counts[:,0].sum()),
                        maximum_normal_error_n=float(abs(actual-cpu_load).max())))

            meter.readback=read_samples
            audit=audit_reward(env,plan,'candidate',manager_rates)
            camera=earlier.Snapshot(env,record)
            observer.begin_collect()

            def publish(k):
                frame=camera.capture()['file']
                observer.publish(record,dict(run_id=run,status='running',purpose=label,controls=k,transitions=k*n,
                    maximum_controls=design['stops_at_control'],learning_updates=0,updates=updates,frame=frame,
                    updated_at=datetime.now(timezone.utc).isoformat()))

            publish(0)
            active=None
            ends=dict(design['timed_windows'])
            for k in range(design['stops_at_control']):
                watch.check()
                assert observer.check(),observer.refusal
                assert perf_counter()-began<design['maximum_wall_seconds']
                if k in ends:
                    torch.cuda.synchronize()
                    active=k,perf_counter()
                action=learner.actor(obs,stochastic_output=True)
                learner.critic(obs)
                obs,reward,terminated,timeout,_=env.step(action)
                assert not terminated.any() and not timeout.any() and torch.isfinite(reward).all()
                endpoint=k+1
                assert env.common_step_counter==endpoint and env._sim_step_counter==5*endpoint
                result.update(controls=endpoint,physics_substeps=5*endpoint)
                if active is not None and endpoint==ends[active[0]]:
                    torch.cuda.synchronize()
                    t=perf_counter()-active[1]
                    throughput=(endpoint-active[0])*n/t
                    windows.append(dict(first=active[0],last=endpoint,seconds=t,transitions_per_second=throughput))
                    updates.append(dict(control=endpoint,transitions_per_second=throughput))
                    active=None
                if endpoint in design['observation_controls']:
                    publish(endpoint)
                    print('LOADED_UPRIGHT_PROFILE_PROGRESS',n,endpoint,flush=True)
            watch.sample()
            raw=torch.stack(trace).cpu().numpy()
            reward=torch.stack(manager_rates).cpu().numpy()
            proof=readback_check(raw,reward,plan,n,independent.weight,force_checks)
            receipt=report/'live-solver-parity.json'
            proof.update(reward_audits=audit,physical_reference_arrays=['normal_force','quaternion_before','quaternion_after'],
                         compared_pulse_controls=plan['term'])
            write(receipt,proof)
            result.update(status='completed',windows=windows,conservative_transitions_per_s=min(row['transitions_per_second'] for row in windows),
                live_parity=rel(receipt),live_parity_sha256=sha(receipt))
    except BaseException as exc:
        result['reason']=repr(exc)
        if isinstance(exc,torch.cuda.OutOfMemoryError) or 'Observed GPU memory ceiling' in str(exc):
            result['status']='memory-ceiling-reached'
        else:
            raise
    finally:
        try:
            if camera is not None:
                camera.close()
            if env is not None:
                env.close()
            result.update(wall_seconds=perf_counter()-began,cpu_seconds=process_time()-cpu_start,
                peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                peak_gpu_mib=max((x['used_mib'] for x in watch.rows),default=0),
                total_gpu_mib=watch.rows[0]['total_mib'] if watch.rows else None,gpu_samples=watch.rows,
                transitions=result['controls']*n,scope='Fixed observer path and actual solved reward, no optimizer.')
            write(saved,result)
            observer.close(result)
        finally:
            current=read(PROJECT)
            current_node=current['goal_execution'][PROFILE_KEY]
            assert current_node['owner']==owner
            del current_node['owner'],current_node['active_profile']
            current_node['reviewed'][str(n)]=dict(result=rel(saved),sha256=sha(saved),status=result['status'])
            current['staged_training_request']['simulation_running']=False
            write(PROJECT,current)
    print('LOADED_UPRIGHT_PROFILE_FINISHED',n,result['status'],result.get('conservative_transitions_per_s'),result['peak_gpu_mib'],flush=True)


def freeze_training():
    project,node,plan=profile_context()
    entries=node['reviewed']
    assert entries and list(entries)==list(map(str,SIZES[:len(entries)]))
    assert len(entries)==len(SIZES) or entries[next(reversed(entries))]['status']=='memory-ceiling-reached'
    rows=[read(ROOT/e['result'],e['sha256']) for e in entries.values()]
    good=[r for r in rows if r['status']=='completed']
    assert len(good)>=2 and all(r['controls']==160 and r['physics_substeps']==800 for r in good)
    for row in good:
        assert row['peak_gpu_mib']/row['total_gpu_mib']<=plan['gpu_memory_fraction_limit']
        assert read(ROOT/row['live_parity'],row['live_parity_sha256'])['signed_credit_verified']
    peak=max(r['conservative_transitions_per_s'] for r in good)
    chosen=min((r for r in good if r['conservative_transitions_per_s']>=.98*peak),key=lambda r:r['worlds'])
    n=chosen['worlds']
    plan=earlier.resize(plan,n)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['runs']={role:f'duck-getup-loaded-upright-friction080-{role}-s{TRAIN_SEED}-{stamp}' for role in ROLES}
    proof=CAPACITY/'selection.json'
    write(proof,dict(environments=n,maximum_measured_conservative_transitions_per_s=peak,
        chosen_conservative_transitions_per_s=chosen['conservative_transitions_per_s'],candidate_receipts=entries,
        pass_rule='Smallest completed size within 2% of the maximum of the slower same-duration timed windows.',
        physics_steps_for_selection=0,learning_updates=0))
    files=[ROOT/row['live_parity'] for row in good]
    files.extend(ROOT/v['result'] for v in entries.values())
    plan['integrity'].update({rel(p):sha(p) for p in (PROFILE,proof,*files)})
    assert not PLAN.exists()
    write(PLAN,plan)
    project['goal_execution'][KEY]=dict(status='cuda-zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),
        runs=plan['runs'],selected_capacity=rel(proof),selected_capacity_sha256=sha(proof),hardware_io=False)
    node.update(status='reviewed',selection=rel(proof),selection_sha256=sha(proof))
    earlier.selected(project,STAGE,PLAN,STAGE+'-zero-ready',proof,
                     'Verify matching initial CUDA actor, critic, observations, scene and optimizer before equal-budget training.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_TRAINING_FROZEN',sha(PLAN),n,plan['runs'],flush=True)


class Observer(earlier.SupportObserver):
    def __init__(self,run_id,report,allowed,**settings):
        with patch.object(earlier,'TITLES',TITLES):
            super().__init__(run_id,report,allowed,**settings)


def run_write(path,payload):
    write(path,payload)
    if path.name=='state.json' and payload.get('run_id','').startswith('duck-getup-loaded-upright-friction080-'):
        document=path.with_name('README.md')
        if not document.exists():
            label=TITLES[payload['arm']]
            document.write_text(f'# 低摩擦起身｜{label}\n\n[训练配置](../../../{rel(PLAN)}) · '
                                '[逐次指标](observation.json) · [原片索引](clips.json)\n',encoding='utf-8')


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze-capacity','profile','freeze-training','zero','match','train'))
    cli.add_argument('argument',nargs='?')
    args=cli.parse_args()
    action,arg=args.action,args.argument
    opts={'profile':tuple(map(str,SIZES)),'zero':ROLES,'train':ROLES}
    assert arg in opts.get(action,(None,))
    auth=partial(earlier.RecordedAuthorization,report=OUT/str(arg)/'authorization-monitor.json')
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,
          patch.multiple(original,KEY=KEY,OUT=OUT,PLAN=PLAN,TERM=TERM,band_rate=loaded_upright_rate,
              original_config=earlier.config,initialize=initialize,audit_band=audit_reward,RunObserver=Observer,
              AuthorizationMonitor=auth,write=run_write,select=select)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze-capacity':freeze_capacity,'profile':lambda:profile(int(arg)),'freeze-training':freeze_training,
         'zero':lambda:original.zero(arg),'match':original.match_zero,'train':lambda:original.train(arg)}[action]()


if __name__=='__main__':
    main()
