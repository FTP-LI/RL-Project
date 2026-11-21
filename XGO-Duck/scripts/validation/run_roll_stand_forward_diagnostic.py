"""Compare direct walking with the archived post-roll walking using the same plant and actor."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_roll_friction_pair as previous
from roll_stand_walk_sequence import FAILED, FORWARD, STAND_START, Sequence

ROOT, P, read, put, sha, rel, require = (previous.ROOT, previous.P, previous.read, previous.put,
                                       previous.sha, previous.rel, previous.require)
pair = previous.pair
STAGE = 'roll-friction-stand-forward'
KEY = 'roll_friction_stand_forward'
PLANS = ROOT/'configs/experiments/roll_friction_stand_forward'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/roll-friction-stand-forward'
ENTRY = Path(__file__).resolve()


class StandForward(Sequence):
    """Use the original stability test, then switch to walking without crossing actors."""

    def __init__(self, plan):
        self.plan, self.criteria = plan, plan['roll']['criteria']
        n = plan['worlds']
        self.phase = np.full(n, STAND_START, dtype=np.int8)
        self.dwell = np.zeros(n, dtype=np.int32)
        self.stand_at = np.full(n, -1, dtype=np.int32)
        self.forward_at = self.stand_at.copy()
        self.phase_changed = self.stand_at.copy()
        # The shared guard sees an empty roll interval. It still checks every
        # captured frame for contact penetration, finite state and joint limits.
        self.roll_begin = plan['maximum_controls'] + 1
        self.roll_end = plan['maximum_controls']

    def update(self, step, state):
        old = self.phase.copy()
        self.phase[~state['active']] = FAILED
        waiting = self.phase == STAND_START
        if step <= self.plan['initial_stand_controls']:
            self.dwell = np.where(waiting & self.stable(state), self.dwell + 1, 0)
        if step == self.plan['initial_stand_controls']:
            ready = waiting & (self.dwell >= self.plan['standing_dwell_controls'])
            self.phase[waiting] = FAILED
            self.phase[ready] = FORWARD
            self.forward_at[ready] = step
        changed = self.phase != old
        self.phase_changed[changed] = step
        self.dwell[changed] = 0
        return changed


def freeze():
    from native_nominal_baseline import verify_sources

    project = read(P)
    task, old = project['staged_training_request'], project['goal_execution']['roll_friction_development']
    require(pair.trial.allowed(project,previous.STAGE) and old['status']=='development-reviewed'
        and not task['simulation_running'] and not task['learning_running'] and sha(previous.MASTER)==old['master_sha256'],
        'The paired roll development stage must be independently closed')
    require(not PLANS.exists() and not REPORT.exists(), 'Preserve the existing stage and files')
    results = ROOT/old['result']
    analysis = read(results)
    require(len(old['scored'])==12 and not analysis['reserved_seed_used'] and not analysis['gate_passed'],
        'Unexpected paired development record')
    earlier = read(previous.MASTER)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case in analysis['paired']:
        key = case['case']
        source_key = key+'-low-friction'
        source_config = previous.PLANS/(source_key+'.json')
        after_score = ROOT/old['scored'][source_key]
        approved = earlier['arms'][source_key]
        proof, plan = read(after_score), read(source_config)
        require(sha(after_score)==old['score_sha256'][source_key] and sha(source_config)==approved['plan_sha256']
            and proof['full_route'] and proof['run_id']==plan['run_id'] and proof['worlds']==128
            and not any(proof['safety'].values()), 'Post-roll reference is missing or changed')
        zero = ROOT/plan['shared_zero']
        zero_receipt = zero.parent/'zero-reference.json'
        zero_info = read(zero_receipt)
        require(zero_info['physics_substeps']==zero_info['learning_updates']==0
            and set(zero_info['files'])=={'startup.npz','endpoint-contact.npz'}, 'Invalid zero-step receipt')
        for name,digest in zero_info['files'].items():
            require(sha(zero/name)==digest, 'The matched startup snapshot changed')
        after_raw = ROOT/'records/evaluations'/proof['run_id']/'raw-score.json'
        counts = (plan['worlds'], plan['initial_stand_controls'], plan['standing_dwell_controls'], plan['forward_controls'],
                  plan['maximum_controls'], plan['minimum_continuous_successes'], plan['walking']['criteria']['yaw_error_max'])
        require(counts==(128,50,25,1000,1550,116,.2) and plan['feedback']['maximum_yaw_input_rad_s']==.15,
            'The previous forward scoring contract differs')
        source_files = [ENTRY, previous.MASTER, results, source_config, after_score, after_raw, zero_receipt,
                        *(zero/name for name in zero_info['files'])]
        plan['frozen_sha256'].update({rel(file):sha(file) for file in source_files})
        plan.update(run_id=f'duck_direct_forward_{key}_{stamp}', stage=STAGE,
            maximum_controls=1050, maximum_sampling_wall_seconds=360, physics_reference_controls=1550,
            roll_controls=0, landing_stand_controls=0, minimum_roll_completion=0,
            maximum_control_transitions=1050*128, maximum_physics_substeps=1050*5,
            route_components={role:plan['route_components'][role] for role in ('stand','forward')},
            route_scope='One second of standing, then twenty seconds of forward walking in one physics episode.',
            comparison_source=rel(after_score), comparison_forward_score=rel(after_raw),
            comparison_zero_receipt=rel(zero_receipt), guarded_output=rel(REPORT/key), page=rel(REPORT/key/'route-view.html'))
        verify_sources(plan)
        prepared[key] = plan

    PLANS.mkdir()
    REPORT.mkdir()
    hashes = {}
    for key,plan in prepared.items():
        folder, path = REPORT/key, PLANS/(key+'.json')
        folder.mkdir()
        put(path,plan)
        hashes[key] = sha(path)
        put(folder/'windows-entry.json',dict(protocol_sha256=hashes[key], evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE, order=list(hashes), protocols=hashes, total_control_transitions=4*1050*128,
        assigned_worlds_per_case=128, forward_yaw_limit_rad_s=.2, minimum_forward_successes=116,
        heldout_seed=2246, heldout_used=False, learning_updates=0, hardware_io=False))
    project['goal_execution'][KEY] = dict(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
        current_evidence=rel(results), next_gate='Measure direct forward under both recorded friction conditions before selecting a control change.')
    pair.trial.write(P,project)
    (REPORT/'current-evaluation.md').write_text('# 站立接前进对照\n\n四组协议已冻结。[阶段方案](../../../configs/experiments/roll_friction_stand_forward/master.json)\n', encoding='utf-8')
    print('STAND_FORWARD_FROZEN',len(hashes),sha(MASTER),flush=True)


def checked(key, status):
    from native_nominal_baseline import verify_sources

    project, master = read(P), read(MASTER)
    run, task = project['goal_execution'][KEY], project['staged_training_request']
    require(pair.trial.allowed(project,STAGE) and run['status']==status and sha(MASTER)==run['master_sha256']
        and key==next((item for item in master['order'] if item not in run['scored']),None)
        and not task['simulation_running'] and not task['learning_running'], 'Diagnostic phase or case changed')
    path = PLANS/(key+'.json')
    require(sha(path)==master['protocols'][key], 'Diagnostic case changed')
    plan = read(path)
    verify_sources(plan)
    return project, path, plan


def collect(key):
    with pair.trial.supervisor.TRAIN_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        collect_locked(key)


def collect_locked(key):
    project, path, plan = checked(key,'ready')
    out = REPORT/key
    zero_src = ROOT/plan['shared_zero']
    reference = read(ROOT/plan['comparison_zero_receipt'])
    (out/'zero').mkdir()
    for name,digest in reference['files'].items():
        shutil.copyfile(zero_src/name,out/'zero'/name)
        require(sha(out/'zero'/name)==digest, 'Matched reset copy differs')

    os.environ.update(MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    import mjlab.envs as env_module
    import common_scene_contract as scene
    import guarded_roll_collector as collector
    import run_continuous_roll_stand_walk as base
    from factor_scene_guard_adapter import factor_guard_bindings, require_complete
    from native_roll_observation_bounded import NativeRollObservationBounded
    from run_local_getup_ppo import MemoryWatch

    limits = read(previous.RANGES)['skills']['roll']
    ranges = dict(foot_friction=limits['training_foot_friction'],vin_range_v=limits['training_voltage'])
    nominal = dict(foot_friction=limits['nominal_foot_friction'],vin_range_v=limits['nominal_voltage'])
    factor = dict(name=plan['factor_arm'],**plan['parameter_factors'],
                  changed_factors=[] if plan['factor_arm']=='nominal' else ['foot_friction'])
    startup = ROOT/plan['reference_root']
    owner, digest = os.getpid(), sha(path)
    project['staged_training_request'].update(current_protocol=rel(path),simulation_running=True)
    project['goal_execution'][KEY].update(status='sampling',worker_pid=owner)
    pair.trial.write(P,project)

    def permitted():
        current = pair.bounded.read_status(P,reader=read)
        task, run = current['staged_training_request'],current['goal_execution'][KEY]
        return (pair.trial.allowed(current,STAGE) and run.get('worker_pid')==owner and run['status']=='sampling'
                and task['simulation_running'] and not task['learning_running']
                and task['current_protocol']==rel(path) and sha(path)==digest)

    configure = base.configuration
    def configure_direct(spec):
        # Keep the original episode clock inside the scene. The collector,
        # rather than an environment reset, ends this shorter branch.
        return configure(dict(spec,maximum_controls=spec['physics_reference_controls']))

    view = NativeRollObservationBounded(plan['run_id'],out,permitted)
    view.state.update(purpose='站立接前进对照',maximum_controls=plan['maximum_controls'])
    watch = MemoryWatch(plan['max_gpu_fraction'])
    result = dict(status='review-required',run_id=plan['run_id'])
    ok = False
    try:
        view.start()
        def allowed():
            watch.check()
            return view.check()

        sample = collector.make_guarded_collector(base,allowed)
        def guarded(*args, **kwargs):
            require_complete(binding)
            view.begin_collect()
            return sample(*args,**kwargs)

        with ExitStack() as stack:
            stack.enter_context(watch.active())
            stack.enter_context(base.torch.no_grad())
            stack.enter_context(patch.object(collector,'Sequence',StandForward))
            stack.enter_context(patch.object(base,'configuration',configure_direct))
            binding = stack.enter_context(factor_guard_bindings(base,scene,env_module,arm=factor,nominal=nominal,
                ranges=ranges,startup_root=startup,reference_sha256=plan['reference_sha256']))
            stack.enter_context(patch.object(base.live,'publish',view.publish))
            stack.enter_context(patch.multiple(base,PLAN=path,OUT=out,STARTUP=startup,PAGE=out/'route-view.html',
                                               checked=lambda:plan,collect=guarded))
            require(permitted(),'Diagnostic approval changed before collecting')
            base.run(False)
            require_complete(binding)
        raw = ROOT/'records/evaluations'/plan['run_id']/'guarded-evaluation.json'
        result.update(read(raw),evidence_sha256=sha(raw))
        ok = True
    except BaseException as error:
        result['reason'] = type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            view.close(result)
            put(out/'collection-result.json',dict(result,gpu_samples=watch.rows))
        finally:
            current = read(P)
            state = current['goal_execution'][KEY]
            if state.get('worker_pid')==owner:
                state.pop('worker_pid')
                state['status'] = 'score-pending' if ok else 'review-required'
                current['staged_training_request']['simulation_running'] = False
                pair.trial.write(P,current)


def score_case(key):
    import roll_stand_walk_sequence as sequence
    from score_roll_friction_route import score

    project, path, plan = checked(key,'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    collected = read(REPORT/key/'collection-result.json')
    require(collected['run_id']==plan['run_id'] and collected['evidence_sha256']==sha(folder/'guarded-evaluation.json'),
            'Collection receipt changed')
    with patch.object(sequence,'Sequence',StandForward):
        row = score(plan,folder)
    require(not any(row['roll_world_pass'] or []), 'A direct walking trial recorded a roll')
    state = project['goal_execution'][KEY]
    full = row['full_route'] and not any(row['safety'].values())
    if full:
        raw_score = read(folder/'raw-score.json')
        require(set(raw_score['forward_at'])<={-1,50}, 'Forward entry occurred outside the frozen transition')
        row['raw_score_sha256'] = sha(folder/'raw-score.json')
        row['forward_entries'] = sum(item['entered_forward'] for item in raw_score['forward'])
    target = REPORT/key/'independent.json'
    put(target,dict(row,seed=plan['seed'],condition=plan['factor_arm'],protocol_sha256=sha(path)))
    state['scored'][key] = dict(path=rel(target),sha256=sha(target))
    state['status'] = 'ready' if full else 'review-required'
    pair.trial.write(P,project)
    print('STAND_FORWARD_SCORED',key,row.get('forward_entries'),row['route_success'],full,flush=True)
    return 0 if full else 2


def forward_summary(rows):
    entered = {item['world']:item for item in rows if item['entered_forward']}
    yaw = {world:item['yaw_error_rad_s'] for world,item in entered.items() if item['yaw_error_rad_s'] is not None}
    faults = Counter(label for item in entered.values() for label in item['failed_checks'])
    return dict(entered=len(entered),passed=sum(item['success'] for item in entered.values()),
                yaw_failure_worlds=faults['yaw_error'],failure_counts=dict(faults),
                median_abs_yaw_rad_s=float(np.median(list(yaw.values()))) if yaw else None),yaw


def compare():
    project,master = read(P),read(MASTER)
    state = project['goal_execution'][KEY]
    require(pair.trial.allowed(project,STAGE) and sha(MASTER)==state['master_sha256'] and state['status']=='ready'
        and set(state['scored'])==set(master['order']), 'The four direct cases are not ready for comparison')
    rows = []
    for key in master['order']:
        path = PLANS/(key+'.json')
        plan, evidence = read(path), state['scored'][key]
        point = read(ROOT/evidence['path'])
        require(sha(path)==master['protocols'][key]==point['protocol_sha256']
                and sha(ROOT/evidence['path'])==evidence['sha256'] and point['full_route'], 'Direct case evidence changed')
        direct = ROOT/'records/evaluations'/point['run_id']/'raw-score.json'
        after = ROOT/plan['comparison_forward_score']
        require(sha(direct)==point['raw_score_sha256'] and sha(after)==plan['frozen_sha256'][rel(after)],
                'Forward scoring evidence changed')
        direct_metrics,direct_yaw = forward_summary(read(direct)['forward'])
        post_metrics,post_yaw = forward_summary(read(after)['forward'])
        shared = sorted(direct_yaw.keys() & post_yaw.keys())
        paired = [post_yaw[i]-direct_yaw[i] for i in shared]
        rows.append(dict(seed=plan['seed'],foot_friction=plan['parameter_factors']['foot_friction'],assigned=128,
            direct=direct_metrics,post_roll=post_metrics,paired_worlds=len(shared),
            paired_median_post_minus_direct_rad_s=float(np.median(paired)) if paired else None,case=key,
            evidence=evidence['path'],post_roll_evidence=plan['comparison_source'],clips=point['clips']))
    output = REPORT/'paired-analysis.json'
    put(output,dict(rows=rows,forward_controls=1000,yaw_limit_rad_s=.2,learning_updates=0,heldout_used=False,hardware_io=False))
    state.update(status='reviewed',result=rel(output),result_sha256=sha(output))
    project['staged_training_request'].update(current_evidence=rel(output),
        next_gate='Use the paired direct walking evidence to isolate the low-friction yaw control failure.')
    pair.trial.write(P,project)
    (REPORT/'current-evaluation.md').write_text('# 站立接前进对照\n\n四组已独立复核。[配对结果](paired-analysis.json)\n',encoding='utf-8')
    print('STAND_FORWARD_COMPLETE',rel(output),flush=True)


def batch():
    master = read(MASTER)
    for key in master['order']:
        project = read(P)
        run = project['goal_execution'][KEY]
        if key in run['scored']:
            continue
        require(pair.trial.allowed(project,STAGE) and run['status'] in ('ready','score-pending'), 'Case awaits review')
        plan = read(PLANS/(key+'.json'))
        condition = '标称摩擦' if plan['factor_arm']=='nominal' else '低摩擦'
        (REPORT/'current-evaluation.md').write_text('# 站立接前进对照\n\n'
            f'已复核{len(run["scored"])}/4组。当前为种子{plan["seed"]}、{condition}。\n\n'
            f'[本组状态]({key}/native-view/native-status.txt) · [采样日志]({key}/collect-command.log)\n', encoding='utf-8')
        actions = ('collect','score') if run['status']=='ready' else ('score',)
        for action in actions:
            args = [sys.executable,'-B','-u',str(ENTRY),action,key]
            log_path = REPORT/key/f'{action}-command.log'
            with log_path.open('x',encoding='utf-8',newline='\n') as log:
                log.write('COMMAND '+subprocess.list2cmdline(args)+'\n')
                log.flush()
                result = subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('STAND_FORWARD_CASE',key,action,result.returncode,rel(log_path),flush=True)
            if result.returncode:
                return 2
    compare()
    return 0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    parser.add_argument('case',nargs='?')
    args = parser.parse_args()
    if args.action in ('collect','score') and not args.case:
        parser.error('Choose the frozen case')
    action = {'freeze':freeze,'batch':batch,'collect':collect,'score':score_case}[args.action]
    raise SystemExit(action(args.case) if args.action in ('collect','score') else action())
