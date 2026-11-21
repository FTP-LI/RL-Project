"""Run the frozen development routes on the official actor and both learners."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import fcntl
import os
import shutil
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_friction_pair as pair

ROOT, P, read, put, sha, rel, require = pair.ROOT, pair.P, pair.read, pair.put, pair.sha, pair.rel, pair.require
REPORT = ROOT / 'records/reports/roll-friction-paired-development'
PLANS = ROOT / 'configs/experiments/roll_friction_paired_development'
MASTER = PLANS / 'master.json'
STAGE = 'roll-friction-paired-development'
RANGES = ROOT / 'records/reports/common-plant-screen/domain-parameter-ranges.json'
TEMPLATE = ROOT / 'configs/experiments/roll_factor_low_continuation_s1242/friction-low.json'
MODELS = ('official', 'uniform', 'low-friction')
LABELS = {'official':'官方滚翻模型','uniform':'均匀摩擦训练对照组','low-friction':'低摩擦加权训练候选组'}


def freeze():
    from native_nominal_baseline import verify_sources

    project, plan = pair.checked({'evaluation-pending'})
    require(not PLANS.exists() and not REPORT.exists(), 'Preserve the current batch')
    template, draft = read(TEMPLATE), read(pair.trial.DRAFT)
    evaluation = plan['evaluation']
    require(evaluation['development_seeds']==[1247,1248] and evaluation['reserved_seed']==2246
            and not evaluation['reserved_seed_used'], 'Development and future holdout sets differ')
    model = {'official': (plan['recipe']['official_onnx'], plan['recipe']['official_sha256'])}
    fit = project['goal_execution']['roll_friction_paired']
    for role in pair.ROLES:
        export = read(pair.REPORT / role / 'export.json')
        prior = read(ROOT / fit['reviews'][role])
        require(prior['run_id']==export['run_id']==plan['arms'][role]['run_id']
                and prior['checkpoint_sha256']==export['checkpoint_sha256']==sha(ROOT/export['checkpoint'])
                and export['model_sha256']==sha(ROOT/export['model']), 'Trained model export changed')
        model[role] = export['model'], export['model_sha256']

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    implementations = [TEMPLATE, RANGES, pair.PLAN, ROOT/'scripts/validation/evaluate_roll_friction_pair.py',
                       ROOT/'scripts/validation/score_roll_friction_route.py']
    implementation_hashes = {rel(file):sha(file) for file in implementations}
    specifications = []
    for seed in evaluation['development_seeds']:
        for condition, mu in (('nominal',1.0),('friction-low',.8)):
            for role in MODELS:
                case = f's{seed}-{condition}'
                key = f'{case}-{role}'
                arm = deepcopy(template)
                m, digest = model[role]
                arm.update(run_id=f'duck_roll_development_{key}_{stamp}', stage=STAGE, seed=seed,
                    factor_arm=condition, parameter_factors=dict(foot_friction=mu, vin_range_v=7.7),
                    route_components=arm['route_components'] | {'roll':m},
                    route_scope='Official and two equal-sample learners; paired fixed-parameter development routes.',
                    heldout_used=False, dynamics_authorized=True)
                for key_from_template in ('admission_record','reference_zero','reference_zero_files','comparison_role','comparison_scope'):
                    arm.pop(key_from_template, None)
                arm.update(guarded_output=rel(REPORT/key), page=rel(REPORT/key/'route-view.html'),
                           shared_zero=rel(REPORT/(case+'-official')/'zero'))
                arm['roll'].update(seed=seed, model=m, model_sha256=digest)
                arm['frozen_sha256'].update(implementation_hashes)
                arm['frozen_sha256'][m] = digest
                require(arm['minimum_roll_completion']==arm['minimum_continuous_successes']==116
                        and arm['maximum_controls']==1550 and arm['worlds']==128, 'Original route gates changed')
                verify_sources(arm)
                specifications.append((key, arm, dict(case=case, role=role, seed=seed,
                    condition=condition, model_sha256=digest, run_id=arm['run_id'])))
    PLANS.mkdir()
    REPORT.mkdir()
    registered = {}
    for key, arm, spec in specifications:
        output, path = REPORT / key, PLANS / (key+'.json')
        output.mkdir()
        put(path, arm)
        registered[key] = dict(spec, plan_sha256=sha(path))
        put(output/'windows-entry.json', dict(protocol_sha256=registered[key]['plan_sha256'], status='native-observer-prepared'))
    put(MASTER, dict(status='frozen-development-evaluation', arms=registered, order=list(registered),
        planned_seed_count=len(evaluation['development_seeds']), reserved_seed=2246, reserved_seed_used=False,
        paired_low_friction_roll_gain_minimum=4, action_rules=draft['development_evaluation'], hardware_io=False))
    project['goal_execution']['roll_friction_development'] = dict(status='ready', master_sha256=sha(MASTER),
        next_arm=next(iter(registered)), scored={}, score_sha256={})
    project['staged_training_request'].update(current_stage=STAGE, current_protocol=rel(MASTER),
        current_protocol_frozen=True, next_gate='Verify same-state routes on the official, uniform, and enriched actors.')
    pair.trial.write(P, project)
    print('ROLL_FRICTION_ROUTES_FROZEN', len(registered), sha(MASTER), flush=True)


def checked(key, state):
    from native_nominal_baseline import verify_sources

    project, master = read(P), read(MASTER)
    task, run = project['staged_training_request'], project['goal_execution']['roll_friction_development']
    require(pair.trial.allowed(project, STAGE) and run['master_sha256']==sha(MASTER) and run['status']==state
        and run['next_arm']==key and not task['simulation_running'] and not task['learning_running']
        and not master['reserved_seed_used'] and not master['hardware_io'], 'Route stage or arm changed')
    path, spec = PLANS/(key+'.json'), master['arms'][key]
    require(sha(path)==spec['plan_sha256'], 'Route plan changed')
    plan = read(path)
    verify_sources(plan)
    return project, master, spec, path, plan


def collect(key, *, zero=False):
    with pair.trial.supervisor.TRAIN_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return collect_locked(key, zero)


def collect_locked(key, zero):
    project, _, spec, path, plan = checked(key, 'ready')
    output = REPORT/key
    if zero:
        require(spec['role']=='official', 'The common reset uses the official policy for each case')
    elif spec['role']!='official':
        baseline = REPORT/(spec['case']+'-official')
        source = baseline/'zero'
        fingerprint = read(baseline/'zero-reference.json')
        (output/'zero').mkdir()
        for name, digest in fingerprint['files'].items():
            require(sha(source/name)==digest, 'Common route reset changed')
            shutil.copyfile(source/name, output/'zero'/name)
            require(sha(output/'zero'/name)==digest, 'Common reset copy changed')
    if not zero:
        require((output/'zero'/'startup.npz').is_file(), 'Collect official zero first')

    os.environ.update(MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    import mjlab.envs as env_module
    import common_scene_contract as scene
    import run_continuous_roll_stand_walk as base
    from factor_scene_guard_adapter import factor_guard_bindings, require_complete
    from guarded_roll_collector import make_guarded_collector
    from mjlab.sim import Simulation
    from native_roll_observation_bounded import NativeRollObservationBounded
    from run_local_getup_ppo import MemoryWatch

    bounds = read(RANGES)['skills']['roll']
    ranges = dict(foot_friction=bounds['training_foot_friction'], vin_range_v=bounds['training_voltage'])
    nominal = dict(foot_friction=bounds['nominal_foot_friction'], vin_range_v=bounds['nominal_voltage'])
    factor = dict(name=plan['factor_arm'], **plan['parameter_factors'],
                  changed_factors=[] if spec['condition']=='nominal' else ['foot_friction'])
    startup = ROOT / plan['reference_root']
    task = project['staged_training_request']
    task.update(current_protocol=rel(path), simulation_running=not zero)
    pair.trial.write(P, project)

    def permitted():
        p = pair.bounded.read_status(P, reader=read)
        t, g = p['staged_training_request'], p['goal_execution']['roll_friction_development']
        return (pair.trial.allowed(p, STAGE) and not t['learning_running'] and t['simulation_running'] == (not zero)
                and t['current_protocol']==rel(path) and g['next_arm']==key and sha(path)==spec['plan_sha256'])

    def no_step(*_args, **_kwargs):
        raise RuntimeError('Route reset cannot advance physics')

    view = None if zero else NativeRollObservationBounded(plan['run_id'], output, permitted)
    watch = MemoryWatch(plan['max_gpu_fraction'])
    result = dict(status='review-required', run_id=plan['run_id'])
    success = False
    try:
        if view:
            condition = '标称摩擦' if spec['condition']=='nominal' else '低摩擦'
            view.state['purpose'] = f'滚翻配对评价 · {LABELS[spec["role"]]} · 种子{spec["seed"]} · {condition}'
            view.start()
            def allowed():
                watch.check()
                return view.check()

            sampler = make_guarded_collector(base, allowed)
            def sampled(*args, **kwargs):
                require_complete(guard)
                view.begin_collect()
                return sampler(*args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(watch.active())
            stack.enter_context(base.torch.no_grad())
            if zero:
                stack.enter_context(patch.object(Simulation, 'step', no_step))
            else:
                stack.enter_context(patch.object(base.live, 'publish', view.publish))
                stack.enter_context(patch.object(base, 'collect', sampled))
            guard = stack.enter_context(factor_guard_bindings(base, scene, env_module, arm=factor, nominal=nominal,
                ranges=ranges, startup_root=startup, reference_sha256=plan['reference_sha256']))
            stack.enter_context(patch.multiple(base, PLAN=path, OUT=output, PAGE=output/'route-view.html', STARTUP=startup,
                                               checked=lambda:plan))
            require(permitted(), 'Route approval changed')
            base.run(zero)
            require_complete(guard)
        if zero:
            state = read(output/'zero/runtime-contract.json')
            require(state['control_steps']==state['physics_substeps']==0, 'Unexpected route reset work')
            files = {name:sha(output/'zero'/name) for name in ('startup.npz','endpoint-contact.npz')}
            put(output/'zero-reference.json', dict(files=files, physics_substeps=0, learning_updates=0))
        else:
            raw = ROOT/'records/evaluations'/plan['run_id']/ 'guarded-evaluation.json'
            result.update(read(raw))
            result['evidence_sha256'] = sha(raw)
        success = True
        return 0
    except BaseException as error:
        result['reason'] = type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            if view:
                view.close(result)
                result['gpu_samples'] = watch.rows
                put(output/'collection-result.json', result)
        finally:
            p = read(P)
            p['staged_training_request']['simulation_running'] = False
            p['goal_execution']['roll_friction_development']['status'] = (
                ('ready' if zero else 'score-pending') if success else 'review-required')
            pair.trial.write(P, p)


def score_arm(key):
    from score_roll_friction_route import score

    project, master, spec, path, plan = checked(key, 'score-pending')
    dest = REPORT/key
    raw = ROOT / 'records/evaluations' / plan['run_id']
    result = read(dest/'collection-result.json')
    require(result['run_id']==spec['run_id'] and result['evidence_sha256']==sha(raw/'guarded-evaluation.json'),
            'Route collection receipt differs')
    analysis = score(plan, raw)
    proof = dest/'independent.json'
    put(proof, dict(analysis, model_sha256=spec['model_sha256'], protocol_sha256=sha(path),
                    seed=spec['seed'], condition=spec['condition'], role=spec['role']))
    run = project['goal_execution']['roll_friction_development']
    run['scored'][key] = rel(proof)
    run['score_sha256'][key] = sha(proof)
    reviewed = not any(analysis['safety'].values()) and (analysis['full_route'] or analysis['stop_reason']=='roll-gate-impossible')
    position = master['order'].index(key)+1
    if reviewed and position < len(master['order']):
        run.update(status='ready',next_arm=master['order'][position])
    else:
        run.update(status='development-complete' if reviewed else 'review-required',next_arm=None)
    pair.trial.write(P, project)
    print('ROLL_FRICTION_ROUTE_SCORED', key, analysis['roll_completion_at_common_prefix'], analysis['route_success'], run['next_arm'], flush=True)
    return 0 if reviewed else 2


def compare():
    project, master = read(P), read(MASTER)
    state = project['goal_execution']['roll_friction_development']
    require(pair.trial.allowed(project,STAGE) and state['status']=='development-complete'
        and sha(MASTER)==state['master_sha256'] and set(state['scored'])==set(master['arms']), 'Development comparisons are incomplete')
    grouped = {}
    for key, spec in master['arms'].items():
        path = ROOT/state['scored'][key]
        row = read(path)
        require(sha(path)==state['score_sha256'][key] and row['run_id']==spec['run_id'] and row['protocol_sha256']==spec['plan_sha256']
                and row['model_sha256']==spec['model_sha256'], 'Development evidence identity changed')
        grouped.setdefault(spec['case'], {})[spec['role']] = row
    results = []
    for case, arms in grouped.items():
        enriched, uniform, official = (arms[role] for role in ('low-friction','uniform','official'))
        first, second = uniform['roll_world_pass'], enriched['roll_world_pass']
        changed = None if first is None or second is None else dict(
            gains=sum(new and not old for old,new in zip(first,second,strict=True)),
            losses=sum(old and not new for old,new in zip(first,second,strict=True)))
        gain = None if changed is None else changed['gains']-changed['losses']
        passed = (enriched['passed_gate'] and (enriched['condition']=='nominal' or
                  gain is not None and gain>=master['paired_low_friction_roll_gain_minimum']))
        results.append(dict(case=case, seed=enriched['seed'], condition=enriched['condition'],
            counts={row['role']:dict(roll=row['roll_completion_at_common_prefix'], route=row['route_success'],
                                     worlds=row['worlds'], passed_original_gate=row['passed_gate'])
                    for row in (official,uniform,enriched)},
            candidate_roll_pairs_against_uniform=changed, candidate_roll_gain_against_uniform=gain, passed=passed))
    passed = all(row['passed'] for row in results)
    output = REPORT/'paired-analysis.json'
    put(output, dict(status='development-replayed', gate_passed=passed, paired=results, reserved_seed=master['reserved_seed'],
                     reserved_seed_used=False, hardware_io=False, learning_updates=0))
    state.update(status='holdout-pending' if passed else 'development-reviewed', result=rel(output))
    project['staged_training_request'].update(next_gate='Apply untouched roll holdout only once after the development gate.'
        if passed else 'Review the paired route evidence and select the next training question.')
    pair.trial.write(P,project)
    print('ROLL_FRICTION_DEVELOPMENT_REVIEWED', passed, rel(output), flush=True)


def batch():
    master = read(MASTER)
    live = REPORT/'current-evaluation.md'

    def launch(action, key):
        path = REPORT/key/f'{action}-command.log'
        cmd = [sys.executable,'-B','-u',__file__,action,key]
        with path.open('x',encoding='utf-8',newline='\n') as log:
            log.write('COMMAND '+subprocess.list2cmdline(cmd)+'\n')
            log.flush()
            result = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f'{key} {action} returned {result.returncode}; see {rel(path)}')

    while True:
        project = read(P)
        current = project['goal_execution']['roll_friction_development']
        require(pair.trial.allowed(project,STAGE) and sha(MASTER)==current['master_sha256'], 'Evaluation stopped before the next arm')
        if current['status']=='development-complete':
            compare()
            live.write_text('# 滚翻配对开发评估\n\n本轮评估已复核。[场景配对结果](paired-analysis.json)\n', encoding='utf-8')
            return 0
        key = current['next_arm']
        require(key in master['arms'] and current['status'] in ('ready','score-pending'), 'Evaluation needs an evidence review')
        arm = master['arms'][key]
        condition = '标称摩擦' if arm['condition']=='nominal' else '低摩擦'
        live.write_text('# 滚翻配对开发评估\n\n'
            f'已核验{len(current["scored"])}/{len(master["arms"])}轮。当前为种子{arm["seed"]}、{condition}、{LABELS[arm["role"]]}。\n\n'
            f'[VS Code当前状态]({key}/native-view/native-status.txt) · [运行命令日志]({key}/sample-command.log)\n', encoding='utf-8')
        print('ROLL_FRICTION_ROUTE_CURRENT',key,len(current['scored']),flush=True)
        if current['status']=='ready':
            if arm['role']=='official' and not (REPORT/key/'zero-reference.json').is_file():
                launch('zero',key)
            launch('sample',key)
        launch('score',key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','zero','sample','score','compare','batch'))
    parser.add_argument('arm', nargs='?')
    args = parser.parse_args()
    if args.action in ('freeze','compare','batch'):
        return {'freeze':freeze,'compare':compare,'batch':batch}[args.action]()
    if args.arm is None:
        parser.error('Development arm required')
    if args.action=='score':
        return score_arm(args.arm)
    return collect(args.arm, zero=args.action=='zero')


if __name__=='__main__':
    raise SystemExit(main())
