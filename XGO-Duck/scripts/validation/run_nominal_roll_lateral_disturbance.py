"""Test the accepted full nominal route under an early-roll lateral impulse."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_roll_timestep_confirmation as confirmed
from check_roll_active_lateral_writer import apply_roll_lateral_push, lateral_world
from roll_stand_walk_sequence import ROLL
from verify_nominal_low_friction_release_integrator_readback import reconstruct_clock


repeat = confirmed.repeat
comparison, driver, verifier, parent = confirmed.comparison, confirmed.driver, confirmed.verifier, confirmed.parent
original, recorder = repeat.original, repeat.recorder
ROOT, PROJECT = confirmed.ROOT, confirmed.PROJECT
read, rel, sha, write = confirmed.read, confirmed.rel, confirmed.sha, confirmed.write
KEY = 'nominal_roll_lateral_disturbance'
STAGE = 'nominal-roll-lateral-disturbance'
PREVIOUS = 'nominal-low-friction-hip-knee-authority'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = confirmed.REPORT.with_name('closed-loop-roll-lateral-disturbance')
ROLES = ('no-push','left-push','right-push')
LABELS = dict(zip(('shared',*ROLES),('共同跌倒、起身与滚翻前段','无扰动参照','左向扰动','右向扰动'),strict=True))
VELOCITY = dict(zip(ROLES,(0.,.08,-.08),strict=True))
SEED, MOTOR_TICKS = 2257, 5
QUALIFY = parent.qualify


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'],project['goal_execution']
    assert parent.allowed(project,PREVIOUS) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    hip = goal['nominal_low_friction_hip_knee_authority']
    assert hip['status']=='reviewed' and not hip['position_preflight_admitted']
    last = read(ROOT/hip['result'],hip['result_sha256'])
    assert last['cohorts']['all']['protected_pairs']==8
    source = goal[confirmed.KEY]
    plan = read(ROOT/source['protocol'],source['protocol_sha256'])
    recorded = read(ROOT/source['result'],source['result_sha256'])
    assert source['passed'] and recorded['nominal_confirmation_passed'] and recorded['result']['complete']==872
    legacy = ROOT/'configs/experiments/continuous_roll_early_lateral_push_draft.json'
    original_trial = read(legacy)
    assert original_trial['injection_control']-original_trial['standing_heading_control']==8
    assert original_trial['groups_horizontal_velocity_m_s']=={'reference':0.,'left-push':.08,'right-push':-.08}
    d,env,seven = (plan[k] for k in ('composition','environment','seven_command'))
    begin,stop = plan['solver_comparison']['fine_solver_interval']
    inject,end = begin+8,d['maximum_controls']
    assert (begin,inject,stop,d['command_entry_control'],end)==(600,608,1050,1125,2125)
    assert (d['foot_friction'],d['voltage_v'],d['physics_period_s'],d['control_period_s'])==(1.,7.7,.004,.02)
    assert (d['worlds'],env['seed'],d['existing_low_friction_reserved_seed'])==(896,2256,2246)
    assert (plan['roll_route']['minimum_roll_completion'],seven['minimum_overall'])==(812,852)
    assert len(seven['groups'])==7 and all(seven['minimum_by_command'][k]==122 for k in seven['groups'])
    for directory in ('records/evaluations','records/experiments'):
        assert not any(f'-s{SEED}-' in item.name or f'_s{SEED}_' in item.name for item in (ROOT/directory).iterdir())
    assert SEED not in (2256,2246)
    counts = [d['requested_pose_counts'][pose] for pose in parent.POSES]
    codes = np.random.default_rng(SEED).permutation(np.repeat(range(4),counts))
    seven['groups'] = parent.balanced_groups(codes,seven['commands'],SEED+70000)
    assert all(len(ids)==128 for ids in seven['groups'].values())
    assert sorted(i for ids in seven['groups'].values() for i in ids)==list(range(len(codes)))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-lateral-disturbance-2ms-s{SEED}-{stamp}'
    d['seed'] = SEED
    env.update(seed=SEED,run_id=run,split=dict(pose_codes=codes.tolist(),
        groups={pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(parent.POSES)}))
    for actor in plan['models'].values():
        actor['seed'] = SEED
    plan.pop('independent_confirmation')
    plan['standing_pair'].update(common_snapshot_control=inject,archived_initial=rel(REPORT/'zero/proof.json'),
                                 roles=list(ROLES))
    controls = inject+len(ROLES)*(end-inject)
    extra_solves = inject-begin+len(ROLES)*(stop-inject)
    launches = MOTOR_TICKS*(controls+extra_solves)
    plan['solver_comparison'].update(roles=list(ROLES),maximum_solver_launches=launches,
        solver_seconds_by_role={role:.002 for role in ROLES})
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=controls,
        maximum_physics_substeps=MOTOR_TICKS*controls,maximum_control_transitions=controls*d['worlds'],
        lateral_disturbance=dict(seed=SEED,source_baseline_seed=2256,low_friction_reserved_seed=2246,
            departure_heading_control=begin,shared_injection_control=inject,signed_velocity_m_s=VELOCITY,
            minimum_eligible_rolling_worlds=812,minimum_rolling_eligibility_fraction=.95,
            intervention='One root horizontal velocity increment before the first solver of the boundary control.',
            eligible='Active and rolling at the boundary with no prior head-ground contact since entering the roll.',
            scoring='Original get-up, roll-and-landing, all seven assigned commands and complete-route gates in all arms.',
            automatic_retry=False,learning_updates=0,hardware_io=False))
    delivery = source['nominal_simulation_delivery']
    index = read(ROOT/delivery['path'],delivery['sha256'])
    assert all(item['model_sha256']==index['policy_roles'][key]['onnx']['sha256'] for key,item in plan['models'].items())
    assert sha(ROOT/hip['result'])==hip['result_sha256']
    plan['references'].update({source['protocol']:source['protocol_sha256'],source['result']:source['result_sha256'],
        hip['result']:hip['result_sha256'],delivery['path']:delivery['sha256']})
    for path in (Path(__file__).resolve(),legacy,Path(apply_roll_lateral_push.__code__.co_filename).resolve(),
                 Path(reconstruct_clock.__code__.co_filename).resolve()):
        plan['references'][rel(path)] = sha(path)
    parent.base.shared.assert_sources(plan)
    observation = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),run_id=run,observer=observation,seed=SEED,
        affected_controls=[begin,inject,stop],worlds=d['worlds'],maximum_batched_motor_controls=controls,
        maximum_solver_launches=launches,maximum_scene_transitions=controls*d['worlds'],physical_steps=0))
    goal[KEY] = dict(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,
        preflight=rel(receipt),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Verify the actual CUDA reset, velocity write and restoration before running all matched routes.')
    write(PROJECT,project)
    print('NOMINAL_LATERAL_DISTURBANCE_FROZEN',sha(PLAN),run,flush=True)


def qualify_writer(env, obs, plan, actors, folder):
    certificate = QUALIFY(env,obs,plan,actors,folder)
    sim = env.sim.data
    saved = sim.qvel.clone()
    reset_pose,ctrl,observed = (parent.cpu(x) for x in (sim.qpos,sim.ctrl,env.obs_buf['actor']))
    n = env.num_envs
    eligible = np.arange(n)%2==0
    addresses = parent.cpu(env.scene['robot'].data.indexing.free_joint_v_adr[:2])
    np.testing.assert_array_equal(addresses,[0,1])
    all_zero = parent.cpu(saved)
    outside = np.ones(all_zero.shape,bool)
    outside[np.ix_(eligible,addresses)] = False
    rows = {}
    rng = torch.cuda.get_rng_state(env.device).clone()
    try:
        for role,magnitude in VELOCITY.items():
            event = apply_roll_lateral_push(env,eligible,reset_pose[:,3:7],magnitude)
            current = parent.cpu(sim.qvel)
            np.testing.assert_array_equal(current[outside],all_zero[outside])
            np.testing.assert_array_equal(event['world_ids'],np.flatnonzero(eligible))
            if magnitude:
                np.testing.assert_allclose(np.linalg.norm(event['measured_world_xy'],axis=-1),abs(magnitude),rtol=0,atol=1e-7)
            rows[role] = dict(affected=int(eligible.sum()) if magnitude else 0,
                error_m_s=float(abs(event['target_world_xy']-event['measured_world_xy']).max()))
            sim.qvel.copy_(saved)
    finally:
        sim.qvel.copy_(saved)
    for values,prior in ((sim.qpos,reset_pose),(sim.qvel,all_zero),(sim.ctrl,ctrl),(env.obs_buf['actor'],observed)):
        np.testing.assert_array_equal(parent.cpu(values),prior)
    assert torch.equal(rng,torch.cuda.get_rng_state(env.device))
    assert env.common_step_counter==env._sim_step_counter==0
    write(folder/'push-writer.json',dict(status='verified-actual-cuda-velocity-write-and-restore',fields=['warp.qvel'],
        checked_horizontal_addresses=addresses.tolist(),fixture_eligible=bool(eligible.any()),eligible_worlds=int(eligible.sum()),
        rows=rows,physical_steps=0,history_and_observation_unchanged=True,protocol_sha256=sha(PLAN)))
    return certificate


def zero():
    with patch.object(parent,'qualify',qualify_writer):
        parent.zero()
    project = read(PROJECT)
    writer = REPORT/'zero/push-writer.json'
    project['goal_execution'][KEY].update(writer=rel(writer),writer_sha256=sha(writer))
    write(PROJECT,project)


def sample():
    project,plan = driver.checked('sample-ready')
    node = project['goal_execution'][KEY]
    proof_path = ROOT/node['zero']
    proof = read(proof_path,node['zero_sha256'])
    writer = read(ROOT/node['writer'],node['writer_sha256'])
    assert proof['protocol_sha256']==writer['protocol_sha256']==sha(PLAN)
    assert not proof['physical_substeps'] and not writer['physical_steps']
    assert writer['status']=='verified-actual-cuda-velocity-write-and-restore'
    assert plan['environment']['seed']==SEED
    driver.sample()


class ImpulseClock(original.SolverClock):
    def __init__(self,env,folder,plan):
        super().__init__(env,folder,plan)
        self.roll_start = self.start
        self.start = plan['standing_pair']['common_snapshot_control']
        self.rows['shared']['engine_changes'] = []
        self.pending_push = None

    def finish_arm(self):
        self.rows[self.role]['end_time_s'] = float(self.sim.data.time[0].item())

    def enter(self,role):
        self.role = role
        self.rows[role] = dict(self.new_arm(),engine_changes=[])
        self.check_model_step(2)

    def control_step(self,action):
        control = self.env.common_step_counter
        if control in (self.roll_start,self.stop):
            step = 2 if control==self.roll_start else 1
            self.select(step)
            self.rows[self.role]['engine_changes'].append(dict(control=control,factor=step,
                time_s=float(self.sim.data.time[0].item())))
        return super().control_step(action)

    def step(self):
        if self.pending_push is not None:
            role,book,arrays = self.pending_push
            assert self.role==role and self.factor==2 and self.env.common_step_counter==self.start
            assert self.env._sim_step_counter==MOTOR_TICKS*self.start+1
            first = parent.cpu(self.sim.data.qvel)
            np.testing.assert_array_equal(first,arrays['after'])
            path = self.folder/role/'impulse.npz'
            np.savez_compressed(path,**arrays,first_solver_qvel=first)
            book.update(archive_sha256=sha(path),first_solver_exact=True)
            self.pending_push = None
        return super().step()


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = ImpulseClock(env,folder,plan)
    base_fork,composition = recorder.StandingFork,recorder.Composition
    names = iter(ROLES)
    counters = original.CollectorLedger(ledger)
    settings = plan['lateral_disturbance']
    start,mark = settings['departure_heading_control'],settings['shared_injection_control']
    book = counters['injections'] = {}
    publish = observer.publish

    class CaptureDeparture(composition):
        def observe_roll(self,control,frame):
            super().observe_roll(control,frame)
            if control==start:
                self.lateral_heading = frame['qpos'][:,3:7].copy()
                self.head_during_prefix = frame['head_contact'].copy()
            if start<control<=mark:
                self.head_during_prefix |= frame['head_contact']
            if control==mark:
                self.rolling_now = (self.roll_route.phase==ROLL)&frame['active']
                self.eligible_at_push = self.rolling_now&~self.head_during_prefix

    class SameRollState(base_fork):
        def __init__(self,plant,controller,actor,memory):
            self.eligible = controller.eligible_at_push
            self.heading = controller.lateral_heading
            rolling = int(controller.rolling_now.sum())
            count = int(self.eligible.sum())
            coverage_path = folder/'impulse-coverage.json'
            mask_path = folder/'impulse-coverage.npz'
            np.savez_compressed(mask_path,heading=self.heading,eligible=self.eligible,already_contacted_head=controller.head_during_prefix)
            write(coverage_path,dict(control=mark,heading_control=start,active_rolling=rolling,eligible=count,
                archive_sha256=sha(mask_path),min_count=settings['minimum_eligible_rolling_worlds'],
                min_fraction=settings['minimum_rolling_eligibility_fraction']))
            counters['impulse_coverage_sha256'] = sha(coverage_path)
            assert count>=settings['minimum_eligible_rolling_worlds']
            assert count/rolling>=settings['minimum_rolling_eligibility_fraction']
            super().__init__(plant,controller,actor,memory)

        def restore(self):
            clock.finish_arm()
            clock.select(2)
            state = super().restore()
            assert state['physical']['counters']==[mark,MOTOR_TICKS*mark]
            role = next(names)
            clock.enter(role)
            magnitude = settings['signed_velocity_m_s'][role]
            before = parent.cpu(env.sim.data.qvel)
            event = apply_roll_lateral_push(env,self.eligible,self.heading,magnitude)
            after = parent.cpu(env.sim.data.qvel)
            visible = self.plant.live_signatures()
            changed = [name for name in self.plant.expected if visible[name]!=self.plant.expected[name]]
            assert changed==(['warp.qvel'] if magnitude else []),changed
            row = book[role] = dict(control=mark,velocity_m_s=magnitude,fields_changed=changed)
            clock.pending_push = role,row,dict(before=before,after=after,**event)
            return state

    def publish_case(path,state):
        publish(path,dict(state,purpose=f'连续滚翻横向扰动｜{LABELS[state["phase"]]}',solver_launches=clock.executed))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,
                             StandingFork=SameRollState,Composition=CaptureDeparture),
              patch.object(env,'step',clock.control_step),patch.object(env.sim,'step',clock.step),
              patch.object(observer,'publish',publish_case)):
            recorder.collect(env,plan,actors,folder,observer,watch,counters)
    finally:
        clock.leave()
        ledger.update(counters,solver_launches=clock.executed,solver_roles=clock.rows)


def boundary(plan,folder,record,trace):
    control = plan['standing_pair']['common_snapshot_control']
    row, = (x for x in record['boundaries'] if x['control']==control)
    path = folder/row['file']
    assert sha(path)==row['sha256']
    with np.load(path,allow_pickle=False) as data:
        first = dict(data)
    np.testing.assert_array_equal(first['mode'],trace['mode'][control])
    return dict(boundary_control=control,worlds_rolling=int((first['mode']==comparison.ROLLING).sum())),first


def verify_push(folder, plan, raw, prefix, role, saved):
    start,split,end = saved['coordinates']
    branch = raw['branches'][role]
    measured = raw['injections'][role]
    strength = plan['lateral_disturbance']['signed_velocity_m_s'][role]
    assert measured['control']==split and measured['velocity_m_s']==strength and measured['first_solver_exact']
    assert measured['fields_changed']==(['warp.qvel'] if strength else [])
    file = folder/role/'impulse.npz'
    assert sha(file)==measured['archive_sha256']
    with np.load(file,allow_pickle=False) as data:
        event = dict(data)
    ids = np.flatnonzero(saved['eligible'])
    rest = prefix['qvel'][-1]
    np.testing.assert_array_equal(event['before'],rest)
    np.testing.assert_array_equal(event['world_ids'],ids)
    _,delta = lateral_world(saved['heading'][ids],strength)
    requested = delta[:,:2].astype(rest.dtype)
    candidate = rest.copy()
    candidate[ids,:2] += requested
    for label,target in (('after',candidate),('first_solver_qvel',candidate),('target_world_xy',requested),
                         ('measured_world_xy',candidate[ids,:2]-rest[ids,:2])):
        np.testing.assert_array_equal(event[label],target,err_msg=label)
    timing = raw['solver_roles'][role]
    reached = branch['controls']
    assert timing['motor_ticks']==MOTOR_TICKS*(reached-split)
    assert timing['solver_launches']==MOTOR_TICKS*(reached-split+min(end,reached)-split)
    _,t0 = reconstruct_clock(0,0,split,{start:2})
    points,final = reconstruct_clock(0,0,reached,{start:2,end:1})
    assert timing['start_time_s']==t0==raw['solver_roles']['shared']['end_time_s'] and timing['end_time_s']==final
    switches = [dict(control=x['control'],factor=x['new_factor'],time_s=x['engine_clock_s']) for x in points if x['control']>split]
    assert timing['engine_changes']==switches
    proof = read(folder/role/'closed-loop-clock.json',timing['audit_sha256'])
    assert proof['role']==role and proof['worlds']==plan['composition']['worlds'] and proof['motor_delay_ticks']==1
    assert proof['checked_every_world_clock'] and proof['unchanged_motor_input_between_solver_launches']
    assert [row['control'] for row in proof['controls']]==list(range(split,split+original.AUDIT_CONTROLS))
    assert proof['controls']==timing['controls']
    for event_row in proof['controls']:
        original.check_control(event_row,2)
    return dict(affected_worlds=len(ids) if strength else 0,sign_m_s=strength,only_qvel_xy_changed=True,
        maximum_xy_rounding_error_m_s=float(abs(event['measured_world_xy']-requested).max()),
        first_cuda_solver_qvel_exact=True,solver_launches=timing['solver_launches'],solver_clock_end_s=final)


def score():
    project,plan = driver.checked('score-pending')
    task,node = project['staged_training_request'],project['goal_execution'][KEY]
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not any(raw[k] for k in ('learning_updates','hardware_io','reserved_seed_used'))
    assert sha(folder/'protocol.json')==sha(PLAN) and sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    start,end = plan['solver_comparison']['fine_solver_interval']
    split = plan['standing_pair']['common_snapshot_control']
    worlds = plan['composition']['worlds']
    assert (plan['environment']['seed'],worlds)==(SEED,896)
    assert tuple(raw['branches'])==ROLES and raw['prefix']['controls']==split
    assert raw['controls']==split+sum(x['controls']-split for x in raw['branches'].values())
    assert raw['transitions']==raw['controls']*worlds<=plan['maximum_control_transitions']
    assert raw['physics_substeps']==MOTOR_TICKS*raw['controls']==sum(x['motor_ticks'] for x in raw['solver_roles'].values())
    assert raw['solver_launches']==sum(x['solver_launches'] for x in raw['solver_roles'].values())
    assert raw['solver_launches']<=plan['solver_comparison']['maximum_solver_launches']
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[split,MOTOR_TICKS*split] and fork['physical']['fields']==len(fork['fields'])
    shared_clock = raw['solver_roles']['shared']
    points,shared_time = reconstruct_clock(0,0,split,{start:2})
    assert shared_clock['end_time_s']==shared_time and shared_clock['solver_launches']==MOTOR_TICKS*(2*split-start)
    assert shared_clock['engine_changes']==[dict(control=start,factor=2,time_s=points[0]['engine_clock_s'])]
    extent = read(folder/'impulse-coverage.json',raw['impulse_coverage_sha256'])
    assert sha(folder/'impulse-coverage.npz')==extent['archive_sha256']
    assert sha(folder/'initial.npz')==raw['initial_sha256'] and sha(folder/'release-poses.npz')==raw['release_sha256']
    with (np.load(folder/'initial.npz',allow_pickle=False) as first,
          np.load(folder/'impulse-coverage.npz',allow_pickle=False) as mask,
          np.load(folder/'release-poses.npz',allow_pickle=False) as pose):
        initial,observed,release = dict(first),dict(mask),pose['poses']
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    fields = tuple(dict.fromkeys((*comparison.EVIDENCE_FIELDS,'qvel')))
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,split,fields,sha)
    np.testing.assert_array_equal(prefix['qpos'][start-1,:,3:7],observed['heading'])
    np.testing.assert_array_equal(prefix['head_contact'][start-1:split].any(axis=0),observed['already_contacted_head'])
    rolling = (prefix['roll_phase'][-1]==ROLL)&prefix['active'][-1]
    np.testing.assert_array_equal(rolling&~observed['already_contacted_head'],observed['eligible'])
    assert extent['active_rolling']==int(rolling.sum()) and extent['eligible']==int(observed['eligible'].sum())
    assert extent['eligible']>=extent['min_count'] and extent['eligible']/extent['active_rolling']>=extent['min_fraction']
    saved = dict(observed,coordinates=(start,split,end))
    rolls, populations, controls = {},{},{}
    for role in ROLES:
        branch = raw['branches'][role]
        intervention = verify_push(folder,plan,raw,prefix,role,saved)
        with patch.object(verifier,'standing_checks',boundary):
            result, population, first_action = verifier.verify_branch(plan,folder/role,branch,prefix,common,initial,fork)
        result.update(impulse=intervention,roll_by_actual_release_pose={
            pose:dict(assigned=int((release==pose).sum()),rolled=int(population['roll'][release==pose].sum()))
            for pose in parent.POSES})
        if population['complete'] is not None:
            result['complete_by_actual_release_pose'] = {
                pose:int(population['complete'][release==pose].sum()) for pose in parent.POSES}
        rolls[role],populations[role],controls[role] = result,population,first_action
    first = controls[ROLES[0]]
    for role in ROLES[1:]:
        for field in ('observed','actor_input','mode','raw','previous_target','actual_target'):
            np.testing.assert_array_equal(controls[role][field],first[field],err_msg=f'{role}/{field}')
        np.testing.assert_array_equal(controls[role]['motor'][0],first['motor'][0],err_msg=f'{role}/first delayed motor')
    matched = {role:verifier.compare(populations[role],populations[ROLES[0]],plan['seven_command']['groups']) for role in ROLES[1:]}
    passed = all(row['passed_development_gate'] for row in rolls.values())
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='verified-independent-nominal-roll-disturbance',protocol_sha256=sha(PLAN),
        collection_sha256=node['collection_sha256'],seed=SEED,worlds=worlds,controls=raw['controls'],transitions=raw['transitions'],
        solver_launches=raw['solver_launches'],verified_fork=fork['physical'],coverage=extent,
        original_gates_passed_in_all_arms=passed,first_observation_policy_target_and_delayed_motor_exact=True,
        gpu_peak_mib=max(x['used_mib'] for x in raw['gpu_samples']),branches=rolls,paired_vs_no_push=matched,
        low_friction_reserved_seed_used=False,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),passed=passed)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    project.update(info)
    task.update(info,next_gate='Continue the independent robustness matrix and the unresolved low-friction full route.'
                if passed else 'Localize loss by perturbation direction and original phase, preserving the fresh independent seed result.')
    write(PROJECT,project)
    print('NOMINAL_LATERAL_DISTURBANCE_SCORED',sha(output),{k:(v['roll_complete'],v['complete']) for k,v in rolls.items()},passed,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','zero','sample','score'))
    args = cli.parse_args()
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(parent,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'zero':zero,'sample':sample,'score':score}[args.action]()
