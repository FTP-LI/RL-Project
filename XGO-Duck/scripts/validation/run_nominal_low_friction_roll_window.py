"""Compare early 1 ms rolling and two 2 ms repeats from one CUDA entry state."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import fcntl
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_timestep_low_friction_development as previous
from review_nominal_roll_closed_loop_timestep import matched_effect
from review_nominal_skill_composition_handoff import replay


base, driver, parent = previous.base, previous.driver, previous.parent
comparison, verifier = previous.comparison, previous.verifier
original, recorder = previous.repeat.original, previous.repeat.recorder
ROOT, PROJECT = previous.ROOT, previous.PROJECT
read, rel, sha, write = previous.read, previous.rel, previous.sha, previous.write
KEY, STAGE = 'nominal_low_friction_roll_window', 'nominal-low-friction-roll-window'
PREPARATION = 'nominal-low-friction-roll-clock-preflight'
CLOCK_KEY = 'nominal_low_friction_roll_clock'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = previous.REPORT.with_name('closed-loop-low-friction-roll-window')
ROLES = ('reference-2ms-first', 'reference-2ms-repeat', 'early-1ms-candidate')
LABELS = dict(zip(('shared', *ROLES), ('共同跌倒、释放与站稳', '2ms参照一', '2ms参照二', '首1.20秒1ms候选'), strict=True))
EARLY_CONTROLS = 60
MOTOR_TICKS = 5


def factor_at(control, role, start, switch, end):
    if role == 'shared' or not start <= control < end:
        return 1
    return 4 if role == ROLES[-1] and control < switch else 2


def expected_solves(controls, role, start, switch, end):
    rolling = max(0, min(controls,end)-start)
    refined = max(0, min(controls,switch)-start) if role == ROLES[-1] else 0
    return MOTOR_TICKS*(controls-start+rolling+2*refined)


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert parent.allowed(project,PREPARATION) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    lower, clock, mechanism = (goal[name] for name in (
        previous.KEY, CLOCK_KEY, previous.KEY+'_mechanism_review'))
    assert all(row['status']=='reviewed' for row in (lower,clock,mechanism)) and not lower['qualified']
    plan = read(ROOT/lower['protocol'],lower['protocol_sha256'])
    earlier = read(ROOT/lower['result'],lower['result_sha256'])
    bench = read(ROOT/clock['evidence'],clock['evidence_sha256'])
    why = read(ROOT/mechanism['evidence'],mechanism['evidence_sha256'])
    assert (bench['status'],bench['sliding_friction'],bench['actuator_hold_ms'])==('verified-cuda-clock',.8,4)
    assert [row['executed_solver_steps'] for row in bench['modes']]==[10,20]
    assert all(row['input_unchanged_within_every_interval'] for row in bench['modes'])
    assert not bench['closed_loop_evaluated'] and not bench['reserved_seed_used']
    d, seven = plan['composition'], plan['seven_command']
    start, end = plan['solver_comparison']['fine_solver_interval']
    switch, limit = start+EARLY_CONTROLS, d['maximum_controls']
    assert (start,switch,end,d['command_entry_control'],limit)==(600,660,1050,1125,2125)
    assert (d['physics_period_s'],d['control_period_s'],d['foot_friction'])==(.004,.02,.8)
    assert d['control_period_s']*EARLY_CONTROLS==1.2
    seed, worlds = plan['environment']['seed'], d['worlds']
    reserved = d['existing_low_friction_reserved_seed']
    assert (seed,worlds,reserved)==(1257,896,2246) and not earlier['reserved_seed_used']
    assert plan['roll_route']['minimum_roll_completion']==812 and seven['minimum_overall']==852
    assert len(seven['groups'])==7
    assert all(len(ids)==128 and seven['minimum_by_command'][key]==122 for key,ids in seven['groups'].items())
    support_losses = [row['world'] for row in why['rolling']]
    assert len(set(support_losses))==6 and bench['worlds'][:6]==support_losses
    saved = ROOT/plan['standing_pair']['archived_initial']
    reset = saved.with_name('reset.npz')
    assert sha(reset)==read(saved)['reset_sha256']
    controls = start+len(ROLES)*(limit-start)
    launches = MOTOR_TICKS*start+sum(expected_solves(limit,role,start,switch,end) for role in ROLES)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-early-clock-friction080-s{seed}-{stamp}'
    plan['environment']['run_id'] = run
    plan['standing_pair'].update(common_snapshot_control=start,roles=list(ROLES))
    plan['solver_comparison'].update(roles=list(ROLES),maximum_solver_launches=launches,
        candidate_early_solver_interval=[start,switch],candidate_early_solver_s=.001,
        solver_seconds_by_role={role:.002 for role in ROLES})
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=controls,
        maximum_physics_substeps=MOTOR_TICKS*controls,maximum_control_transitions=controls*worlds,
        roll_window=dict(reserved_seed=reserved,archived_support_loss_worlds=support_losses,
            promotion='Both passing 2 ms repeats take priority. Otherwise prefer a 1 ms arm passing all '
                      'original gates; one passing 2 ms arm alone requires another development check.'))
    for row, field in ((lower,'protocol'),(lower,'result'),(lower,'collection'),(clock,'protocol'),
                       (clock,'evidence'),(mechanism,'evidence')):
        plan['references'][row[field]] = row[field+'_sha256']
    for path in (saved,reset,Path(__file__).resolve(),Path(matched_effect.__code__.co_filename).resolve()):
        plan['references'][rel(path)] = sha(path)
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),observer=observer,physical_substeps=0,worlds=worlds,
        matched_entry_control=start,maximum_batched_controls=controls,
        maximum_solver_launches=launches,maximum_scene_transitions=controls*worlds))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,
        preflight=rel(receipt),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Collect one common entry plus three fixed-budget arms and score the original gates.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_ROLL_WINDOW_FROZEN',sha(PLAN),run,flush=True)


class WindowClock(original.SolverClock):
    def __init__(self, env, folder, plan):
        super().__init__(env,folder,plan)
        self.switch = plan['solver_comparison']['candidate_early_solver_interval'][1]

    def enter(self, role):
        self.role = role
        self.rows[role] = dict(self.new_arm(),transitions=[])
        self.check_model_step(1)

    def control_step(self, action):
        k = self.env.common_step_counter
        if self.role != 'shared':
            arm = self.rows[self.role]
            if k == self.start:
                self.early_started = monotonic()
            factor = factor_at(k,self.role,self.start,self.switch,self.stop)
            if factor != self.factor:
                before = self.factor
                self.select(factor)
                arm['transitions'].append(dict(control=k,prior_factor=before,new_factor=factor,
                    engine_clock_s=float(self.sim.data.time[0].item())))
        result = super().control_step(action)
        if self.role != 'shared' and k+1 == self.switch:
            self.rows[self.role]['early_window_wall_s'] = monotonic()-self.early_started
        return result


def collect(env, plan, actors, folder, observer, watch, ledger):
    fields = ROOT/plan['friction_development']['startup_fields']
    assert env.common_step_counter==env._sim_step_counter==0
    with np.load(fields,allow_pickle=False) as expected:
        for name in sorted(base.RESET_FIELDS):
            actual = parent.cpu(getattr(env.sim.model,name))
            np.testing.assert_array_equal(actual,expected['expanded_'+name],err_msg=name)
    ledger['actual_slide_friction'] = plan['composition']['foot_friction']
    clock = WindowClock(env,folder,plan)
    assigned = iter(ROLES)
    base_fork = recorder.StandingFork
    publish = observer.publish
    counters = original.CollectorLedger(ledger)

    class RollEntryFork(base_fork):
        def restore(self):
            clock.leave()
            state = super().restore()
            assert state['physical']['counters']==[clock.start,MOTOR_TICKS*clock.start]
            clock.enter(next(assigned))
            return state

    def publish_window(path, state):
        publish(path,dict(state,purpose=f'低摩擦滚翻步长对照｜{LABELS[state["phase"]]}',
            solver_launches=clock.executed))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=RollEntryFork),
              patch.object(env,'step',clock.control_step), patch.object(env.sim,'step',clock.step),
              patch.object(observer,'publish',publish_window)):
            recorder.collect(env,plan,actors,folder,observer,watch,counters)
    finally:
        clock.leave()
        ledger.update(counters,solver_launches=clock.executed,solver_roles=clock.rows)


def check_arm_clock(folder, role, reached, fork_clock, clocks, start, switch, end, worlds):
    clock = clocks[role]
    assert clock['start_time_s']==fork_clock and clock['motor_ticks']==MOTOR_TICKS*(reached-start)
    assert clock['solver_launches']==expected_solves(reached,role,start,switch,end)
    measured = read(folder/role/'closed-loop-clock.json',clock['audit_sha256'])
    assert measured['role']==role and measured['worlds']==worlds and measured['motor_delay_ticks']==1
    assert measured['checked_every_world_clock'] and measured['unchanged_motor_input_between_solver_launches']
    assert measured['controls']==clock['controls']
    assert [row['control'] for row in clock['controls']]==list(range(start,start+original.AUDIT_CONTROLS))
    for row in measured['controls']:
        original.check_control(row,factor_at(row['control'],role,start,switch,end))
    points = [start,switch,end] if role==ROLES[-1] else [start,end]
    assert [row['control'] for row in clock['transitions']]==[k for k in points if k<reached]
    for event in clock['transitions']:
        k = event['control']
        assert event['prior_factor']==factor_at(k-1,role,start,switch,end)
        assert event['new_factor']==factor_at(k,role,start,switch,end)
    assert clock['early_window_wall_s']>0
    return dict(solver_launches=clock['solver_launches'],early_window_wall_s=clock['early_window_wall_s'],
        early_scene_transitions_per_second=(switch-start)*worlds/clock['early_window_wall_s'],
        actual_switches=clock['transitions'])


def score():
    project, plan = driver.checked('score-pending')
    goal, task = project['goal_execution'], project['staged_training_request']
    node, folder = goal[KEY], ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert raw['actual_slide_friction']==.8 and not (raw['hardware_io'] or raw['learning_updates'] or raw['reserved_seed_used'])
    assert sha(folder/'protocol.json')==sha(PLAN) and sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    start,end = plan['solver_comparison']['fine_solver_interval']
    switch = plan['solver_comparison']['candidate_early_solver_interval'][1]
    worlds, clocks = plan['composition']['worlds'], raw['solver_roles']
    assert raw['prefix']['controls']==start and tuple(raw['branches'])==ROLES
    assert raw['controls']==start+sum(row['controls']-start for row in raw['branches'].values())
    assert raw['transitions']==raw['controls']*worlds<=plan['maximum_control_transitions']
    assert raw['physics_substeps']==MOTOR_TICKS*raw['controls']==sum(x['motor_ticks'] for x in clocks.values())
    assert raw['solver_launches']==sum(x['solver_launches'] for x in clocks.values())<=plan['solver_comparison']['maximum_solver_launches']
    assert clocks['shared']['solver_launches']==clocks['shared']['motor_ticks']==MOTOR_TICKS*start
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[start,MOTOR_TICKS*start]
    assert fork['physical']['fields']==len(fork['fields']) and fork['fields']==sorted(fork['field_signatures'])
    initial_path, release_path = folder/'initial.npz',folder/'release-poses.npz'
    assert sha(initial_path)==raw['initial_sha256'] and sha(release_path)==raw['release_sha256']
    with np.load(initial_path,allow_pickle=False) as stored:
        initial = dict(stored)
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,start,comparison.EVIDENCE_FIELDS,sha)
    chosen = plan['roll_window']['archived_support_loss_worlds']
    entry, outcomes, populations, roll = {}, {}, {}, {}
    for role in ROLES:
        branch = raw['branches'][role]
        timing = check_arm_clock(folder,role,branch['controls'],clocks['shared']['end_time_s'],clocks,
                                 start,switch,end,worlds)
        with patch.object(verifier,'standing_checks',comparison.branch_first):
            outcome, populations[role], entry[role] = verifier.verify_branch(
                plan,folder/role,branch,prefix,common,initial,fork)
        stitched = [dict(part,file=f'{name}/{part["file"]}') for name,segment in (
            ('shared',raw['prefix']),(role,branch)) for part in segment['chunks']]
        _, entered, passed, reason, peak, _ = replay(plan,folder,dict(chunks=stitched))
        np.testing.assert_array_equal(passed,populations[role]['roll'])
        np.testing.assert_array_equal(entered,entry[role]['mode']==comparison.ROLLING)
        outcome.update(timing=timing,early_window_solver_s=.001 if role==ROLES[-1] else .002,
            roll_entry=int(entered.sum()),roll_reasons=dict(Counter(reason)),
            archived_loss_recheck=[dict(world=world,reason=reason[world],
                peak_supported_deg=float(peak[1,world])) for world in chosen])
        outcomes[role] = outcome
        roll[role] = dict(passed=passed,reason=reason,peak_supported_deg=peak[1])
    for role in ROLES[1:]:
        for field in ('observed','actor_input','mode','raw','previous_target','actual_target'):
            np.testing.assert_array_equal(entry[role][field],entry[ROLES[0]][field],err_msg=field)
        np.testing.assert_array_equal(entry[role]['motor'][0],entry[ROLES[0]]['motor'][0])
    pairs = {}
    groups = plan['seven_command']['groups']
    for left,right in ((ROLES[1],ROLES[0]),(ROLES[-1],ROLES[0]),(ROLES[-1],ROLES[1])):
        pairs[f'{left}-vs-{right}'] = dict(
            terminal=verifier.compare(populations[left],populations[right],groups),
            roll=matched_effect(roll[left],roll[right],np.ones(worlds,bool)))
    passed = [outcomes[role]['passed_development_gate'] for role in ROLES]
    preferred = ROLES[0] if all(passed[:2]) else ROLES[-1] if passed[-1] else None
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='verified-low-friction-roll-window',protocol_sha256=sha(PLAN),
        collection_sha256=node['collection_sha256'],seed=plan['environment']['seed'],worlds=worlds,
        verified_fork=fork['physical'],first_control_and_first_delayed_motor_exact=True,
        transitions=raw['transitions'],solver_launches=raw['solver_launches'],results=outcomes,pairs=pairs,
        peak_gpu_mib=max(x['used_mib'] for x in raw['gpu_samples']),
        preferred_for_independent_check=preferred,learning_updates=0,hardware_io=False,reserved_seed_used=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),preferred=preferred)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    task.update(info,next_gate='Freeze an independent full-route check with the selected clock.' if preferred else
        'Use the three matched branches to locate the original gates remaining below threshold.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_ROLL_WINDOW_SCORED',sha(output),
        {role:(row['roll_complete'],row['complete']) for role,row in outcomes.items()},preferred,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.multiple(parent,config=base.configured,PLAN=PLAN),
          patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()
