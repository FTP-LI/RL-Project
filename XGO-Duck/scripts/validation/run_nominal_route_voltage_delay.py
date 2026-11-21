"""Evaluate independent supply and actuator delay factors on the full accepted route."""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_roll_timestep_confirmation as accepted
import verify_nominal_skill_composition as verifier
from getup_release_fork_state import signature
from run_nominal_roll_support_transfer import layout
from selective_friction_guard import RESET_FIELDS, STATIC_FIELDS
from verify_nominal_low_friction_release_integrator_readback import reconstruct_clock


parent, clock_base = accepted.parent, accepted.repeat.original
ROOT, PROJECT, read, rel, sha, write = accepted.ROOT, accepted.PROJECT, accepted.read, accepted.rel, accepted.sha, accepted.write
KEY = 'nominal_route_voltage_delay'
STAGE = 'nominal-route-voltage-delay'
PRIOR = 'nominal-low-friction-right-support-contact-timing'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = accepted.REPORT.with_name('closed-loop-voltage-delay')
SEED = 2258
NOMINAL = 'nominal-7p7v-4ms'
ARMS = {
    NOMINAL: dict(supply_v=7.7,actuator_delay_ticks=1,changed=[]),
    'voltage-7p4v': dict(supply_v=7.4,actuator_delay_ticks=1,changed=['supply_v']),
    'delay-8ms': dict(supply_v=7.7,actuator_delay_ticks=2,changed=['actuator_delay_ticks']),
}
LABELS = {NOMINAL:'标称7.7V与4ms舵机延迟','voltage-7p4v':'7.4V供电','delay-8ms':'8ms舵机延迟'}
PREPARE, QUALIFY, COLLECT = parent.config,parent.qualify,parent.collect


def arm_key(name):
    return KEY + '_' + name.replace('-','_')


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert parent.allowed(project,PRIOR) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    nominal,lateral,timing = (goal[x] for x in (accepted.KEY,'nominal_roll_lateral_disturbance',
                                                'nominal_low_friction_right_support_contact_timing'))
    assert nominal['passed'] and lateral['passed'] and timing['status']=='reviewed'
    source = read(ROOT/nominal['protocol'],nominal['protocol_sha256'])
    nominal_result = read(ROOT/nominal['result'],nominal['result_sha256'])
    assert nominal_result['result']['complete']==872 and nominal_result['nominal_confirmation_passed']
    legacy = ROOT/'configs/experiments/roll_domain_parameter_cross_check_voltage-low.json'
    assert read(legacy)['parameter_arm']['factors']==dict(foot_friction=1.,vin_range=7.4)
    delivery = nominal['nominal_simulation_delivery']
    package = read(ROOT/delivery['path'],delivery['sha256'])
    d,env,commands = (source[x] for x in ('composition','environment','seven_command'))
    assert (env['seed'],d['worlds'],d['foot_friction'],d['voltage_v'])==(2256,896,1.,7.7)
    assert (d['release_control'],d['getup_to_stand_control'],d['roll_start_control'],
            d['roll_deadline_control'],d['command_entry_control'],d['maximum_controls'])==(100,550,600,1050,1125,2125)
    assert d['existing_low_friction_reserved_seed']==2246 and SEED not in (2246,2256,2257)
    assert source['roll_route']['minimum_roll_completion']==812 and commands['minimum_overall']==852
    assert all(commands['minimum_by_command'][name]==122 for name in commands['groups'])
    for location in ('records/evaluations','records/experiments'):
        assert not any(f'-s{SEED}-' in p.name or f'_s{SEED}_' in p.name for p in (ROOT/location).iterdir())
    for name,model in source['models'].items():
        assert model['model_sha256']==package['policy_roles'][name]['onnx']['sha256']
    parent.base.shared.assert_sources(source)
    counts = [d['requested_pose_counts'][pose] for pose in parent.POSES]
    codes = np.random.default_rng(SEED).permutation(np.repeat(range(4),counts))
    groups = parent.balanced_groups(codes,commands['commands'],SEED+70000)
    assert all(len(ids)==128 for ids in groups.values()) and sorted(i for ids in groups.values() for i in ids)==list(range(len(codes)))
    pinned = {node[name]:node[name+'_sha256'] for node in (nominal,lateral,timing)
              for name in ('protocol','result')}
    pinned[delivery['path']] = delivery['sha256']
    for file in (legacy,Path(__file__).resolve(),Path(layout.__code__.co_filename).resolve(),
                 Path(reconstruct_clock.__code__.co_filename).resolve()):
        pinned[rel(file)] = sha(file)
    for name,digest in pinned.items():
        assert sha(ROOT/name)==digest
    start,end = source['solver_comparison']['fine_solver_interval']
    horizon = d['maximum_controls']
    assert [start,end]==[600,1050] and source['solver_comparison']['maximum_solver_launches']==5*(horizon+end-start)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plans = PLAN.with_suffix('')
    plans.mkdir()
    REPORT.mkdir()
    receipt = parent.preflight()
    nodes = {}
    for name,arm in ARMS.items():
        assert arm['changed']==[key for key in ('supply_v','actuator_delay_ticks') if arm[key]!=ARMS[NOMINAL][key]]
        spec = deepcopy(source)
        spec.pop('standing_pair')
        spec.pop('independent_confirmation')
        run = f'duck-route-{name}-2ms-s{SEED}-{stamp}'
        spec['composition']['seed'] = SEED
        spec['composition']['voltage_v'] = arm['supply_v']
        spec['environment'].update(seed=SEED,run_id=run,split=dict(pose_codes=codes.tolist(),
            groups={pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(parent.POSES)}))
        for item in spec['models'].values():
            item['seed'] = SEED
        spec['seven_command']['groups'] = groups
        spec['references'].update(pinned)
        spec['solver_comparison'].update(roles=[name],solver_seconds_by_role={name:.002})
        spec.update(stage=STAGE,record=f'records/evaluations/{run}',named_factor=dict(arm,name=name),
            maximum_unique_controls=horizon,maximum_physics_substeps=5*horizon,maximum_control_transitions=horizon*d['worlds'])
        path = plans/f'{name}.json'
        write(path,spec)
        (REPORT/name).mkdir()
        nodes[name] = dict(protocol=rel(path),protocol_sha256=sha(path),status='zero-ready',run_id=run,hardware_io=False)
        goal[arm_key(name)] = nodes[name].copy()
    master = dict(seed=SEED,worlds=d['worlds'],arms=nodes,reference=NOMINAL,
        original_sources_sha256=pinned,policy_manifest=delivery,observer=receipt,
        maximum_controls_by_arm=horizon,maximum_total_control_transitions=horizon*d['worlds']*len(ARMS),
        maximum_total_solver_launches=5*(horizon+end-start)*len(ARMS),
        frozen_before_sampling=True,hardware_io=False,learning_updates=0,low_friction_reserved_seed=2246)
    write(PLAN,master)
    goal[KEY] = dict(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify matching physical fields and actual reset across all three factor arms before any rollout.')
    project.update(info)
    write(PROJECT,project)
    print('NOMINAL_VOLTAGE_DELAY_FROZEN',sha(PLAN),list(nodes),master['maximum_total_control_transitions'],flush=True)


def configuration(plan):
    cfg = PREPARE(plan)
    servo, = cfg.scene.entities['robot'].articulation.actuators
    factor = plan['named_factor']
    assert servo.vin_range==(7.7,7.7) and (servo.delay_min_lag,servo.delay_max_lag)==(1,1)
    assert servo.vin_min==7. and cfg.events['foot_friction'].params['ranges']==(1.,1.)
    servo.vin_range = (factor['supply_v'],)*2
    servo.delay_min_lag = servo.delay_max_lag = factor['actuator_delay_ticks']
    return cfg


def physical_arrays(env):
    model = env.sim.mj_model
    return {**{'static_'+name:np.asarray(getattr(model,name)) for name in sorted(STATIC_FIELDS)},
            **{'expanded_'+name:parent.cpu(getattr(env.sim.model,name)) for name in sorted(RESET_FIELDS)}}


def qualify(env,obs,plan,actors,folder):
    proof = QUALIFY(env,obs,plan,actors,folder)
    actuator, channels = layout(env,plan['walking']['joint_names'])
    factor = plan['named_factor']
    delay = factor['actuator_delay_ticks']
    assert env.common_step_counter==env._sim_step_counter==0
    assert actuator.cfg.delay_min_lag==actuator.cfg.delay_max_lag==delay
    assert actuator._delay_buffer is not None and env.cfg.decimation==5 and (env.physics_dt,env.step_dt)==(.004,.02)
    nominal,gain = parent.cpu(actuator.vin_tensor),parent.cpu(actuator.vin_drop_gain)
    assert nominal.shape==gain.shape==(env.num_envs,1)
    np.testing.assert_allclose(nominal,factor['supply_v'],atol=5e-7,rtol=0)
    assert (gain>0).all()
    path = folder/'factor-state.npz'
    np.savez_compressed(path,**physical_arrays(env),configured_voltage=nominal,vin_drop_gain=gain)
    write(folder/'factor-witness.json',dict(factor=factor,physics_steps=0,arrays_sha256=sha(path),
        channels=channels,vin_floor=float(actuator.cfg.vin_min),motor_tick_s=env.physics_dt,actor_tick_s=env.step_dt))
    return proof


class MotorClock(clock_base.SolverClock):
    def __init__(self,env,folder,plan):
        super().__init__(env,folder,plan)
        self.actuator,self.channels = layout(env,plan['walking']['joint_names'])
        self.calls = self.actuator.apply_delay
        self.window_begin,self.window_end = self.start-1,self.start+3
        self.motor_rows,self.audit_rows,self.changes = [],[],[]
        self.pending = None

    def apply_delay(self,command):
        k = self.env.common_step_counter
        if not self.window_begin<=k<self.window_end:
            return self.calls(command)
        assert self.pending is None
        columns = self.channels['actuator_columns']
        requested = parent.cpu(command.position_target[:,columns])
        previous_motor = parent.cpu(self.actuator._prev_motor_torque)
        result = self.calls(command)
        self.pending = dict(control=k,motor_interval=self.env._sim_step_counter,
            requested=requested,delayed=parent.cpu(result.position_target[:,columns]),previous_motor=previous_motor)
        return result

    def control_step(self,action):
        k = self.env.common_step_counter
        if k in (self.start,self.stop):
            factor = 2 if k==self.start else 1
            self.select(factor)
            self.changes.append(dict(control=k,factor=factor,time_s=float(self.sim.data.time[0].item())))
        if not self.start<=k<self.window_end:
            return self.environment_step(action)
        begin = self.sim.data.time.clone()
        row = dict(control=k,events=[],elapsed_solver_s_world_0=[])
        self.audit = row
        try:
            result = self.environment_step(action)
        finally:
            self.audit = None
        elapsed = self.sim.data.time.clone()-begin
        torch.testing.assert_close(elapsed,torch.full_like(elapsed,.02),atol=1e-5,rtol=0)
        assert row['events']==['solver']*10
        row['actor_elapsed_s_world_0'] = float(elapsed[0].item())
        self.audit_rows.append(row)
        return result

    def step(self):
        if self.pending is not None:
            row = self.pending
            self.pending = None
            assert row['motor_interval']==self.env._sim_step_counter
            row['effective_supply_v'] = parent.cpu(self.actuator._bam_model.actuator.vin)
            row['motor_torque_request_nm'] = parent.cpu(self.sim.data.ctrl[:,self.channels['motor_ids']])
            self.motor_rows.append(row)
        return super().step()

    def record(self,ledger):
        self.leave()
        ledger.update(solver_launches=self.executed,actuator_clock=dict(self.rows['shared'],
                      controls=self.audit_rows,changes=self.changes))
        if self.motor_rows:
            archive = self.folder/'motor-boundary.npz'
            np.savez_compressed(archive,**{key:np.stack([row[key] for row in self.motor_rows])
                                         for key in self.motor_rows[0]})
            ledger['motor_boundary_sha256'] = sha(archive)


def collect(env,plan,actors,folder,observer,watch,ledger):
    clock = MotorClock(env,folder,plan)
    factor = plan['named_factor']['name']
    published = observer.publish
    weights = signature({name:actor.state_dict() for name,actor in actors.items()})

    def publish(record,state):
        published(record,dict(state,purpose=f'连续动作供电与时延｜{LABELS[factor]}',solver_launches=clock.executed))

    try:
        with (patch.object(observer,'publish',publish),patch.object(env,'step',clock.control_step),
              patch.object(env.sim,'step',clock.step),patch.object(clock.actuator,'apply_delay',clock.apply_delay),
              patch.object(env,'_reset_idx',side_effect=AssertionError('No in-episode reset'))):
            COLLECT(env,plan,actors,folder,observer,watch,ledger)
            ledger['actor_weights_unchanged'] = weights==signature({name:actor.state_dict() for name,actor in actors.items()})
            assert ledger['actor_weights_unchanged']
            if 'video' in ledger:
                publish(folder,dict(read(folder/'state.json'),latest_clip=ledger['video']['file']))
    finally:
        clock.record(ledger)


def load_master():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert parent.allowed(project,STAGE) and not project['staged_training_request']['simulation_running']
    master = read(PLAN,node['protocol_sha256'])
    assert (master['seed'],master['worlds'])==(SEED,896) and tuple(master['arms'])==tuple(ARMS)
    return project,node,master


@contextmanager
def arm_context(name,expected):
    project,_,plan = load_master()
    node = project['goal_execution'][arm_key(name)]
    entry = plan['arms'][name]
    assert node['status']==expected and node['protocol_sha256']==entry['protocol_sha256']
    path = ROOT/node['protocol']
    assert sha(path)==entry['protocol_sha256']
    task = project['staged_training_request']
    task.update(current_protocol=rel(path),current_protocol_frozen=True)
    write(PROJECT,project)
    with (patch.multiple(parent,KEY=arm_key(name),STAGE=STAGE,PLAN=path,REPORT=REPORT/name,
                         config=configuration,qualify=qualify,collect=collect),
          patch.multiple(verifier,KEY=arm_key(name),PLAN=path,REPORT=REPORT/name),
          patch.object(accepted.comparison.feedback,'components',accepted.driver.isolated_components)):
        yield node,read(path)


def zero():
    project,node,master = load_master()
    assert node['status']=='zero-ready'
    for name in ARMS:
        with arm_context(name,'zero-ready'):
            parent.zero()
    project,node,master = load_master()
    inputs,files = {},{}
    for name in ARMS:
        directory=REPORT/name/'zero'
        p = project['goal_execution'][arm_key(name)]
        z = read(directory/'proof.json',p['zero_sha256'])
        assert z['physical_substeps']==0 and z['protocol_sha256']==p['protocol_sha256']
        witness = read(directory/'factor-witness.json')
        assert witness['factor']==dict(ARMS[name],name=name) and witness['physics_steps']==0
        paths={'reset.npz':z['reset_sha256'],'factor-state.npz':witness['arrays_sha256'],
               'factor-witness.json':sha(directory/'factor-witness.json'),'proof.json':p['zero_sha256']}
        for filename,digest in paths.items():
            assert sha(directory/filename)==digest
            files[name+'/'+filename]=digest
        with (np.load(directory/'reset.npz',allow_pickle=False) as start,
              np.load(directory/'factor-state.npz',allow_pickle=False) as factors):
            inputs[name] = dict(start)|{key:factors[key] for key in factors.files if key!='configured_voltage'}
        if name!=NOMINAL:
            assert inputs[name].keys()==inputs[NOMINAL].keys()
            for key,value in inputs[name].items():
                np.testing.assert_array_equal(value,inputs[NOMINAL][key],err_msg=f'{name}/{key}')
    receipt=REPORT/'zero-comparison.json'
    write(receipt,dict(status='verified-matched-cuda-zero',protocol_sha256=sha(PLAN),seed=SEED,
        same_reset_qpos_qvel_motor_and_actor_observation=True,same_compiled_and_runtime_physics_arrays=True,
        files=files,physical_steps=0,hardware_io=False))
    node.update(status='sample-ready',zero=rel(receipt),zero_sha256=sha(receipt))
    project['staged_training_request'].update(current_evidence=rel(receipt),
        next_gate='Run the three frozen full routes and verify the measured delay and supply under load.')
    write(PROJECT,project)
    print('NOMINAL_VOLTAGE_DELAY_ZERO_VERIFIED',sha(receipt),flush=True)


def compare_process_zero(folder, previous):
    for file in ('reset.npz','factor-state.npz'):
        with (np.load(folder/file,allow_pickle=False) as x,np.load(previous/file,allow_pickle=False) as y):
            assert x.files==y.files
            for field in x.files:
                np.testing.assert_array_equal(x[field],y[field],err_msg=field)
    assert read(folder/'factor-witness.json')['factor']==read(previous/'factor-witness.json')['factor']


def verify_motor(plan,raw,folder,zero_path):
    name = plan['named_factor']['name']
    spec = ARMS[name]
    literal = read(zero_path/'factor-witness.json')
    assert literal['factor']==dict(spec,name=name)
    lag = spec['actuator_delay_ticks']
    start,stop = plan['solver_comparison']['fine_solver_interval']
    archive = folder/'motor-boundary.npz'
    assert sha(archive)==raw['motor_boundary_sha256']
    with np.load(archive,allow_pickle=False) as data:
        rows = dict(data)
    with np.load(zero_path/'factor-state.npz',allow_pickle=False) as data:
        nominal,gain = data['configured_voltage'],data['vin_drop_gain']
    np.testing.assert_array_equal(rows['control'],np.repeat(np.arange(start-1,start+3),5))
    np.testing.assert_array_equal(rows['motor_interval'],np.arange(5*(start-1)+1,5*(start+3)+1))
    np.testing.assert_array_equal(rows['delayed'][lag:],rows['requested'][:-lag])
    assert (rows['requested'][5]!=rows['requested'][4]).any()
    np.testing.assert_array_equal(rows['delayed'][5+lag],rows['requested'][5])
    previous = rows['previous_motor']
    voltage = rows['effective_supply_v']
    expected = (nominal-gain*np.abs(previous).sum(axis=-1,keepdims=True)).clip(min=literal['vin_floor'])
    np.testing.assert_allclose(voltage,expected,atol=1e-6,rtol=0.)
    measured = raw['actuator_clock']
    assert measured['motor_ticks']==5*raw['controls']==raw['physics_substeps']
    assert measured['solver_launches']==raw['solver_launches']==5*(raw['controls']+min(raw['controls'],stop)-start)
    changes,end_time = reconstruct_clock(0,0,raw['controls'],{start:2,stop:1})
    expected_changes = [dict(control=x['control'],factor=x['new_factor'],time_s=x['engine_clock_s']) for x in changes]
    assert measured['changes']==expected_changes and measured['end_time_s']==end_time
    assert [x['control'] for x in measured['controls']]==list(range(start,start+3))
    for row in measured['controls']:
        np.testing.assert_allclose(row['elapsed_solver_s_world_0'],[[.002,.004]]*5,atol=2e-6,rtol=0)
        assert abs(row['actor_elapsed_s_world_0']-.02)<=1e-5
    first, = (x for x in raw['boundaries'] if x['control']==start)
    rp = folder/first['file']
    assert sha(rp)==first['sha256']
    with np.load(rp,allow_pickle=False) as data:
        np.testing.assert_array_equal(rows['motor_torque_request_nm'][5:10],data['motor'][:,:,literal['channels']['motor_ids']])
    return dict(status='verified-motor-delay-and-supply',observed_fixed_lag_ticks=lag,observed_delay_s=.004*lag,
        voltage_reset_v=spec['supply_v'],effective_voltage_min_v=float(voltage.min()),effective_voltage_median_v=float(np.median(voltage)),
        effective_voltage_max_v=float(voltage.max()),maximum_supply_model_error_v=float(abs(voltage-expected).max()),
        first_requested_roll_control=start,motor_intervals_audited=len(voltage),worlds=plan['composition']['worlds'],
        two_solver_steps_share_same_motor_input=True,record_sha256=raw['motor_boundary_sha256'],
        nominal_roll_clock_preserved=True,solver_launches=raw['solver_launches'],
        gpu_peak_mib=max(x['used_mib'] for x in raw['gpu_samples']))


def sample(name):
    _,node,_ = load_master()
    assert node['status']=='sample-ready'
    receipt = read(ROOT/node['zero'],node['zero_sha256'])
    assert receipt['status']=='verified-matched-cuda-zero'
    for path,digest in receipt['files'].items():
        group,file = path.split('/',1)
        assert sha(REPORT/group/'zero'/file)==digest
    with arm_context(name,'sample-ready'):
        parent.sample()


def score(name):
    with arm_context(name,'score-pending') as (node,plan):
        folder = ROOT/plan['record']
        raw = read(folder/'collection.json',node['collection_sha256'])
        assert raw['actor_weights_unchanged']
        assert not any(raw[k] for k in ('learning_updates','hardware_io','reserved_seed_used'))
        assert raw['controls']>=603,raw.get('reason')
        compare_process_zero(folder/'zero',REPORT/name/'zero')
        audit = verify_motor(plan,raw,folder,REPORT/name/'zero')
        verifier.score()
    project,_,_ = load_master()
    out=REPORT/name/'actuator-audit.json'
    write(out,audit)
    project['goal_execution'][arm_key(name)].update(actuator_audit=rel(out),actuator_audit_sha256=sha(out))
    write(PROJECT,project)
    print('NOMINAL_VOLTAGE_DELAY_ARM_VERIFIED',name,sha(out),audit['observed_delay_s'],audit['effective_voltage_median_v'],flush=True)


def consolidate():
    project,node,master = load_master()
    assert node['status']=='sample-ready'
    rows = {}
    total,launches = 0,0
    for name in ARMS:
        item = project['goal_execution'][arm_key(name)]
        assert item['status']=='reviewed'
        scorecard = read(ROOT/item['result'],item['result_sha256'])
        witness = read(ROOT/item['actuator_audit'],item['actuator_audit_sha256'])
        assert witness['status']=='verified-motor-delay-and-supply'
        assert witness['observed_fixed_lag_ticks']==ARMS[name]['actuator_delay_ticks']
        rows[name] = dict(score=scorecard,motor=witness,result=rel(ROOT/item['result']),result_sha256=item['result_sha256'],
            run_id=item['run_id'])
        total += scorecard['transitions']
        launches += witness['solver_launches']
    assert total<=master['maximum_total_control_transitions'] and launches<=master['maximum_total_solver_launches']
    nominal = rows[NOMINAL]['score']['complete']
    paired_count_difference = {name:row['score']['complete']-nominal if nominal is not None and row['score']['complete'] is not None else None
                               for name,row in rows.items() if name!=NOMINAL}
    complete = all(row['score']['passed_development_gate'] for row in rows.values())
    path = REPORT/'analysis.json'
    assert not path.exists()
    write(path,dict(status='verified-independent-supply-and-actuation-delay',protocol_sha256=sha(PLAN),seed=SEED,
        worlds=master['worlds'],total_transitions=total,solver_launches=launches,arms=rows,
        complete_count_difference_from_nominal=paired_count_difference,all_original_gates_passed=complete,
        no_in_episode_reset=True,parameter_changes_are_independent=True,learning_updates=0,hardware_io=False,
        low_friction_reserved_seed_used=False))
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(path))
    node.update(status='reviewed',result=rel(path),result_sha256=sha(path),passed=complete)
    project.update(info)
    project['staged_training_request'].update(info,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Preserve tested voltage and delay endpoints and continue the independent low-friction full-route gate.'
        if complete else 'Localize the failed factor by original phase. Use the early support timing for the separate low-friction route.')
    write(PROJECT,project)
    print('NOMINAL_VOLTAGE_DELAY_VERIFIED',sha(path),
          {name:(row['score']['roll_complete'],row['score']['complete']) for name,row in rows.items()},complete,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','zero','sample','score','consolidate'))
    cli.add_argument('arm',nargs='?',choices=tuple(ARMS))
    args=cli.parse_args()
    assert (args.arm is not None)==(args.action in ('sample','score'))
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.arm:
            {'sample':sample,'score':score}[args.action](args.arm)
        else:
            {'freeze':freeze,'zero':zero,'consolidate':consolidate}[args.action]()
