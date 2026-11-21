"""Verify that bounded early position offsets reach the correct motor directions."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_right_support_signal import NAME as SIGNAL, checkpoint, mask


KEY = 'nominal_low_friction_right_support_position'
STAGE = 'nominal-low-friction-right-support-position'
PRIOR = 'nominal-low-friction-right-support-signal'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-right-support-position'
JOINTS = ('right_hip_pitch','right_knee')
AMOUNTS = (.02,.04,.06)


def freeze():
    project = read(PROJECT)
    permitted(project,PRIOR)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert KEY not in goal and not (PLAN.exists() or REPORT.exists())
    prior = goal[SIGNAL]
    assert prior['status']=='reviewed' and prior['actor_input_gate_passed'] and prior['geometry_replay_passed']
    previous = read(ROOT/prior['result'],prior['result_sha256'])
    signal = read(ROOT/prior['protocol'],prior['protocol_sha256'])
    assert previous['protocol_sha256']==prior['protocol_sha256'] and signal['control']==136
    assert previous['all_archived_actor_observations_passed'] and previous['independent_nominal_geometry_passed']
    source = {prior[k]:prior[k+'_sha256'] for k in ('protocol','result')}
    for item in signal['sources']:
        if not all(branch is not None for branch in item['branches']):
            continue
        for name in ('protocol','collection'):
            row = item[name]
            source[row['path']] = row['sha256']
    helper = ROOT/'scripts/validation/review_roll_lateral_pitch_authority.py'
    source[rel(helper)] = sha(helper)
    for file,digest in source.items():
        assert sha(ROOT/file)==digest
    plan = dict(status='frozen-fixed-state-position-authority',stage=STAGE,signal_plan=prior['protocol'],
        signal_result=prior['result'],sources_sha256=source,control=136,seed=1257,joints=JOINTS,
        minimum_torque_change_nm=.005,minimum_triggered_missed_coverage=.8,maximum_joint_model_headroom_rad=.06,
        candidate_magnitudes_rad=AMOUNTS,signs=(-1,1),voltage_v=[7.,7.7],physics_period_s=.004,
        fixed_archived_joint_kinematics=True,real_delay_buffer=True,integrations=0,learning_updates=0,hardware_io=False,
        script_sha256=sha(Path(__file__)),low_friction_reserved_seed=2246)
    REPORT.mkdir()
    write(PLAN,plan)
    goal[KEY] = dict(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Replay the real one-substep motor delay and bound delivered right hip and knee torque at both supply endpoints.')
    write(PROJECT,project)
    print('RIGHT_SUPPORT_POSITION_FROZEN',sha(PLAN),flush=True)


def evaluate(spec):
    import torch
    from mjlab.scene import Scene
    from mjlab.utils.buffers import DelayBuffer

    from review_roll_lateral_pitch_authority import torque
    from run_nominal_skill_composition import config

    assert not torch.cuda.is_initialized()
    signal = read(ROOT/spec['signal_plan'])
    observed = read(ROOT/spec['signal_result'])
    collected = [item for item in signal['sources'] if item['branches'][0] is not None]
    plan = read(ROOT/collected[0]['protocol']['path'],collected[0]['protocol']['sha256'])
    cfg = config(dict(plan,composition={**plan['composition'],'foot_friction':1.}))
    assert cfg.sim.mujoco.timestep==spec['physics_period_s']==.004 and cfg.decimation==5
    scene = Scene(cfg.scene,device='cpu')
    plant = scene.compile()
    wrapper, = scene['robot'].actuators
    motor = wrapper._bam_model.actuator
    assert type(motor).__name__=='STS3215Actuator' and motor.max_current is None
    servo, = cfg.scene.entities['robot'].articulation.actuators
    assert (servo.delay_min_lag,servo.delay_max_lag)==(1,1)
    assert spec['voltage_v']==[servo.vin_min,servo.vin_range[0]]
    tick = cfg.sim.mujoco.timestep
    max_change = motor.model.max_velocity.value*tick
    ids = np.array([plant.joint('robot/'+name).id for name in spec['joints']])
    ii = plant.jnt_qposadr[ids]
    vv = plant.jnt_dofadr[ids]
    acts = [np.flatnonzero(plant.actuator_trnid[:,0]==i).item() for i in ids]
    assert np.all(plant.actuator_gear[acts,0]==1.)
    force = plant.actuator_forcerange[acts]
    pos_min,pos_max = plant.jnt_range[ids].T
    direction = np.array(spec['signs'])
    results = {}
    for item in collected:
        info,entry = item['protocol'],item['collection']
        current,raw = read(ROOT/info['path'],info['sha256']),read(ROOT/entry['path'],entry['sha256'])
        assert current['environment']['seed']==spec['seed'] and current['composition']['foot_friction']==.8
        folder = (ROOT/entry['path']).parent
        k = spec['control']
        columns = [current['walking']['joint_names'].index(name) for name in spec['joints']]
        np.testing.assert_equal(np.asarray(current['environment']['joint_limits'])[columns],plant.jnt_range[ids])
        with np.load(folder/'release-routing.npz',allow_pickle=False) as routed:
            assert sha(folder/'release-routing.npz')==raw['release_routing_sha256']
            latched = routed['latched_supine']&~routed['right']
        for role in item['branches']:
            record,storage = raw['branches'][role],folder/role
            states = checkpoint(storage,record['chunks'],k,('qpos','qvel','active'))
            final = checkpoint(storage,record['chunks'],signal['entry_control'],('roll_phase',))['roll_phase']
            start = signal['release_control']
            mark, = (x for x in record['boundaries'] if x['control']==start)
            assert sha(storage/mark['file'])==mark['sha256']
            with np.load(storage/mark['file'],allow_pickle=False) as saved:
                previous = saved['previous_target'][:,columns].astype(np.float64)
            file = storage/'pre-roll-standing.npz'
            assert sha(file)==record['standing_sha256']
            with np.load(file,allow_pickle=False) as data:
                step = k-start
                trip = mask(data['actor_input'][step,:,3:6],latched&states['active'],signal)
                targets = data['actual_target'][:step+1,:,columns]
            names = np.flatnonzero(trip)
            expected = observed['results'][item['key']][role]['actor_observation']
            missed = final[names]!=1
            assert int(missed.sum())==expected['missed_triggered'] and int((~missed).sum())==expected['passed_triggered']
            previous = previous[names]
            targets = targets[:,names].astype(np.float64)
            assert (targets>=pos_min-1e-7).all() and (targets<=pos_max+1e-7).all()
            assert (previous>=pos_min-1e-7).all() and (previous<=pos_max+1e-7).all()
            before = np.stack((np.tile(pos_min,(len(names),1)),np.tile(pos_max,(len(names),1))))
            delay = DelayBuffer(min_lag=1,max_lag=1,batch_size=len(names),device='cpu')
            delay.append(torch.from_numpy(previous))
            delay.append(torch.from_numpy(previous))
            for target in targets[:-1]:
                for j in range(5):
                    delay.append(torch.from_numpy(target))
                    applied = delay.compute().numpy()
                    np.testing.assert_array_equal(applied,previous if j==0 else target)
                    before = np.clip(applied,before-max_change,before+max_change)
                previous = target
            delay.append(torch.from_numpy(targets[-1]))
            common_delayed = delay.compute().numpy()
            np.testing.assert_array_equal(common_delayed,previous)
            before = np.clip(common_delayed,before-max_change,before+max_change)
            width = float(abs(before[0]-before[1]).max())
            assert width<=1e-10
            delay.append(torch.from_numpy(targets[-1]))
            np.testing.assert_array_equal(delay.compute().numpy(),targets[-1])
            q = states['qpos'][np.ix_(names,ii)].astype(np.float64)
            dq = states['qvel'][np.ix_(names,vv)].astype(np.float64)
            raw_target = targets[-1]
            outputs = {}
            for amount in spec['candidate_magnitudes_rad']:
                alternative = np.clip(raw_target+amount*direction,pos_min,pos_max)
                assert (abs(alternative-raw_target)<=spec['maximum_joint_model_headroom_rad']+1e-7).all()
                per_voltage,meets = {},np.ones(len(names),bool)
                for voltage in spec['voltage_v']:
                    old,_ = torque(motor,voltage,before[0],raw_target,q,dq,tick)
                    new,_ = torque(motor,voltage,before[0],alternative,q,dq,tick)
                    signed = (np.clip(new,*force.T)-np.clip(old,*force.T))*direction
                    assert (signed>=-1e-12).all()
                    meets &= (signed>=spec['minimum_torque_change_nm']).all(axis=1)
                    per_voltage[str(voltage)] = signed.tolist()
                outputs[str(amount)] = dict(both_measured_joint_signs_meet_in_missed=int((meets&missed).sum()),
                    both_measured_joint_signs_meet_in_passed=int((meets&~missed).sum()),signed_torque_changes_nm=per_voltage,
                    maximum_projected_target_change_rad=float(abs(alternative-raw_target).max()))
            results[role] = dict(source=item['key'],scenario_ids=names.tolist(),missed_at_g600=missed.tolist(),
                triggered_missed_count=int(missed.sum()),triggered_admitted_count=int((~missed).sum()),
                internal_goal_bracket_at_effective_tick_rad=width,outputs=outputs)
    admissible = [amount for amount in spec['candidate_magnitudes_rad']
        if all(row['outputs'][str(amount)]['both_measured_joint_signs_meet_in_missed']/row['triggered_missed_count']
               >=spec['minimum_triggered_missed_coverage'] for row in results.values())]
    assert not torch.cuda.is_initialized()
    return dict(status='verified-archived-fixed-state-position-authority',protocol_sha256=sha(PLAN),
        compared_at_prior_q_dq=True,same_initial_motor_tick=True,checked_joint_order=list(JOINTS),
        maximum_firmware_goal_change_per_motor_tick_rad=max_change,results=results,
        recommended_position_offset_rad=min(admissible) if admissible else None,
        steps=0,learning_updates=0,hardware_io=False,reserved_seed_used=False)


def score(check=False):
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'zero-ready')
    if not check:
        permitted(project,STAGE)
    spec = read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,digest in spec['sources_sha256'].items():
        assert sha(ROOT/path)==digest
    result = evaluate(spec)
    out = REPORT/'analysis.json'
    if check:
        assert read(out,node['result_sha256'])==result
    else:
        assert not out.exists()
        write(out,result)
        preferred = result['recommended_position_offset_rad']
        node.update(status='reviewed',result=rel(out),result_sha256=sha(out),position_offset_rad=preferred)
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(out))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=
            'Freeze a development matched-release position residual and verify actual servo input after delay and the original action gates.'
            if preferred is not None else 'Use the fixed-state voltage response and joint-specific saturation to choose a different action timing.')
        write(PROJECT,project)
    print('RIGHT_SUPPORT_POSITION_VERIFIED',sha(out),result['recommended_position_offset_rad'],flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    args = cli.parse_args()
    freeze() if args.action=='freeze' else score(args.action=='check')
