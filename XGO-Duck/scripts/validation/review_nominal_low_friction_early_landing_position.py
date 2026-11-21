"""Select a causal early landing trigger and verify an attainable servo-position bias."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

import review_nominal_low_friction_right_foot_landing_authority as landing
import review_nominal_low_friction_right_support_signal as previous
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY = 'nominal_low_friction_early_landing_position'
STAGE = 'nominal-low-friction-early-landing-position'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-early-transfer-authority'
OUTPUT = REPORT/'servo-position.json'


def freeze():
    project=read(PROJECT)
    permitted(project,landing.STAGE)
    goal=project['goal_execution']
    foot=goal[landing.KEY]
    matched=read(ROOT/foot['result'],foot['result_sha256'])
    assert foot['status']=='reviewed' and matched['selected_profile']=='left_hip_roll/-1'
    p=read(previous.PLAN)
    sources={rel(previous.PLAN):sha(previous.PLAN),foot['protocol']:foot['protocol_sha256'],foot['result']:foot['result_sha256']}
    episodes=[]
    for row in p['sources']:
        if row['branches']==[None]:
            continue
        for field in ('protocol','collection'):
            sources[row[field]['path']]=row[field]['sha256']
        episodes.append(row)
    for name in ('review_roll_lateral_pitch_authority.py','review_nominal_low_friction_right_support_signal.py'):
        source=Path(__file__).with_name(name)
        sources[rel(source)]=sha(source)
    assert not PLAN.exists() and not OUTPUT.exists() and KEY not in goal
    write(PLAN,dict(stage=STAGE,sources=sources,script_sha256=sha(Path(__file__)),episodes=episodes,
        release_control=100,decision_control=120,reference_control=116,entry_control=600,actuator='left_hip_roll',sign=-1,
        target_decrement_rad=[.005,.01,.015,.02,.025,.03,.04],minimum_signed_torque_nm=.012,maximum_signed_torque_nm=.06,
        minimum_aligned_miss_fraction=.8,voltages_v=[7.,7.7],
        focus_scene_ids=sorted(int(w) for w in matched['observed_foot_loading']),
        minimum_focused_miss_fraction=.75,minimum_focused_miss_in_archive=2,maximum_admitted_exposure_fraction=.05,
        pitch_lower_deg=[42.,46.,50.,54.,56.,58.,60.,62.],pitch_upper_deg=[70.,76.,82.,88.,94.],
        sagittal_progress_upper_deg=[0.,4.,8.,12.,16.,20.,24.,180.],absolute_roll_upper_deg=[12.,18.,24.,32.,45.,90.],
        right_foot_force_used_as_policy_input=False,additional_physical_steps=0,learning_updates=0,hardware_io=False,reserved_seed=2246))
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify a single early trigger and a delayed position command using original actor inputs.')
    write(PROJECT,project)
    print('EARLY_LANDING_POSITION_FROZEN',sha(PLAN),flush=True)


def features(now,earlier):
    def pitch(gravity):
        return np.rad2deg(np.arctan2(gravity[:,0],-gravity[:,2]))

    angle=pitch(now)
    absolute_roll=np.rad2deg(np.arctan2(abs(now[:,1]),np.hypot(now[:,0],now[:,2])))
    return angle,pitch(earlier)-angle,absolute_roll


def position_trigger(now,earlier,active,rule):
    angle,progress,roll=features(now,earlier)
    return (active & (angle>=rule['pitch_min_deg']) & (angle<=rule['pitch_max_deg'])
        & (progress<=rule['progress_max_deg']) & (roll<=rule['roll_max_deg']))


def archive_sequences(spec):
    k,earlier,start=(spec[name] for name in ('decision_control','reference_control','release_control'))
    archives={}
    for readout in spec['episodes']:
        protocol,entry=(readout[key] for key in ('protocol','collection'))
        plan,record=(read(ROOT/item['path'],item['sha256']) for item in (protocol,entry))
        assert (plan['environment']['seed'],plan['composition']['foot_friction'])==(1257,.8)
        assert record['prefix']['controls']==start and record['actor_weights_unchanged'] and not record['reserved_seed_used']
        folder=(ROOT/entry['path']).parent
        routed_path=folder/'release-routing.npz'
        assert sha(routed_path)==record['release_routing_sha256']
        with np.load(routed_path,allow_pickle=False) as z:
            latched=z['latched_supine']&~z['right']
        col=plan['walking']['joint_names'].index(spec['actuator'])
        for role in readout['branches']:
            branch=record['branches'][role]
            path=folder/role
            states=previous.checkpoint(path,branch['chunks'],k,('qpos','qvel','active'))
            terminal=previous.checkpoint(path,branch['chunks'],spec['entry_control'],('roll_phase',))
            stand=path/'pre-roll-standing.npz'
            assert sha(stand)==branch['standing_sha256']
            with np.load(stand,allow_pickle=False) as z:
                actor=z['actor_input']
                now,old=(actor[control-start,:,3:6].copy() for control in (k,earlier))
                assert (z['mode'][k-start,latched]==0).all() and np.isfinite(now).all() and np.isfinite(old).all()
                targets=z['actual_target'][:k-start+1,:,col:col+1].astype(np.float64)
            # A target written at G100 belongs to the segment that starts at
            # that state, before the next integration step.
            mark,=(x for x in branch['boundaries'] if x['control']==start)
            bound=path/mark['file']
            assert sha(bound)==mark['sha256']
            with np.load(bound,allow_pickle=False) as z:
                initial=z['previous_target'][:,col:col+1].astype(np.float64)
            live=latched & states['active']
            admitted=terminal['roll_phase']==1
            focus=np.isin(np.arange(len(live)),spec['focus_scene_ids'])&live&~admitted
            assert focus.sum()>=spec['minimum_focused_miss_in_archive']
            archives[f'{readout["key"]}/{role}']=dict(plan=plan,now=now,old=old,active=live,missed=live&~admitted,
                admitted=live&admitted,focus=focus,position=states['qpos'],velocity=states['qvel'],targets=targets,previous=initial)
    return archives


def pick_trigger(archives,spec):
    candidate_keys=('pitch_lower_deg','pitch_upper_deg','sagittal_progress_upper_deg','absolute_roll_upper_deg')
    rule_keys=('pitch_min_deg','pitch_max_deg','progress_max_deg','roll_max_deg')
    ranking=[]
    best_infeasible=None
    for values in product(*(spec[key] for key in candidate_keys)):
        rule=dict(zip(rule_keys,values,strict=True))
        groups={}
        for name,data in archives.items():
            mask=position_trigger(data['now'],data['old'],data['active'],rule)
            counts={f'{label}_{prefix}':int(bits.sum()) for label in ('focus','missed','admitted')
                    for prefix,bits in (('total',data[label]),('triggered',mask&data[label]))}
            groups[name]=counts
        rows=list(groups.values())
        cover=min(row['focus_triggered']/row['focus_total'] for row in rows)
        exposure=max(row['admitted_triggered']/row['admitted_total'] for row in rows)
        rank=(-cover,exposure,-sum(row['missed_triggered'] for row in rows),values)
        if exposure<=spec['maximum_admitted_exposure_fraction']:
            if best_infeasible is None or rank<best_infeasible[0]:
                best_infeasible=(rank,rule,groups)
            if cover>=spec['minimum_focused_miss_fraction'] and all(row['focus_triggered']>=spec['minimum_focused_miss_in_archive'] for row in rows):
                ranking.append((rank,rule,groups))
    best=min(ranking,key=lambda item:item[0]) if ranking else None
    return best,best_infeasible,len(ranking)


def actuator_authority(archives,rule,spec):
    import torch
    from mjlab.scene import Scene
    from mjlab.utils.buffers import DelayBuffer
    from review_roll_lateral_pitch_authority import torque
    from run_nominal_skill_composition import config

    sample=next(iter(archives.values()))['plan']
    cfg=config(dict(sample,composition={**sample['composition'],'foot_friction':1.}))
    scene=Scene(cfg.scene,device='cpu')
    plant=scene.compile()
    wrapper,=scene['robot'].actuators
    motor=wrapper._bam_model.actuator
    servo,=cfg.scene.entities['robot'].articulation.actuators
    assert type(motor).__name__=='STS3215Actuator' and motor.max_current is None
    assert (servo.delay_min_lag,servo.delay_max_lag,cfg.sim.mujoco.timestep,cfg.decimation)==(1,1,.004,5)
    assert spec['voltages_v']==[servo.vin_min,servo.vin_range[0]]
    joint=plant.joint('robot/'+spec['actuator'])
    qaddr,daddr=joint.qposadr.item(),joint.dofadr.item()
    actuator=np.flatnonzero(plant.actuator_trnid[:,0]==joint.id).item()
    assert plant.actuator_gainprm[actuator,0]==plant.actuator_gear[actuator,0]==1.
    lo,hi=joint.range
    taumax=plant.actuator_forcerange[actuator]
    tick=cfg.sim.mujoco.timestep
    max_move=motor.model.max_velocity.value*tick
    checks={}
    for name,row in archives.items():
        eligible=position_trigger(row['now'],row['old'],row['active'],rule)
        assert eligible.any() and (eligible&row['missed']).any()
        tgt,previous=row['targets'][:,eligible],row['previous'][eligible]
        assert (tgt>=lo-1e-7).all() and (tgt<=hi+1e-7).all()
        delay=DelayBuffer(min_lag=1,max_lag=1,batch_size=int(eligible.sum()),device='cpu')
        delay.append(torch.from_numpy(previous))
        delay.append(torch.from_numpy(previous))
        reach=np.stack((np.full_like(previous,lo),np.full_like(previous,hi)))
        for target in tgt[:-1]:
            for i in range(5):
                delay.append(torch.from_numpy(target))
                low=delay.compute().numpy()
                np.testing.assert_array_equal(low,previous if i==0 else target)
                reach=np.clip(low,reach-max_move,reach+max_move)
            previous=target
        old=tgt[-1]
        delay.append(torch.from_numpy(old))
        np.testing.assert_array_equal(delay.compute().numpy(),previous)
        reach=np.clip(previous,reach-max_move,reach+max_move)
        assert np.max(abs(reach[0]-reach[1]))<=1e-10
        delay.append(torch.from_numpy(old))
        np.testing.assert_array_equal(delay.compute().numpy(),old)
        q=row['position'][eligible,qaddr,None].astype(np.float64)
        dq=row['velocity'][eligible,daddr,None].astype(np.float64)
        choices={}
        for size in spec['target_decrement_rad']:
            new=np.clip(old+spec['sign']*size,lo,hi)
            deltas=[]
            for supply in spec['voltages_v']:
                base,_=torque(motor,supply,reach[0],old,q,dq,tick)
                modified,_=torque(motor,supply,reach[0],new,q,dq,tick)
                delta=(np.clip(modified,*taumax)-np.clip(base,*taumax))*spec['sign']
                assert (delta>=-1e-12).all()
                deltas.append(delta.ravel())
            min_delta=np.minimum(*deltas)
            max_delta=np.maximum(*deltas)
            miss=row['missed'][eligible]
            frac=float((min_delta[miss]>=spec['minimum_signed_torque_nm']).mean())
            choices[str(size)]=dict(minimum_missed_fraction=frac,maximum_signed_torque_nm=float(max_delta.max()),
                count_missed=int(miss.sum()),deltas_at_voltage_nm=[v.tolist() for v in deltas])
        checks[name]=dict(worlds=np.flatnonzero(eligible).tolist(),figures=choices)
    assert not torch.cuda.is_initialized()
    options=[size for size in spec['target_decrement_rad'] if all(
        row['figures'][str(size)]['minimum_missed_fraction']>=spec['minimum_aligned_miss_fraction']
        and row['figures'][str(size)]['maximum_signed_torque_nm']<=spec['maximum_signed_torque_nm'] for row in checks.values())]
    return options[0] if options else None,checks


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,digest in spec['sources'].items():
        assert sha(ROOT/path)==digest,path
    archives=archive_sequences(spec)
    best,closest,count=pick_trigger(archives,spec)
    selected,responses=actuator_authority(archives,best[1],spec) if best else (None,{})
    result=dict(status='verified-causal-early-landing-position',protocol_sha256=sha(PLAN),eligible_trigger_count=count,
        rule=best[1] if best else None,rule_scores=best[2] if best else None,
        best_protection_only={'rule':closest[1],'scores':closest[2]} if closest else None,
        actuator=spec['actuator'],recommended_decrement_rad=selected,responses=responses,
        additional_physical_steps=0,learning_updates=0,holdout_used=False,hardware_io=False)
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),decrement_rad=selected,
            report=rel(REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Run the frozen 896-world paired early landing trial with matching 4 ms servo inputs and original get-up/roll-entry scoring.'
            if selected else 'Use the causal early trigger or the servo response witness to refine and validate the observable closed-loop input.'))
        write(PROJECT,project)
    print('EARLY_LANDING_POSITION_VERIFIED',sha(OUTPUT),count,result['rule'],selected,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    option=cli.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
