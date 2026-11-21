"""Compare observed-tilt leg recovery from a shared pre-trigger CUDA snapshot."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import preflight_nominal_low_friction_upright_target as geometry
from getup_stand_handoff import NOISE
from nominal_skill_composition_controller import GETUP
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_seven_command_score import recovery_and_standing
import run_nominal_low_friction_roll_window as window
from verify_nominal_skill_composition import PHYSICAL, STAND, check_video, read_trace


ROOT, PROJECT = window.ROOT, window.PROJECT
read, rel, sha, write = window.read, window.rel, window.sha, window.write
base, driver, parent, recorder = window.base, window.driver, window.parent, window.recorder
KEY, STAGE = 'nominal_low_friction_upright_recovery', 'nominal-low-friction-upright-recovery'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = geometry.REPORT.with_name('low-friction-upright-recovery')
ROLES = ('original-first', 'original-repeat', 'observed-leg-recovery')
LABELS = dict(zip(('shared',*ROLES),('共同跌倒、释放与起身','原策略参照一','原策略参照二','持续前倾后的腿部恢复'),strict=True))


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert parent.allowed(project,geometry.STAGE) and not task['simulation_running']
    assert not PLAN.exists() and not REPORT.exists() and KEY not in goal and not torch.cuda.is_initialized()
    upstream, pose = (goal[k] for k in (window.KEY,geometry.KEY))
    assert upstream['status']==pose['status']=='reviewed'
    plan = read(ROOT/upstream['protocol'],upstream['protocol_sha256'])
    evidence = read(ROOT/pose['evidence'],pose['evidence_sha256'])
    limits = read(ROOT/pose['protocol'],pose['protocol_sha256'])
    assert evidence['physical_steps']==0 and not evidence['reserved_seed_used']
    d = plan['composition']
    assert (d['foot_friction'],d['existing_low_friction_reserved_seed'],plan['environment']['seed'],d['worlds'])==(.8,2246,1257,896)
    assert plan['environment']['common_actor_noise']==NOISE
    detector = limits['detector']
    fork,end = detector['controls'][0],d['roll_start_control']
    assert (fork,end,d['getup_to_stand_control'])==(250,600,550) and detector['successive_controls']==25
    templates, joints = [], None
    for cohort,episode in evidence['batches'].items():
        for row in episode:
            chosen = row['targets']['12.0']['legs-only']
            if not chosen['feasible']:
                continue
            if joints is None:
                joints = chosen['joint_names']
            assert chosen['joint_names']==joints
            q = np.asarray(chosen['joint_target_rad'])
            templates.append(dict(source_cohort=cohort,source_world=row['world'],start=(q-chosen['joint_change_rad']).tolist(),
                                  target=q.tolist()))
    assert len(templates)==25 and len(joints)==10
    order = [plan['walking']['joint_names'].index(name) for name in joints]
    lower,upper = np.asarray(plan['environment']['joint_limits'])[order].T
    assert all((np.asarray(x['target'])>=lower-1e-9).all() and (np.asarray(x['target'])<=upper+1e-9).all()
               for x in templates)
    saved = ROOT/plan['standing_pair']['archived_initial']
    assert sha(saved.with_name('reset.npz'))==read(saved)['reset_sha256']
    source_paths = (Path(__file__).resolve(),ROOT/upstream['protocol'],ROOT/upstream['observed_tilt_readback'],
                    ROOT/pose['protocol'],ROOT/pose['evidence'])
    plan['references'].update({rel(path):sha(path) for path in source_paths})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-tilt-leg-friction080-s1257-{stamp}'
    controls = fork+len(ROLES)*(end-fork)
    plan['environment']['run_id'] = run
    plan['standing_pair'].update(common_snapshot_control=fork,roles=list(ROLES))
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=controls,
        maximum_control_transitions=controls*d['worlds'],maximum_physics_substeps=5*controls,
        upright_recovery=dict(detector=detector,joints=joints,joint_indices=order,templates=templates,end_control=end,
            max_template_error_rad=.06,max_target_from_observed_rad=limits['joint_delta_cap_rad'],
            max_slew_rad_per_control=.04,abort_min_body_up=.45,abort_max_gyro_rad_s=5.,
            target_end_control=d['getup_to_stand_control'],return_end_control=end-1,
            min_left_gain_against_each_reference=2,reserved_seed=2246))
    REPORT.mkdir()
    write(PLAN,plan)
    proof = REPORT/'preflight.json'
    write(proof,dict(protocol_sha256=sha(PLAN),geometry_sha256=pose['evidence_sha256'],templates=len(templates),
        observer=observer,matched_snapshot_control=fork,maximum_control_transitions=controls*d['worlds'],
        physical_steps=0,learning_updates=0,hardware_io=False))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,
                     preflight=rel(proof),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(proof))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Verify observed triggers, bounded applied targets and the original standing gates.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_RECOVERY_FROZEN',sha(PLAN),run,flush=True)


class Recovery:
    def __init__(self, env, plan):
        self.design = plan['upright_recovery']
        self.worlds,self.device = env.num_envs,env.device
        def tensor(value):
            return torch.as_tensor(value,device=self.device,dtype=torch.float32)
        self.cols = self.design['joint_indices']
        self.home = tensor(plan['environment']['model_home'])[self.cols]
        self.lower,self.upper = tensor(plan['environment']['joint_limits'])[self.cols].T
        self.template_start = tensor([row['start'] for row in self.design['templates']])
        self.template_target = tensor([row['target'] for row in self.design['templates']])
        self.pitch = tensor(np.deg2rad(self.design['detector']['signed_pitch_degrees']))
        self.bounds = self.design['detector']['controls']

    def reset(self, role):
        self.role = role
        self.dwell = torch.zeros(self.worlds,dtype=torch.int32,device=self.device)
        self.trigger = torch.zeros(self.worlds,dtype=torch.bool,device=self.device)
        self.active = self.trigger.clone()
        self.returning = self.trigger.clone()
        self.targets = torch.zeros(self.worlds,len(self.cols),dtype=torch.float32,device=self.device)
        self.events = []
        self.rows = []

    def apply(self, term, previous, control, action):
        if control<self.bounds[0] or control>self.design['return_end_control']:
            return
        observed = action.actor_input
        g = observed[:,3:6]
        speed = observed[:,:3].norm(dim=-1)
        pitch = torch.atan2(g[:,0],g[:,1:].norm(dim=-1))
        getup = torch.as_tensor(action.mode==GETUP,device=self.device)
        if control<=self.bounds[1]:
            good = (pitch>=self.pitch[0]) & (pitch<=self.pitch[1]) & (
                speed<=self.design['detector']['maximum_angular_speed_observation_norm']) & getup
            self.dwell = torch.where(good,self.dwell+1,0)
            worlds = torch.nonzero((self.dwell>=self.design['detector']['successive_controls'])&~self.trigger).flatten()
            if worlds.numel():
                self.trigger[worlds] = True
                measured = observed[worlds,6:20][:,self.cols]+self.home
                distances = (measured[:,None]-self.template_start[None]).abs()
                best = distances.square().mean(dim=-1).argmin(dim=1)
                error = distances[torch.arange(len(worlds),device=self.device),best].amax(dim=-1)
                valid = error<=self.design['max_template_error_rad']
                cap = self.design['max_target_from_observed_rad']
                target = torch.clamp(torch.clamp(self.template_target[best],min=measured-cap,max=measured+cap),
                                     min=self.lower,max=self.upper)
                self.targets[worlds] = target
                self.active[worlds] = valid & (self.role==ROLES[-1])
                for world,index,near,accept,seed,raw in zip(worlds.tolist(),best.tolist(),error.tolist(),valid.tolist(),
                                                           measured.tolist(),target.tolist(),strict=True):
                    self.events.append(dict(world=world,control=control,template=index,match_max_rad=near,
                        template_admissible=accept,observed_joints=seed,bounded_target=raw))
        if self.role!=ROLES[-1] or not bool(self.active.any()):
            return
        up = -g[:,2]/g.norm(dim=-1).clamp_min(.5)
        self.active &= getup
        self.returning |= self.active & ((up<self.design['abort_min_body_up'])|
            (speed>self.design['abort_max_gyro_rad_s'])|(control>=self.design['target_end_control']))
        ids = torch.nonzero(self.active).flatten()
        if not ids.numel():
            return
        columns = torch.as_tensor(self.cols,device=self.device)
        rows = ids[:,None]
        old = previous[rows,columns]
        nominal = term._processed_actions[rows,columns].clone()
        selection = self.returning[ids,None]
        desired = torch.where(selection,nominal,self.targets[ids])
        rate = self.design['max_slew_rad_per_control']
        gap = desired-old
        actual = torch.clamp(torch.where(gap.abs()<=rate,desired,old+gap.clamp(-rate,rate)),
                             min=self.lower,max=self.upper)
        assert torch.max(torch.abs(actual-old)).item()<=rate+1e-6
        term._processed_actions[rows,columns] = actual
        term.previous_target[rows,columns] = actual
        term.projected_destination[rows,columns] = actual
        reached = selection.flatten() & (actual==nominal).all(dim=1)
        self.active[ids[reached]] = False
        self.rows.extend(dict(control=control,world=i,returning=flag,previous=a,original=b,applied=c)
            for i,flag,a,b,c in zip(ids.tolist(),selection.flatten().tolist(),old.tolist(),nominal.tolist(),
                                    actual.tolist(),strict=True))


def collect(env, plan, actors, folder, observer, watch, ledger):
    with np.load(ROOT/plan['friction_development']['startup_fields'],allow_pickle=False) as expected:
        for name in sorted(base.RESET_FIELDS):
            np.testing.assert_array_equal(parent.cpu(getattr(env.sim.model,name)),expected['expanded_'+name])
    bound = plan['standing_pair']['common_snapshot_control']
    truncated = dict(plan,composition={**plan['composition'],'maximum_controls':plan['upright_recovery']['end_control']})
    candidate = Recovery(env,plan)
    phase, roles = 'shared', iter(ROLES)
    base_fork, apply, publish = recorder.StandingFork, recorder.Composition.apply, observer.publish
    counters = window.original.CollectorLedger(ledger)
    paths, histories = {}, {}

    def save():
        if phase=='shared':
            return
        file = folder/phase/'recovery-actions.json'
        write(file,dict(events=candidate.events,applied=candidate.rows))
        paths[phase] = sha(file)
        histories[phase] = len(candidate.events)

    class TriggerFork(base_fork):
        def restore(self):
            nonlocal phase
            save()
            state = super().restore()
            assert state['physical']['counters']==[bound,5*bound]
            phase = next(roles)
            candidate.reset(phase)
            return state

    def apply_recovery(controller, term, previous, control, action):
        apply(controller,term,previous,control,action)
        if phase!='shared':
            candidate.apply(term,previous,control,action)

    def publish_recovery(path,state):
        publish(path,dict(state,purpose=f'低摩擦起身恢复｜{LABELS[state["phase"]]}'))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=TriggerFork),
              patch.object(recorder.Composition,'apply',apply_recovery),patch.object(observer,'publish',publish_recovery)):
            recorder.collect(env,truncated,actors,folder,observer,watch,counters)
    finally:
        save()
        ledger.update(counters,recovery_action_sha256=paths,detected_worlds=histories,
                      actual_slide_friction=plan['composition']['foot_friction'])


def score():
    project,plan = driver.checked('score-pending')
    goal,task = project['goal_execution'],project['staged_training_request']
    node,folder = goal[KEY],ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert raw['actual_slide_friction']==.8 and not (raw['learning_updates'] or raw['hardware_io'] or raw['reserved_seed_used'])
    assert raw['transitions']==plan['maximum_control_transitions'] and raw['physics_substeps']==plan['maximum_physics_substeps']
    start, end = plan['standing_pair']['common_snapshot_control'],plan['upright_recovery']['end_control']
    assert raw['prefix']['controls']==start
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['fields']==len(fork['fields']) and fork['physical']['counters']==[start,5*start]
    initial_path = folder/'initial.npz'
    assert sha(initial_path)==raw['initial_sha256'] and sha(folder/'release-poses.npz')==raw['release_sha256']
    with np.load(initial_path,allow_pickle=False) as saved:
        initial = dict(saved)
    fields = tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'original_feet','roll_phase')))
    shared = read_trace(folder/'shared',raw['prefix']['chunks'],1,start,fields,sha)
    d,plant,study = plan['composition'],plan['environment'],plan['upright_recovery']
    left = np.asarray(plan['seven_command']['groups']['turn-left'])
    joints = raw['qpos_joint_addresses']
    bounds = np.asarray(plant['joint_limits'])
    tolerance = np.deg2rad(plant['applied_joint_limit_tolerance_deg'])
    outcomes, entry, first_actions = {},{},{}
    for role,branch in raw['branches'].items():
        path = folder/role
        assert branch['status']=='collected' and branch['controls']==end
        assert read(path/'restored-fork.json',branch['restore_sha256'])=={k:fork[k] for k in ('physical','controller_sha256')}
        boundary, = (row for row in branch['boundaries'] if row['control']==start)
        file = path/boundary['file']
        assert sha(file)==boundary['sha256']
        with np.load(file,allow_pickle=False) as data:
            first_actions[role] = {key:data[key] for key in ('observed','actor_input','mode','raw','actual_target','motor')}
        trace = read_trace(path,branch['chunks'],start+1,end,fields,sha)
        trace = {key:np.concatenate((shared[key],trace[key])) for key in fields}
        q = trace['qpos'][...,joints]
        legal = ((q>=bounds[:,0]-tolerance)&(q<=bounds[:,1]+tolerance)).all(axis=-1)
        physical = dict(trace,foot_contact=trace['original_feet'],joint_positions=q,joint_envelope_ok=legal)
        passed, standing, original = recovery_and_standing(plan,folder,trace,initial,physical)
        sequence = Sequence(plan['roll_route'])
        for control in range(d['getup_to_stand_control'],end+1):
            frame = {key:trace[key][control-1] for key in STATE_FIELDS}
            if control==end:
                frame['active'] = frame['active'] & (trace['mode'][control-1]==STAND)
            sequence.update(control-d['getup_to_stand_control'],frame)
            np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][control-1])
        entry[role] = sequence.phase==ROLL
        events = read(path/'recovery-actions.json',raw['recovery_action_sha256'][role])
        changed = {row['world'] for row in events['applied'] if row['original']!=row['applied']}
        assert len(events['events'])==raw['detected_worlds'][role]
        assert all(row['control']>=start+study['detector']['successive_controls']-1 for row in events['events'])
        if role!=ROLES[-1]:
            assert not changed and not events['applied']
        if events['applied']:
            applied,prior = (np.asarray([row[key] for row in events['applied']]) for key in ('applied','previous'))
            selected = bounds[study['joint_indices']]
            assert ((applied>=selected[:,0]-1e-6)&(applied<=selected[:,1]+1e-6)).all()
            assert abs(applied-prior).max()<=study['max_slew_rad_per_control']+1e-6
            assert changed<={row['world'] for row in events['events'] if row['template_admissible']}
        motion = {}
        for item in events['events']:
            world,k = item['world'],item['control']
            if world not in changed:
                continue
            observed_at = np.asarray([k,min(k+25,end),min(k+75,end),d['getup_to_stand_control'],end])
            attitude = rotation(trace['quaternion'][observed_at-1,world])
            tilt = np.rad2deg(np.arccos(np.clip(attitude[:,2,2],-1.,1.)))
            first_clear = np.flatnonzero(~trace['head_contact'][k:,world])
            physical_window = trace['foot_contact'][k:min(k+75,end),world].all(axis=-1)
            motion[world] = dict(control=observed_at.tolist(),actual_tilt_deg=tilt.tolist(),
                first_head_clear_control=None if not len(first_clear) else int(k+1+first_clear[0]),
                both_feet_contact_samples_first_75=int(physical_window.sum()))
        outcomes[role] = dict(original_getup=original,timely_final_getup=int(passed.sum()),standing=int(standing.sum()),
            roll_entry=int(entry[role].sum()),left_roll_entry=int(entry[role][left].sum()),
            roll_entry_by_command={name:int(entry[role][ids].sum()) for name,ids in plan['seven_command']['groups'].items()},
            triggered_worlds=len(events['events']),changed_worlds=sorted(changed),motion_after_change=motion,
            changed_that_passed_original_entry=sorted(world for world in changed if entry[role][world]),
            head_contact_last_25_by_changed={int(world):int(trace['head_contact'][-25:,world].sum()) for world in sorted(changed)},
            video=check_video(path,dict(branch,camera_command=raw['camera_command']),trace,initial))
    assert tuple(outcomes)==ROLES
    for role in ROLES[1:]:
        for key in first_actions[role]:
            # Later actuator intervals can differ due to physical numerical drift.
            a,b = first_actions[role][key],first_actions[ROLES[0]][key]
            np.testing.assert_array_equal(a[0] if key=='motor' else a,b[0] if key=='motor' else b,err_msg=key)
    candidate = outcomes[ROLES[-1]]
    rescued = entry[ROLES[-1]]&~entry[ROLES[0]]&~entry[ROLES[1]]
    proven = [world for world in candidate['changed_worlds'] if rescued[world]]
    compared = {role:dict(overall_delta=candidate['roll_entry']-outcomes[role]['roll_entry'],
        left_delta=candidate['left_roll_entry']-outcomes[role]['left_roll_entry']) for role in ROLES[:-1]}
    eligible = (candidate['original_getup']['functional_coverage_gate'] and len(proven)>=2 and
        all(row['overall_delta']>=0 and row['left_delta']>=study['min_left_gain_against_each_reference'] for row in compared.values()))
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(status='verified-observed-tilt-recovery',protocol_sha256=sha(PLAN),
        collection_sha256=node['collection_sha256'],matched_snapshot=fork['physical'],
        transitions=raw['transitions'],peak_gpu_mib=max(row['used_mib'] for row in raw['gpu_samples']),
        results=outcomes,paired_entries=compared,changed_and_rescued_against_both=proven,
        ready_for_full_roll_comparison=bool(eligible),reserved_seed_used=False,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),eligible=bool(eligible))
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    task.update(info,next_gate='Evaluate the selected recovery on the complete rolling and seven-command route.' if eligible else
        'Review the actual first-foot and head response to distinguish commanded offsets from settling in the tilted pose.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_RECOVERY_SCORED',sha(output),
        {role:(row['roll_entry'],row['left_roll_entry'],len(row['changed_worlds'])) for role,row in outcomes.items()},
        eligible,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.multiple(parent,config=base.configured,PLAN=PLAN),
          patch.object(window.comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()
