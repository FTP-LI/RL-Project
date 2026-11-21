"""Test a causal takeover by the archived fallback in the continuous route."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from audit_nominal_supine_early_support import KEY, REPORT
from audit_nominal_supine_terminal_stance import attitude
from getup_stand_handoff import projected_slew
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from run_nominal_roll_closed_loop_timestep import CollectorLedger
from run_nominal_supine_terminal_stance import driver, recorder, comparison, head_foot_contact
from routed_getup_sequence import GETUP, STAND
from verify_nominal_skill_composition import check_video, read_trace


ROOT, PROJECT = comparison.ROOT, comparison.PROJECT
read, rel, sha, write = comparison.read, comparison.rel, comparison.sha, comparison.write
PREP = 'nominal-supine-early-support-preparation'
STAGE = 'nominal-supine-early-takeover'
PLAN = ROOT/'configs/experiments/nominal_supine_early_takeover_pair.json'
ROLES = ('original-getup', 'archived-fallback-takeover')
LABELS = {'shared':'共同跌倒与起身','original-getup':'原策略','archived-fallback-takeover':'归档仰卧策略接手'}
FORK, END = 120, 600
OLD_TEACHER = ROOT/'configs/experiments/getup_archived_policy_comparison_s1161.json'
OLD_TEACHER_SHA = '00a55a314960eacacea7187b809e5e773c47dfe4903976d50526e4e1ca9388a5'


class ObservedGate:
    def __init__(self, n, rule):
        self.rule = rule
        self.dual = np.zeros(n,dtype=np.int32)
        self.head = np.zeros(n,dtype=np.int32)
        self.align = np.full(n,-1,dtype=np.int32)
        self.trigger = self.align.copy()

    def snapshot(self):
        return tuple(x.copy() for x in (self.dual,self.head,self.align,self.trigger))

    def restore(self, state):
        for target, source in zip((self.dual,self.head,self.align,self.trigger),state,strict=True):
            np.copyto(target,source)

    def observe(self, k, supine, dual, head, pitch, roll, tilt, fit):
        rule = self.rule
        eligible = supine & dual & (tilt <= rule['alignment']['maximum_root_tilt_deg'])
        eligible &= abs(roll) <= rule['alignment']['maximum_absolute_roll_deg']
        self.dual[:] = np.where(eligible,self.dual+1,0)
        self.head[:] = np.where(head,self.head+1,0)
        self.align[(self.align<0)&(self.dual>=rule['alignment']['minimum_dual_foot_reads'])] = k
        good = (self.trigger<0)&(self.align>=0)&(k-self.align<=rule['candidate_window_after_first_dual_reads'])
        good &= supine & dual & fit & (self.head>=rule['candidate_minimum_head_contact_reads'])
        good &= (pitch>=rule['candidate_minimum_root_pitch_deg'])&(pitch<=rule['candidate_maximum_root_pitch_deg'])
        good &= abs(roll)<=rule['candidate_maximum_absolute_roll_deg']
        self.trigger[good] = k


def rehearsal(rule, cohort):
    path = ROOT/cohort['samples']
    assert sha(path) == cohort['samples_sha256']
    with np.load(path,allow_pickle=False) as source:
        rows = dict(source)
    gate = ObservedGate(len(rows['worlds']),rule)
    actual_supine = np.ones(len(rows['worlds']),dtype=bool)
    for i,k in enumerate(rows['controls']):
        dual = rows['native_feet'][i].all(axis=1) & rows['active'][i]
        gate.observe(k,actual_supine,dual,rows['head_foot_reconstructed'][i].any(axis=1),
            rows['pitch_deg'][i],rows['roll_deg'][i],rows['tilt_deg'][i],rows['foot_flags_match'][i])
    expected = np.where(rows['first_any_usable']>=0,rows['first_any_usable']+rows['controls'][0],-1)
    np.testing.assert_array_equal(gate.trigger,expected)
    assert (gate.trigger[gate.trigger>=0]>FORK).all()
    return dict(seed=cohort['seed'],worlds=len(actual_supine),triggered=int((gate.trigger>=0).sum()),
        earliest=int(gate.trigger[gate.trigger>=0].min()))


def freeze():
    project = read(PROJECT)
    task = project['staged_training_request']
    node = project['goal_execution'][KEY]
    assert driver.parent.allowed(project,PREP) and not task['simulation_running']
    assert node['status']=='archive-audit-reviewed' and not PLAN.exists()
    audit = read(ROOT/node['analysis'],node['analysis_sha256'])
    rule = read(ROOT/node['design'],node['design_sha256'])
    terminal = project['goal_execution']['nominal_supine_terminal_stance']
    original = read(ROOT/terminal['protocol'],terminal['protocol_sha256'])
    teacher = read(OLD_TEACHER,OLD_TEACHER_SHA)['policies']['side-to-supine-replay']
    fallback = original['models']['getup-fallback']
    assert sha(ROOT/teacher['model']) == teacher['model_sha256'] == fallback['model_sha256']
    assert original['models']['getup-supine']['model_sha256'] != fallback['model_sha256']
    assert original['policy_home']['getup-supine'] == original['policy_home']['getup-fallback'] == original['environment']['model_home']
    periods = original['composition']
    assert periods['roll_start_control']==END and FORK>periods['release_control']
    assert original['roll_route']['standing_dwell_controls']==rule['original_pre_roll_dwell_controls']
    assert original['roll_route']['roll']['criteria']['tilt_max_deg']==rule['original_pre_roll_tilt_deg']
    measurements = [rehearsal(rule,row) for row in audit['cohorts']]
    original.pop('terminal_stance')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    original['environment']['run_id'] = f'duck-getup-early-takeover-s1257-{stamp}'
    controls = FORK+len(ROLES)*(END-FORK)
    original.update(stage=STAGE,record=f'records/evaluations/{original["environment"]["run_id"]}',
        maximum_unique_controls=controls,maximum_physics_substeps=controls*5,
        maximum_control_transitions=controls*original['environment']['num_envs'])
    original['standing_pair']['common_snapshot_control'] = FORK
    original['early_takeover'] = dict(fork_control=FORK,end_control=END,roles=list(ROLES),eligibility=rule,
        teacher='getup-fallback',target_slew_rad_s=2.,minimum_gain_forward_roll_entry=2,reserved_seeds=rule['reserved_seeds'])
    for source,fields in ((node,('design','analysis')),(terminal,('protocol','support_chronology'))):
        for field in fields:
            original['references'][source[field]] = source[field+'_sha256']
    entry = Path(__file__).resolve()
    original['references'].update({rel(OLD_TEACHER):OLD_TEACHER_SHA,teacher['model']:teacher['model_sha256'],
        rel(entry):sha(entry)})
    for row in audit['cohorts']:
        original['references'][row['samples']] = row['samples_sha256']
    driver.parent.base.shared.assert_sources(original)
    observer = driver.parent.preflight()
    write(PLAN,original)
    certificate = REPORT/'pair-preflight.json'
    write(certificate,dict(protocol_sha256=sha(PLAN),observation=observer,rehearsal=measurements,
        target_policy_sha256=fallback['model_sha256'],comparison_controls=controls,physical_substeps=0))
    node.update(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(certificate),
        run_id=original['environment']['run_id'])
    task.update(current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        current_protocol_execution='early-takeover-frozen',current_evidence=rel(certificate))
    project.update(current_protocol_execution=task['current_protocol_execution'],current_evidence=task['current_evidence'])
    write(PROJECT,project)
    print('SUPINE_EARLY_TAKEOVER_FROZEN',sha(PLAN),node['run_id'],flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    spec,d = plan['early_takeover'],plan['composition']
    short = dict(plan,composition={**d,'maximum_controls':spec['end_control']})
    role,roles = 'shared',iter(ROLES)
    base_fork,base_ctl = recorder.StandingFork,recorder.Composition
    counters = CollectorLedger(ledger)
    model = env.sim.mj_model
    head = model.geom('robot/head_collision').id
    feet = [model.geom(f'robot/{side}_foot_collision').id for side in ('left','right')]
    limits = torch.as_tensor(plan['environment']['joint_limits'],device=env.device,dtype=torch.float32)
    home = torch.as_tensor(plan['environment']['model_home'],device=env.device,dtype=torch.float32)
    teacher = actors[spec['teacher']]
    td = driver.parent.base.shared.td
    rule = spec['eligibility']

    class Takeover(base_ctl):
        def __init__(self,*args):
            super().__init__(*args)
            self.relative_history = torch.zeros((env.num_envs,14),device=env.device)
            self.supine = np.zeros(env.num_envs,dtype=bool)
            self.gate = ObservedGate(env.num_envs,rule)
            self.branch_evidence = {}

        def observe_roll(self,k,frame):
            super().observe_roll(k,frame)
            if k==d['release_control']:
                poses = recorder.classify_release(recorder.projected_gravity(frame['quaternion']),
                    plan['environment']['criteria']['release_alignment_min'])
                self.supine[:] = np.asarray(poses)=='supine'
            if not rule['first_control']<=k<=rule['last_control']:
                return
            pitch,roll,tilt = attitude(frame['quaternion']).T
            dual = frame['active'] & frame['foot_contact'].all(axis=1)
            route = actors['getup']
            current = self.supine
            if route.parent.supine_at_release is not None:
                current = current & recorder.cpu(route.parent.supine_at_release & ~route.right)
            self.gate.observe(k,current,dual,head_foot_contact(env,head,feet),pitch,roll,tilt,True)

        def apply(self,term,previous,control,action):
            super().apply(term,previous,control,action)
            if role!=ROLES[-1]:
                return
            use = (self.gate.trigger>=0)&(action.mode==GETUP)
            ids = np.flatnonzero(use)
            if not len(ids):
                return
            fresh = self.first_applied[ids]<0
            selected = torch.as_tensor(ids,device=env.device)
            adapted = action.actor_input[selected].clone()
            # Homes are identical. Start with the outgoing model's real raw
            # history, then carry the incoming model's own prediction history.
            adapted[:,34:48] = torch.where(torch.as_tensor(fresh,device=env.device)[:,None],
                                           adapted[:,34:48],self.relative_history[selected])
            raw = teacher(td('actor',adapted))
            self.relative_history[selected] = raw
            desired = raw+home
            before = previous[selected]
            applied,destination = projected_slew(before,desired,limits,env.step_dt,spec['target_slew_rad_s'])
            assert (abs(applied-before)<=spec['target_slew_rad_s']*env.step_dt+3e-7).all()
            if fresh.any():
                for field,tensor in (('actor_input',adapted),('raw',raw),('previous_target',before),('applied_target',applied),
                                     ('parent_raw',action.raw[selected])):
                    self.first_rows[field].extend(recorder.cpu(tensor)[fresh])
                self.first_rows['worlds'].extend(ids[fresh])
            self.first_applied[ids[fresh]] = control+1
            self.applied_count[ids] += 1
            for field,value in (('requested_target',desired),('projected_destination',destination),
                                ('_processed_actions',applied),('previous_target',applied)):
                getattr(term,field)[selected] = value

        def finish(self,arm):
            self.branch_evidence[arm] = dict(trigger=self.gate.trigger.copy(),first_applied=self.first_applied.copy(),
                applied_count=self.applied_count.copy(),**self.first_rows)

        def begin(self):
            self.first_applied = np.full(env.num_envs,-1,dtype=np.int32)
            self.applied_count = np.zeros(env.num_envs,dtype=np.int32)
            self.first_rows = {k:[] for k in ('actor_input','raw','previous_target','applied_target','parent_raw','worlds')}
            self.relative_history.zero_()

    class SameStart(base_fork):
        def __init__(self,*args):
            super().__init__(*args)
            self.detector = self.control.gate.snapshot()
            assert (self.detector[-1]<0).all()

        def restore(self):
            nonlocal role
            if role!='shared':
                self.control.finish(role)
            state = super().restore()
            self.control.gate.restore(self.detector)
            for current,expected in zip(self.control.gate.snapshot(),self.detector,strict=True):
                np.testing.assert_array_equal(current,expected)
            role = next(roles)
            self.control.begin()
            return state

    native = observer.publish
    controller = None

    def build(*args):
        nonlocal controller
        controller = Takeover(*args)
        return controller

    def publish(path,values):
        native(path,dict(values,purpose=f'仰卧起身早段对照｜{LABELS[values["phase"]]}'))

    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=SameStart,Composition=build),
              patch.object(observer,'publish',publish)):
            recorder.collect(env,short,actors,folder,observer,watch,counters)
    finally:
        ledger.update(counters)
        if role!='shared' and controller is not None:
            controller.finish(role)
            for arm,samples in controller.branch_evidence.items():
                target = folder/arm/'takeover.npz'
                np.savez_compressed(target,**samples)
                ledger['branches'][arm]['takeover_sha256'] = sha(target)


def score():
    project,plan = driver.checked('score-pending')
    node = project['goal_execution'][KEY]
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    initial_file = folder/'initial.npz'
    assert sha(initial_file)==raw['initial_sha256'] and raw['actor_weights_unchanged']
    assert raw['transitions']==plan['maximum_control_transitions'] and raw['resets']==1
    with np.load(initial_file,allow_pickle=False) as items:
        initial = dict(items)
    d = plan['composition']
    fields = ('mode','roll_phase','qpos',*STATE_FIELDS)
    common = read_trace(folder/'shared',raw['prefix']['chunks'],1,FORK,fields,sha)
    results,passes = {},{}
    forward = plan['seven_command']['groups']['forward']
    weights = plan['models']['getup-fallback']
    teacher = driver.parent.onnx_session(ROOT/weights['model'])
    fork = read(folder/'fork.json',raw['fork_sha256'])
    first = None
    for arm in ROLES:
        branch = raw['branches'][arm]
        path = folder/arm
        assert read(path/'restored-fork.json',branch['restore_sha256']) == {k:fork[k] for k in ('physical','controller_sha256')}
        leaf = read_trace(path,branch['chunks'],FORK+1,END,fields,sha)
        trace = {k:np.concatenate((common[k],leaf[k])) for k in fields}
        sequence = Sequence(plan['roll_route'])
        for k in range(d['getup_to_stand_control'],END+1):
            frame = {key:trace[key][k-1] for key in STATE_FIELDS}
            if k==END:
                frame['active'] = frame['active'] & (trace['mode'][k-1]==STAND)
            sequence.update(k-d['getup_to_stand_control'],frame)
            np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][k-1])
        passes[arm] = sequence.phase==ROLL
        mark, = (x for x in branch['boundaries'] if x['control']==FORK)
        assert sha(path/mark['file'])==mark['sha256']
        with np.load(path/mark['file'],allow_pickle=False) as saved:
            current = {k:saved[k] for k in ('observed','actor_input','raw','previous_target')}
            current['motor'] = saved['motor'][0]
        if first is None:
            first = current
        for k,value in first.items():
            np.testing.assert_array_equal(value,current[k],err_msg=f'{arm}/{k}')
        evidence = path/'takeover.npz'
        assert sha(evidence)==branch['takeover_sha256']
        with np.load(evidence,allow_pickle=False) as values:
            x = dict(values)
        activated = np.flatnonzero(x['first_applied']>=0)
        matched = np.flatnonzero(x['trigger']>=0)
        if arm==ROLES[0]:
            assert not len(activated) and not x['applied_count'].any()
        else:
            assert len(activated)
            np.testing.assert_array_equal(activated, np.sort(x['worlds']))
            np.testing.assert_array_equal(x['first_applied'][activated],x['trigger'][activated]+1)
            expected = np.concatenate([teacher.run(None,{'obs':row[None]})[0] for row in x['actor_input']])
            np.testing.assert_allclose(expected,x['raw'],atol=2e-5,rtol=2e-5)
            assert np.max(abs(x['raw']-x['parent_raw']))>1e-5
            limits = np.asarray(plan['environment']['joint_limits'])
            desired = np.clip(x['raw']+plan['environment']['model_home'],limits[:,0],limits[:,1])
            target = x['previous_target']+np.clip(desired-x['previous_target'],-.04,.04)
            np.testing.assert_allclose(x['applied_target'],target,atol=4e-7,rtol=0)
        results[arm] = dict(roll_entry=int(passes[arm].sum()),forward_roll_entry=int(passes[arm][forward].sum()),
            triggered=matched.tolist(),triggered_roll_entries=int(passes[arm][matched].sum()),applied=activated.tolist(),
            record=raw['run_id'],video=check_video(path,dict(branch,camera_command=raw['camera_command']),trace,initial))
    gained = passes[ROLES[1]]&~passes[ROLES[0]]
    lost = passes[ROLES[0]]&~passes[ROLES[1]]
    net = int(gained[forward].sum()-lost[forward].sum())
    eligible = net>=plan['early_takeover']['minimum_gain_forward_roll_entry'] and not lost.any()
    path = REPORT/'pair-analysis.json'
    assert not path.exists()
    write(path,dict(protocol_sha256=sha(PLAN),collection_sha256=node['collection_sha256'],seed=plan['environment']['seed'],
        roles=results,gained_roll_entry=np.flatnonzero(gained).tolist(),lost_roll_entry=np.flatnonzero(lost).tolist(),
        forward_net=net,advance_to_integrated_trial=eligible,transitions=raw['transitions'],learning_updates=0,hardware_io=False))
    node.update(status='reviewed',pair_analysis=rel(path),pair_analysis_sha256=sha(path),advance_to_integrated_trial=eligible)
    task = project['staged_training_request']
    task.update(current_protocol_execution='early-takeover-reviewed',current_evidence=rel(path))
    project.update(current_protocol_execution=task['current_protocol_execution'],current_evidence=task['current_evidence'])
    write(PROJECT,project)
    print('SUPINE_EARLY_TAKEOVER_SCORED',sha(path),{k:v['roll_entry'] for k,v in results.items()},net,flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.object(driver.parent,'PLAN',PLAN),patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__=='__main__':
    main()
