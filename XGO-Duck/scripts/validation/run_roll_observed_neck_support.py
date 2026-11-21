"""Compare one gravity-timed neck target intervention to two identical fixed actors."""

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import recover_roll_head_recontact_load as recovery
import run_roll_head_recontact_load_probe as probe
import verify_roll_checkpoint_selection as scorer
from native_nominal_baseline import verify_sources
from record_io import write
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


ENTRY = Path(__file__).resolve()
KEY = 'roll_observed_neck_support'
STAGE = 'roll-observed-neck-support'
PLANS = recovery.PLANS.with_name('observed-neck-support')
MASTER = PLANS/'master.json'
DESIGN = PLANS.with_name('observed_neck_support.json')
REPORT = recovery.REPORT.with_name('observed-neck-support')
SOURCE = recovery.PLANS/'master.json'
REFERENCES = ('first-unmodified-reference', 'second-unmodified-reference')
CANDIDATE = 'observation-timed-neck-candidate'
ROLES = (*REFERENCES, CANDIDATE)
LABELS = {'shared':'共同站立', REFERENCES[0]:'原策略第一次参照', REFERENCES[1]:'原策略第二次参照',
          CANDIDATE:'按观测相位调节颈部目标'}


def forward_phase(gravity):
    x,y,z = np.moveaxis(np.asarray(gravity,dtype=np.float64),-1,0)
    lateral = np.sqrt(np.maximum(1.-y*y,0.))
    return np.rad2deg(np.arctan2(x,-z/np.maximum(lateral,1e-6))) % 360.


def window(observed, phase, bridge, design):
    gravity = observed[...,design['projected_gravity_indices']]
    angle = forward_phase(gravity)
    a,b,c,d = design['phase_degrees']
    ramp = np.clip(np.minimum((angle-a)/(b-a),(d-angle)/(d-c)),0.,1.)
    active = (phase==ROLL) & ~bridge & (observed[...,design['pitch_angular_speed_index']]>0.)
    active &= abs(gravity[...,1]) <= design['maximum_abs_lateral_gravity']
    return np.where(active, design['increment_rad']*ramp, 0.).astype(np.float32)


def project_neck(target, observed, phase, bridge, design, joint, limits):
    original = target[...,joint]
    adjustment = window(observed,phase,bridge,design)
    legal = np.clip(original+adjustment,*np.asarray(limits,dtype=np.float32))
    return np.where(adjustment>0.,legal,original)


def archive_preflight(plan, raw, role, measured, design, hashes):
    branch = raw/'clips'/role
    manifest = read(raw/'guarded-evaluation.json')['branches'][role]['trace_sha256']
    files = {name:branch/name for name in ('commands.npz','trace.npz')}
    require(all(sha(path)==manifest[name] for name,path in files.items()),'Archived source trace differs')
    hashes.update({rel(path):manifest[name] for name,path in files.items()})
    bounds = plan['walking']['joint_limits']
    joint = plan['walking']['joint_names'].index(design['joint'])
    with np.load(files['commands.npz'],allow_pickle=False) as commands, np.load(files['trace.npz'],allow_pickle=False) as trace:
        observed, phase, bridge = (commands[key] for key in ('observed','phase_used','bridge'))
        targets = commands['applied_targets']
        gravity = -rotation(trace['qpos'][:-1,:,3:7])[...,2,:]
        angular = trace['angular_velocity'][:-1]
        np.testing.assert_allclose(observed[...,3:6][phase==ROLL],gravity[phase==ROLL],atol=2e-4,rtol=0.)
        np.testing.assert_array_equal(observed[...,1][phase==ROLL]>0.,angular[...,1][phase==ROLL]>0.)
        applied = project_neck(targets,observed,phase,bridge,design,joint,bounds[joint])
    controls = np.arange(len(phase))+plan['checkpoint_selection']['plant']['shared_snapshot_control']+1
    delta = applied-targets[...,joint]
    require(delta.min()>=-1e-7 and delta.max()<=design['increment_rad']+1e-7 and not (delta[phase!=ROLL]!=0.).any(),
            'Candidate changes a target outside its declared roll support window')
    changes = delta>1e-6
    leads = np.asarray([(row['control']-1-controls[0],row['world']) for row in measured['samples_by_world']],dtype=int)
    require(len(leads)==measured['samples'] and np.all((leads[:,0]>=0)&(leads[:,0]<len(controls))),
            'Missing archived control before the measured release')
    action = delta[tuple(leads.T)]
    qualified = action>=design['increment_rad']/2.
    require(qualified.mean()>=design['minimum_original_actor_release_coverage'],
            'The causal phase window misses most measured reference releases')
    changed_controls = controls[changes.any(axis=1)]
    return dict(affected_worlds=int(changes.any(axis=0).sum()),affected_control_worlds=int(changes.sum()),
                first_and_last_changed_control=changed_controls[[0,-1]].tolist(),
                released_controls_checked=len(leads),releases_with_half_or_more_offset_one_control_earlier=int(qualified.sum()),
                maximum_delta_rad=float(delta.max()),projected_gravity_and_pitch_direction_match=True)


class PhaseObserver(RecordedObserver):
    def start(self):
        self.state['purpose'] = '滚翻颈部相位对照｜固定策略'
        super().start()


def freeze():
    project = read(P)
    task, goals = project['staged_training_request'], project['goal_execution']
    design = read(DESIGN)
    require(probe.endpoint.previous.direct.pair.trial.allowed(project,'roll-support-target-authority-review')
            and not (task['learning_running'] or task['simulation_running'] or PLANS.exists() or REPORT.exists()),
            'Archived authority review must be the current idle project stage')
    authority_file = ROOT/design['authority_review']
    authority = read(authority_file)
    master = read(SOURCE)
    origin = goals[recovery.KEY]
    require(origin['status']=='reviewed' and sha(SOURCE)==origin['master_sha256'] and master['reserved_seed']==2246
            and not master['reserved_seed_used'] and sha(authority_file)==design['authority_review_sha256']
            and authority['new_physical_steps']==authority['learning_updates']==0,'Source support evidence changed')
    comparison = ROOT/origin['result']
    require(sha(comparison)==origin['result_sha256'],'Archived contact evidence changed')
    rows = read(comparison)['arms']
    deps = (ENTRY,DESIGN,authority_file,SOURCE,comparison,Path(recovery.__file__),
            Path(probe.__file__),Path(scorer.__file__))
    base_hash = {rel(path):sha(path) for path in deps}
    stamps = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared,readbacks = {},{}
    for case in probe.CASES:
        source = SOURCE.with_name(case+'.json')
        plan, result = read(source), rows[case]
        require(sha(source)==master['protocols'][case] and plan['worlds']==design['scenes_per_seed']==128
                and plan['minimum_roll_completion']==design['roll_gate']==116 and plan['maximum_controls']==500
                and plan['checkpoint_selection']['assessment']['minimum_paired_success_gain_against_each_reference']
                ==design['paired_net_gain_against_each_reference']==4,'Original action acceptance changed')
        raw = ROOT/'records/evaluations'/result['run_id']
        receipt = raw/'guarded-evaluation.json'
        require(sha(receipt)==result['collection_sha256'],'Archived sampler receipt differs')
        hashes = base_hash | {rel(source):sha(source),rel(receipt):sha(receipt)}
        readbacks[case] = {role:archive_preflight(plan,raw,role,authority['seeds'][case][role],design,hashes)
                           for role in probe.ROLE_NAMES[:2]}
        out = REPORT/case
        plan['checkpoint_selection']['role_source'] = dict.fromkeys(ROLES,plan['roll']['model'])
        plan.update(stage=STAGE,run_id=f'duck_roll_observed_neck_support_s{plan["seed"]}_{stamps}',
                    guarded_output=rel(out),page=rel(out/'route-view.html'),earlier_roll=plan['roll'],
                    observed_neck_support=design,route_scope='Identical fixed actors with one observation-timed neck target correction')
        plan['frozen_sha256'].update(hashes)
        verify_sources(plan)
        prepared[case] = plan
    require([item['seed'] for item in prepared.values()]==design['development_seeds'],'Development seed order changed')
    REPORT.mkdir()
    PLANS.mkdir()
    admission = REPORT/'admission.json'
    put(admission,dict(observation=preflight(),source_sha256=base_hash,archived_projection=readbacks,initial_physical_steps=0))
    protocols={}
    for case,plan in prepared.items():
        out, path = REPORT/case, PLANS/(case+'.json')
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,plan)
        protocols[case] = sha(path)
        put(out/'windows-entry.json',dict(protocol_sha256=protocols[case],evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=list(probe.CASES),protocols=protocols,physical_transition_budget=master['physical_transition_budget'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),admission=rel(admission),scored={})
    authority_state = goals['roll_support_target_authority']
    authority_state.update(status='reviewed',result=rel(authority_file),result_sha256=sha(authority_file),motor_readback_samples=1045)
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='roll-observed-neck-support-ready',current_evidence=rel(admission),
                next_gate='Independently compare the observation-timed neck target against both identical policy references.')
    project['current_objective'] = 'Evaluate one observation-timed neck target correction under the unchanged roll and later route gates.'
    write(P,project)
    publish({})
    print('ROLL_OBSERVED_NECK_SUPPORT_FROZEN',sha(MASTER),readbacks,flush=True)


def collect(case):
    from run_roll_stand_forward_diagnostic import checked
    import run_continuous_roll_stand_walk as base

    _,_,plan = checked(case,'ready')
    base_policy = base.PolicyControl
    design = plan['observed_neck_support']
    joint = plan['walking']['joint_names'].index(design['joint'])
    bounds = plan['walking']['joint_limits'][joint]

    class PhasePolicy(base_policy):
        def __init__(self, p, networks, homes, sim_home, limits):
            super().__init__(p,networks,homes,sim_home,limits)
            self.reference_actor = networks['roll']

        def next(self, k, obs, sequence):
            target,row = super().next(k,obs,sequence)
            if self.networks['roll'] is not self.reference_actor and (row['phase_used']==ROLL).any():
                # The certified actor history and policy clipping are untouched. Only
                # the physical neck command changes, before normal actuator delay and limits.
                actual = project_neck(row['applied_targets'],base.cpu(obs),row['phase_used'],row['bridge'],design,joint,bounds)
                target[:,joint] = base.torch.as_tensor(actual,device=target.device)
            return target,row

    with patch.object(base,'PolicyControl',PhasePolicy):
        probe.endpoint.previous.sample(case)


def score(case):
    original = scorer.check_policy
    roles = iter(ROLES)

    def check(commands,startup,plan,phase):
        role=next(roles)
        approved=commands['applied_targets'].copy()
        joint=plan['walking']['joint_names'].index(plan['observed_neck_support']['joint'])
        if role==CANDIDATE:
            approved[...,joint] = project_neck(approved,commands['observed'],phase,commands['bridge'],plan['observed_neck_support'],
                                               joint,plan['walking']['joint_limits'][joint])
        np.testing.assert_allclose(commands['actual_motor_target'],approved,atol=1e-6,rtol=0.)
        changed = abs(approved[...,joint]-commands['applied_targets'][...,joint])>1e-6
        require(changed.any()==(role==CANDIDATE),'The matched physical target correction was not isolated to the candidate')
        projected = dict(commands,actual_motor_target=commands['applied_targets'])
        return dict(original(projected,startup,plan,phase),physical_neck_target_verified=True,
                    changed_target_controls=int(changed.sum()),changed_target_worlds=int(changed.any(axis=0).sum()))

    with patch.object(scorer,'check_policy',check):
        probe.endpoint.previous.score(case)
    require(next(roles,None) is None,'Independent readback skipped an actor')


def publish(rows, active=None):
    lines = ['# 按观测相位调节颈部目标','', '每个种子128个场景。三组使用同一滚翻模型与第49步站立快照。', '',
             '| 种子 | 原策略第一次参照 | 原策略第二次参照 | 按观测相位调节颈部目标 | 候选配对净增 |',
             '| --- | ---: | ---: | ---: | ---: |']
    links=[]
    for case,row in rows.items():
        scores=' | '.join(f'{row["arms"][role]["roll_success"]}/128' for role in ROLES)
        gains='／'.join(f'{row["comparisons"][role]["net"]:+d}' for role in REFERENCES)
        lines.append(f'| {row["seed"]} | {scores} | {gains} |')
        media=[f'[{LABELS[role]}原片](../../../../../{row["arms"][role]["video"]["path"]})' for role in ROLES]
        links.extend(('',f'[种子{row["seed"]}的独立评分]({case}/independent.json) · '+' · '.join(media)))
    if active:
        lines.extend(('',f'正在采样{active}，已复核{len(rows)}/2组。', '',
                      f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
                      f'[实景]({active}/native-view/native-latest.jpg) · [命令日志]({active}/collect-command.log)'))
    elif (REPORT/'comparison.json').exists():
        lines.extend(('','[完整评分](comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines+links)+'\n',encoding='utf-8')


@contextmanager
def bindings():
    paths = dict(KEY=KEY,STAGE=STAGE,PLANS=PLANS,REPORT=REPORT,ENTRY=ENTRY)
    with patch.multiple(probe,**paths,MASTER=MASTER,ROLE_NAMES=ROLES,LoadObserver=PhaseObserver,instrument=recovery.instrument):
        with patch.multiple(probe.endpoint,**paths,MASTER=MASTER,ROLES=ROLES,CANDIDATE=CANDIDATE,
                            REFERENCES=REFERENCES,LABELS=LABELS,publish=publish), probe.scope():
            yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    parser.add_argument('case',nargs='?',choices=probe.CASES)
    args=parser.parse_args()
    if args.action=='freeze':
        return freeze()
    with bindings():
        if args.action=='batch':
            return probe.endpoint.previous.batch()
        if args.case is None:
            parser.error('Select a frozen development case')
        return {'collect':collect,'score':score}[args.action](args.case)


if __name__=='__main__':
    main()
