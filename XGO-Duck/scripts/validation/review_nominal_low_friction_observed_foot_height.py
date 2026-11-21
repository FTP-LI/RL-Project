"""Screen an observed right-foot-height trigger without simulator contact inputs."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

import review_nominal_low_friction_early_landing_position as position_screen
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY = 'nominal_low_friction_observed_foot_height'
STAGE = 'nominal-low-friction-observed-foot-height'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
OUTPUT = position_screen.REPORT/'observed-foot-height.json'


def repair_freeze():
    project=read(PROJECT)
    permitted(project,STAGE)
    goal=project['goal_execution']
    node=goal[KEY]
    assert node['status']=='archive-ready' and not OUTPUT.exists()
    spec=read(PLAN,node['protocol_sha256'])
    archived_plan=position_screen.REPORT/'observed-height-protocol-before-fix.json'
    archived_script=position_screen.REPORT/'observed-height-script-before-fix.py.txt'
    diagnostic=position_screen.REPORT/'observed-height-coordinate-diagnostic.log'
    assert sha(archived_plan)==node['protocol_sha256'] and sha(archived_script)==spec['script_sha256']
    assert 'DIAGNOSTIC_EXIT=0' in diagnostic.read_text(encoding='utf-8')
    limit=spec.pop('maximum_reconstructed_site_error_m')
    assert limit==1e-6
    spec.update(maximum_reconstructed_relative_height_error_m=limit,script_sha256=sha(Path(__file__)),
        reconstruction_baseline='Compare the relative vertical output used by the trigger. Retain the full position error for context.',
        zero_step_scope_correction={rel(path):sha(path) for path in (archived_plan,archived_script,diagnostic)})
    write(PLAN,spec)
    node['protocol_sha256']=sha(PLAN)
    info=dict(current_protocol_execution=STAGE+'-coordinate-corrected',current_evidence=rel(PLAN))
    project.update(info)
    project['staged_training_request'].update(info)
    write(PROJECT,project)
    print('OBSERVED_FOOT_HEIGHT_RELATIVE_AXIS_FROZEN',sha(PLAN),flush=True)


class FootKinematics:
    def __init__(self,plan):
        import mujoco
        from mjlab.scene import Scene
        from run_nominal_skill_composition import config

        cfg=config(dict(plan,composition={**plan['composition'],'foot_friction':1.}))
        self.model=Scene(cfg.scene,device='cpu').compile()
        self.data=mujoco.MjData(self.model)
        self.mj=mujoco
        self.addresses=np.array([self.model.joint('robot/'+n).qposadr.item() for n in plan['walking']['joint_names']])
        assert self.model.nq==21 and set(self.addresses.tolist())==set(range(7,21))
        self.sites=[self.model.site('robot/'+n+'_foot').id for n in ('left','right')]

    def world_sites(self,configuration):
        self.data.qpos[:]=configuration
        self.mj.mj_kinematics(self.model,self.data)
        return self.data.site_xpos[self.sites].copy()

    def relative_height(self,gravity,joints):
        norm=np.linalg.norm(gravity,axis=1,keepdims=True)
        assert (norm>.5).all()
        up=-gravity/norm
        q=np.asarray(self.model.qpos0).copy()
        q[:7]=[0.,0.,0.,1.,0.,0.,0.]
        heights=[]
        for values,vertical in zip(joints,up,strict=True):
            q[self.addresses]=values
            feet=self.world_sites(q)
            heights.append(float((feet[1]-feet[0])@vertical))
        assert self.data.time==0.
        return np.asarray(heights)


def extract(source,rule_spec):
    archives=position_screen.archive_sequences(source)
    kin=FootKinematics(next(iter(archives.values()))['plan'])
    proof={}
    for entry in source['episodes']:
        path=ROOT/entry['collection']['path']
        raw=read(path,entry['collection']['sha256'])
        for role in entry['branches']:
            name=f'{entry["key"]}/{role}'
            row=archives[name]
            active=row['active']
            home=np.asarray(row['plan']['policy_home']['getup-supine'])
            assert home.shape==(14,) and row['targets'].shape[1]==len(active)
            branch=raw['branches'][role]
            folder=path.parent/role
            trace=position_screen.previous.checkpoint(folder,branch['chunks'],source['decision_control'],('foot_positions',))
            physical=np.asarray([kin.world_sites(q) for q in row['position'][active]])
            stored=trace['foot_positions'][active]
            geometric_error=float(abs(stored-physical).max())
            relative=stored[:,1,2].astype(np.float64)-stored[:,0,2]
            vertical_error=float(abs(relative-(physical[:,1,2]-physical[:,0,2])).max())
            assert vertical_error<=rule_spec['maximum_reconstructed_relative_height_error_m'],vertical_error
            file=folder/'pre-roll-standing.npz'
            assert sha(file)==branch['standing_sha256']
            with np.load(file,allow_pickle=False) as z:
                observed=z['actor_input']
                at,old=(observed[step-source['release_control'],active].copy() for step in
                        (rule_spec['decision_control'],rule_spec['reference_control']))
            np.testing.assert_array_equal(at[:,3:6],row['now'][active])
            np.testing.assert_array_equal(old[:,3:6],row['old'][active])
            now,before=(kin.relative_height(item[:,3:6],item[:,6:20]+home) for item in (at,old))
            error=now-relative
            p95=float(np.percentile(abs(error),95))
            assert p95<=rule_spec['maximum_observed_relative_height_error_p95_m'],(name,p95)
            current=np.zeros(len(active)); current[active]=now*1000.
            delta=np.zeros(len(active)); delta[active]=(now-before)*1000.
            row.update(relative_right_height_mm=current,relative_right_height_change_mm=delta)
            proof[name]=dict(absolute_world_site_error_m=geometric_error,reconstructed_relative_height_error_m=vertical_error,
                observed_relative_height_p95_error_m=p95,
                focus=[dict(world=int(w),relative_right_height_mm=float(current[w]),relative_right_height_change_mm=float(delta[w]))
                       for w in np.flatnonzero(row['focus'])])
    return archives,proof


def pick(archives,spec):
    features={name:(row,np.rad2deg(np.arctan2(row['now'][:,0],-row['now'][:,2]))) for name,row in archives.items()}
    cases=tuple(product(*(spec[k] for k in ('min_pitch_deg','max_pitch_deg','minimum_relative_right_height_mm',
        'maximum_relative_right_height_mm','minimum_right_height_change_mm'))))
    best,nearest,n=None,None,0
    for lo,hi,low,high,trend in cases:
        if low>=high:
            continue
        groups={}
        masks={}
        for name,(row,pitch) in features.items():
            h=row['relative_right_height_mm']
            mask=(row['active']&(pitch>=lo)&(pitch<=hi)&(h>=low)&(h<=high)&(row['relative_right_height_change_mm']>=trend))
            masks[name]=mask
            groups[name]={f'{key}_{variant}':int(bits.sum()) for key in ('focus','missed','admitted')
                          for variant,bits in (('total',row[key]),('triggered',mask&row[key]))}
        records=list(groups.values())
        cover=min(v['focus_triggered']/v['focus_total'] for v in records)
        rate=max(v['admitted_triggered']/v['admitted_total'] for v in records)
        score=(-cover,rate,-sum(v['missed_triggered'] for v in records),lo,hi,low,high,trend)
        if rate>spec['maximum_admitted_exposure_fraction']:
            continue
        rule=dict(pitch_min_deg=lo,pitch_max_deg=hi,relative_right_height_min_mm=low,
            relative_right_height_max_mm=high,relative_right_height_change_min_mm=trend)
        if nearest is None or score<nearest[0]:
            nearest=(score,rule,groups,masks)
        if cover>=spec['minimum_focused_miss_fraction'] and all(v['focus_triggered']>=spec['minimum_focused_miss_in_archive'] for v in records):
            n+=1
            if best is None or score<best[0]:
                best=(score,rule,groups,masks)
    return best,nearest,n


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,checksum in spec['zero_step_scope_correction'].items():
        assert sha(ROOT/path)==checksum,path
    old=read(ROOT/spec['source_protocol'],spec['source_protocol_sha256'])
    read(ROOT/spec['source_result'],spec['source_result_sha256'])
    assert sha(Path(position_screen.__file__))==old['script_sha256']
    for path,checksum in old['sources'].items():
        assert sha(ROOT/path)==checksum,path
    archives,proof=extract(old,spec)
    best,nearest,count=pick(archives,spec)
    selected,responses=None,{}
    if best:
        triggered={name:dict(row,active=best[3][name]) for name,row in archives.items()}
        identity=dict(pitch_min_deg=-180.,pitch_max_deg=180.,progress_max_deg=360.,roll_max_deg=90.)
        selected,responses=position_screen.actuator_authority(triggered,identity,old)
    result=dict(status='verified-actor-relative-foot-height',protocol_sha256=sha(PLAN),qualified_rules=count,
        rule=best[1] if best else None,scores=best[2] if best else None,
        strongest_protecting_rule=dict(rule=nearest[1],scores=nearest[2]) if nearest else None,
        actuator=old['actuator'],recommended_decrement_rad=selected,servo_replay=responses,geometry=proof,
        physical_steps=0,learning_updates=0,holdout_used=False,hardware_io=False)
    if check:
        assert result==read(OUTPUT,node['result_sha256'])
    else:
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),decrement_rad=selected,
            report=rel(position_screen.REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Run same-state CUDA early landing perturbation with a causal encoder guard.' if selected else
            'Verify a bounded same-state landing diagnostic around G120 while preserving the candidate deployment gate.'))
        write(PROJECT,project)
    print('OBSERVED_FOOT_HEIGHT_VERIFIED',sha(OUTPUT),count,result['rule'],selected,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('repair-freeze','score','check'))
    option=cli.parse_args().action
    repair_freeze() if option=='repair-freeze' else score(option=='check')
