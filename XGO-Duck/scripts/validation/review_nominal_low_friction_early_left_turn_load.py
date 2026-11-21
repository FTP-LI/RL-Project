"""Compare early measured load transfer for the four persistent left-turn misses."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_first_support_effort import tilt


KEY = 'nominal_low_friction_early_left_turn_load'
STAGE = 'nominal-low-friction-early-left-turn-load'
PRIOR = 'nominal-low-friction-right-foot-window'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-right-support-trial/support-timing'
OUTPUT = REPORT/'early-left-turn-load.json'


def freeze():
    project = read(PROJECT)
    permitted(project,PRIOR)
    task,goal = project['staged_training_request'],project['goal_execution']
    previous = goal['nominal_low_friction_right_foot_window']
    loads = goal['nominal_low_friction_support_effort_limit_readback']
    assert previous['status']==loads['status']=='reviewed' and not previous['actuator_preflight_admitted']
    worlds = read(ROOT/previous['protocol'],previous['protocol_sha256'])['left_turn_persistent_worlds']
    source = {node[field]:node[field+'_sha256'] for node,fields in ((loads,('protocol','collection','result')),
        (previous,('protocol','result'))) for field in fields}
    assert len(worlds)==4 and KEY not in goal and not PLAN.exists() and not OUTPUT.exists()
    archives = dict(loads=loads['collection'],pair_matches=loads['result'],load_protocol=loads['protocol'])
    write(PLAN,dict(sources=source,archives=archives,script_sha256=sha(Path(__file__)),seed=1257,foot_friction=.8,
        release_control=100,motor_update_ms=4,worlds=worlds,windows_after_release_ms=[[320,400],[400,480],
            [480,560],[560,640],[640,720]],matched_pair_minimum=3,minimum_pair_median_upright_deficit_deg=5.,
        minimum_pair_median_left_right_load_share_difference=.10,low_friction_reserved_seed=2246,
        additional_physical_steps=0,learning_updates=0,hardware_io=False))
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    goal[KEY] = dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Read early load share and pitch on the four matched left-turn misses before the G136 correction window.')
    project.update(info)
    write(PROJECT,project)
    print('EARLY_LEFT_TURN_LOAD_FROZEN',sha(PLAN),flush=True)


def reading(force,before,after,threshold,weight):
    head,left,right = force[:,1:].T
    total = head+left+right
    two = (left>threshold)&(right>threshold)&(left+right>=.5*weight)
    return dict(upright_progress_deg=float(tilt(before[0])-tilt(after[-1])),
        head_fraction_of_contact_impulse=float(head.sum()/max(total.sum(),1e-9)),
        left_minus_right_fraction_of_contact_impulse=float((left-right).sum()/max(total.sum(),1e-9)),
        two_foot_support_fraction=float(two.mean()),sole_left_loaded_fraction=float(((left>threshold)&(right<=threshold)).mean()),
        sole_right_loaded_fraction=float(((right>threshold)&(left<=threshold)).mean()))


def score(check):
    project = read(PROJECT)
    permitted(project,STAGE)
    node = project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec = read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for name,digest in spec['sources'].items():
        assert sha(ROOT/name)==digest,name
    raw_path = ROOT/spec['archives']['loads']
    raw = read(raw_path)
    original = read(ROOT/spec['archives']['load_protocol'])
    assert original['environment']['seed']==spec['seed'] and original['composition']['foot_friction']==spec['foot_friction']
    assert raw['actor_weights_unchanged'] and not raw['reserved_seed_used']
    metadata = read(raw_path.parent/raw['support_effort']['path'],raw['support_effort']['sha256'])
    entry, = (row for row in metadata['windows'] if row['window']=='release')
    assert metadata['contact_channels']==['ground','head','left_foot','right_foot']
    path = raw_path.parent/entry['file']
    assert sha(path)==entry['sha256']
    with np.load(path,allow_pickle=False) as report:
        steps,force,before,after = (report[k] for k in ('substep','normal_force_n','orientation_before','orientation_after'))
    np.testing.assert_array_equal(steps,np.arange(501,876))
    np.testing.assert_array_equal(before[1:],after[:-1])
    assert np.isfinite(force).all() and (force>=0).all()
    pairs = read(ROOT/spec['archives']['pair_matches'])['release_support']['matches']
    matched = {row['missed']['world']:row['admitted']['world'] for row in pairs if row['missed']['world'] in spec['worlds']}
    assert sorted(matched)==spec['worlds'] and len(set(matched.values()))==len(matched)
    weight = metadata['nominal_weight_n']
    windows={}
    for a,b in spec['windows_after_release_ms']:
        select = slice(a//spec['motor_update_ms'],b//spec['motor_update_ms'])
        rows=[]
        for lost,kept in matched.items():
            missed,admitted = (reading(force[select,i],before[select,i],after[select,i],.02*weight,weight) for i in (lost,kept))
            delta={key:admitted[key]-missed[key] for key in admitted}
            rows.append(dict(missed_world=lost,admitted_world=kept,missed=missed,admitted=admitted,admitted_minus_missed=delta))
        summary={}
        for key in rows[0]['admitted_minus_missed']:
            values=np.asarray([row['admitted_minus_missed'][key] for row in rows])
            summary[key]=dict(median=float(np.median(values)),positive=int((values>0).sum()),negative=int((values<0).sum()))
        pitch=summary['upright_progress_deg']
        share=summary['left_minus_right_fraction_of_contact_impulse']
        passes = pitch['median']>=spec['minimum_pair_median_upright_deficit_deg'] and pitch['positive']>=spec['matched_pair_minimum']
        consistent = max(share['positive'],share['negative'])>=spec['matched_pair_minimum']
        passes &= consistent and abs(share['median'])>=spec['minimum_pair_median_left_right_load_share_difference']
        windows[f'{a}-{b}'] = dict(summary=summary,pairs=rows,support_shift_window=bool(passes))
    selected = next((name for name,row in windows.items() if row['support_shift_window']),None)
    result = dict(status='verified-development-load-readback',protocol_sha256=sha(PLAN),windows=windows,
        first_support_shift_window_ms=selected,raw_sha256=sha(path),additional_physical_steps=0,learning_updates=0,
        low_friction_reserved_seed_used=False,hardware_io=False)
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),earliest_support_shift_window_ms=selected,
            report=rel(REPORT/'README.md'))
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        task.update(info,next_gate='Verify actual force and motor changes before selecting a matched early transfer intervention.' if selected else
            'Compare early pitch and head unloading without a left-right load-shift assumption, then select the next preflight.')
        project.update(info)
        write(PROJECT,project)
    print('EARLY_LEFT_TURN_LOAD_VERIFIED',sha(OUTPUT),selected,
        {name:(row['summary']['upright_progress_deg'],row['summary']['left_minus_right_fraction_of_contact_impulse'])
         for name,row in windows.items()},flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','score','check'))
    option=parser.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
