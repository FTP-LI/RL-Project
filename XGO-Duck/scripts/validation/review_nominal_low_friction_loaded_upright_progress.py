"""Check an early supported-upright reward before another paired training run."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import replay_nominal_low_friction_early_hip_roll as replay
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_first_support_effort import tilt


KEY='nominal_low_friction_loaded_upright_progress'
STAGE='nominal-low-friction-loaded-upright-progress'
PLAN=ROOT/'configs/experiments'/f'{KEY}.json'
OUTPUT=replay.authority.REPORT/'loaded-upright-progress.json'


def freeze():
    project=read(PROJECT)
    permitted(project,replay.STAGE)
    goal=project['goal_execution']
    last=goal[replay.KEY]
    auth=goal[replay.authority.KEY]
    assert read(ROOT/last['result'],last['result_sha256'])['eligible_for_development_followup'] is False
    assert KEY not in goal and not PLAN.exists() and not OUTPUT.exists()
    authority=read(ROOT/auth['protocol'],auth['protocol_sha256'])
    sources={node[k]:node[k+'_sha256'] for node in (auth,last) for k in ('protocol','result')}
    for label in ('load_trace','load_metadata','effort_result','prior_experiment'):
        path=authority[label]
        sources[path]=authority['sources'][path]
    write(PLAN,dict(stage=STAGE,sources=sources,script_sha256=sha(Path(__file__)),authority_protocol=auth['protocol'],
        load_trace=authority['load_trace'],load_metadata=authority['load_metadata'],effort_result=authority['effort_result'],
        first_search_elapsed_ms=400,settled_right_reads=3,substep_ms=4,pairs=13,focus=authority['focus_missed_worlds'],
        windows_relative_to_right_confirmation_ms=[[0,80],[80,160]],foot_load_fraction=.02,combined_foot_load_fraction=.5,
        minimum_improved_pairs=10,minimum_improved_focus=3,minimum_median_paired_supported_progress_difference_deg=3.,
        shape='Signed upright-angle progress weighted by min(total foot load/body weight,1) while both feet support.',
        reserved_seed=2246,physical_steps=0,learning_updates=0,hardware_io=False))
    goal[KEY]=dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify contact-conditioned upright-angle progress relative to the first sustained right-foot load.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_PROGRESS_FROZEN',sha(PLAN),flush=True)


def repair_freeze():
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']=='archive-ready' and not OUTPUT.exists()
    spec=read(PLAN,node['protocol_sha256'])
    folder=OUTPUT.parent
    plan=folder/'loaded-upright-progress-protocol-before-fix.json'
    script=folder/'loaded-upright-progress-script-before-fix.py.txt'
    log=folder/'loaded-upright-progress-command.log'
    assert sha(plan)==node['protocol_sha256'] and sha(script)==spec['script_sha256']
    assert 'TypeError: Object of type int64 is not JSON serializable' in log.read_text(encoding='utf-8')
    spec.update(script_sha256=sha(Path(__file__)),serialization_correction={rel(p):sha(p) for p in (plan,script,log)})
    write(PLAN,spec)
    node['protocol_sha256']=sha(PLAN)
    write(PROJECT,project)
    print('LOADED_UPRIGHT_PROGRESS_SERIALIZATION_CORRECTION',sha(PLAN),flush=True)


def measure(spec,weight,forces,before,after,world):
    start=spec['first_search_elapsed_ms']//spec['substep_ms']
    use=forces[:,world,1:]
    head,left,right=use.T
    threshold=spec['foot_load_fraction']*weight
    n=spec['settled_right_reads']
    runs=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(right[start:]>threshold,n).all(axis=-1))
    assert len(runs)>0,world
    first=int(start+runs[0]+n-1)
    progress=tilt(before[:,world])-tilt(after[:,world])
    total=left+right
    supported=(left>threshold)&(right>threshold)&(total>=spec['combined_foot_load_fraction']*weight)
    values={}
    for a,b in spec['windows_relative_to_right_confirmation_ms']:
        start=first+1+a//spec['substep_ms']
        stop=first+1+b//spec['substep_ms']
        window=slice(start,stop)
        assert stop<=len(progress)
        controlled=supported[window]*np.minimum(total[window]/weight,1.)
        values[f'{a}-{b}']=dict(supported_upright_progress_deg=float((controlled*progress[window]).sum()),
            unweighted_upright_progress_deg=float(progress[window].sum()),
            qualifying_load_reads=int(supported[window].sum()),supported_phase_reads=stop-start,
            relative_head_impulse=float(head[window].sum()/max((head[window]+total[window]).sum(),1e-9)))
    return dict(right_confirmation_since_release_ms=int((first+1)*spec['substep_ms']),windows=values)


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'archive-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,checksum in spec.get('serialization_correction',{}).items():
        assert sha(ROOT/path)==checksum,path
    for path,checksum in spec['sources'].items():
        assert sha(ROOT/path)==checksum,path
    metadata=read(ROOT/spec['load_metadata'])
    assert metadata['contact_channels']==['ground','head','left_foot','right_foot']
    data=read(ROOT/spec['effort_result'])
    assert (data['seed'],data['foot_sliding_friction'])==(1257,.8)
    pairs=data['release_support']['matches']
    assert len(pairs)==spec['pairs'] and len({pair['admitted']['world'] for pair in pairs})==len(pairs)
    with np.load(ROOT/spec['load_trace'],allow_pickle=False) as a:
        steps,forces,before,after=(a[k] for k in ('substep','normal_force_n','orientation_before','orientation_after'))
    np.testing.assert_array_equal(steps,np.arange(501,876))
    np.testing.assert_array_equal(before[1:],after[:-1])
    rows=[]
    for pair in pairs:
        miss,good=(pair[label]['world'] for label in ('missed','admitted'))
        row={label:measure(spec,metadata['nominal_weight_n'],forces,before,after,i)
             for label,i in (('missed',miss),('admitted',good))}
        rows.append(dict(missed_id=miss,admitted_id=good,focus=miss in spec['focus'],**row))
    timing=np.asarray([r['missed']['right_confirmation_since_release_ms']-r['admitted']['right_confirmation_since_release_ms'] for r in rows])
    assert sum(row['focus'] for row in rows)==len(spec['focus'])
    comparisons={}
    selected=None
    for w in rows[0]['missed']['windows']:
        deltas={k:[r['admitted']['windows'][w][k]-r['missed']['windows'][w][k] for r in rows]
            for k in ('supported_upright_progress_deg','unweighted_upright_progress_deg','qualifying_load_reads','relative_head_impulse')}
        progress=np.asarray(deltas['supported_upright_progress_deg'])
        focused=progress[[r['focus'] for r in rows]]
        gap=float(np.median(progress))
        ready=(int((progress>0.).sum())>=spec['minimum_improved_pairs'] and int((focused>0.).sum())>=spec['minimum_improved_focus']
               and gap>=spec['minimum_median_paired_supported_progress_difference_deg'])
        comparisons[w]=dict(by_measure={k:dict(paired_median=float(np.median(v)),positive_pairs=int((np.asarray(v)>0.).sum()),
            differences=v) for k,v in deltas.items()},positive_focus_pairs=int((focused>0.).sum()),passes=bool(ready))
        if ready and selected is None:
            selected=w
    result=dict(status='verified-matched-loaded-upright-progress',protocol_sha256=sha(PLAN),comparisons=comparisons,pairs=rows,
        original_four_ms_first_earlier_admitted=int((timing>0.).sum()),first_load_time_difference_ms=timing.tolist(),
        recommended_reward_window_ms=selected,physical_steps=0,learning_updates=0,holdout_used=False,hardware_io=False)
    if check:
        assert read(OUTPUT,node['result_sha256'])==result
    else:
        write(OUTPUT,result)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),reward_window_ms=selected,
            report=rel(replay.authority.REPORT/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Verify the selected loaded-upright term on GPU in the same parent training interface, then freeze a matched training comparison.'
            if selected else 'Use phase-specific right-foot support and original terminal posture to select a better targeted reset-and-reward objective.'))
        write(PROJECT,project)
    print('LOADED_UPRIGHT_PROGRESS_VERIFIED',sha(OUTPUT),result['original_four_ms_first_earlier_admitted'],selected,
        {w:(r['by_measure']['supported_upright_progress_deg']['paired_median'],r['by_measure']['supported_upright_progress_deg']['positive_pairs'],r['positive_focus_pairs'])
         for w,r in comparisons.items()},flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','repair-freeze','score','check'))
    option=cli.parse_args().action
    if option in ('freeze','repair-freeze'):
        {'freeze':freeze,'repair-freeze':repair_freeze}[option]()
    else:
        score(option=='check')
