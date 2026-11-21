"""Verify the supported-upright scoring clock on the local CUDA device."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import review_nominal_low_friction_loaded_upright_progress as previous
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write
from review_nominal_low_friction_first_support_effort import tilt


KEY='nominal_low_friction_loaded_upright_device'
STAGE='nominal-low-friction-loaded-upright-device'
PLAN=ROOT/'configs/experiments'/f'{KEY}.json'
OUT=previous.OUTPUT.with_name('loaded-upright-device.json')
MODULE=ROOT/'scripts/training/getup_loaded_upright_progress.py'


def freeze():
    project=read(PROJECT)
    permitted(project,previous.STAGE)
    goal=project['goal_execution']
    earlier=goal[previous.KEY]
    assert earlier['reward_window_ms']=='0-80' and earlier['status']=='reviewed'
    source=read(ROOT/earlier['protocol'],earlier['protocol_sha256'])
    archive=source['load_trace']
    metadata=source['load_metadata']
    assert KEY not in goal and not PLAN.exists() and not OUT.exists()
    sources={earlier[k]:earlier[k+'_sha256'] for k in ('protocol','result')}
    sources.update({k:source['sources'][k] for k in (archive,metadata)})
    sources[rel(MODULE)]=sha(MODULE)
    write(PLAN,dict(stage=STAGE,sources=sources,script_sha256=sha(Path(__file__)),source=earlier['protocol'],
        source_result=earlier['result'],archive=archive,metadata=metadata,worlds=896,shape=dict(
        foot_fraction=.02,both_feet_fraction=.5,settled_reads=3,window_reads=20,maximum_tick_progress_deg=1.5),
        first_read_elapsed_ms=400,maximum_absolute_progress_error_deg=.0002,
        maximum_aggregate_error_deg=.001,minimum_positive_pairs=10,minimum_positive_focus=3,minimum_bounded_median_gap_deg=3.,
        physical_steps=0,learning_updates=0,hardware_io=False,holdout_seed=2246))
    goal[KEY]=dict(status='device-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info=dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    project['staged_training_request'].update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Compare CUDA accumulated progress, exact first-load clock and bounded reward against saved 4 ms samples.')
    write(PROJECT,project)
    print('LOADED_UPRIGHT_DEVICE_FROZEN',sha(PLAN),flush=True)


def reward_reference(loads,before,after,settings,weight):
    right=loads[...,3]>weight*settings['foot_fraction']
    spans=np.lib.stride_tricks.sliding_window_view(right,settings['settled_reads'],axis=0).all(axis=-1)
    first=np.where(spans.any(axis=0),spans.argmax(axis=0)+settings['settled_reads']-1,-1)
    ticks=np.arange(len(loads))[:,None]
    matches=ticks==first
    active=(first>=0)&(ticks>first)&(ticks<=first+settings['window_reads'])
    left=loads[...,2]
    sum_feet=loads[...,2]+loads[...,3]
    supported=(left>weight*settings['foot_fraction'])&right&(sum_feet>=weight*settings['both_feet_fraction'])
    progress=(tilt(before)-tilt(after))*np.clip(sum_feet/weight,0.,1.)
    raw=np.where(active&supported,progress,0.)
    bounded=np.clip(raw/settings['maximum_tick_progress_deg'],-1.,1.)/settings['window_reads']
    return raw,bounded,matches,active


def evaluate(spec):
    import torch
    from getup_loaded_upright_progress import LoadedUprightProgress

    assert torch.cuda.is_available()
    device=torch.device('cuda:0')
    metadata=read(ROOT/spec['metadata'])
    assert metadata['contact_channels']==['ground','head','left_foot','right_foot']
    weight=metadata['nominal_weight_n']
    published=read(ROOT/spec['source_result'])
    source=read(ROOT/spec['source'])
    shape=spec['shape']
    start=spec['first_read_elapsed_ms']//source['substep_ms']
    with np.load(ROOT/spec['archive'],allow_pickle=False) as data:
        clocks,loads,before,after=(data[k][start:] for k in ('substep','normal_force_n','orientation_before','orientation_after'))
    np.testing.assert_array_equal(clocks,np.arange(501+start,876))
    assert loads.shape==(len(clocks),spec['worlds'],4)
    np.testing.assert_array_equal(before[1:],after[:-1])
    reference=reward_reference(loads,before,after,shape,weight)
    clock=LoadedUprightProgress(spec['worlds'],device,body_weight=weight,**shape)
    forces,q_before,q_after=(torch.from_numpy(x).to(device=device) for x in (loads,before,after))
    enabled=torch.ones(spec['worlds'],dtype=torch.bool,device=device)
    with torch.inference_mode():
        entries=[clock.advance(f,q,p,enabled) for f,q,p in zip(forces,q_before,q_after,strict=True)]
        groups=[torch.stack(items).cpu().numpy() for items in zip(*entries,strict=True)]
        reset=torch.arange(spec['worlds'],device=device)%2==0
        clock.reset(reset)
        _,after_reset,first,active=clock.advance(forces[0],q_before[0],q_after[0],reset)
        assert not first.any() and not active.any() and not after_reset.any()
        assert (clock.elapsed[reset]<0).all()
    raw,unit,confirmed,active=groups
    baseline,unit_ref,first_ref,active_ref=reference
    np.testing.assert_array_equal(confirmed,first_ref)
    np.testing.assert_array_equal(active,active_ref)
    error=float(abs(raw-baseline).max())
    aggregate=float(abs(raw.sum(axis=0)-baseline.sum(axis=0)).max())
    assert error<=spec['maximum_absolute_progress_error_deg'] and aggregate<=spec['maximum_aggregate_error_deg']
    assert (abs(unit).sum(axis=0)<=1.+1e-6).all()
    np.testing.assert_allclose(unit,unit_ref,atol=spec['maximum_absolute_progress_error_deg']/
        (shape['maximum_tick_progress_deg']*shape['window_reads']),rtol=0)
    gains=[]
    unbounded=raw.sum(axis=0)
    bounded=unit.sum(axis=0)*(shape['maximum_tick_progress_deg']*shape['window_reads'])
    for pair in published['pairs']:
        miss,admitted=pair['missed_id'],pair['admitted_id']
        for label,index in (('missed',miss),('admitted',admitted)):
            expected=pair[label]['windows']['0-80']['supported_upright_progress_deg']
            assert abs(float(unbounded[index])-expected)<=spec['maximum_aggregate_error_deg']
            endpoint=np.flatnonzero(confirmed[:,index])
            assert len(endpoint)==1 and int((endpoint[0]+start+1)*source['substep_ms'])==pair[label]['right_confirmation_since_release_ms']
        gains.append(dict(miss=miss,admitted=admitted,focus=pair['focus'],
            bounded_paired_progress_difference_deg=float(bounded[admitted]-bounded[miss])))
    positive=sum(row['bounded_paired_progress_difference_deg']>0. for row in gains)
    focus=sum(row['focus'] and row['bounded_paired_progress_difference_deg']>0. for row in gains)
    median=float(np.median([row['bounded_paired_progress_difference_deg'] for row in gains]))
    coverage=(positive>=spec['minimum_positive_pairs'] and focus>=spec['minimum_positive_focus']
              and median>=spec['minimum_bounded_median_gap_deg'])
    return dict(status='verified-archived-cuda-reward-arithmetic',protocol_sha256=sha(PLAN),real_cuda_device=torch.cuda.get_device_name(device),
        worlds=spec['worlds'],archived_examples=int(np.prod(loads.shape[:2])),confirmed_clocks=int(confirmed.sum()),
        first_and_last_reward_reads_exact=True,partial_reset_verified=True,maximum_tick_error_deg=error,maximum_accumulated_error_deg=aggregate,
        positive_pairs=positive,positive_focus_pairs=focus,median_bounded_progress_difference_deg=median,pairs=gains,
        eligible_for_live_sampler=bool(coverage),physics_steps=0,learning_updates=0,hardware_io=False,holdout_used=False)


def score(check):
    project=read(PROJECT)
    permitted(project,STAGE)
    node=project['goal_execution'][KEY]
    assert node['status']==('reviewed' if check else 'device-ready')
    spec=read(PLAN,node['protocol_sha256'])
    assert sha(Path(__file__))==spec['script_sha256']
    for path,checksum in spec['sources'].items():
        assert sha(ROOT/path)==checksum,path
    proof=evaluate(spec)
    supported=proof['eligible_for_live_sampler']
    if check:
        assert read(OUT,node['result_sha256'])==proof
    else:
        write(OUT,proof)
        node.update(status='reviewed',result=rel(OUT),result_sha256=sha(OUT),live_sampler_admitted=supported,
            report=rel(OUT.parent/'README.md'))
        info=dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUT))
        project.update(info)
        project['staged_training_request'].update(info,next_gate=(
            'Verify the live solved-contact reward bridge and benchmark the paired training sampler at selected local capacity.' if supported
            else 'Review bounded reward on the original paired signals before connecting it to training.'))
        write(PROJECT,project)
    print('LOADED_UPRIGHT_CUDA_VERIFIED',sha(OUT),proof['archived_examples'],proof['maximum_tick_error_deg'],
        proof['positive_pairs'],proof['positive_focus_pairs'],proof['median_bounded_progress_difference_deg'],supported,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    option=cli.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
