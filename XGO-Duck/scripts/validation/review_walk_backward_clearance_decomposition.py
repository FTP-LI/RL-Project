"""Attribute archived sole clearance changes using exact mixed-pose geometry."""
from math import factorial
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
import torch

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import arrays, onset, position_only, SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_clearance_decomposition.json'
OUT = ROOT/'records/reports/common-walk-backward-clearance-decomposition'


def effects(geometry, before, after, stance, groups, tolerance):
    """Average each marginal over every order, including mixed-pose interactions."""
    count = len(groups)
    values = np.empty((1<<count,2))
    for mask in range(len(values)):
        pose = before.copy()
        for index,coords in enumerate(groups.values()):
            if mask&(1<<index):
                pose[coords] = after[coords]
        measure = position_only(geometry,pose,stance)
        values[mask] = measure['swing_clearance_mm'],measure['stance_clearance_mm']
    whole = position_only(geometry,after,stance)
    end = np.array([whole['swing_clearance_mm'],whole['stance_clearance_mm']])
    if count==3:
        np.testing.assert_allclose(values[-1],end,rtol=0,atol=tolerance)
    contributions = np.zeros((count,2))
    for index in range(count):
        bit = 1<<index
        for mask in range(len(values)):
            if not mask&bit:
                size = mask.bit_count()
                weight = factorial(size)*factorial(count-size-1)/factorial(count)
                contributions[index] += weight*(values[mask|bit]-values[mask])
    np.testing.assert_allclose(contributions.sum(axis=0),values[-1]-values[0],rtol=0,atol=tolerance)
    return values,contributions,end


def stage_rows(geometry, trace, old, stance, spec, detailed):
    frames = {row['offset']:row for row in old['frames']}
    q = {offset:trace['qpos'][frame['control']-1,old['world']] for offset,frame in frames.items()}
    saved = {offset:np.array([row['swing_clearance_mm'],row['stance_clearance_mm']]) for offset,row in frames.items()}
    groups = dict(zip(spec['group_factors'],(slice(0,3),slice(3,7),slice(7,None)),strict=True))
    swing = 'right' if stance=='left' else 'left'
    limbs = {name:np.array([geometry.model.joint(f'robot/{swing}_{name}').qposadr.item()])
             for name in spec['individual_swing_factors'][2:]}
    detail_groups = dict(root_translation=slice(0,3),root_orientation=slice(3,7),**limbs)
    tol = spec['closure_tolerance_mm']
    windows = {}
    for name,(a,b) in spec['windows_controls'].items():
        values,coefficients,end = effects(geometry,q[a],q[b],stance,groups,tol)
        np.testing.assert_allclose(values[0],saved[a],rtol=0,atol=tol)
        np.testing.assert_allclose(end,saved[b],rtol=0,atol=tol)
        change = saved[b]-saved[a]
        closure = float(np.max(abs(coefficients.sum(axis=0)-change)))
        row = dict(total_swing_mm=float(change[0]),total_stance_mm=float(change[1]),
                   swing_mm=dict(zip(groups,coefficients[:,0].tolist(),strict=True)),
                   stance_mm=dict(zip(groups,coefficients[:,1].tolist(),strict=True)),
                   closure_mm=closure)
        if detailed:
            combinations,detail,whole = effects(geometry,q[a],q[b],stance,detail_groups,tol)
            # The opposite leg and head cannot move this sole at a fixed root.
            np.testing.assert_allclose(combinations[-1,0],whole[0],rtol=0,atol=tol)
            np.testing.assert_allclose(detail[:,0].sum(),change[0],rtol=0,atol=tol)
            row['individual_swing_mm'] = dict(zip(detail_groups,detail[:,0].tolist(),strict=True))
        windows[name] = row
    return dict(world=old['world'],windows=windows)


def describe(rows, stage, key):
    values = np.asarray([row['windows'][stage][key] for row in rows])
    return np.quantile(values,[.25,.5,.75]).tolist()


def components(rows, stage, name, factors):
    return {factor:np.quantile([row['windows'][stage][name][factor] for row in rows],[.25,.5,.75]).tolist()
            for factor in factors}


def run(seed, cohort, natural_spec, old, geometry, scene, spec):
    proof_path = ROOT/cohort['source']
    assert sha(proof_path)==cohort['source_sha256']
    proof = read(proof_path)
    assert proof['status']=='passed-integrity' and proof['seed']==seed
    assert not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    fixed_scene = proof_path.with_name('cpu-scene-fields.npz')
    assert sha(fixed_scene)==natural_spec['scene_sha256']
    with np.load(fixed_scene) as saved:
        compare_arrays(scene,{name:saved[name] for name in saved.files})
    role = natural_spec['reference_role']
    path = record/f'trace-{role}.npz'
    branch = arrays(path,old['trace_sha256'])
    if 'scores' in proof:
        prefix = arrays(record/'trace-shared-prefix.npz',proof['shared_prefix_sha256'])
        branch = {key:np.concatenate((prefix[key],branch[key])) for key in branch}
    assert branch['qpos'].shape[0]==300
    events = []
    for entry in old['events']:
        stance = entry['stance']
        start = stage_rows(geometry,branch,entry['start'],stance,spec,True)
        stalled = [stage_rows(geometry,branch,row,stance,spec,False) for row in entry['matched_stalls']]
        events.append(dict(stance=stance,start=start,matched_stalls=stalled))
    starts = [row['start'] for row in events]
    stalls = [row for event in events for row in event['matched_stalls']]
    assert len(starts)==old['late_starters'] and len(stalls)==old['aligned_stall_windows']
    summary = {}
    for stage in spec['windows_controls']:
        paired = [event['start']['windows'][stage]['total_swing_mm']-
                  float(np.median([row['windows'][stage]['total_swing_mm'] for row in event['matched_stalls']]))
                  for event in events]
        summary[stage] = dict(
            starter_swing_change_mm=describe(starts,stage,'total_swing_mm'),
            stalled_swing_change_mm=describe(stalls,stage,'total_swing_mm'),
            starter_minus_aligned_stall_median_mm=np.quantile(paired,[.25,.5,.75]).tolist(),
            starter_components_mm=components(starts,stage,'swing_mm',spec['group_factors']),
            stalled_components_mm=components(stalls,stage,'swing_mm',spec['group_factors']),
            starter_individual_swing_mm=components(starts,stage,'individual_swing_mm',spec['individual_swing_factors']))
    return dict(seed=seed,late_starters=len(starts),aligned_stall_windows=len(stalls),summary=summary,events=events)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-kinematics-only'
    assert not any(spec[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==spec['stage'] and ROOT/stage['current_protocol']==PLAN
    for key in ('natural_evidence','geometry_reader'):
        path,expected = spec[key]
        assert sha(ROOT/path)==expected
    source = read(ROOT/spec['natural_evidence'][0])
    natural_plan = ROOT/spec['natural_protocol']
    assert source['protocol_sha256']==sha(natural_plan)
    assert source['status']=='passed-natural-clearance-review' and not source['heldout_used']
    natural = read(natural_plan)
    transfer_path,expected = natural['transfer_source']
    assert sha(ROOT/transfer_path)==expected
    transfer = read(ROOT/transfer_path)
    base = read(onset.PLAN)
    cfg,_ = onset.config(onset.environment(base,spec['seeds'][0]))
    model = Scene(cfg.scene,device='cpu').compile()
    geometry,scene = SoleGeometry(model),model_arrays(model)
    groups = zip(spec['seeds'],transfer['cohorts'],source['cohorts'],strict=True)
    results = [run(seed,cohort,natural,old,geometry,scene,spec) for seed,cohort,old in groups]
    assert sum(row['late_starters'] for row in results)==59
    max_error = max(record['closure_mm'] for batch in results for event in batch['events']
                    for case in (event['start'],*event['matched_stalls']) for record in case['windows'].values())
    assert max_error<=spec['closure_tolerance_mm'] and not torch.cuda.is_initialized()
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-clearance-decomposition',protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),quantiles=[.25,.5,.75],cohorts=results,max_closure_mm=max_error,
          physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_CLEARANCE_DECOMPOSITION',[(row['seed'],{window:{
        name:values[1] for name,values in part['starter_components_mm'].items()}
        for window,part in row['summary'].items()}) for row in results],max_error,flush=True)


if __name__=='__main__':
    main()
