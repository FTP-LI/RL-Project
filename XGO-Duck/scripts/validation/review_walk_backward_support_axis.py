"""Resolve pre-contact attitude contributions in the initial body frame."""
from math import factorial
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import arrays, onset, position_only, SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_support_axis.json'
OUT = ROOT/'records/reports/common-walk-backward-support-axis'
CONTEXTS = ((False,False,1/3),(True,False,1/6),(False,True,1/6),(True,True,1/3))


def attitude_parts(geometry, before, after, stance):
    initial = Rotation.from_quat(np.roll(before[3:7],-1))
    final = Rotation.from_quat(np.roll(after[3:7],-1))
    vector = (initial.inv()*final).as_rotvec()
    poses = []
    for mask in range(8):
        if mask in (0,7):
            poses.append((before if mask==0 else after)[3:7])
        else:
            partial = vector*np.array([bool(mask&(1<<axis)) for axis in range(3)])
            poses.append(np.roll((initial*Rotation.from_rotvec(partial)).as_quat(),1))
    parts = np.zeros(3)
    for translation,joints,weight in CONTEXTS:
        mixed = before.astype(float).copy()
        if translation:
            mixed[:3] = after[:3]
        if joints:
            mixed[7:] = after[7:]
        clear = []
        for quat in poses:
            mixed[3:7] = quat
            clear.append(position_only(geometry,mixed,stance)['swing_clearance_mm'])
        for axis in range(3):
            bit = 1<<axis
            for mask in range(8):
                if not mask&bit:
                    rank = mask.bit_count()
                    parts[axis] += weight*factorial(rank)*factorial(2-rank)/6*(clear[mask|bit]-clear[mask])
    return np.rad2deg(vector),parts


def compare(geometry, trace, previous, decomposed, stance, offsets, spec):
    frames = {frame['offset']:frame['control']-1 for frame in previous['frames']}
    result = {}
    for window in spec['windows']:
        first,last = (frames[k] for k in offsets[window])
        before,after = trace['qpos'][[first,last],previous['world']]
        vector,parts = attitude_parts(geometry,before,after,stance)
        total = decomposed['windows'][window]['swing_mm']['root_orientation']
        error = abs(float(parts.sum())-total)
        assert error<=spec['closure_tolerance_mm']
        sign = 1 if stance=='right' else -1
        result[window] = dict(roll_toward_stance_deg=float(sign*vector[0]),
             rotation_vector_deg=dict(zip(spec['axis_order'],vector.tolist(),strict=True)),
             contribution_mm=dict(zip(spec['axis_order'],parts.tolist(),strict=True)),closure_mm=error)
    return result


def run(old, input_cohort, prior, natural_spec, geometry, scene, offsets, spec):
    receipt = ROOT/input_cohort['source']
    assert sha(receipt)==input_cohort['source_sha256']
    proof = read(receipt)
    assert proof['status']=='passed-integrity' and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    scene_source = receipt.with_name('cpu-scene-fields.npz')
    assert sha(scene_source)==natural_spec['scene_sha256']
    with np.load(scene_source) as stored:
        compare_arrays(scene,{name:stored[name] for name in stored.files})
    path = record/f'trace-{natural_spec["reference_role"]}.npz'
    trace = arrays(path,old['trace_sha256'])
    if 'scores' in proof:
        prefix = arrays(record/'trace-shared-prefix.npz',proof['shared_prefix_sha256'])
        trace = {key:np.concatenate((prefix[key],trace[key])) for key in trace}
    events = []
    for row,previous in zip(old['events'],prior['events'],strict=True):
        stance = row['stance']
        assert stance==previous['stance'] and row['start']['world']==previous['start']['world']
        start = compare(geometry,trace,row['start'],previous['start'],stance,offsets,spec)
        matched = []
        for case,detail in zip(row['matched_stalls'],previous['matched_stalls'],strict=True):
            assert case['world']==detail['world']
            matched.append(compare(geometry,trace,case,detail,stance,offsets,spec))
        events.append(dict(world=row['start']['world'],stance=stance,start=start,aligned_stalls=matched))
    summary = {}
    for window in spec['windows']:
        chosen = [event['start'][window] for event in events]
        stalled = [row[window] for event in events for row in event['aligned_stalls']]
        summary[window] = {
            group:dict(roll_toward_stance_deg=np.quantile([v['roll_toward_stance_deg'] for v in values],[.25,.5,.75]).tolist(),
                       contribution_mm={axis:np.quantile([v['contribution_mm'][axis] for v in values],[.25,.5,.75]).tolist()
                                        for axis in spec['axis_order']})
            for group,values in (('starters',chosen),('aligned_stalls',stalled))}
    leading = summary['precontact']['starters']
    passed = (leading['roll_toward_stance_deg'][1]>=spec['minimum_seed_median_roll_toward_stance_deg'] and
              leading['contribution_mm']['roll'][1]>=spec['minimum_seed_median_roll_contribution_mm'])
    return dict(seed=old['seed'],worlds=len(events),summary=summary,events=events,roll_design_gate=passed)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-attitude-only'
    assert not any(spec[name] for name in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    active = read(ROOT/'project.json')['staged_training_request']
    assert active['current_stage']==spec['stage'] and ROOT/active['current_protocol']==PLAN
    file,digest = spec['decomposition_evidence']
    assert sha(ROOT/file)==digest
    proof = read(ROOT/file)
    assert proof['status']=='passed-saved-clearance-decomposition' and not proof['heldout_used']
    decomposition = ROOT/spec['decomposition_protocol']
    assert sha(decomposition)==proof['protocol_sha256']
    attribution = read(decomposition)
    natural_file,expected = attribution['natural_evidence']
    assert sha(ROOT/natural_file)==expected
    natural = read(ROOT/natural_file)
    natural_plan = ROOT/attribution['natural_protocol']
    assert sha(natural_plan)==natural['protocol_sha256']
    options = read(natural_plan)
    transfer_path,expected = options['transfer_source']
    assert sha(ROOT/transfer_path)==expected
    transfer = read(ROOT/transfer_path)
    base = read(onset.PLAN)
    cfg,_ = onset.config(onset.environment(base,spec['seeds'][0]))
    model = Scene(cfg.scene,device='cpu').compile()
    geometry,scene = SoleGeometry(model),model_arrays(model)
    cohorts = []
    for seed,cohort,raw,prior in zip(spec['seeds'],transfer['cohorts'],natural['cohorts'],proof['cohorts'],strict=True):
        assert seed==cohort['seed']==raw['seed']==prior['seed']
        cohorts.append(run(raw,cohort,prior,options,geometry,scene,attribution['windows_controls'],spec))
    assert sum(row['worlds'] for row in cohorts)==59 and not torch.cuda.is_initialized()
    accepted = sum(row['roll_design_gate'] for row in cohorts)>=spec['minimum_seed_medians_passing']
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-attitude-review',protocol_sha256=sha(PLAN),
         script_sha256=sha(Path(__file__)),cohorts=cohorts,roll_target_geometry_admitted=accepted,
         dynamic_execution_authorized=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_SUPPORT_AXIS',[(row['seed'],row['summary']['precontact']['starters'],row['roll_design_gate'])
          for row in cohorts],accepted,flush=True)


if __name__=='__main__':
    main()
