"""Read back the bounded actuator waveform against real saved walking commands."""
from pathlib import Path

import numpy as np
import torch

from collect_walk_backward_lateral_probe import OUT,PLAN,SEEDS
from record_io import write
from s1_reference_inputs import ROOT,read,sha


DRAFT = ROOT/'configs/experiments/common_walk_backward_actuator_support_draft.json'


def pulse_weight(step,start,up,hold,down):
    age = step-start-1
    if age<=0:
        return 0.
    if age<=up:
        return age/up
    if age<=up+hold:
        return 1.
    return max(0.,1.-(age-up-hold)/down)


def apply_support_target(nominal,shift,limits,weight):
    return torch.maximum(torch.minimum(nominal+weight*shift,limits[:,1]),limits[:,0])


def main():
    target = OUT/'actuator-target-preflight.json'
    assert not target.exists() and not torch.cuda.is_initialized()
    plan = read(DRAFT)
    geometry_path = ROOT/plan['source']
    assert sha(geometry_path)==plan['source_sha256']
    geometry = read(geometry_path)
    assert geometry['status']=='passed-saved-bounded-kinematics' and not geometry['heldout_used']
    spec = read(PLAN)
    bounds = torch.tensor(spec['environment']['joint_limits'],dtype=torch.float32)
    names = spec['environment']['joint_names']
    original_geometry = OUT/'kinematics-preflight.json'
    assert sha(original_geometry)==geometry['source_sha256']
    legs = [names.index(name) for name in read(original_geometry)['joints']]
    settings = plan['design']
    start = settings['start_control']
    spans = [settings[f'{phase}_controls'] for phase in ('ramp_up','hold','ramp_down')]
    weights = [pulse_weight(step,start,*spans) for step in range(start+1,start+sum(spans)+3)]
    assert weights[0]==weights[-1]==0. and max(weights)==1.
    assert np.max(abs(np.diff(weights)))<=1/min(spans)+1e-12
    cap = settings['maximum_relative_joint_target_rad']
    receipts = []
    for cohort,seed in zip(geometry['cohorts'],SEEDS,strict=True):
        assert cohort['seed']==seed
        proof = read(OUT/f'seed{seed}-analysis.json')
        record = ROOT/proof['record']
        trace = record/'trace-backward.npz'
        assert sha(trace)==proof['trace_sha256']['backward']
        ids = [case['world'] for case in cohort['rows']]
        assert ids==proof['common_near_static']
        with np.load(trace) as archive:
            baseline = torch.from_numpy(archive['applied_joint_targets'][:len(weights),ids].copy())
            action = torch.from_numpy(archive['actions'][:len(weights),ids].copy())
        baseline_original = baseline.clone()
        raw_original = action.clone()
        by_side = {}
        for side in ('left','right'):
            offsets = torch.zeros_like(baseline[0])
            offsets[:,legs] = torch.tensor([case[side]['target_rad'] for case in cohort['rows']],dtype=baseline.dtype)
            assert all(case[side]['feasible'] for case in cohort['rows']) and offsets.abs().max()<=cap+1e-8
            sent = torch.stack([apply_support_target(base,offsets,bounds,weight)
                                for base,weight in zip(baseline,weights,strict=True)])
            correction = sent-baseline
            assert sent.isfinite().all() and (sent>=bounds[:,0]).all() and (sent<=bounds[:,1]).all()
            assert correction.abs().max()<=cap+1e-7
            torch.testing.assert_close(sent[[0,-1]],baseline[[0,-1]],atol=0,rtol=0)
            skipped = sorted(set(range(len(names)))-set(legs))
            torch.testing.assert_close(sent[:,:,skipped],baseline[:,:,skipped],atol=0,rtol=0)
            residual = (baseline+torch.tensor(weights)[:,None,None]*offsets-sent).abs()
            by_side[side] = dict(maximum_applied_delta_rad=float(correction.abs().max()),
                max_projection_residual_rad=float(residual.max()),
                rows_with_projection=int((residual>1e-7).any(dim=-1).sum()))
        torch.testing.assert_close(baseline,baseline_original,atol=0,rtol=0)
        torch.testing.assert_close(action,raw_original,atol=0,rtol=0)
        receipts.append(dict(seed=seed,worlds=ids,trace_sha256=sha(trace),directions=by_side))
    write(target,dict(status='passed-saved-actuator-targets',draft_sha256=sha(DRAFT),source_sha256=sha(Path(__file__)),
        weights=weights,cohorts=receipts,model_input_and_raw_action_unchanged=True,physical_substeps=0,
        learning_updates=0,heldout_used=False))
    print('BACKWARD_ACTUATOR_TARGETS',[(r['seed'],r['directions']) for r in receipts],flush=True)


if __name__=='__main__':
    main()
