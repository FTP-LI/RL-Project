"""Prepare successful-state replay and analytic CPU checks; no PPO runner.

Keep only original official end-hold successes and inputs before8s. All
requested poses and fixed input times remain explicit, enabling balanced
sampling. Evidence comes from saved development data, never the roll heldout.
Toy gradients verify the objective and detached teacher, not policy learning.
"""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_teacher_anchor import mean_anchor_loss,anchored_objective,balanced_indices

OUT=ROOT/'records/reports/getup-teacher-anchor'


def main():
    assert not OUT.exists()
    draftpath=ROOT/'configs/experiments/getup_teacher_anchor_draft.json';d=read(draftpath)
    a=read(ROOT/d['basis']);assert a['status']=='passed-saved-data-only'
    source=ROOT/d['replay']['source'];assert sha(source)==a['archive_sha256']
    basis=read(ROOT/'configs/experiments/getup_policy_drift_audit.json')
    original=read(ROOT/basis['basis']);trace=ROOT/original['record']/'coverage-trace.npz'
    assert sha(trace)==original['trace_sha256']==a['source_trace_sha256']
    assert sha(ROOT/d['teacher_model'])==d['teacher_sha256']==a['models']['original']['sha256']
    with np.load(trace) as f:poses=f['requested_pose'].copy()
    with np.load(source) as f:
        held=f['original_end_hold'].copy();times=f['input_state_times'].copy();phase=times<8.
        inputs=f['inputs'][phase][:,held].reshape(-1,61).copy()
        targets=f['original_mean'][phase][:,held].reshape(-1,14).copy()
    times=times[phase];assert len(times)==8 and int(held.sum())==295 and len(inputs)==2360
    pose=np.broadcast_to(poses[held],(len(times),int(held.sum()))).reshape(-1).copy()
    when=np.repeat(times,int(held.sum()))
    names=['prone','supine','left-side','right-side'];assert set(pose)==set(names)
    cells={f'{p}|{t:g}':np.flatnonzero((pose==p)&(when==t)).tolist() for p in names for t in times}
    assert len(cells)==32 and all(len(v)>0 for v in cells.values())
    mean=torch.full((2,14),.1,requires_grad=True);target=torch.zeros(2,14,requires_grad=True)
    loss=mean_anchor_loss(mean,target,.1);torch.testing.assert_close(loss,torch.tensor(7.))
    loss.backward();torch.testing.assert_close(mean.grad,torch.full_like(mean,5.));assert target.grad is None
    exact=target.detach().clone().requires_grad_();zero=mean_anchor_loss(exact,target,.1)
    zero.backward();assert zero.item()==0 and not exact.grad.any()
    policy=torch.tensor(.2,requires_grad=True);assert anchored_objective(policy,loss,0.) is policy
    global_rng=torch.get_rng_state().clone();generator=torch.Generator(device='cpu');generator.manual_seed(43031)
    selected=balanced_indices(cells,generator,8);assert len(selected)==256
    for name,indices in cells.items():assert int(torch.isin(selected,torch.tensor(indices)).sum())==8,name
    assert torch.equal(global_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    OUT.mkdir();file=OUT/'anchor-data.npz'
    np.savez_compressed(file,observations=inputs,teacher_targets=targets,requested_pose=pose,input_state_seconds=when)
    write(OUT/'cells.json',cells)
    proof=dict(status='passed-cpu-preparation-only',draft_sha256=sha(draftpath),audit_sha256=sha(ROOT/d['basis']),
        observations=2360,input_dimensions=61,action_dimensions=14,cells=32,
        cell_sizes={k:len(v) for k,v in cells.items()},times_seconds=times.tolist(),original_success_worlds=295,
        teacher_sha256=d['teacher_sha256'],data_sha256=sha(file),cells_sha256=sha(OUT/'cells.json'),
        source_archive_sha256=sha(source),analytic_loss=7.,analytic_gradient=5.,teacher_gradient_absent=True,
        coefficient0_tensor_identity=True,exact_teacher_zero_gradient=True,
        balanced_batch_size=256,cpu_global_rng_unchanged=True,cuda_initialized=False,
        physics_steps=0,optimizer_steps=0,ppo_integration_pending=True,matched_value_admission_pending=True,
        dynamic_protocol_frozen=False,hardware_io=False,
        helper_sha256=sha(ROOT/'scripts/training/getup_teacher_anchor.py'),script_sha256=sha(Path(__file__)))
    write(OUT/'preparation.json',proof)
    print('GETUP_TEACHER_ANCHOR_CPU_PREPARED',len(inputs),'samples,32balancedcells; no learning/physics',flush=True)


if __name__=='__main__':main()
