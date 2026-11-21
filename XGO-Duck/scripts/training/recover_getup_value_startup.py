"""Declared zero-step type repair; preserve parent sources and value budget.

torch.testing does not dispatch TorchArray.__torch_function__. Convert only
that exact wrapper to a same-dtype tensor copy in startup comparisons. No
tolerance, physical state, RNG, reward, network, data split or budget changes.
The original0step failure remains a separate immutable record.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import inspect
import json
import shutil
import torch
import getup_value_bootstrap as base
from mjlab.sim.sim_data import TorchArray

ROOT=base.ROOT
PARENT_PLAN=base.PLAN
PARENT_OUT=base.OUT
PLAN=ROOT/'configs/experiments/getup_value_bootstrap_startup_recovery.json'
OUT=PARENT_OUT/'startup-recovery'
original_startup=base.startup
original_assert_close=torch.testing.assert_close


def comparable_close(a,b,**kwargs):
    if isinstance(a,TorchArray):a=a.clone()
    if isinstance(b,TorchArray):b=b.clone()
    return original_assert_close(a,b,**kwargs)


def startup(*args,**kwargs):
    torch.testing.assert_close=comparable_close
    try:return original_startup(*args,**kwargs)
    finally:torch.testing.assert_close=original_assert_close


def freeze():
    from check_getup_value_bootstrap import check
    import warp as wp
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    p=deepcopy(base.read(PARENT_PLAN));old=base.read(PARENT_OUT/'progress.json');record=Path(old['path'])
    assert old['status']=='failed-preserved' and old['completed_updates']==old['actor_updates']==old['control_steps']==0
    assert base.read(record/'runtime.json')['physics_substeps']==0
    assert 'TorchArray' in old['error'] and not (record/'value-data.npz').exists()
    for name,digest in p['source_sha256'].items():assert base.sha(ROOT/name)==digest,name
    # Exercise the actual installed CPU wrapper, including a deliberately wrong
    # value to prove that normalization does not weaken the equality check.
    wrapped=TorchArray(wp.array([1.,2.],dtype=wp.float32,device='cpu'))
    comparable_close(torch.tensor([1.,2.]),wrapped,atol=0,rtol=0)
    rejected=False
    try:comparable_close(torch.tensor([1.,3.]),wrapped,atol=0,rtol=0)
    except AssertionError:rejected=True
    assert rejected and not torch.cuda.is_initialized()
    proof=check(p)
    p['startup_recovery']=dict(parent_protocol=str(PARENT_PLAN.relative_to(ROOT)),parent_protocol_sha256=base.sha(PARENT_PLAN),
        parent_record=str(record.relative_to(ROOT)),parent_runtime_sha256=base.sha(record/'runtime.json'),
        original_physics_steps=0,change='Normalize exact TorchArray wrappers to same-dtype clones only during startup assert_close',
        comparison_tolerances_unchanged=True,budgets_and_learning_semantics_unchanged=True)
    for name in ('scripts/training/recover_getup_value_startup.py','scripts/validation/verify_getup_value_startup_recovery.py'):
        p['source_sha256'][name]=base.sha(ROOT/name)
    p['library_source_sha256'][inspect.getfile(TorchArray)]=base.sha(Path(inspect.getfile(TorchArray)))
    OUT.mkdir()
    shutil.copy2(PARENT_OUT/'cpu-scene-fields.npz',OUT/'cpu-scene-fields.npz')
    base.write(PLAN,p);pre=base.read(PARENT_OUT/'preflight.json')
    pre.update(protocol_sha256=base.sha(PLAN),fixture=proof,startup_type_positive_and_negative_fixture=True,
        parent_preflight_sha256=base.sha(PARENT_OUT/'preflight.json'))
    base.write(OUT/'preflight.json',pre)
    print('VALUE_ZERO_STEP_RECOVERY_FROZEN',base.sha(PLAN),'new_budget=0',flush=True)


def activate():
    base.PLAN=PLAN;base.OUT=OUT;base.startup=startup


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');a=parser.parse_args()
    if a.freeze:freeze()
    else:activate();base.run()
