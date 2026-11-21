"""Explicit logging-only repair after a one-control failed value attempt.

The frozen parent reads a nonexistent public processed_actions field after
the first physics control. Preserve its1024transitions and zero gradients.
Expose a read-only alias for the installed private tensor; no action math,
reward, reset, noise, RNG or tolerance changes. Freeze a new400control first
episode dataset, separately accounting the failed1024transitions. This is a
fresh measurement, not restoration from an incomplete physics checkpoint.
CPU checks prove exact tensor identity and reject a wrong comparison before
new live startup. Both parent code and evidence remain immutable.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,subprocess,sys
import torch
import getup_interface_value as base
from getup_stand_handoff import HandoffAction

ROOT=base.ROOT
PARENT_PLAN=base.PLAN
PARENT_OUT=base.OUT
PLAN=ROOT/'configs/experiments/getup_interface_value_readback_recovery.json'
OUT=PARENT_OUT/'readback-recovery'


def install_readback():
    assert not hasattr(HandoffAction,'processed_actions')
    HandoffAction.processed_actions=property(lambda self:self._processed_actions)


def freeze():
    from dataclasses import asdict
    from evaluate_roll_reset_pair import differences
    from check_getup_stand_handoff import real_processor
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    p=deepcopy(base.checked());state=base.read(PARENT_OUT/'progress.json');record=Path(state['path'])
    assert state['status']=='failed-preserved' and state['control_steps']==1
    assert state['actor_updates']==state['critic_updates']==0
    assert "no attribute 'processed_actions'" in state['error']
    assert base.read(record/'runtime.json')['control_transitions']==1024
    assert not (record/'value-data.npz').exists()
    cfg=base.prepare(p)[0];before=asdict(cfg)
    install_readback()
    action=HandoffAction.__new__(HandoffAction)
    action._processed_actions=torch.tensor([[1.,2.]],dtype=torch.float32)
    assert action.processed_actions is action._processed_actions
    assert action.processed_actions.data_ptr()==action._processed_actions.data_ptr()
    torch.testing.assert_close(action.processed_actions,torch.tensor([[1.,2.]]),atol=0,rtol=0)
    rejected=False
    try:torch.testing.assert_close(action.processed_actions,torch.tensor([[1.,3.]]),atol=0,rtol=0)
    except AssertionError:rejected=True
    assert rejected and not HandoffAction.processed_actions.fset
    fixture=real_processor(cfg.env,p['model_home'],p['standing_home'])
    assert not differences(before,asdict(base.prepare(p)[0]))
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=base.sha(Path(__file__))
    p['readback_recovery']=dict(parent_protocol=str(PARENT_PLAN.relative_to(ROOT)),
        parent_protocol_sha256=base.sha(PARENT_PLAN),parent_record=str(record.relative_to(ROOT)),
        parent_runtime_sha256=base.sha(record/'runtime.json'),parent_control_steps=1,
        discarded_control_transitions=1024,parent_actor_updates=0,parent_critic_updates=0,
        change='Read-only exact alias processed_actions -> _processed_actions',
        new_measurement_transitions=409600,total_transitions_including_failed_attempt=410624,
        continuation_or_restore=False,configuration_and_tolerances_unchanged=True)
    p.update(run_id='duck_getup_interface-value-readback_reference_s42_'+datetime.now().strftime('%Y%m%dT%H%M%S'),
             frozen_at=datetime.now(timezone.utc).isoformat())
    OUT.mkdir();shutil.copy2(PARENT_OUT/'cpu-scene-fields.npz',OUT/'cpu-scene-fields.npz')
    base.write(PLAN,p);pre=base.read(PARENT_OUT/'preflight.json')
    pre.update(protocol_sha256=base.sha(PLAN),source_count=len(p['source_sha256']),fixture=fixture,
        readonly_readback_identity_checked=True,negative_fixture_rejected=True,
        parent_protocol_sha256=base.sha(PARENT_PLAN),original_failure_controls=1,
        new_measurement_controls=400,total_transitions_including_failed_attempt=410624)
    base.write(OUT/'preflight.json',pre)
    print('READBACK_RECOVERY_FROZEN',base.sha(PLAN),'parent_controls=1;new_controls=400;actor_updates=0',flush=True)


def activate():
    install_readback();base.PLAN,base.OUT=PLAN,OUT


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:
        activate()
        if args.verify:
            import verify_getup_interface_value as verify
            verify.main()
        else:
            base.run()
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--verify'],check=True)
