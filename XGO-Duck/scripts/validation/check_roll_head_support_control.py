"""Exercise the actual pulse target overlay and its fork inventory without physics."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from record_io import write
from roll_head_support_control import HeadSupportRollPolicy, PULSE_FIELDS, ProjectedRollPolicy, pulse_snapshot_slots
from roll_stand_walk_sequence import ROLL, STAND_LANDING
from s1_reference_inputs import ROOT, read, sha


def main():
    draft = ROOT/'configs/experiments/continuous_roll_head_support_pulse_draft.json'
    parent = ROOT/'configs/experiments/continuous_roll_early_lateral_push.json'
    path = ROOT/'records/reports/continuous-roll-early-lateral-push/head-roll-overlay-cpu.json'
    assert not path.exists() and not torch.cuda.is_initialized()
    p = dict(read(parent),worlds=3,head_support_pulse=read(draft)['candidate'])
    limits = torch.tensor(p['walking']['joint_limits'])
    joint = p['walking']['joint_names'].index('head_roll')
    original = torch.zeros((3,len(limits)))
    original[1,joint] = limits[joint,0]+.005
    raw = torch.full_like(original,.25)

    def init(self,spec,networks,homes,sim_home,bounds):
        self.limits = bounds
        self.history = raw.clone()

    def next_policy(self,step,obs,phase):
        return original.clone(),dict(applied_targets=original.numpy().copy(),phase_used=phase)

    phase = np.array([ROLL,ROLL,STAND_LANDING])
    with patch.object(ProjectedRollPolicy,'__init__',init), patch.object(ProjectedRollPolicy,'next',next_policy):
        policy = HeadSupportRollPolicy(p,{},None,original,limits)
        slots = pulse_snapshot_slots([],policy)

        def step(k,degrees):
            obs = torch.zeros((3,61))
            angle = np.deg2rad(degrees)
            obs[:,3],obs[:,5] = float(np.sin(angle)),float(-np.cos(angle))
            target,row = policy.next(k,obs,phase)
            torch.testing.assert_close(policy.history,raw,rtol=0,atol=0)
            torch.testing.assert_close(policy.previous,target,rtol=0,atol=0)
            return target,row

        for k,degrees in enumerate((0,30,60,90),1):
            target,row = step(k,degrees)
            torch.testing.assert_close(target,original,rtol=0,atol=0)
            assert not row['pulse_selected'].any()
        assert [label for label,_,_ in slots] == [f'controller.pulse.{name}' for name in PULSE_FIELDS]
        saved = [getattr(owner,name).copy() for _,owner,name in slots]
        for k,degrees in enumerate((121,145,165),5):
            target,row = step(k,degrees)
            torch.testing.assert_close(target,original,rtol=0,atol=0)
            assert row['pulse_selected'].tolist() == [k<7,k<7,False] and not row['pulse_enabled'].any()
        for (_,owner,name),value in zip(slots,saved,strict=True):
            setattr(owner,name,value.copy())
        policy.pulse_enabled = True
        tested = []
        for k,degrees in enumerate((121,145,165),5):
            target,row = step(k,degrees)
            expected = original.clone()
            if k<7:
                expected[0,joint] -= .02
                expected[1,joint] = limits[joint,0]
            torch.testing.assert_close(target,expected,rtol=0,atol=0)
            np.testing.assert_array_equal(row['applied_targets'],expected.numpy())
            np.testing.assert_array_equal(row['pulse_applied_delta_rad'],(expected-original)[:,joint].numpy())
            assert row['pulse_selected'].tolist() == [k<7,k<7,False]
            tested.append(row['pulse_applied_delta_rad'].tolist())
    assert not torch.cuda.is_initialized()
    files = (draft,parent,Path(__file__),Path(__file__).with_name('roll_head_support_control.py'))
    write(path,dict(status='passed-cpu-control-overlay',physical_substeps=0,learning_updates=0,
        nominal_unchanged=True,actor_history_unchanged=True,clamped_at_joint_lower_bound=True,
        trigger_restorable=True,tested_delta_rad=tested,
        sha256={file.resolve().relative_to(ROOT).as_posix():sha(file) for file in files}))
    print('ROLL_HEAD_SUPPORT_OVERLAY_CPU_OK',tested,flush=True)


if __name__ == '__main__':
    main()
