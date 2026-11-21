"""Causal, reversible left-hip target pulse for a fixed-policy comparison."""

from pathlib import Path

import numpy as np
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT/"configs/experiments/getup_supine_left_hip_bilateral_pulse_draft.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture/pulse-component"


class LeftHipPulse:
    """Track one first exposure per environment at completed control boundaries."""

    def __init__(self,supine,rule,pulse):
        self.supine = supine.clone()
        self.bilateral_seen = torch.zeros_like(supine)
        self.trigger = torch.full_like(supine,-1,dtype=torch.int32)
        self.direction = torch.zeros_like(supine,dtype=torch.float32)
        self.min_head = rule["minimum_head_right_foot_force_n"]
        self.min_sole = rule["minimum_sole_force_n"]
        self.min_gap = float(np.deg2rad(rule["minimum_absolute_left_hip_gap_deg"]))
        self.start,self.end = (round(t/.02) for t in rule["window_seconds"])
        self.width = rule["consecutive_substeps"]
        self.pulse = torch.tensor(pulse,device=supine.device,dtype=torch.float32)

    def observe(self,control,loads,tracking_gap):
        assert loads.shape == (self.width,len(self.supine),6)
        assert tracking_gap.shape == self.supine.shape
        feet = loads[...,:2] >= self.min_sole
        bilateral = feet.all(dim=2).all(dim=0)
        if self.start <= control <= self.end:
            left_only = (feet[...,0]&~feet[...,1]).all(dim=0)
            head = (loads[...,3] >= self.min_head).all(dim=0)
            ready = (self.supine & ~self.bilateral_seen & left_only & head &
                     (tracking_gap.abs() >= self.min_gap) & (self.trigger < 0))
            self.trigger.masked_fill_(ready,control)
            self.direction.copy_(torch.where(ready,torch.sign(tracking_gap),self.direction))
        self.bilateral_seen.logical_or_(bilateral)

    def target(self,control,original,limits,polarity):
        assert polarity in (-1,0,1)
        if not polarity:
            return original
        step = control-self.trigger-1
        active = (self.trigger >= 0)&(step >= 0)&(step < len(self.pulse))
        delta = self.direction*polarity*self.pulse[step.clamp(0,len(self.pulse)-1).long()]
        lower,upper = limits
        safe = active&(original >= lower)&(original <= upper)
        return torch.where(safe,(original+delta).clamp(lower,upper),original)

    def snapshot(self):
        return tuple(value.clone() for value in (self.trigger,self.direction,self.bilateral_seen))

    def restore(self,saved):
        for value,original in zip((self.trigger,self.direction,self.bilateral_seen),saved,strict=True):
            value.copy_(original)


def archive_rehearsal():
    project = read(ROOT/"project.json")
    p = read(DRAFT)
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == p["stage"]
    assert ROOT/project["goal_execution"]["current_protocol"] == DRAFT
    assert not any(p[key] for key in ("physical_substeps_authorized","learning_updates","heldout_used","hardware_io"))
    previous = read(OUT/"analysis.json")
    assert previous["status"] == "passed-symmetric-left-hip-zero-readback" and not previous["physical_substeps"]
    result = OUT/"runtime-rehearsal.json"
    assert not result.exists()
    rule_path,expected_path,capture = (ROOT/p[key] for key in ("exposure_protocol","exposure_result","capture_protocol"))
    rule,expected = read(rule_path),read(expected_path)
    assert expected["protocol_sha256"] == sha(rule_path) and expected["selected_for_bounded_pulse_preflight"] == p["selected_trigger"]
    evidence = read(ROOT/rule["source"])
    assert evidence["protocol_sha256"] == sha(capture)
    record = ROOT/evidence["record"]
    actual = read(record/"evaluation.json")
    clock = record/actual["force_clock"]["archive"]
    assert sha(clock) == expected["force_clock_sha256"] == previous["archive"]["archive_sha256"]
    supine = torch.tensor([row["release_pose"] == "supine" for row in evidence["original_timing"]["episodes"]])
    with np.load(clock,allow_pickle=False) as data:
        stride = rule["consecutive_substeps"]
        controls = data["substep"][stride-1::stride]//stride
        loads = data["load_n"].reshape(len(controls),stride,len(supine),6)
        target = data["target"][stride-1::stride]
    expected_trigger = np.full(len(supine),-1,np.int32)
    expected_sign = np.zeros(len(supine),np.float32)
    examples = expected["groups"][p["selected_trigger"]]["examples"]
    for row in examples:
        expected_trigger[row["world"]] = row["control"]
        expected_sign[row["world"]] = np.sign(row["left_hip_gap_rad"])
    torch.set_num_threads(3)
    control = LeftHipPulse(supine,rule,p["target_pulse_rad"])
    boundary = control.snapshot()
    limits = (-1.570796326795,1.57079632679479)
    trace = {}
    for polarity in (0,1,-1):
        control.restore(boundary)
        observed = []
        for k,load,raw in zip(controls,loads,target,strict=True):
            base = torch.from_numpy(raw[:,1].copy())
            observed.append((control.target(int(k),base,limits,polarity)-base).numpy())
            gap = torch.from_numpy(raw[:,1]-raw[:,2])
            control.observe(int(k),torch.from_numpy(load),gap)
        np.testing.assert_array_equal(control.trigger.numpy(),expected_trigger)
        np.testing.assert_array_equal(control.direction.numpy(),expected_sign)
        trace[polarity] = np.stack(observed)
    assert not trace[0].any()
    np.testing.assert_allclose(trace[1],-trace[-1],atol=2e-7,rtol=0)
    assert (abs(trace[1]).max() <= p["maximum_correction_rad"]+1e-7 and
            int(np.count_nonzero(trace[1])) == previous["archive"]["arms"]["toward-target"]["changed_rows"])
    outside = torch.full_like(control.direction,limits[1]+.01)
    for sign in (1,-1):
        assert torch.equal(control.target(126,outside,limits,sign),outside)
    write(result,dict(status="passed-archived-runtime-rehearsal",draft_sha256=sha(DRAFT),source_sha256=sha(Path(__file__)),
        previous_component_sha256=sha(OUT/"analysis.json"),exposure_result_sha256=sha(expected_path),
        input_clock_sha256=sha(clock),worlds=len(supine),controls=len(controls),exposures=len(examples),
        modified_controls_per_direction=int(np.count_nonzero(trace[1])),trigger_and_sign_exact=True,
        input_boundaries_restored_between_arms=True,maximum_target_delta_rad=float(abs(trace[1]).max()),
        physical_substeps=0,learning_updates=0,hardware_io=False,heldout_used=False))
    print("LEFT_HIP_PULSE_RUNTIME_ARCHIVE",len(examples),int(np.count_nonzero(trace[1])),flush=True)


if __name__ == "__main__":
    archive_rehearsal()
