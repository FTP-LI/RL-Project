"""Verify the original HOME target ramp from real pre-trigger commands."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from getup_stand_handoff import projected_slew
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_return_preflight.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial/second-attempt-return-preflight.json"


def excess(target,limits):
    return np.maximum(np.maximum(limits[:,0]-target,target-limits[:,1]),0.)


def check(seed,proof,source,p,screen):
    record = ROOT/proof["record"]
    case = read(record/"protocol.json")
    env = case["environment"]
    dt = p["dt_seconds"]
    assert env["seed"] == seed and env["criteria"]["step_dt"] == dt
    assert env["handoff_settings"]["rate_rad_s"] == p["target_rate_rad_s"]
    role = proof["role"]
    path = record/f"trace-{role}.npz"
    receipt = read(source)
    assert sha(path) == proof["trace_sha256"] == receipt["arms"][role]["trace_sha256"]
    rows = proof["triggers"]
    assert all(row["release_pose"] == "supine" and not row["original_passed"] for row in rows)
    worlds = np.asarray([row["world"] for row in rows])
    controls = np.asarray([row["first_decision_control"] for row in rows])
    release = round(screen["release_seconds"]/dt)
    boundary = controls-release-1
    with np.load(path,allow_pickle=False) as trace:
        np.testing.assert_allclose(trace["time"][boundary],controls*dt,atol=1e-8,rtol=0.)
        previous = trace["applied_targets"][boundary,worlds].copy()
        joints = trace["joint_positions"][boundary,worlds]
    home = np.asarray(env["simulation_home"],dtype=np.float32)
    limits = np.asarray(env["joint_limits"],dtype=np.float32)
    assert (home >= limits[:,0]).all() and (home <= limits[:,1]).all()
    ramp = [torch.from_numpy(previous)]
    goal = torch.from_numpy(np.broadcast_to(home,previous.shape).copy())
    total = p["return_controls"]
    rate = p["target_rate_rad_s"]
    for _ in range(total):
        actual,destination = projected_slew(ramp[-1],goal,torch.from_numpy(limits),dt,rate)
        assert torch.equal(destination,goal)
        ramp.append(actual)
    targets = torch.stack(ramp).numpy()
    step = np.abs(np.diff(targets,axis=0))
    assert step.max() <= dt*rate+1e-6
    expected = previous+np.clip(home-previous,-np.arange(total+1)[:,None,None]*dt*rate,
                                np.arange(total+1)[:,None,None]*dt*rate)
    np.testing.assert_allclose(targets,expected,atol=4e-6,rtol=0.)
    assert (np.diff(excess(targets,limits),axis=0) <= 1e-6).all()
    last_motion = total-p["settle_allowance_controls"]
    err = abs(targets-home).max(axis=-1)
    reachable = err[last_motion] <= p["maximum_target_home_error_rad"]
    model_path = ROOT/env["model"]
    assert sha(model_path) == env["model_sha256"]
    names = ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map["joint_names"].split(",")
    assert len(names) == previous.shape[1]
    examples = [dict(world=int(world),trigger_control=int(controls[i]),
        start_max_target_home_error_rad=float(err[0,i]),
        start_max_measured_home_distance_rad=float(abs(joints[i]-home).max()),
        target_error_at_settle_start_rad=float(err[last_motion,i]),terminal_target_error_rad=float(err[-1,i]),
        largest_start_target_gap_joint=names[int(abs(previous[i]-home).argmax())],
        target_reachable=bool(reachable[i])) for i,world in enumerate(worlds)]
    count = int(reachable.sum())
    return dict(seed=seed,triggered=len(worlds),reachable_by_settle_start=count,maximum_target_step_rad=float(step.max()),
        initial_target_out_of_range=int(excess(previous,limits).any(axis=1).sum()),
        path_matches_closed_form=True,range_excess_never_increases=True,
        terminal_target_home_error_rad=float(err[-1].max()),
        admitted=count/len(worlds) >= p["minimum_reachable_trigger_fraction_per_seed"],examples=examples)


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-second-attempt-return-preflight" and not OUT.exists()
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == p["stage"]
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    for key in ("screen_protocol","screen_result","slew_source"):
        assert sha(ROOT/p[key]) == p[key+"_sha256"]
    screen,result = (read(ROOT/p[key]) for key in ("screen_protocol","screen_result"))
    assert result["protocol_sha256"] == p["screen_protocol_sha256"] and result["component_preparation_admitted"]
    assert p["return_controls"]*p["dt_seconds"] == screen["return_to_start_budget_seconds"]
    torch.set_num_threads(2)
    outputs = []
    for row in result["results"]:
        seed = row["seed"]
        source,checksum,role = screen["archives"][str(seed)]
        source = ROOT/source
        assert sha(source) == checksum and role == row["role"]
        outputs.append(check(seed,row,source,p,screen))
    decision = all(row["admitted"] for row in outputs)
    write(OUT,dict(status="completed-saved-second-attempt-return-preflight",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),rows_by_seed=outputs,return_target_gate_passed=decision,
        physical_response_tested=False,physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    for row in outputs:
        print("GETUP_SECOND_ATTEMPT_RETURN",row["seed"],row["reachable_by_settle_start"],row["triggered"],
              row["terminal_target_home_error_rad"],row["admitted"],flush=True)


if __name__ == "__main__":
    main()
