"""Check the short retreat target on the same archived pre-trigger commands."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from check_getup_second_attempt_return import excess
from getup_stand_handoff import projected_slew
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_neck_retreat.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial/second-attempt-neck-retreat.json"


def main():
    p = read(PLAN)
    state = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-neck-retreat-target-preflight" and not OUT.exists()
    assert state["current_phase"] == p["stage"] and state["goal_execution"]["status"] == "active"
    assert ROOT/state["goal_execution"]["current_protocol"] == PLAN
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    full,plan = (ROOT/p[key] for key in ("full_home_result","full_home_plan"))
    previous,base = read(full),read(plan)
    assert sha(full) == p["full_home_result_sha256"] and sha(plan) == previous["protocol_sha256"]
    assert not previous["return_target_gate_passed"]
    for key in ("screen_result","screen_protocol","slew_source"):
        assert sha(ROOT/base[key]) == base[key+"_sha256"]
    proof,screen = (read(ROOT/base[key]) for key in ("screen_result","screen_protocol"))
    assert proof["component_preparation_admitted"] and proof["protocol_sha256"] == base["screen_protocol_sha256"]
    dt,rate = base["dt_seconds"],base["target_rate_rad_s"]
    end = base["return_controls"]
    first_settle = end-base["settle_allowance_controls"]
    release = round(screen["release_seconds"]/dt)
    outputs = []
    torch.set_num_threads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for row in proof["results"]:
        case = ROOT/row["record"]
        env = read(case/"protocol.json")["environment"]
        model = ROOT/env["model"]
        assert sha(model) == env["model_sha256"]
        names = ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map["joint_names"].split(",")
        col = names.index(p["neck_joint"])
        limits = np.array(env["joint_limits"],dtype=np.float32)
        home = np.array(env["simulation_home"],dtype=np.float32)
        home[col] = limits[col,0]+p["neck_goal_inside_lower_limit_rad"]
        assert not excess(home,limits).any()
        clock = np.array([entry["first_decision_control"] for entry in row["triggers"]])
        ids = np.array([entry["world"] for entry in row["triggers"]])
        file = case/f'trace-{row["role"]}.npz'
        assert sha(file) == row["trace_sha256"]
        ix = clock-release-1
        with np.load(file,allow_pickle=False) as t:
            np.testing.assert_allclose(t["time"][ix],clock*dt,atol=1e-8,rtol=0.)
            begin = t["applied_targets"][ix,ids].copy()
            measured_neck = t["joint_positions"][ix,ids,col]
        destination = torch.from_numpy(np.broadcast_to(home,begin.shape).copy())
        path = [torch.from_numpy(begin)]
        for _ in range(end):
            path.append(projected_slew(path[-1],destination,torch.from_numpy(limits),dt,rate)[0])
        path = torch.stack(path).numpy()
        step = float(abs(np.diff(path,axis=0)).max())
        assert step <= rate*dt+1e-6 and (np.diff(excess(path,limits),axis=0) <= 1e-6).all()
        extent = np.arange(end+1)[:,None,None]*dt*rate
        np.testing.assert_allclose(path,begin+np.clip(home-begin,-extent,extent),atol=4e-6,rtol=0.)
        final_error = abs(path-home).max(axis=-1)
        legal = ~excess(path[-1],limits).any(axis=1)
        success = (final_error[first_settle] <= p["maximum_target_home_error_rad"])&legal
        reach = int(success.sum())
        check = dict(seed=row["seed"],triggers=len(ids),commands_reachable=reach,peak_target_step_rad=step,
            neck_column=col,neck_goal_rad=float(home[col]),neck_lower_limit_rad=float(limits[col,0]),
            measured_neck_start_range_rad=[float(measured_neck.min()),float(measured_neck.max())],
            worst_settle_error_rad=float(final_error[first_settle].max()),worst_terminal_error_rad=float(final_error[-1].max()),
            passed=reach/len(ids) >= p["minimum_reachable_trigger_fraction_per_seed"])
        outputs.append(check)
        print("GETUP_NECK_RETREAT_TARGET",check,flush=True)
    write(OUT,dict(status="completed-saved-neck-retreat-target-preflight",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),rows_by_seed=outputs,retreat_target_gate_passed=all(r["passed"] for r in outputs),
        original_commands_remain_unchanged_in_archive=True,physical_response_tested=False,
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))


if __name__ == "__main__":
    main()
