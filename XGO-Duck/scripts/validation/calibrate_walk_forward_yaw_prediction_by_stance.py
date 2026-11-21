"""Check separate stance confidence cutoffs with the original accuracy and coverage gates."""

from pathlib import Path

import numpy as np
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
from calibrate_walk_forward_yaw_prediction import restore
import fit_walk_forward_yaw_prediction as fit


def cutoff(prediction, truth, precision_limit, coverage_limit):
    amplitude = abs(prediction)
    order = np.argsort(amplitude)
    levels, start = np.unique(amplitude[order],return_index=True)
    correct = np.sign(prediction[order]) == np.sign(truth[order])
    remaining = len(order)-start
    precision = correct[::-1].cumsum()[::-1][start]/remaining
    coverage = remaining/len(order)
    candidates = np.flatnonzero((precision >= precision_limit) & (coverage >= coverage_limit))
    i = int(candidates[0]) if len(candidates) else int(np.where(coverage >= coverage_limit,precision,-1.).argmax())
    return dict(selected_threshold_rad_s=float(levels[i]) if len(candidates) else None,
        compared_cutoffs=len(levels), precision=float(precision[i]), coverage=float(coverage[i]),
        closest_or_selected_threshold_rad_s=float(levels[i]))


def main():
    torch.set_num_threads(4)
    plan = read(fit.PLAN)
    model_path = fit.OUT/"forward-yaw-predictor-cpu.json"
    common_path = fit.OUT/"forward-yaw-predictor-exact-calibration.json"
    model, common = read(model_path), read(common_path)
    assert model["protocol_sha256"] == sha(fit.PLAN)
    assert common["original_predictor_sha256"] == sha(model_path) and common["selected_threshold_rad_s"] is None
    assert not common["verification"]
    observer_path = fit.OUT/"forward-stance-observation-cpu.json"
    assert model["support_observer_sha256"] == sha(observer_path)
    observer_plan = ROOT/plan["support_protocol"]
    support, state = read(observer_plan), read(observer_path)
    assert state["protocol_sha256"] == sha(observer_plan)
    observer = StanceObserver(state["model"],support["minimum_score_separation"])
    predict = restore(model["model"],plan)
    cases = {key:row for group in plan["fit_cases"] for key,row in fit.load(group,plan,observer).items()}
    assert {key:row["source_receipt_sha256"] for key,row in cases.items()} == model["fit_sources"]
    indices = [next(iter(cases.values()))["worlds"].index(world) for world in model["calibration_worlds"]]
    x,y,phase = (np.concatenate([row[k][:,indices] for row in cases.values()]) for k in ("x","y","phase"))
    pred = predict(x)
    calibration = {name: cutoff(pred[phase == i],y[phase == i],
        plan["minimum_calibration_direction_precision"],plan["minimum_verification_coverage"])
        for i,name in enumerate(fit.SIDES,1)}
    thresholds = {name:row["selected_threshold_rad_s"] for name,row in calibration.items()}
    verification, pulse = {}, {}
    if all(value is not None for value in thresholds.values()):
        for group in support["verification"]:
            for key,row in fit.load(group,plan,observer).items():
                values = predict(row["x"])
                sides = {name:fit.metrics(values,row["y"],row["phase"],thresholds[name])[name] for name in fit.SIDES}
                verification[key] = dict(seed=row["seed"],source_receipt_sha256=row["source_receipt_sha256"],
                    sides=sides,admitted=all(side["precision"] >= plan["minimum_verification_direction_precision"] and
                        side["coverage"] >= plan["minimum_verification_coverage"] for side in sides.values()))
        pulse = {name:fit.pulse(plan,observer,predict,thresholds[name])[name] for name in fit.SIDES}
    passed = bool(verification and all(row["admitted"] for row in verification.values()) and
                  all(row["admitted"] for row in pulse.values()))
    assert not torch.cuda.is_initialized()
    write(fit.OUT/"forward-yaw-predictor-stance-calibration.json",dict(status="checked-stance-specific-cutoffs",
        source_sha256=sha(Path(__file__)),original_predictor_sha256=sha(model_path),common_cutoff_sha256=sha(common_path),
        protocol_sha256=sha(fit.PLAN),calibration=calibration,thresholds_rad_s=thresholds,
        verification=verification,pulse_branch_reselection=pulse,admitted_for_new_mixed_pulse=passed,
        physical_substeps=0,policy_updates=0,heldout_used=False))
    print("FORWARD_YAW_STANCE_CALIBRATION",passed,calibration,verification,pulse,flush=True)


if __name__ == "__main__":
    main()
