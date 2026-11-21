"""Check every distinct calibration cutoff for the frozen predictor weights."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
import fit_walk_forward_yaw_prediction as fit


def restore(model, plan):
    mean, scale = (np.asarray(model[key],np.float32) for key in ("mean","scale"))
    a,b = plan["hidden_sizes"]
    network = nn.Sequential(nn.Linear(len(mean),a),nn.Tanh(),nn.Linear(a,b),nn.Tanh(),nn.Linear(b,1))
    network.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in model["weights"].items()})
    network.eval()

    @torch.inference_mode()
    def predict(data):
        return network(torch.from_numpy((data-mean)/scale)).squeeze(-1).numpy()

    return predict


def exact_cutoff(pred,actual,phase,plan):
    thresholds = np.unique(abs(pred[phase > 0]))
    valid = np.ones(len(thresholds),bool)
    both_precision = []
    both_coverage = []
    for side in (1,2):
        keep = phase == side
        amplitudes = abs(pred[keep])
        order = amplitudes.argsort()
        equal = np.sign(pred[keep][order]) == np.sign(actual[keep][order])
        location = np.searchsorted(amplitudes[order],thresholds,side="left")
        remaining = len(order)-location
        correct = np.r_[equal[::-1].cumsum()[::-1],0][location]
        coverage = remaining/len(order)
        precision = np.divide(correct,remaining,out=np.zeros(len(remaining)),where=remaining > 0)
        valid &= (coverage >= plan["minimum_verification_coverage"]) & (precision >= plan["minimum_calibration_direction_precision"])
        both_precision.append(precision)
        both_coverage.append(coverage)
    admitted = np.flatnonzero(valid)
    feasible_coverage = np.min(both_coverage,axis=0) >= plan["minimum_verification_coverage"]
    best = np.where(feasible_coverage,np.min(both_precision,axis=0),-1.).argmax()
    return float(thresholds[admitted[0]]) if len(admitted) else None, dict(
        candidate_cutoffs=len(thresholds), admitted_cutoffs=len(admitted),
        best_minimum_precision_at_required_coverage=float(min(side[best] for side in both_precision)),
        best_covered_cutoff_rad_s=float(thresholds[best]),
        best_cutoff_side_precision=[float(side[best]) for side in both_precision],
        best_cutoff_side_coverage=[float(side[best]) for side in both_coverage])


def main():
    torch.set_num_threads(4)
    plan, source = read(fit.PLAN), fit.OUT/"forward-yaw-predictor-cpu.json"
    original = read(source)
    assert original["protocol_sha256"] == sha(fit.PLAN) and original["selected_threshold_rad_s"] is None
    support_file = ROOT/plan["support_protocol"]
    support = read(support_file)
    observer_file = fit.OUT/"forward-stance-observation-cpu.json"
    observer_record = read(observer_file)
    assert sha(observer_file) == original["support_observer_sha256"] and observer_record["protocol_sha256"] == sha(support_file)
    observer = StanceObserver(observer_record["model"],support["minimum_score_separation"])
    predict = restore(original["model"],plan)
    sources = {key:row for group in plan["fit_cases"] for key,row in fit.load(group,plan,observer).items()}
    assert {key:row["source_receipt_sha256"] for key,row in sources.items()} == original["fit_sources"]
    indices = [next(iter(sources.values()))["worlds"].index(world) for world in original["calibration_worlds"]]
    x,y,phase = (np.concatenate([row[key][:,indices] for row in sources.values()]) for key in ("x","y","phase"))
    pred = predict(x)
    assert y.size == original["calibration_rows"]
    for value, expected in original["calibration_curve"].items():
        assert fit.metrics(pred,y,phase,float(value)) == expected, value
    threshold, calibration = exact_cutoff(pred,y,phase,plan)
    verification, pulse = {}, {}
    if threshold is not None:
        for group in support["verification"]:
            for key,row in fit.load(group,plan,observer).items():
                prediction = predict(row["x"])
                sides = fit.metrics(prediction,row["y"],row["phase"],threshold)
                verification[key] = dict(seed=row["seed"],source_receipt_sha256=row["source_receipt_sha256"],
                    sides=sides,mae_rad_s=float(abs(prediction-row["y"]).mean()),
                    admitted=all(side["precision"] >= plan["minimum_verification_direction_precision"] and
                                 side["coverage"] >= plan["minimum_verification_coverage"] for side in sides.values()))
        pulse = fit.pulse(plan,observer,predict,threshold)
    passed = bool(threshold is not None and all(v["admitted"] for v in verification.values()) and
                  all(v["admitted"] for v in pulse.values()))
    assert not torch.cuda.is_initialized()
    write(fit.OUT/"forward-yaw-predictor-exact-calibration.json",dict(status="checked-exact-calibration-thresholds",
        source_sha256=sha(Path(__file__)),original_predictor_sha256=sha(source),protocol_sha256=sha(fit.PLAN),
        original_grid_reproduced=True,original_weights_preserved=True,calibration=calibration,selected_threshold_rad_s=threshold,
        verification=verification,pulse_branch_reselection=pulse,admitted_for_new_mixed_pulse=passed,
        physical_substeps=0,policy_updates=0,heldout_used=False))
    print("FORWARD_YAW_EXACT_CALIBRATION",passed,threshold,calibration,verification,pulse,flush=True)


if __name__ == "__main__":
    main()
