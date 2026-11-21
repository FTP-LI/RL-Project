"""Fit the same causal observation window directly to next-step yaw direction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
from calibrate_walk_forward_yaw_prediction_by_stance import cutoff
import fit_walk_forward_yaw_prediction as base


PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_direction_fit.json"
OUT = base.OUT


def fit(x,y,plan,variant):
    mean,scale = x.mean(axis=0),x.std(axis=0).clip(1e-6)
    normalized = torch.from_numpy((x-mean)/scale)
    target = torch.from_numpy((y > 0).astype(np.float32))
    amount = np.clip(abs(y)/variant["full_weight_at_abs_rate_rad_s"],0.,1.)
    floor = variant["weak_rate_weight"]
    weight = torch.from_numpy(floor+(1.-floor)*amount)
    a,b = plan["hidden_sizes"]
    network = nn.Sequential(nn.Linear(x.shape[1],a),nn.Tanh(),nn.Linear(a,b),nn.Tanh(),nn.Linear(b,1))
    optimizer = torch.optim.AdamW(network.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"])
    history = []
    for _ in range(plan["fit_epochs"]):
        losses = []
        for ids in torch.randperm(len(target)).split(plan["batch"]):
            optimizer.zero_grad(set_to_none=True)
            logits = network(normalized[ids]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits,target[ids],weight=weight[ids])
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    network.eval()

    @torch.inference_mode()
    def predict(data):
        return network(torch.from_numpy((data-mean)/scale)).squeeze(-1).numpy()

    weights = {key:value.tolist() for key,value in network.state_dict().items()}
    return predict,dict(mean=mean.tolist(),scale=scale.tolist(),weights=weights,activation="tanh",
        observation_columns=plan["observation_columns"],input_window=plan["window"],output="direction-logit"),history


def main():
    torch.set_num_threads(6)
    variant = read(PLAN)
    assert variant["status"] == "frozen-archive-direction-fit" and variant["require_same_per_side_calibration_and_verification_gates"]
    assert not any(variant[key] for key in ("heldout_used","physical_substeps","policy_updates","hardware_io"))
    source, fitted, calibration_path = (ROOT/variant[key] for key in ("base_protocol","base_fit_receipt","base_stance_calibration"))
    plan, old, prior = (read(path) for path in (source,fitted,calibration_path))
    assert old["protocol_sha256"] == prior["protocol_sha256"] == sha(source)
    assert prior["original_predictor_sha256"] == sha(fitted) and not prior["admitted_for_new_mixed_pulse"]
    assert not prior["verification"]
    torch.manual_seed(plan["split_seed"])
    support_file = ROOT/plan["support_protocol"]
    support = read(support_file)
    observed = OUT/"forward-stance-observation-cpu.json"
    proof = read(observed)
    assert proof["protocol_sha256"] == sha(support_file) and sha(observed) == old["support_observer_sha256"]
    observer = StanceObserver(proof["model"],support["minimum_score_separation"])
    training = {key:row for group in plan["fit_cases"] for key,row in base.load(group,plan,observer).items()}
    assert {key:row["source_receipt_sha256"] for key,row in training.items()} == old["fit_sources"]
    worlds = next(iter(training.values()))["worlds"]
    assert all(row["worlds"] == worlds and row["seed"] == old["fit_seed"] for row in training.values())
    c = np.array([worlds.index(world) for world in old["calibration_worlds"]])
    f = np.setdiff1d(np.arange(len(worlds)),c)
    x = np.concatenate([row["x"][:,f].reshape(-1,row["x"].shape[-1]) for row in training.values()])
    y = np.concatenate([row["y"][:,f].ravel() for row in training.values()])
    assert len(y) == old["fit_rows"]
    predict, model, loss = fit(x,y,plan,variant)
    c_x,c_y,phase = (np.concatenate([row[k][:,c] for row in training.values()]) for k in ("x","y","phase"))
    assert c_y.size == old["calibration_rows"]
    pred = predict(c_x)
    calibration = {name:cutoff(pred[phase == i],c_y[phase == i],plan["minimum_calibration_direction_precision"],
        plan["minimum_verification_coverage"]) for i,name in enumerate(base.SIDES,1)}
    # The shared cutoff helper uses the historical field name. Here the values are logits.
    thresholds = {name:row.pop("selected_threshold_rad_s") for name,row in calibration.items()}
    for row in calibration.values():
        row["closest_or_selected_threshold_logit"] = row.pop("closest_or_selected_threshold_rad_s")
    verify, pulse = {}, {}
    if all(value is not None for value in thresholds.values()):
        for group in support["verification"]:
            for key,row in base.load(group,plan,observer).items():
                assert row["seed"] != old["fit_seed"]
                result = predict(row["x"])
                sides = {name:base.metrics(result,row["y"],row["phase"],thresholds[name])[name] for name in base.SIDES}
                verify[key] = dict(seed=row["seed"],source_receipt_sha256=row["source_receipt_sha256"],sides=sides,
                    admitted=all(value["precision"] >= plan["minimum_verification_direction_precision"] and
                        value["coverage"] >= plan["minimum_verification_coverage"] for value in sides.values()))
        pulse = {name:base.pulse(plan,observer,predict,thresholds[name])[name] for name in base.SIDES}
    admitted = bool(verify and all(row["admitted"] for row in verify.values()) and all(row["admitted"] for row in pulse.values()))
    assert not torch.cuda.is_initialized()
    write(OUT/"forward-yaw-direction-fit-cpu.json",dict(status="checked-causal-yaw-direction-fit",
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(PLAN),base_fit_sha256=sha(fitted),
        base_calibration_sha256=sha(calibration_path),fit_rows=len(y),calibration_rows=c_y.size,
        fit_loss_by_epoch=loss,calibration=calibration,thresholds_logit=thresholds,verification=verify,
        pulse_branch_reselection=pulse,model=model,admitted_for_new_mixed_pulse=admitted,
        physical_substeps=0,policy_updates=0,heldout_used=False))
    print("FORWARD_YAW_DIRECTION_CPU",admitted,thresholds,calibration,verify,pulse,flush=True)


if __name__ == "__main__":
    main()
