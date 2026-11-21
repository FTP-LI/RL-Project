"""Fit a small causal next-control yaw predictor without changing any policy."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
import run_walk_forward_stance_response as pulse_trial


PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_prediction.json"
OUT = pulse_trial.STAGE
SIDES = ("left-only","right-only")


def features(obs, start, plan):
    h, width = plan["window"], plan["observation_columns"]
    rows = np.arange(start,len(obs))
    return np.concatenate([obs[rows-offset,...,:width] for offset in range(h-1,-1,-1)],axis=-1)


def load(group, plan, observer):
    protocol, source = (ROOT/group[key] for key in ("protocol","receipt"))
    spec, proof = read(protocol), read(source)
    assert proof["protocol_sha256"] == sha(protocol) and not proof["heldout_used"]
    out = {}
    for key in group["cases"]:
        entry = proof["records"][key]
        record,p = ROOT/entry["path"],spec["cases"][key]["environment"]
        verify = record/"verification.json"
        assert sha(verify) == entry["verification_sha256"]
        receipt = read(verify)
        worlds = p["groups"]["forward"]
        blocks = []
        for name,digest in zip(("shared-prefix","actuator-feedback"),receipt["trace_hashes"],strict=True):
            path = record/f"trace-{name}.npz"
            assert sha(path) == digest
            with np.load(path,allow_pickle=False) as f:
                assert f["active"][:,worlds].all()
                blocks.append((f["observations"][:,worlds],f["yaw_rate"][:,worlds]))
        obs,yaw = (np.concatenate(parts) for parts in zip(*blocks,strict=True))
        out[key] = dict(seed=p["seed"],worlds=worlds,source_receipt_sha256=sha(verify),
            x=features(obs,50,plan),y=yaw[50:],phase=observer.predict(obs[50:]),gyro=obs[50:,:,2])
    return out


def fit(x, y, plan):
    mean, scale = x.mean(axis=0), x.std(axis=0).clip(1e-6)
    torch.set_num_threads(6)
    torch.manual_seed(plan["split_seed"])
    normalized = torch.from_numpy((x-mean)/scale)
    target = torch.from_numpy(y[:,None])
    a,b = plan["hidden_sizes"]
    model = nn.Sequential(nn.Linear(x.shape[1],a),nn.Tanh(),nn.Linear(a,b),nn.Tanh(),nn.Linear(b,1))
    optimizer = torch.optim.AdamW(model.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"])
    losses = []
    for _ in range(plan["fit_epochs"]):
        values = []
        for batch in torch.randperm(len(target)).split(plan["batch"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(normalized[batch]),target[batch])
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
    model.eval()

    @torch.inference_mode()
    def predict(data):
        return model(torch.from_numpy((data-mean)/scale)).squeeze(-1).numpy()

    weights = {name: value.tolist() for name,value in model.state_dict().items()}
    return predict, dict(mean=mean.tolist(),scale=scale.tolist(),weights=weights,
        activation="tanh",input_window=plan["window"],observation_columns=plan["observation_columns"]), losses


def metrics(pred, actual, phase, threshold):
    result = {}
    for i,name in enumerate(SIDES,1):
        support = phase == i
        keep = support & (abs(pred) >= threshold)
        n = int(keep.sum())
        result[name] = dict(eligible=n,support_rows=int(support.sum()),
            coverage=float(n/support.sum()),
            precision=float((np.sign(pred[keep]) == np.sign(actual[keep])).mean()) if n else 0.)
    return result


def pulse(plan, observer, predict, threshold):
    spec = pulse_trial.checked()
    item = next(iter(spec["cases"].values()))
    record = ROOT/item["record"]
    proof = read(pulse_trial.OUT/"analysis.json")
    assert proof["protocol_sha256"] == sha(pulse_trial.PLAN) and proof["both_directions_admitted"]
    traces, applied = {}, {}
    for role in pulse_trial.ROLES:
        source = record/f"trace-{role}.npz"
        witness = record/f"pulse-inputs-{role}.npz"
        assert sha(source) == proof["roles"][role]["trace_sha256"] and sha(witness) == proof["roles"][role]["pulse_sha256"]
        with np.load(witness,allow_pickle=False) as f:
            ids, phase_saved, applied[role] = (f[k] for k in ("world","phase","applied_yaw"))
        with np.load(source,allow_pickle=False) as f:
            traces[role] = f["yaw_rate"][:3,ids].astype(np.float64)
            if role == "reference-a":
                first = f["observations"][:1,ids]
    prefix = record/"trace-shared-prefix.npz"
    assert sha(prefix) == proof["shared_prefix_sha256"]
    with np.load(prefix,allow_pickle=False) as f:
        before = f["observations"][-(plan["window"]-1):,ids]
    phase = observer.predict(first[0])
    np.testing.assert_array_equal(phase,phase_saved)
    value = predict(features(np.concatenate((before,first)),plan["window"]-1,plan))[0]
    choose_negative = value > 0.
    change = np.where(choose_negative,applied["negative"],applied["positive"])
    selected = np.where(choose_negative[None,:],traces["negative"],traces["positive"])
    a,b = (traces[k] for k in pulse_trial.ROLES[:2])
    gain = (abs(a)+abs(b))/2-abs(selected)
    spread = abs(abs(a)-abs(b))
    result = {}
    rng = np.random.default_rng(plan["split_seed"]+1)
    for i,name in enumerate(SIDES,1):
        use = (phase == i) & (abs(value) >= threshold) & (change != applied["reference-a"])
        n = int(use.sum())
        if not n:
            result[name] = dict(worlds=[],admitted=False)
            continue
        gains = gain[0,use]
        ci = np.quantile(rng.choice(gains,size=(2000,n),replace=True).mean(axis=1),[.025,.975]).tolist()
        result[name] = dict(worlds=ids[use].tolist(), first_step_mean_gain_rad_s=float(gains.mean()),
            first_step_gain_ci95=ci, mean_repeat_spread_rad_s=float(spread[0,use].mean()),
            actual_sign_accuracy=float((np.sign(value[use]) == np.sign((a[0,use]+b[0,use])/2)).mean()),
            admitted=bool(n >= plan["minimum_guarded_pulse_worlds_per_side"] and ci[0] > 0. and
                          gains.mean() > spread[0,use].mean()))
    return result


def main():
    plan = read(PLAN)
    assert plan["status"] == "frozen-archive-predictor-screen"
    assert not any(plan[k] for k in ("heldout_used","physical_substeps","policy_updates","hardware_io"))
    support_file = ROOT/plan["support_protocol"]
    support = read(support_file)
    receipt = OUT/"forward-stance-observation-cpu.json"
    proof = read(receipt)
    assert proof["protocol_sha256"] == sha(support_file) and proof["admitted_for_control_design"]
    observer = StanceObserver(proof["model"],support["minimum_score_separation"])
    fit_cases = {key: row for group in plan["fit_cases"] for key,row in load(group,plan,observer).items()}
    verification = {key: row for group in support["verification"] for key,row in load(group,plan,observer).items()}
    reference = next(iter(fit_cases.values()))
    assert all(row["worlds"] == reference["worlds"] and row["seed"] == 1141 for row in fit_cases.values())
    assert all(row["seed"] != reference["seed"] for row in verification.values())
    rng = np.random.default_rng(plan["split_seed"])
    calibration = np.sort(rng.choice(len(reference["worlds"]),size=plan["calibration_worlds_per_case"],replace=False))
    fitting = np.setdiff1d(np.arange(len(reference["worlds"])),calibration)
    x = np.concatenate([row["x"][:,fitting].reshape(-1,row["x"].shape[-1]) for row in fit_cases.values()])
    y = np.concatenate([row["y"][:,fitting].ravel() for row in fit_cases.values()])
    predict, model, history = fit(x,y,plan)
    c_x,c_y,c_phase = (np.concatenate([row[key][:,calibration] for row in fit_cases.values()]) for key in ("x","y","phase"))
    c_pred = predict(c_x)
    calibration_curve = {str(value): metrics(c_pred,c_y,c_phase,value) for value in plan["calibration_thresholds_rad_s"]}
    admissible = [float(value) for value, row in calibration_curve.items()
                  if all(side["precision"] >= plan["minimum_calibration_direction_precision"] and
                         side["coverage"] >= plan["minimum_verification_coverage"] for side in row.values())]
    threshold = min(admissible,default=None)
    val, pulse_check = {}, {}
    if threshold is not None:
        for key,row in verification.items():
            pred = predict(row["x"])
            sides = metrics(pred,row["y"],row["phase"],threshold)
            val[key] = dict(seed=row["seed"],sides=sides,source_receipt_sha256=row["source_receipt_sha256"],
                predicted_mae_rad_s=float(abs(pred-row["y"]).mean()),gyro_mae_rad_s=float(abs(row["gyro"]-row["y"]).mean()),
                admitted=all(side["precision"] >= plan["minimum_verification_direction_precision"] and
                             side["coverage"] >= plan["minimum_verification_coverage"] for side in sides.values()))
        pulse_check = pulse(plan,observer,predict,threshold)
    passed = bool(threshold is not None and all(row["admitted"] for row in val.values()) and
                  all(row["admitted"] for row in pulse_check.values()))
    assert not torch.cuda.is_initialized()
    write(OUT/"forward-yaw-predictor-cpu.json",dict(status="checked-causal-yaw-predictor",
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(PLAN),support_observer_sha256=sha(receipt),
        fit_seed=reference["seed"],fit_rows=len(x),calibration_rows=c_y.size,calibration_worlds=[reference["worlds"][i] for i in calibration],
        fit_sources={key: row["source_receipt_sha256"] for key,row in fit_cases.items()},fit_mse_by_epoch=history,
        selected_threshold_rad_s=threshold,calibration_curve=calibration_curve,verification=val,pulse_branch_reselection=pulse_check,
        model=model,admitted_for_new_mixed_pulse=passed,physical_substeps=0,policy_updates=0,heldout_used=False))
    print("FORWARD_YAW_PREDICTOR_CPU",passed,"threshold",threshold,
        {key: row["sides"] for key,row in val.items()},pulse_check,flush=True)


if __name__ == "__main__":
    main()
