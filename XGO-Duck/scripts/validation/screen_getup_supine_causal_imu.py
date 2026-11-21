"""Screen a fixed causal IMU history on archived supine recovery episodes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release
from verify_getup_supine_early_signal_readback import normalized, tilt


PLAN = ROOT/"configs/experiments/getup_supine_causal_imu_screen.json"
OUT = ROOT/"records/reports/getup-supine-causal-imu-screen"


def features(history):
    gravity = normalized(history[...,3:6])
    angles = tilt(gravity)
    times = np.arange(len(angles),dtype=np.float64)
    centered = times-times.mean()
    trend = centered @ angles / (centered @ centered) * times[-1]
    pitch = history[...,1].astype(np.float64)
    return np.column_stack((angles[-1],trend,angles[-1]-angles.min(axis=0),
        abs(gravity[...,1]).mean(axis=0),pitch.mean(axis=0),np.sqrt(np.square(pitch).mean(axis=0))))


def load_batch(seed,role,settings,protocol,previous):
    receipt_path,expected = protocol["archives"][str(seed)]
    receipt = ROOT/receipt_path
    assert sha(receipt) == expected
    report = read(receipt)
    assert report["status"] == "passed-independent-verification" and not report["heldout_used"]
    directory = ROOT/report["record"]
    case = read(directory/"protocol.json")
    assert sha(directory/"protocol.json") == report["protocol_sha256"]
    env, result = case["environment"],report["arms"][role]
    assert env["seed"] == seed and env["criteria"]["step_dt"] == .02
    truth = next(row for row in previous["batches"] if row["seed"] == seed)["roles"][role]
    path = directory/f"trace-{role}.npz"
    assert sha(path) == truth["trace_sha256"] == result["trace_sha256"]
    control = settings["decision_control"]
    controls = np.asarray(settings["history_controls"])
    start = 101
    source_rows = controls+1-start
    with np.load(path,allow_pickle=False) as trace:
        np.testing.assert_allclose(trace["time"][source_rows],(controls+1)*.02,atol=1e-8,rtol=0)
        inputs = trace["actor_input"]
        history = inputs[source_rows,:,:6]
        np.testing.assert_array_equal(history,trace["observations"][source_rows,:,:6])
        release = np.array(classify_release(normalized(inputs[0,:,3:6]),env["criteria"]["release_alignment_min"]))
    selected = [row for row in truth["rows"] if row["actual_release"] == "supine"]
    ids = np.array([row["world"] for row in selected])
    np.testing.assert_array_equal(ids,np.flatnonzero(release == "supine"))
    assert all(result["timing"]["episodes"][i]["release_pose"] == "supine" for i in ids)
    passed = np.asarray([row["passed"] for row in selected],bool)
    useful = np.asarray([not row["passed"] and row["jam_onset"] is not None
                         and row["jam_onset"]-control >= settings["minimum_lead_controls"] for row in selected],bool)
    assert useful.any() and (~passed).any() and passed.any()
    x = features(history[:,ids])
    assert x.shape == (len(ids),len(settings["features"])) and np.isfinite(x).all()
    return dict(seed=seed,role=role,worlds=ids,features=x,passed=passed,useful=useful,
                trace=path.relative_to(ROOT).as_posix(),trace_sha256=truth["trace_sha256"])


def fit(batches,penalty):
    x = np.concatenate([row["features"] for row in batches])
    y = np.concatenate([row["useful"] for row in batches]).astype(np.float64)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0),1e-6)
    matrix = np.column_stack((np.ones(len(x)),(x-center)/scale))
    positive = y.sum()
    weight = np.where(y == 1,.5/positive,.5/(len(y)-positive))

    def objective(coeff):
        z = matrix @ coeff
        value = weight @ (np.logaddexp(0,z)-y*z)+.5*penalty*np.square(coeff[1:]).sum()
        grad = matrix.T @ (weight*(expit(z)-y))
        grad[1:] += penalty*coeff[1:]
        return value,grad

    fitted = minimize(objective,np.zeros(matrix.shape[1]),jac=True,method="L-BFGS-B",options=dict(ftol=1e-12))
    assert fitted.success and np.isfinite(fitted.x).all()
    return dict(center=center.tolist(),scale=scale.tolist(),intercept=float(fitted.x[0]),
        coefficient=fitted.x[1:].tolist(),positive=int(positive),episodes=len(x))


def predict(model,x):
    return ((x-model["center"])/model["scale"]) @ model["coefficient"]+model["intercept"]


def metrics(batch,scores,cut,settings):
    eligible = scores >= cut
    passed,useful = batch["passed"],batch["useful"]
    n, m = int((~passed).sum()),int(passed.sum())
    coverage = float((eligible & useful).sum()/n)
    rate = float((eligible & passed).sum()/m)
    return dict(seed=batch["seed"],role=batch["role"],failed=n,successful=m,
        useful_activations=int((eligible & useful).sum()),success_activations=int((eligible & passed).sum()),
        useful_failure_coverage=coverage,success_activation_rate=rate,
        passed=bool(coverage >= settings["minimum_useful_failure_coverage"]
                    and rate <= settings["maximum_success_activation_rate"]))


def main():
    settings = read(PLAN)
    assert settings["status"] == "frozen-archive-screen-only" and not OUT.exists()
    assert not any(settings[k] for k in ("physical_substeps","ppo_updates","heldout_used","hardware_io"))
    assert settings["history_controls"] == list(range(settings["decision_control"]-6,settings["decision_control"]+1))
    p = read(ROOT/"project.json")
    assert p["goal_execution"]["status"] == "active" and p["current_phase"] == settings["stage"]
    assert ROOT/p["staged_training_request"]["current_protocol"] == PLAN
    archive = ROOT/settings["archive_protocol"]
    previous_plan = ROOT/settings["previous_protocol"]
    previous_readback = ROOT/settings["previous_readback"]
    protocol,previous = read(archive),read(previous_readback)
    assert previous["protocol_sha256"] == sha(previous_plan) and not previous["signal_gate"]["actor"]
    assert settings["minimum_lead_controls"] == protocol["minimum_lead_controls_before_recontact"]
    for key in ("minimum_useful_failure_coverage","maximum_success_activation_rate"):
        assert settings[key] == protocol[key]
    with ThreadPoolExecutor(max_workers=2) as pool:
        development = list(pool.map(lambda seed:load_batch(seed,settings["training_role"],settings,protocol,previous),
                                    settings["reference_seeds"]))
    folds = [fit([other],settings["l2"]) for other in reversed(development)]
    predictions = [predict(model,row["features"]) for model,row in zip(folds,development,strict=True)]
    cuts = np.unique(np.concatenate(predictions))
    cuts = np.r_[cuts,np.nextafter(cuts[-1],np.inf)]
    eligible = []
    for cutoff in cuts:
        scores = [metrics(row,z,cutoff,settings) for row,z in zip(development,predictions,strict=True)]
        if all(row["success_activation_rate"] <= settings["maximum_success_activation_rate"] for row in scores):
            eligible.append((float(cutoff),scores))
    cutoff, results = max(eligible,key=lambda item:(min(row["useful_failure_coverage"] for row in item[1]),
        sum(row["useful_activations"] for row in item[1]),-sum(row["success_activations"] for row in item[1]),item[0]))
    transfer, model = [],None
    selected = all(row["passed"] for row in results)
    if selected:
        model = fit(development,settings["l2"])
        for role in settings["transfer_roles"]:
            batch = load_batch(settings["transfer_seed"],role,settings,protocol,previous)
            scores = predict(model,batch["features"])
            transfer.append(dict(**metrics(batch,scores,cutoff,settings),trace_sha256=batch["trace_sha256"]))
    output = dict(status="completed-archive-only-supine-imu-screen",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)), input_receipts={str(path.relative_to(ROOT)):sha(path)
            for path in (archive,previous_plan,previous_readback)},candidate_threshold_logit=cutoff,
        thresholds_checked=len(cuts), reference_folds=[dict(**result, model=model,trace_sha256=row["trace_sha256"])
            for result,model,row in zip(results,folds,development,strict=True)],transfer=transfer,refit=model,
        reference_admitted=selected,admitted_for_physical_preflight=selected and all(row["passed"] for row in transfer),
        physical_substeps=0,ppo_updates=0,heldout_used=False,hardware_io=False)
    OUT.mkdir()
    write(OUT/"analysis.json",output)
    print("SUPINE_CAUSAL_IMU_SCREEN",results,transfer,output["admitted_for_physical_preflight"],flush=True)


if __name__ == "__main__":
    main()
