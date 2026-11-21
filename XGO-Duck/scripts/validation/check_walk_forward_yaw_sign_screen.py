"""Check a causal yaw sign guard on saved forward walks and measured pulse branches."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
from review_common_walk_yaw_forecast import predict
import run_walk_forward_stance_response as response


PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_sign_screen.json"
OUT = response.STAGE


def inputs(observed, previous, p, observer, forecast, rule):
    columns = [20+p["joint_names"].index(f"{side}_hip_pitch") for side in ("left","right")]
    now = np.stack((observed[...,2],observed[...,columns].sum(axis=-1)),axis=-1)
    then = np.stack((previous[...,2],previous[...,columns].sum(axis=-1)),axis=-1)
    features = np.concatenate((now,now-then),axis=-1)
    score = predict(features, forecast)
    gyro = now[...,0]
    phase = observer.predict(observed)
    eligible = ((phase > 0) & (abs(gyro) >= rule["minimum_abs_observed_gyro_rad_s"]) &
                (abs(score) >= rule["minimum_abs_forecast_score"]) & (np.sign(gyro) == np.sign(score)))
    return phase, np.where(eligible,np.sign(score),0).astype(np.int8)


def archive(group, rule, observer, forecast):
    protocol, source = (ROOT/group[k] for k in ("protocol","receipt"))
    spec, result = read(protocol), read(source)
    assert result["protocol_sha256"] == sha(protocol) and not result["heldout_used"]
    rows = {}
    for key in group["cases"]:
        item = result["records"][key]
        record, p = ROOT/item["path"], spec["cases"][key]["environment"]
        receipt = record/"verification.json"
        assert sha(receipt) == item["verification_sha256"]
        proof = read(receipt)
        selected = p["groups"]["forward"]
        traces = []
        for name, fingerprint in zip(("shared-prefix","actuator-feedback"),proof["trace_hashes"],strict=True):
            path = record/f"trace-{name}.npz"
            assert sha(path) == fingerprint
            with np.load(path, allow_pickle=False) as f:
                assert f["active"][:,selected].all()
                traces.append((f["observations"][:,selected],f["yaw_rate"][:,selected],f["contact"][:,selected]))
        obs, yaw, contacts = (np.concatenate(block) for block in zip(*traces,strict=True))
        assert obs.shape[:2] == (1000,32)
        phase, sign = inputs(obs[50:],obs[49:-1],p,observer,forecast,rule)
        live = contacts[49:-1,:,0].astype(int)+2*contacts[49:-1,:,1]
        target = np.sign(yaw[50:])
        sides = {}
        for name,i in (("left-only",1),("right-only",2)):
            found = phase == i
            keep = found & (sign != 0)
            precision = float((sign[keep] == target[keep]).mean())
            coverage = float(keep.sum()/found.sum())
            sides[name] = dict(support_rows=int(found.sum()), guarded_rows=int(keep.sum()),
                next_step_sign_precision=precision, guarded_coverage=coverage,
                wrong_support_rows=int((keep & (live != i)).sum()),
                admitted=bool(precision >= rule["minimum_direction_precision"] and
                              coverage >= rule["minimum_coverage_of_recognized_support"]))
        rows[key] = dict(source_receipt_sha256=sha(receipt), sides=sides)
    return rows


def pulse(rule, observer, forecast):
    spec = response.checked()
    item = next(iter(spec["cases"].values()))
    p, record = item["environment"], ROOT/item["record"]
    result = read(ROOT/rule["pulse_response"])
    assert result["protocol_sha256"] == sha(response.PLAN) and result["both_directions_admitted"]
    assert not result["admitted_for_sustained_feedback_preflight"]
    prefix = record/"trace-shared-prefix.npz"
    assert sha(prefix) == result["shared_prefix_sha256"]
    traces, applied = {}, {}
    for role in response.ROLES:
        proof = result["roles"][role]
        trace = record/f"trace-{role}.npz"
        witness = record/f"pulse-inputs-{role}.npz"
        assert sha(trace) == proof["trace_sha256"] and sha(witness) == proof["pulse_sha256"]
        with np.load(witness, allow_pickle=False) as f:
            ids = f["world"]
            phase_saved = f["phase"]
            applied[role] = f["applied_yaw"]
        with np.load(trace, allow_pickle=False) as f:
            traces[role] = f["yaw_rate"][:3,ids].astype(np.float64)
            if role == "reference-a":
                observed = f["observations"][0,ids]
    with np.load(prefix, allow_pickle=False) as f:
        previous = f["observations"][-1,ids]
    phase, sign = inputs(observed,previous,p,observer,forecast,rule)
    np.testing.assert_array_equal(phase,phase_saved)
    base = applied["reference-a"]
    np.testing.assert_array_equal(base,applied["reference-b"])
    # Negative measured pulses oppose positive estimated yaw, and vice versa.
    selected = np.where(sign == 1,applied["negative"],applied["positive"])
    eligible = (sign != 0) & (selected != base)
    selected_yaw = np.where(sign[None,:] == 1,traces["negative"],traces["positive"])
    a,b = (traces[k] for k in response.ROLES[:2])
    gain = (abs(a)+abs(b))/2-abs(selected_yaw)
    spread = abs(abs(a)-abs(b))
    out = {}
    rng = np.random.default_rng(1157)
    for name,i in (("left-only",1),("right-only",2)):
        keep = eligible & (phase == i)
        count = int(keep.sum())
        if not count:
            out[name] = dict(worlds=[],admitted=False)
            continue
        world_gain = gain[0,keep]
        ci = np.quantile(rng.choice(world_gain,size=(2000,count),replace=True).mean(axis=1),[.025,.975]).tolist()
        out[name] = dict(worlds=ids[keep].tolist(), negative_pulses=int((keep & (sign == 1)).sum()),
            positive_pulses=int((keep & (sign == -1)).sum()),
            baseline_actual_sign_accuracy=float((sign[keep] == np.sign((a[0,keep]+b[0,keep])/2)).mean()),
            mean_absolute_yaw_gain_per_step_rad_s=gain[:,keep].mean(axis=1).tolist(),
            immediate_mean_gain_ci95=ci,
            immediate_reference_repeat_spread_rad_s=float(spread[0,keep].mean()),
            admitted=bool(count >= rule["minimum_guarded_pulse_worlds_per_side"] and
                          ci[0] > 0. and world_gain.mean() > spread[0,keep].mean()))
    return out


def main():
    rule = read(PLAN)
    assert rule["status"] == "frozen-archive-cpu-screen" and rule["require_gyro_and_forecast_sign_agreement"]
    assert not any(rule[k] for k in ("heldout_used","physical_substeps","policy_updates","hardware_io"))
    observer_plan = ROOT/rule["stance_observation"]
    settings = read(observer_plan)
    observation_file = OUT/"forward-stance-observation-cpu.json"
    observation = read(observation_file)
    assert observation["protocol_sha256"] == sha(observer_plan) and observation["admitted_for_control_design"]
    observer = StanceObserver(observation["model"],settings["minimum_score_separation"])
    forecast_path = ROOT/rule["actuator_forecast"]
    sources = read(forecast_path)
    assert sources["status"] == "passed-saved-linear-response-check" and not sources["heldout_used"]
    forecast = sources["forecast_models"][0]
    archival = {key: row for group in settings["verification"] for key,row in archive(group,rule,observer,forecast).items()}
    measured = pulse(rule,observer,forecast)
    admit = all(side["admitted"] for row in archival.values() for side in row["sides"].values())
    admit = admit and all(side["admitted"] for side in measured.values())
    write(OUT/"forward-yaw-sign-screen-cpu.json",dict(status="checked-causal-yaw-sign-screen",
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(PLAN), observer_sha256=sha(observation_file),
        forecast_sha256=sha(forecast_path), pulse_result_sha256=sha(ROOT/rule["pulse_response"]),
        archived_development=archival, pulse_branch_reselection=measured,
        admitted_for_new_mixed_pulse=admit, physical_substeps=0, policy_updates=0, heldout_used=False))
    print("FORWARD_YAW_SIGN_CPU",admit,{key:{side:row["next_step_sign_precision"] for side,row in value["sides"].items()}
        for key,value in archival.items()},measured,flush=True)


if __name__ == "__main__":
    main()
