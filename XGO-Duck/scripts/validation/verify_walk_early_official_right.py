"""Independently verify the actual early official actor choices and outcomes."""

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import run_walk_early_official_right as trial
import verify_walk_backward_paired_eval as fixed
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_walk_right_startup_seven_command import REPLAY_STEPS, await_case, right_difference


def trace_decision(trace, p, spec, official, base, decision, arm):
    start, last, limit = (spec["detector"][key] for key in
                         ("first_observation_control", "last_observation_control", "absolute_mean_observed_gyro_max_rad_s"))
    assert last+1 == decision["control"] == spec["detector"]["decision_control"]
    ids = sorted(p["groups"]["turn-right"])
    obs, active = trace["observations"], trace["active"]
    gyro = obs[start-1:last, ids, 2]
    complete = active[start-1:last, ids].all(axis=0) & np.isfinite(gyro).all(axis=0)
    mean = np.where(np.isfinite(gyro), gyro, 0.).mean(axis=0, dtype=np.float64)
    fired = complete & active[last, ids] & (abs(mean) <= limit)
    np.testing.assert_array_equal(decision["mean_gyro_rad_s"], mean)
    np.testing.assert_array_equal(decision["complete_window"], complete)
    np.testing.assert_array_equal(decision["active_at_decision"], active[last, ids])
    assert decision["right_worlds"] == ids and decision["triggered_worlds"] == np.asarray(ids)[fired].tolist()
    guarded = []
    if arm != trial.ARMS[0]:
        assert set(decision["candidates"]) == {str(world) for world in decision["triggered_worlds"]}
        low, high = np.asarray(p["joint_limits"], np.float32).T
        home = np.asarray(p["model_home"], np.float32)
        cap = np.asarray(spec["target_step_cap_rad"], np.float32)
        for world in decision["triggered_worlds"]:
            row = decision["candidates"][str(world)]
            inputs = {"obs": obs[last, world:world+1]}
            alternate = official.run(None, inputs)[0][0]
            normal = np.clip(home + base.run(None, inputs)[0][0], low, high)
            np.testing.assert_allclose(row["official_raw"], alternate, atol=2e-5, rtol=2e-5)
            previous = trace["applied_joint_targets"][last-1, world]
            np.testing.assert_array_equal(np.asarray(row["previous_target"], np.float32), previous)
            request = home + alternate
            target = np.clip(request, low, high)
            delta = abs(target-previous)
            np.testing.assert_allclose(row["target_delta"], delta, atol=2e-5, rtol=2e-5)
            excessive = np.flatnonzero(delta > cap+1e-6).tolist()
            limited = np.flatnonzero((request != target) & (normal != low) & (normal != high)).tolist()
            assert row["excessive_joints"] == excessive and row["newly_limited_joints"] == limited
            assert row["finite"] == bool(np.isfinite(target).all())
            if row["finite"] and not (excessive or limited):
                guarded.append(world)
    assert decision["guarded_worlds"] == guarded
    selected = guarded if arm == trial.ARMS[-1] else []
    assert decision["selected_worlds"] == selected
    uses_official = np.zeros_like(active)
    uses_official[:, p["groups"]["forward"]] = True
    uses_official[last:, selected] = True
    return uses_official


class RoutedOrt:
    def __init__(self, trace, use_official, sessions):
        self.rows = iter((bool(use_official[i, j]), trace["observations"][i, j:j+1])
                         for i in REPLAY_STEPS for j in np.flatnonzero(trace["active"][i]))
        self.sessions = sessions

    def run(self, outputs, inputs):
        official, row = next(self.rows)
        np.testing.assert_array_equal(row, inputs["obs"])
        return self.sessions["official" if official else "base"].run(outputs, inputs)


@torch.inference_mode()
def verify_case(spec, item, sessions, actors):
    p, folder = item["environment"], ROOT/item["record"]
    await_case(folder)
    arm = item["arm"]
    zero_path, decision_path = (folder/name for name in ("early-route-zero.json", "right-decision.json"))
    zero, decision = read(zero_path), read(decision_path)
    assert zero["status"] == "passed-actual-cuda-early-route" and zero["physical_substeps"] == 0
    assert zero["protocol_sha256"] == sha(trial.PLAN)
    assert zero["fixture_sha256"] == sha(trial.OUT/"archived-input-fixture.npz")
    ready = read(folder/"windows-observer-ready.json")
    assert ready["observer_host"] == "Windows" and ready["protocol_sha256"] == sha(trial.PLAN)
    route = read(folder/"model-routing.json")
    assert route["arm"] == arm and route["base_inference_rows"] == route["right_inference_rows"] == 192
    assert route["official_sha256"] == spec["models"]["official"][1]
    assert route["base_sha256"] == route["right_sha256"] == spec["models"]["base"][1]
    paths = [folder/"trace-shared-prefix.npz", folder/"trace-actuator-feedback.npz"]
    fields = ("observations", "actions", "active", "applied_joint_targets")
    pieces = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            pieces.append({key: f[key] for key in fields})
    data = {key: np.concatenate([piece[key] for piece in pieces]) for key in fields}
    flags = trace_decision(data, p, spec, sessions["official"], sessions["base"], decision, arm)
    check_inputs = fixed.scorer.verify_inputs

    def replay(trace, env, settings, desired, role, unused):
        np.testing.assert_array_equal(np.asarray(route["external_commands"], np.float32), desired)
        model = RoutedOrt(trace, flags, sessions)
        result = check_inputs(trace, env, settings, desired, role, model)
        assert next(model.rows, None) is None
        return result

    with patch.object(fixed, "PLAN", trial.PLAN), patch.object(fixed.scorer, "verify_inputs", replay):
        receipt = fixed.verify_case(spec, item)
    assert [sha(path) for path in paths] == receipt["trace_hashes"]
    assert zero["native_start_sha256"] == receipt["native_start_sha256"]
    counts, errors = {}, {}
    for role, mask in (("official", flags), ("base", ~flags)):
        mask = mask & data["active"]
        rows, expected = torch.from_numpy(data["observations"][mask]), data["actions"][mask]
        actual = torch.cat([actors[role](trial.native.rollout.td(chunk)) for chunk in rows.split(8192)]).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        counts[role] = len(rows)
        errors[role] = float(abs(actual-expected).max())
    receipt.update(status="passed-independent-right-turn-seven-command-readback", early_route_zero_sha256=sha(zero_path),
        decision_sha256=sha(decision_path), triggered_worlds=decision["triggered_worlds"],
        guarded_worlds=decision["guarded_worlds"], selected_worlds=decision["selected_worlds"],
        verified_active_actions=counts, maximum_nominal_action_errors=errors)
    first = {key: data[key][0].copy() for key in ("observations", "actions", "applied_joint_targets")}
    return receipt, first


def outcome(spec, receipts, first):
    assert len({row["native_start_sha256"] for row in receipts.values()}) == 1
    for values in list(first.values())[1:]:
        for key, value in values.items():
            np.testing.assert_array_equal(value, first[trial.ARMS[0]][key])
    parent, shadow, candidate = (receipts[arm] for arm in trial.ARMS)
    failed = lambda row: sorted(e["world"] for e in row["scores"]["turn-right"]["episode_scores"] if not e["success"])
    expected = spec["expected_triggered_worlds"]
    matched = (failed(parent) == failed(shadow) == expected and
               all(row["triggered_worlds"] == expected for row in receipts.values()) and
               shadow["guarded_worlds"] == candidate["guarded_worlds"] == expected)
    comparisons = {}
    for arm in trial.ARMS[:2]:
        reference, other = receipts[arm]["scores"], candidate["scores"]
        retention = fixed.compare(reference, other, spec["maximum_speed_loss_m_s"])
        right = right_difference(reference, other)
        comparisons[arm] = dict(retention=retention, right=right, gate=(
            len(right["additional_success_worlds"]) >= 1 and not right["lost_success_worlds"] and
            retention["forward_retained"] and abs(retention["forward_speed_change_m_s"]) <= spec["maximum_speed_loss_m_s"]
            and retention["backward_net"] >= 0))
    seven = all(row["functional_development_pass"] for row in candidate["scores"].values())
    passed = matched and seven and all(item["gate"] for item in comparisons.values())
    return dict(eligible_for_fresh_development=passed, reference_and_trigger_matched=matched,
        reference_failed_right_worlds={arm: failed(receipts[arm]) for arm in trial.ARMS[:2]},
        candidate_failed_right_worlds=failed(candidate), candidate_seven_commands_passed=seven,
        candidate_comparisons=comparisons, all_first_observations_actions_and_targets_bitwise_exact=True)


def main():
    spec = trial.checked()
    torch.set_num_threads(3)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    actors, sessions = {}, {}
    for name in ("base", "official"):
        path, checksum = spec["models"][name]
        sessions[name] = ort.InferenceSession(str(ROOT/path), options, providers=["CPUExecutionProvider"])
        actors[name] = trial.native.rollout.make_actor(dict(model=path, model_sha256=checksum), "cpu")
    receipts, first, records = {}, {}, {}
    for key, item in spec["cases"].items():
        arm = item["arm"]
        row, first[arm] = verify_case(spec, item, sessions, actors)
        receipts[arm] = row
        path = ROOT/item["record"]/"verification.json"
        write(path, row)
        records[key] = dict(record=item["record"], verification_sha256=sha(path))
        print("EARLY_OFFICIAL_RIGHT_VERIFIED", arm, row["triggered_worlds"], row["selected_worlds"],
              {name: score["successes"] for name, score in row["scores"].items()}, flush=True)
    execution = read(trial.OUT/"execution.json")
    assert [x["case"] for x in execution] == list(spec["cases"]) and all(x["returncode"] == 0 for x in execution)
    total = sum(row["transitions"] for row in receipts.values())
    steps = sum(row["physical_substeps"] for row in receipts.values())
    assert total == spec["maximum_transitions"] and steps == spec["maximum_batched_substeps"]
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-independent-early-official-right-readback", protocol_sha256=sha(trial.PLAN),
        records=records, actual_transitions=total, batched_physical_substeps=steps, learning_updates=0,
        heldout_used=False, hardware_io=False, **outcome(spec, receipts, first))
    write(trial.OUT/"analysis.json", result)
    print("EARLY_OFFICIAL_RIGHT_DECISION", result["eligible_for_fresh_development"], flush=True)


if __name__ == "__main__":
    main()
