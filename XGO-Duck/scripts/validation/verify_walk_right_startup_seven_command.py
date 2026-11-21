"""Replay the recorded right-only model paths and score all original seven commands."""

from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_walk_right_startup_seven_command import ARMS, OUT, PLAN, checked, native
import verify_walk_backward_paired_eval as original


REPLAY_STEPS = (0, 49, 50, 99, 199, 499, 749, 999)


def model_ids(p, item):
    ids = np.full(p["parallel_worlds"], "base", dtype="U32")
    ids[p["groups"]["forward"]] = "official"
    ids[p["groups"]["turn-right"]] = item["right"]
    return ids


class IndexedReadback:
    def __init__(self, trace, names, sessions):
        active, obs = trace["active"], trace["observations"]
        self.rows = iter((names[j], obs[i, j:j+1])
                         for i in REPLAY_STEPS for j in np.flatnonzero(active[i]))
        self.sessions = sessions

    def run(self, outputs, inputs):
        name, expected = next(self.rows)
        np.testing.assert_array_equal(inputs["obs"], expected)
        return self.sessions[name].run(outputs, inputs)


def await_case(folder):
    exit_path, observer = folder / "execution-exit.txt", folder / "windows-view-verification.json"
    deadline = monotonic()+450
    while not (exit_path.is_file() and observer.is_file()):
        if exit_path.is_file() and exit_path.read_text().strip() not in {"", "0"}:
            raise RuntimeError(f"{folder.name} exited before readback")
        if monotonic() >= deadline:
            raise TimeoutError(folder.name)
        sleep(2)


@torch.inference_mode()
def verify_every_nominal(folder, receipt, names, actors):
    counts = {name: 0 for name in dict.fromkeys(names)}
    errors = {name: 0. for name in counts}
    for i, filename in enumerate(("trace-shared-prefix.npz", "trace-actuator-feedback.npz")):
        path = folder / filename
        assert sha(path) == receipt["trace_hashes"][i]
        with np.load(path, allow_pickle=False) as f:
            active, obs, actions = (f[key] for key in ("active", "observations", "actions"))
        for name in counts:
            mask = active & (names == name)[None, :]
            expected = actions[mask]
            rows = torch.from_numpy(obs[mask])
            if not len(rows):
                continue
            actual = torch.cat([actors[name](native.rollout.td(chunk)) for chunk in rows.split(8192)]).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            counts[name] += len(rows)
            errors[name] = max(errors[name], float(np.abs(actual-expected).max()))
    assert all(count > 0 for count in counts.values())
    return dict(verified_active_actions=counts, maximum_nominal_action_errors=errors)


def verify_case(spec, item, sessions, actors):
    p, folder = item["environment"], ROOT / item["record"]
    await_case(folder)
    ready_path, observation_path = (folder / filename for filename in
        ("windows-observer-ready.json", "windows-view-verification.json"))
    ready, observer, route = (read(path) for path in (ready_path, observation_path, folder / "model-routing.json"))
    assert ready["status"] == "starting-page-observed" and ready["protocol_sha256"] == sha(PLAN)
    assert ready["observer_host"] == observer["observer_host"] == "Windows" and ready["run_id"] == folder.name
    assert observer["observations"][0]["status"] == "starting" and not observer["observations"][0]["controls"]
    assert route["status"] == "passed-right-route-startup" and route["physical_substeps"] == 0
    assert route["arm"] == item["arm"] == read(folder / "evaluation.json")["arm"]
    assert route["official_worlds"] == sorted(p["groups"]["forward"])
    assert route["right_worlds"] == sorted(p["groups"]["turn-right"])
    assert route["base_inference_rows"] == route["right_inference_rows"] == 192
    assert route["official_sha256"] == p["model_sha256"] == spec["models"]["official"][1]
    assert route["base_sha256"] == spec["models"]["base"][1] and route["right_sha256"] == spec["models"][item["right"]][1]
    extra = {}
    for name in dict.fromkeys(("base", item["right"])):
        path = folder / f"{name}-startup" / "startup.json"
        start = read(path)
        assert start["status"] == "passed" and not start["physical_steps"]
        assert start["model_sha256"] == spec["models"][name][1]
        assert start["cpu_cuda_rng_unchanged"] and start["complete_processor_memory_restored"]
        extra[name] = sha(path)

    names = model_ids(p, item)
    verify_inputs = original.scorer.verify_inputs

    def routed_inputs(trace, env, settings, desired, role, unused_session):
        np.testing.assert_array_equal(np.asarray(route["external_commands"], np.float32), desired)
        replay = IndexedReadback(trace, names, sessions)
        result = verify_inputs(trace, env, settings, desired, role, replay)
        assert next(replay.rows, None) is None
        return result

    with patch.object(original, "PLAN", PLAN), patch.object(original.scorer, "verify_inputs", routed_inputs):
        receipt = original.verify_case(spec, item)
    receipt.update(verify_every_nominal(folder, receipt, names, actors),
        status="passed-independent-right-turn-seven-command-readback",
        model_routing_sha256=sha(folder / "model-routing.json"), startup_readbacks=extra,
        windows_observer_ready_sha256=sha(ready_path), right_model_sha256=route["right_sha256"])
    return receipt


def right_difference(base, other):
    a, b = (scores["turn-right"]["episode_scores"] for scores in (base, other))
    assert [x["world"] for x in a] == [x["world"] for x in b]
    passed = [x["world"] for x, y in zip(a, b, strict=True) if not x["success"] and y["success"]]
    lost = [x["world"] for x, y in zip(a, b, strict=True) if x["success"] and not y["success"]]
    return dict(additional_success_worlds=passed, lost_success_worlds=lost, net=len(passed)-len(lost))


def same_start(spec, seed, receipts):
    cohort = [receipts[f"{seed}-{arm}"] for arm in ARMS]
    assert len({row["native_start_sha256"] for row in cohort}) == 1
    p = spec["cases"][f"{seed}-{ARMS[0]}"]["environment"]
    right = np.asarray(p["groups"]["turn-right"])
    kept = np.ones(p["parallel_worlds"], bool)
    kept[right] = False
    first = []
    for row in cohort:
        path = ROOT / row["record"] / "trace-shared-prefix.npz"
        assert sha(path) == row["trace_hashes"][0]
        with np.load(path, allow_pickle=False) as f:
            first.append({name: f[name][0] for name in ("observations", "actions", "requested_joint_targets",
                                                        "applied_joint_targets")})
    for other in first[1:]:
        np.testing.assert_array_equal(other["observations"], first[0]["observations"])
        for name in ("actions", "requested_joint_targets", "applied_joint_targets"):
            np.testing.assert_array_equal(other[name][kept], first[0][name][kept])
    deltas = {arm: float(np.abs(row["actions"][right]-first[0]["actions"][right]).max())
              for arm, row in zip(ARMS[1:], first[1:], strict=True)}
    assert all(value > 0 for value in deltas.values())
    return dict(matched_start_sha256=cohort[0]["native_start_sha256"], non_right_first_outputs_bitwise_exact=True,
                right_first_max_action_difference=deltas)


def summarize(spec, receipts):
    cohorts = {}
    for seed in spec["development_seeds"]:
        start = same_start(spec, seed, receipts)
        scores = {arm: receipts[f"{seed}-{arm}"]["scores"] for arm in ARMS}
        functional = {arm: all(item["functional_development_pass"] for item in score.values()) for arm, score in scores.items()}
        parent = scores[ARMS[0]]
        retained = {}
        for arm in ARMS[1:]:
            change = original.compare(parent, scores[arm], spec["maximum_speed_loss_m_s"])
            change["forward_within_absolute_speed_tolerance"] = abs(change["forward_speed_change_m_s"]) <= spec["maximum_speed_loss_m_s"]
            change.update(right=right_difference(parent, scores[arm]))
            retained[arm] = change
        candidate = right_difference(scores[ARMS[1]], scores[ARMS[2]])
        cohorts[str(seed)] = dict(scores=scores, functional_pass=functional, versus_parent=retained,
            right_candidate_versus_control=candidate, **start)
    eligible = {ARMS[0]: all(row["functional_pass"][ARMS[0]] for row in cohorts.values())}
    for arm in ARMS[1:]:
        eligible[arm] = all(row["functional_pass"][arm] and row["versus_parent"][arm]["forward_retained"]
                           and row["versus_parent"][arm]["forward_within_absolute_speed_tolerance"]
                           and row["versus_parent"][arm]["backward_net"] >= 0 for row in cohorts.values())
    deltas = [row["right_candidate_versus_control"]["net"] for row in cohorts.values()]
    shaping_gain = eligible[ARMS[2]] and min(deltas) >= 0 and sum(deltas) > 0
    return cohorts, eligible, shaping_gain


def main():
    spec = checked()
    torch.set_num_threads(3)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(ROOT / path), options, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items()}
    actors = {name: native.rollout.make_actor(dict(model=path, model_sha256=checksum), "cpu")
              for name, (path, checksum) in spec["models"].items()}
    receipts = {}
    for key, item in spec["cases"].items():
        row = verify_case(spec, item, sessions, actors)
        receipts[key] = row
        write(ROOT / item["record"] / "verification.json", row)
        print("RIGHT_STARTUP_EVAL_VERIFIED", key, {name: score["successes"] for name, score in row["scores"].items()}, flush=True)
    execution = read(OUT / "execution.json")
    assert [x["case"] for x in execution] == list(spec["cases"]) and all(x["returncode"] == 0 for x in execution)
    cohorts, eligible, gain = summarize(spec, receipts)
    transitions = sum(row["transitions"] for row in receipts.values())
    steps = sum(row["physical_substeps"] for row in receipts.values())
    assert transitions == spec["maximum_transitions"] and steps == spec["maximum_batched_substeps"]
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-independent-right-turn-seven-command-comparison", protocol_sha256=sha(PLAN),
        cohorts=cohorts, eligible_for_independent_confirmation=eligible, shaping_gain_on_development_counts=gain,
        actual_transitions=transitions, batched_physical_substeps=steps, learning_updates=0, heldout_used=False,
        hardware_io=False, records={key: dict(path=row["record"],
        verification_sha256=sha(ROOT / row["record"] / "verification.json")) for key, row in receipts.items()})
    write(OUT / "analysis.json", result)
    print("RIGHT_STARTUP_EVAL_DECISION", eligible, gain, flush=True)


if __name__ == "__main__":
    main()
