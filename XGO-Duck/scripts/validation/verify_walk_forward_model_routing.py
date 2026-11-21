"""Independently replay both model paths and score the six frozen cases."""

from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import verify_walk_backward_paired_eval as original
from command_input_mapping import transform
from record_io import write
from run_walk_forward_model_routing import OUT, PLAN, checked, native
from s1_reference_inputs import ROOT, read, sha


class NominalReadback:
    """For replay only, use longitudinal input after proving it matches the external command."""

    def __init__(self, official, learned):
        self.official = official
        self.learned = learned

    def run(self, outputs, inputs):
        row = inputs["obs"]
        assert row.shape == (1, 61)
        session = self.official if row[0, 48] > 0.0 else self.learned
        return session.run(outputs, inputs)


@torch.inference_mode()
def verify_all_nominal(folder, p, receipt, spec):
    actors = {name: native.rollout.make_actor(dict(model=path, model_sha256=digest), "cpu")
              for name, (path, digest) in spec["models"].items()}
    forward = np.zeros(p["parallel_worlds"], dtype=bool)
    forward[p["groups"]["forward"]] = True
    counts, errors = dict(official=0, control=0), dict(official=0., control=0.)
    for index, file in enumerate(("trace-shared-prefix.npz", "trace-actuator-feedback.npz")):
        path = folder / file
        assert sha(path) == receipt["trace_hashes"][index]
        with np.load(path, allow_pickle=False) as trace:
            active, obs, raw = (trace[name] for name in ("active", "observations", "actions"))
        for name, mask in (("official", forward), ("control", ~forward)):
            selected = active & mask[None]
            inputs, expected = torch.from_numpy(obs[selected]), raw[selected]
            actual = torch.cat([actors[name](native.rollout.td(part)) for part in inputs.split(8192)]).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            counts[name] += len(actual)
            errors[name] = max(errors[name], float(abs(actual-expected).max()))
    return dict(nominal_active_inputs_per_model=counts, maximum_all_nominal_action_errors=errors)


def verify_case(spec, item, sessions):
    folder = ROOT / item["record"]
    ready = folder / "windows-observer-ready.json"
    windows = folder / "windows-view-verification.json"
    view, observation = read(ready), read(windows)
    assert view["status"] == "starting-page-observed" and view["observer_host"] == observation["observer_host"] == "Windows"
    assert view["protocol_sha256"] == sha(PLAN) and view["run_id"] == folder.name
    assert observation["observations"][0]["status"] == "starting" and observation["observations"][0]["controls"] == 0
    if item["arm"] != "routed":
        receipt = original.verify_case(spec, item)
    else:
        p = item["environment"]
        route = read(folder / "model-routing.json")
        secondary = read(folder / "learned-startup" / "startup.json")
        assert route["status"] == "passed-two-model-startup" and route["physical_substeps"] == 0
        assert route["official_worlds"] == sorted(p["groups"]["forward"])
        assert route["official_sha256"] == p["model_sha256"] == spec["models"]["official"][1]
        assert route["learned_sha256"] == secondary["model_sha256"] == spec["models"]["control"][1]
        assert secondary["status"] == "passed" and secondary["physical_steps"] == 0
        assert secondary["complete_processor_memory_restored"] and secondary["cpu_cuda_rng_unchanged"]
        assert route["observer_ready_sha256"] == sha(ready)
        verify_inputs = original.scorer.verify_inputs

        def guarded_inputs(trace, env, settings, desired, role, session):
            nominal_x = trace["observations"][..., 48]
            mapped_x = transform(desired)[:, 0]
            np.testing.assert_array_equal(nominal_x, np.broadcast_to(mapped_x, nominal_x.shape))
            np.testing.assert_array_equal(np.flatnonzero(mapped_x > 0), route["official_worlds"])
            np.testing.assert_array_equal(np.asarray(route["external_commands"], dtype=np.float32), desired)
            return verify_inputs(trace, env, settings, desired, role, session)

        pair = NominalReadback(sessions["official"], sessions["control"])
        readback = SimpleNamespace(SessionOptions=ort.SessionOptions, InferenceSession=lambda *args, **kwargs: pair)
        with patch.object(original, "ort", readback), patch.object(original.scorer, "verify_inputs", guarded_inputs):
            receipt = original.verify_case(spec, item)
        receipt.update(verify_all_nominal(folder, p, receipt, spec),
                       model_routing_sha256=sha(folder / "model-routing.json"),
                       learned_startup_sha256=sha(folder / "learned-startup" / "startup.json"))
    receipt["windows_observer_ready_sha256"] = sha(ready)
    return receipt


def main():
    spec = checked()
    torch.set_num_threads(3)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(ROOT / path), sess_options=options, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items()}
    receipts = {}
    with patch.object(original, "PLAN", PLAN):
        for key, item in spec["cases"].items():
            folder = ROOT / item["record"]
            exit_path = folder / "execution-exit.txt"
            windows = folder / "windows-view-verification.json"
            deadline = monotonic() + 440
            while not (exit_path.is_file() and windows.is_file()):
                if exit_path.is_file() and exit_path.read_text().strip() not in {"", "0"}:
                    raise RuntimeError(f"{key} exited before readback")
                if monotonic() >= deadline:
                    raise TimeoutError(f"{key} did not publish both completion files")
                sleep(2)
            receipt = verify_case(spec, item, sessions)
            receipts[key] = receipt
            write(folder / "verification.json", receipt)
            print("FORWARD_ROUTING_VERIFIED", key, {k: v["successes"] for k, v in receipt["scores"].items()}, flush=True)
    execution = read(OUT / "execution.json")
    assert [row["case"] for row in execution] == list(spec["cases"]) and all(row["returncode"] == 0 for row in execution)
    cohorts = {}
    for seed in spec["development_seeds"]:
        matched = {arm: receipts[f"{seed}-{arm}"] for arm in spec["arms"]}
        assert len({row["native_start_sha256"] for row in matched.values()}) == 1
        scores = {arm: row["scores"] for arm, row in matched.items()}
        functional = {arm: all(score["functional_development_pass"] for score in rows.values()) for arm, rows in scores.items()}
        official = original.compare(scores["official"], scores["routed"], spec["maximum_speed_loss_m_s"])
        control = original.compare(scores["control"], scores["routed"], spec["maximum_speed_loss_m_s"])
        eligible = functional["routed"] and official["forward_retained"] and control["backward_net"] >= 0
        cohorts[str(seed)] = dict(scores=scores, functional_pass=functional, routed_vs_official=official,
                                  routed_vs_control=control, routed_eligible=eligible)
    transitions = sum(row["transitions"] for row in receipts.values())
    substeps = sum(row["physical_substeps"] for row in receipts.values())
    assert transitions == spec["maximum_transitions"] and substeps == spec["maximum_batched_substeps"]
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-independent-fixed-routing-readback", protocol_sha256=sha(PLAN), cohorts=cohorts,
        eligible_for_same_scene_development=all(row["routed_eligible"] for row in cohorts.values()),
        records={key: dict(path=row["record"], verification_sha256=sha(ROOT / row["record"] / "verification.json"))
                 for key, row in receipts.items()},
        actual_transitions=transitions, physical_substeps=substeps, learning_updates=0, heldout_used=False, hardware_io=False)
    write(OUT / "analysis.json", result)
    print("FORWARD_ROUTING_DECISION", result["eligible_for_same_scene_development"], flush=True)


if __name__ == "__main__":
    main()
