"""Read back each full cohort and enforce the unchanged seven-command gates."""

from argparse import ArgumentParser
from time import monotonic, sleep
from unittest.mock import patch

import onnxruntime as ort
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_forward_margin as trial
import verify_walk_right_startup_seven_command as verifier


def evaluate(spec, cohort):
    local = trial.scoped(spec, cohort)
    directory = trial.OUT/cohort
    receipts = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    torch.set_num_threads(3)
    sessions = {name: ort.InferenceSession(str(ROOT/path), options, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items()}
    actors = {name: trial.native.rollout.make_actor(dict(model=path, model_sha256=checksum), "cpu")
              for name, (path, checksum) in spec["models"].items()}
    with patch.object(verifier, "PLAN", trial.PLAN):
        for key, item in local["cases"].items():
            settings = dict(local, feedback=dict(local["feedback"], maximum_yaw_input_rad_s=item["forward_cap_rad_s"]))
            receipts[key] = row = verifier.verify_case(settings, item, sessions, actors)
            write(ROOT/item["record"]/"verification.json", row)
            print("FORWARD_MARGIN_CASE", key, {name: score["successes"] for name, score in row["scores"].items()}, flush=True)
        seed = next(iter(local["cases"].values()))["seed"]
        shared = verifier.same_start(local, seed, receipts)
    ledger = directory/"execution.json"
    deadline = monotonic()+10
    while monotonic() < deadline and (not ledger.exists() or len(read(ledger)) != len(receipts)):
        sleep(.2)
    executions = read(ledger)
    assert [row["case"] for row in executions] == list(receipts) and all(row["returncode"] == 0 for row in executions)
    scores = {arm: receipts[f"{seed}-{arm}"]["scores"] for arm in trial.ARMS}
    parent, control, wider = (scores[arm] for arm in trial.ARMS)
    paired = {}
    for arm, baseline in zip(trial.ARMS[:2], (parent, control), strict=True):
        effect = verifier.original.compare(baseline, wider, spec["maximum_speed_loss_m_s"])
        effect["forward_speed_inside_absolute_tolerance"] = abs(effect["forward_speed_change_m_s"]) <= spec["maximum_speed_loss_m_s"]
        effect["right"] = verifier.right_difference(baseline, wider)
        effect["mean_forward_yaw_improvement_rad_s"] = baseline["forward"]["yaw_error"]-wider["forward"]["yaw_error"]
        paired[arm] = effect
    gates = dict(original_seven_commands=all(row["functional_development_pass"] for row in wider.values()),
        all_forward_success_counts_retained=all(row["forward_retained"] for row in paired.values()),
        all_forward_speed_changes_within_tolerance=all(row["forward_speed_inside_absolute_tolerance"] for row in paired.values()),
        no_backward_net_loss=all(row["backward_net"] >= 0 for row in paired.values()),
        no_right_net_loss=all(row["right"]["net"] >= 0 for row in paired.values()),
        minimum_yaw_improvement=paired[trial.ARMS[1]]["mean_forward_yaw_improvement_rad_s"] >= spec["minimum_yaw_improvement"][cohort])
    passed = all(gates.values())
    transitions = sum(row["transitions"] for row in receipts.values())
    steps = sum(row["physical_substeps"] for row in receipts.values())
    assert transitions == spec["maximum_transitions"]//2 and steps == spec["maximum_batched_substeps"]//2
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-independent-forward-feedback-margin-readback", protocol_sha256=sha(trial.PLAN), cohort=cohort,
        seed=seed, scores=scores, paired=paired, gates=gates, passed_development_gate=passed,
        promoted_to_fresh_development=cohort == "known" and passed,
        actual_transitions=transitions, physical_substeps=steps, learning_updates=0, heldout_used=False,
        hardware_io=False, records={key: dict(path=row["record"],
        verification_sha256=sha(ROOT/row["record"]/"verification.json")) for key, row in receipts.items()}, **shared)
    write(directory/"analysis.json", result)
    if cohort == "known" and not passed:
        for item in trial.scoped(spec, "fresh")["cases"].values():
            record, page = (ROOT/item[k] for k in ("record", "page"))
            state = read(record/"state.json")
            assert state["status"] == "starting" and state["controls"] == 0
            state.update(status="stopped-review-required", purpose="已知种子未通过，未启动新种子")
            trial.native.publish(record, page, state)
    print("FORWARD_MARGIN_DECISION", cohort, passed, gates, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=trial.COHORTS, required=True)
    cohort = parser.parse_args().cohort
    spec = trial.checked()
    trial.admit(spec, cohort)
    evaluate(spec, cohort)
