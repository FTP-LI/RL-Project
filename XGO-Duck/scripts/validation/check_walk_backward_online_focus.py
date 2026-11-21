"""Read back the proposed online actor weights using the real backward capture."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
from getup_ppo_math import masked_surrogate
from walk_backward_online_focus import BackwardStallFocus, weighted_surrogate

PLAN = ROOT / "configs/experiments/common_walk_backward_online_focus_draft.json"
OUT = ROOT / "records/reports/common-walk-backward-online-focus/readback.json"
TERMINALS = ("terminated", "truncated", "fell_over", "nan_state", "out_of_bounds")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(plan):
    receipts = {}
    for name in ("capture_result", "continuation_audit", "prefix_audit", "retention"):
        path, expected = plan[name]
        assert sha(ROOT / path) == expected, name
        receipts[name] = read(ROOT / path)
    capture = receipts["capture_result"]
    assert capture["status"] == "passed-stall-capture-integrity" and not capture["capture_admitted"]
    record = ROOT / capture["record"]
    prefix_path = record / "trace-shared-prefix.npz"
    continuation_path = record / "trace-backward.npz"
    assert sha(prefix_path) == capture["shared_prefix_sha256"]
    assert sha(continuation_path) == capture["trace_sha256"]["backward"]
    with np.load(prefix_path, allow_pickle=False) as prefix, np.load(continuation_path, allow_pickle=False) as post:
        def combined(field):
            return np.concatenate((prefix[field], post[field]))

        speed = combined("velocity")[..., 0].copy()
        active = combined("active")
        ended = np.logical_or.reduce([~active, *(combined(field) for field in TERMINALS)])
    assert speed.shape == active.shape == ended.shape == (300, 896)
    selection_path = record / "fork-selected-states.npz"
    assert sha(selection_path) == capture["selected_state_sha256"]
    with np.load(selection_path, allow_pickle=False) as selection:
        expected = selection["ids"]
    retention = receipts["retention"]
    assert retention["status"] == "passed-six-command-retention-integrity"
    assert len(retention["samples_per_command"]) == 6
    assert set(retention["samples_per_command"].values()) == {512} and retention["total_samples"] == 3072
    bank = (ROOT / plan["retention"][0]).with_name("anchors.npz")
    assert sha(bank) == retention["bank_sha256"]
    return speed, active, ended, expected, retention


def check_loss(device):
    log_prob = torch.tensor([0.15, -0.4, 0.1, 0.6], device=device, requires_grad=True)
    old = torch.zeros_like(log_prob)
    advantages = torch.tensor([1.0, -2.0, 3.0, -4.0], device=device)
    active = torch.tensor([True, True, False, True], device=device)
    reference = masked_surrogate(log_prob, old, advantages, active)
    unweighted = weighted_surrogate(log_prob, old, advantages, active, torch.ones_like(log_prob))
    torch.testing.assert_close(unweighted, reference)
    torch.testing.assert_close(torch.autograd.grad(unweighted, log_prob)[0],
                               torch.autograd.grad(reference, log_prob)[0])

    log_prob = torch.tensor([0.0, 0.0, float("nan")], device=device, requires_grad=True)
    old = torch.zeros_like(log_prob)
    advantages = torch.tensor([1.0, 2.0, float("nan")], device=device)
    active = torch.tensor([True, True, False], device=device)
    weights = torch.tensor([1.0, 3.0, 100_000.0], device=device)
    loss = weighted_surrogate(log_prob, old, advantages, active, weights)
    gradient = torch.autograd.grad(loss, log_prob)[0]
    torch.testing.assert_close(loss, torch.tensor(-1.75, device=device))
    torch.testing.assert_close(gradient, torch.tensor([-0.25, -1.5, 0.0], device=device))
    empty = weighted_surrogate(log_prob, old, advantages, torch.zeros_like(active), weights)
    torch.testing.assert_close(empty, torch.zeros_like(empty))
    torch.testing.assert_close(torch.autograd.grad(empty, log_prob)[0], torch.zeros_like(log_prob))
    return {"control_loss_and_gradient_match": True, "candidate_example_loss": loss.item(),
            "candidate_example_gradient": gradient.tolist(), "empty_actor_batch_zero": True}


def replay(plan, source, device):
    speed, active, ended, expected, _ = source
    speed, active, ended = (torch.as_tensor(data, device=device) for data in (speed, active, ended))
    horizon, worlds = active.shape
    block = plan["block_controls"]
    multiplier = plan["focused_actor_loss_multiplier"]
    tracker = BackwardStallFocus(worlds, block, plan["maximum_absolute_block_speed_m_s"],
                                 plan["focus_stop_control"], multiplier, device)
    ages = torch.arange(horizon + 2, device=device)[:, None].expand(-1, worlds)
    control_weights, candidate_weights = [], []
    for t in range(horizon):
        control_weights.append(tracker.actor_weights(ages[t], focused=False))
        candidate_weights.append(tracker.actor_weights(ages[t], focused=True))
        tracker.observe(speed[t], ages[t + 1], ended[t])
    control = torch.stack(control_weights).cpu().numpy()
    candidate = torch.stack(candidate_weights).cpu().numpy()
    active = active.cpu().numpy()
    assert np.all(control == 1)
    focus = candidate == multiplier
    assert np.all(focus | (candidate == 1))
    assert not focus[:2 * block].any()
    np.testing.assert_array_equal(np.flatnonzero(focus[2 * block]), expected)
    assert np.all(~focus[2 * block + 1:] | focus[2 * block])
    weighted_counts = (focus & active).sum(axis=1)
    steps = np.flatnonzero(weighted_counts) + 1
    assert (steps[0], steps[-1]) == (2 * block + 1, plan["focus_stop_control"])
    assert (tracker.actor_weights(ages[horizon], focused=True) == 1).all().item()

    synthetic = BackwardStallFocus(4, block, plan["maximum_absolute_block_speed_m_s"],
                                   plan["focus_stop_control"], multiplier, device)
    ages4 = torch.arange(2 * block + 2, device=device)[:, None].expand(-1, 4)
    speed4 = torch.zeros(4, device=device)
    ended4 = torch.tensor([False, False, True, False], device=device)
    for t in range(1, 2 * block + 1):
        # One earlier termination and one missing speed must each block tagging.
        finite = speed4 if t != block else torch.tensor([0.0, 0.0, 0.0, float("nan")], device=device)
        synthetic.observe(finite, ages4[t], ended4 & (t == block))
    expected_before_reset = torch.tensor([multiplier, multiplier, 1.0, 1.0], device=device)
    torch.testing.assert_close(synthetic.actor_weights(ages4[2 * block], True), expected_before_reset)
    synthetic.observe(speed4, torch.tensor([2 * block + 1, 1, 1, 1], device=device),
                      torch.tensor([True, False, False, False], device=device))
    torch.testing.assert_close(synthetic.actor_weights(ages4[2 * block + 1], True), torch.ones(4, device=device))
    return {"selected_ids": expected.tolist(), "first_weighted_control": int(steps[0]),
            "last_weighted_control": int(steps[-1]), "weighted_active_transitions": int(weighted_counts.sum()),
            "active_transitions": int(active.sum()), "all_control_weights_one": True,
            "no_early_late_or_reset_leak": True, "loss": check_loss(device)}


def main():
    plan = read(PLAN)
    assert plan["status"] == "draft-online-weight-readback-only"
    assert plan["physical_substeps_authorized"] == plan["learning_updates_authorized"] == 0
    assert not plan["heldout_used"] and not plan["hardware_io"]
    source = sources(plan)
    assert torch.cuda.is_available()
    cpu = replay(plan, source, torch.device("cpu"))
    cuda = replay(plan, source, torch.device("cuda:0"))
    assert cpu == cuda
    result = {"status": "passed-real-trace-weight-readback", "plan_sha256": sha(PLAN),
              "script_sha256": sha(Path(__file__)),
              "weight_source_sha256": sha(ROOT / "scripts/training/walk_backward_online_focus.py"),
              "source_receipts": {name: plan[name] for name in
                                  ("capture_result", "continuation_audit", "prefix_audit", "retention")},
              "cpu_cuda_exact_match": True, "cuda_device": torch.cuda.get_device_name(0), "replay": cpu,
              "anchor_counts": source[-1]["samples_per_command"], "physical_substeps": 0,
              "learning_updates": 0, "heldout_used": False, "hardware_io": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(OUT)
    print(json.dumps({key: value for key, value in result.items() if key != "replay"}, indent=2))
    print("ONLINE_FOCUS_REPLAY", {key: value for key, value in cpu.items() if key != "selected_ids"})


if __name__ == "__main__":
    main()
