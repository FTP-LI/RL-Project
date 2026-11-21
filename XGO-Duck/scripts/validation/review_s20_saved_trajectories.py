"""Replay completed S20 traces on CPU, without starting a simulator or learning.

The original verifier used the same scoring functions.  This reader checks
the saved bytes and recomputes the full route, while explicitly comparing the
landing bridge with the independently accepted S17 route.  It does not modify
the historical plans or the original recorded results.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_metrics import summarize
from roll_stand_walk_sequence import FAILED, ROLL, STAND_LANDING, STATE_FIELDS, Sequence, score_forward

OUT = ROOT / "records/reports/roll-domain-parameter-random-s20"
PROTOCOL = ROOT / "configs/experiments/roll_domain_parameter_offline_review_s20.json"
S20_DRAFT = ROOT / "configs/experiments/roll_domain_parameter_random_s20.json"
S17_PLAN = ROOT / "configs/experiments/route_official_independent_s1235.json"
S17_COMMANDS = ROOT / "records/training-archive/evaluations/duck_switch_roll-stand-walk_development_s1235_20261114T104326/commands.npz"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def checked_input(path, digest):
    actual = sha(path)
    if actual != digest:
        raise AssertionError(f"Saved input hash differs: {path}")
    return actual


def first_impossible(seq, k, previous, plan):
    """Reproduce the historical post-hoc diagnostic, not an online stop."""
    if previous is not None:
        return previous
    if seq.roll_begin <= k <= seq.roll_end:
        certified = seq.roll_latch.first_ready >= 0
        possible = certified | (seq.roll_latch.intact & (seq.roll_latch.step < plan["roll_controls"]))
        if int(possible.sum()) < plan["minimum_roll_completion"]:
            return {"reason": "original-roll-threshold-impossible", "control": k,
                    "possible": int(possible.sum()), "certified": int(certified.sum()),
                    "required": plan["minimum_roll_completion"]}
    potential = int(((seq.phase != FAILED) | ((seq.forward_at >= 0) &
                    (k > seq.forward_at + plan["forward_controls"]))).sum())
    if potential < plan["minimum_continuous_successes"]:
        return {"reason": "sequence-threshold-impossible", "control": k,
                "possible": potential, "required": plan["minimum_continuous_successes"]}
    return None


def replay_roll(plan, trace, sequence):
    start, horizon = sequence.roll_begin, plan["roll_controls"]
    observed = min(len(trace["active"]) - start, horizon + 1)
    end = np.where(sequence.stand_at >= 0, sequence.stand_at, start + observed - 1)
    eligible = trace["phase_next"][start] == ROLL
    part = {key: trace[key][start:start + observed].copy() for key in STATE_FIELDS}
    part["active"] &= eligible[None] & (np.arange(start, start + observed)[:, None] <= end[None])
    padded = {key: np.concatenate((value, np.repeat(value[-1:], horizon + 1 - observed, axis=0)))
              for key, value in part.items()}
    padded["active"][observed:] = False
    padded["time"] = np.arange(horizon + 1) * 0.02
    score = summarize(padded, plan["roll"]["criteria"])
    np.testing.assert_array_equal([row["success"] for row in score["episodes"]], sequence.stand_at >= 0)
    return score


def replay_arm(item, draft, parent_plan):
    plan_path = ROOT / item["protocol"]
    plan = read(plan_path)
    saved = read(ROOT / item["output"] / "analysis.json")
    source_hashes = {}
    for file, expected in saved["source_sha256"].items():
        source_hashes[file] = checked_input(ROOT / file, expected)
    record = ROOT / item["run"]
    needed = set(STATE_FIELDS) | {"phase_next", "velocity", "joint_positions", "foot_positions", "foot_speed"}
    with np.load(record / "trace.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in needed}
    with np.load(record / "commands.npz", allow_pickle=False) as archive:
        phases = archive["phase_used"]
        bridge = archive["bridge"].astype(bool)
    last = len(phases)
    assert last == 1550 and trace["active"].shape == (last + 1, plan["worlds"])
    np.testing.assert_array_equal(phases, trace["phase_next"][:-1])
    sequence = Sequence(plan)
    first_stop = None
    for k in range(last + 1):
        sequence.update(k, {key: trace[key][k] for key in STATE_FIELDS})
        np.testing.assert_array_equal(sequence.phase, trace["phase_next"][k])
        if k < last:
            expected_bridge = ((sequence.phase == STAND_LANDING) &
                (k + 1 - sequence.phase_changed <= plan["cross_model_bridge_controls"]))
            np.testing.assert_array_equal(expected_bridge, bridge[k])
        first_stop = first_impossible(sequence, k, first_stop, plan)
    roll = replay_roll(plan, trace, sequence)
    finished = np.where((sequence.forward_at >= 0) &
                        (sequence.forward_at + plan["forward_controls"] <= last), sequence.forward_at, -1)
    forward = score_forward(trace, finished, plan)
    counts = {"worlds": plan["worlds"], "roll_completed": roll["successes"],
              "entered_forward": int((sequence.forward_at >= 0).sum()),
              "finished_forward_window": int((finished >= 0).sum()),
              "continuous_successes": sum(row["success"] for row in forward)}
    assert counts == saved["counts"] and first_stop == saved["early_stop"]
    assert [row["reasons"] for row in roll["episodes"]] == [row["reasons"] for row in saved["roll"]["episodes"]]
    assert [row["failed_checks"] for row in forward] == [row["failed_checks"] for row in saved["forward"]]
    for actual, expected in zip(forward, saved["forward"], strict=True):
        for key in ("signed_speed_m_s", "yaw_error_rad_s", "contact_slip_m_s", "block_error_m_s"):
            if actual.get(key) is not None:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=0)
    roll_reasons = Counter(reason for row in roll["episodes"] for reason in row["reasons"])
    forward_reasons = Counter(reason for row in forward for reason in row["failed_checks"])
    yaw = [row["yaw_error_rad_s"] for row in forward if row.get("yaw_error_rad_s") is not None]
    max_head = float(np.max(trace["head_penetration_m"][trace["active"]], initial=0.0))
    head_limit = plan["roll"]["criteria"]["head_penetration_max_m"]
    head_over = int(np.any((trace["head_penetration_m"] > head_limit) & trace["active"], axis=0).sum())
    joint_over = int(np.any(~trace["joint_envelope_ok"] & trace["active"], axis=0).sum())
    factors = plan["parameter_arm"]["factors"]
    changed = [name for name, value in factors.items() if float(value) != float(draft["nominal"][name])]
    return {
        "name": item["name"], "factors": factors, "changed_factors_vs_nominal": changed,
        "claimed_single_factor": plan["parameter_screen"]["single_factor"],
        "counts": counts, "score_reproduced": True,
        "passed_development_gate": bool(saved["passed_development_gate"]),
        "roll_failure_reasons": dict(roll_reasons), "forward_failure_reasons": dict(forward_reasons),
        "yaw_error_median_rad_s": float(np.median(yaw)) if yaw else None,
        "head_penetration_max_full_episode_m": max_head,
        "head_penetration_over_limit_worlds_full_episode": head_over,
        "joint_envelope_violated_worlds_full_episode": joint_over,
        "actual_bridge_steps_by_phase": {str(int(phase)): int((bridge & (phases == phase)).sum())
                                          for phase in np.unique(phases)},
        "recorded_impossible_gate": first_stop,
        "recorded_full_controls": last,
        "source_sha256": source_hashes,
    }


def main():
    protocol = read(PROTOCOL)
    assert not protocol["training"] and not protocol["new_simulation"] and not protocol["hardware_io"]
    progress, draft = read(OUT / "progress.json"), read(S20_DRAFT)
    assert progress["status"] == "stopped-by-user" and progress["completed"] == 6
    parent = read(ROOT / draft["parent_protocol"])
    arms = []
    for item in progress["results"]:
        arms.append(replay_arm(item, draft, parent))
        print("S20_OFFLINE_REPLAY", item["name"], arms[-1]["counts"], flush=True)
    s17 = read(S17_PLAN)
    with np.load(S17_COMMANDS, allow_pickle=False) as archive:
        s17_bridge = int(archive["bridge"].sum())
    first = read(ROOT / progress["results"][0]["protocol"])
    changed_sources = []
    for name in sorted(parent["frozen_sha256"].keys() & first["frozen_sha256"].keys()):
        if name.startswith("scripts/") and parent["frozen_sha256"][name] != first["frozen_sha256"][name]:
            changed_sources.append({"path": name, "parent_sha256": parent["frozen_sha256"][name],
                                    "s20_sha256": first["frozen_sha256"][name],
                                    "current_sha256": sha(ROOT / name)})
    result = {
        "status": "completed-offline-review-paused", "checked_at": datetime.now(timezone.utc).isoformat(),
        "physical_substeps": 0, "learning_updates": 0, "hardware_io": False,
        "method": "CPU replay of six saved raw NPZ trajectories through archived scoring functions; input hashes checked",
        "arms": arms,
        "controller_comparison": {
            "s20_roll_bridge": "ProjectedRollPolicy removes entry slew in ROLL",
            "s20_landing_bridge_controls": first["cross_model_bridge_controls"],
            "s17_landing_bridge_controls": s17["cross_model_bridge_controls"],
            "s17_recorded_bridge_control_world_steps": s17_bridge,
            "scope": "Different seeds and controllers; these traces do not isolate the causal effect of either physical factor.",
        },
        "metadata_findings": {
            "one_factor_flag_in_all_arm_plans": True,
            "completed_arms_with_two_factors_changed": [a["name"] for a in arms if len(a["changed_factors_vs_nominal"]) == 2],
            "parameter_sampling": "Each arm has fixed values for all worlds; this is a finite grid, not per-world randomized sampling.",
            "parent_vs_s20_script_hash_changes": changed_sources,
            "script_content_equivalence_proven": False,
        },
        "stopping_review": {
            "draft_rule": draft["stopping_rule"],
            "posthoc_diagnostic_not_online_stop": True,
            "source_scope": "run_continuous_roll_stand_walk.py rollout and run_roll_domain_parameter_screen.py arm loop",
            "safety_rule_enforced_in_arm_loop": False,
            "safety_rule_observed_trigger": any(a["head_penetration_over_limit_worlds_full_episode"] or
                                                 a["joint_envelope_violated_worlds_full_episode"] for a in arms),
            "note": "The saved impossible-threshold field is computed during replay after collection; full-horizon collection continued. The arm loop does not branch on the scored safety outcome. A new guarded protocol is required before resuming.",
        },
        "not_claimed": ["causal attribution to friction alone", "validity across the full parameter rectangle", "equivalence to S17", "per-world domain-randomization training benefit"],
        "input_sha256": {relative(path): sha(path) for path in [PROTOCOL, S20_DRAFT, OUT / "progress.json",
                          S17_PLAN, S17_COMMANDS, Path(__file__), ROOT / "scripts/validation/roll_stand_walk_sequence.py",
                          ROOT / "scripts/validation/roll_metrics.py"]},
    }
    (OUT / "offline-review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "arms": len(arms),
                      "s17_bridge": s17_bridge, "s20_bridge": first["cross_model_bridge_controls"],
                      "safety_trigger_observed": result["stopping_review"]["safety_rule_observed_trigger"]}, indent=2))


if __name__ == "__main__":
    main()
