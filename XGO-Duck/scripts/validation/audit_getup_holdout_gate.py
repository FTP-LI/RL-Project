"""Recount the frozen getup holdout from saved traces, without simulation.

This audit checks the standing signals and original right-side paired gate. It
does not rerun a policy, check video, select a new model, or use the holdout
for tuning. The historical scores and decisions remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/getup_right_robustness_holdout_s2065.json"
REPORT = ROOT / "records/reports/getup-right-robustness/holdout"
POSES = ("prone", "supine", "left-side", "right-side")
ROLES = ("reference-a", "reference-b", "candidate")
FIELDS = ("time", "quaternion", "height", "angular_velocity", "foot_contact", "joint_envelope_ok", "active")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in FIELDS}


def standing_flags(trace: dict[str, np.ndarray], criteria: dict) -> tuple[np.ndarray, np.ndarray]:
    """Count consecutive valid controls as in the saved independent verifier."""
    dt = criteria["step_dt"]
    release, deadline, width = (round(criteria[key] / dt) for key in
                                ("freeze_seconds", "deadline_seconds", "hold_seconds"))
    quat = trace["quaternion"].astype(np.float64)
    vertical = 1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
    stable = ((vertical >= np.cos(np.deg2rad(criteria["tilt_max_deg"])))
              & (np.abs(trace["height"] - criteria["nominal_height_m"]) <= criteria["height_tolerance_m"])
              & (np.linalg.norm(trace["angular_velocity"], axis=-1) <= criteria["angular_speed_max_rad_s"])
              & trace["foot_contact"].all(axis=-1) & trace["joint_envelope_ok"] & trace["active"])
    sums = np.cumsum(stable, axis=0, dtype=np.int16)
    sums = np.concatenate((np.zeros((1, sums.shape[1]), np.int16), sums))
    windows = sums[width:] - sums[:-width] == width
    if len(windows) != deadline + 1:
        raise ValueError("Saved trace length disagrees with the frozen standing window")
    return windows[release:].any(axis=0), windows[-1]


def group_results(passed: np.ndarray, values: np.ndarray, rate: float, minimum: int) -> dict:
    result = {}
    for pose in POSES:
        selected = values == pose
        n, k = int(selected.sum()), int((selected & passed).sum())
        required = ceil(rate * n - 1e-10)
        result[pose] = {"successes": k, "episodes": n, "required": required,
                        "passed": n >= minimum and k >= required}
    return result


def main(output: Path) -> None:
    if not output.resolve().is_relative_to((ROOT / "records/reports/getup-gate-clarification").resolve()):
        raise ValueError("Audit output must remain in the dedicated report folder")
    if output.exists():
        raise FileExistsError("Preserve existing audit evidence")
    plan, analysis, decision = read(PLAN), read(REPORT / "analysis.json"), read(REPORT / "decision.json")
    plan_hash, score_hash = sha(PLAN), sha(REPORT / "analysis.json")
    if analysis["protocol_sha256"] != plan_hash or decision["analysis_sha256"] != score_hash:
        raise ValueError("Frozen protocol or scored data changed")
    if analysis["seed"] != 2065 or not decision["heldout_used"] or decision["holdout_eligible_for_tuning"]:
        raise ValueError("Unexpected holdout scope")
    record = (ROOT / plan["record"]).resolve()
    if not record.is_relative_to(ROOT.resolve()):
        raise ValueError("Recorded run is outside this project")
    if sha(record / "protocol.json") != plan_hash:
        raise ValueError("Run protocol differs from frozen protocol")
    state, release = read(record / "evaluation.json"), read(record / "release-coverage.json")
    if sha(record / "release-coverage.npz") != release["poses_sha256"]:
        raise ValueError("Release pose file changed")
    if sha(record / "route-masks.npz") != state["route_masks_sha256"]:
        raise ValueError("Routing masks changed")
    with np.load(record / "release-coverage.npz", allow_pickle=False) as z:
        actual = z["poses"]
    with np.load(record / "initial.npz", allow_pickle=False) as z:
        requested = z["requested_pose"]
    with np.load(record / "route-masks.npz", allow_pickle=False) as z:
        masks = {"supine": z["supine"], "right-side": z["right"]}
    if len(actual) != 3072 or actual.shape != requested.shape or not np.isin(actual, POSES).all():
        raise ValueError("Unexpected release or request coverage")
    expected_requested = np.asarray([POSES[code] for code in plan["environment"]["split"]["pose_codes"]])
    np.testing.assert_array_equal(requested, expected_requested)
    gate = plan["dual_route"]["gate"]
    criteria = plan["environment"]["criteria"]
    if gate != plan["holdout_admission"]["required_gate"]:
        raise ValueError("Original route and holdout gates differ")
    prefix_path = record / "trace-shared-prefix.npz"
    prefix_hash = sha(prefix_path)
    prefix = load_fields(prefix_path)
    scores, flags, trace_hashes = {}, {}, {}
    for role in ROLES:
        path = record / f"trace-{role}.npz"
        digest = sha(path)
        if digest != analysis["arms"][role]["trace_sha256"]:
            raise ValueError(f"Saved trace hash differs: {role}")
        trace_hashes[role] = digest
        branch = load_fields(path)
        trace = {key: np.concatenate((prefix[key], branch[key]), axis=0) for key in FIELDS}
        np.testing.assert_allclose(trace["time"], np.arange(1, 551) * criteria["step_dt"], atol=1e-12, rtol=0)
        timely, final = standing_flags(trace, criteria)
        original = analysis["arms"][role]
        episodes = original["timely"]["episodes"]
        np.testing.assert_array_equal([row["env_index"] for row in episodes], np.arange(len(actual)))
        np.testing.assert_array_equal([row["release_pose"] for row in episodes], actual)
        np.testing.assert_array_equal([row["requested_pose"] for row in episodes], requested)
        np.testing.assert_array_equal(timely, [row["hold_start_seconds"] is not None for row in episodes])
        np.testing.assert_array_equal(final, original["final"]["stable_by_environment"])
        flags[role] = timely & final
        rate = gate["timely_and_final_each_requested_and_actual_pose_and_overall"]
        requested_scores = group_results(flags[role], requested, rate, 1)
        actual_scores = group_results(flags[role], actual, rate, gate["minimum_actual_pose_count"])
        total = int(flags[role].sum())
        scores[role] = {"overall": {"successes": total, "episodes": len(actual),
                                    "required": ceil(rate * len(actual) - 1e-10)},
                        "requested": requested_scores, "actual": actual_scores}
        if total != original["combined"]["overall"]["timely_holds"]:
            raise ValueError(f"Combined score differs for {role}")
        print(f"RAW_TRACE_AUDITED {role}: {total}/{len(actual)}", flush=True)
        del trace, branch
    paired = {}
    right = actual == "right-side"
    for role in ROLES[:2]:
        wins = np.flatnonzero(right & flags["candidate"] & ~flags[role]).tolist()
        losses = np.flatnonzero(right & flags[role] & ~flags["candidate"]).tolist()
        paired[role] = {"wins": wins, "losses": losses, "net": len(wins) - len(losses)}
        old = decision["paired_right"][role]
        if wins != old["wins"] or losses != old["losses"] or paired[role]["net"] != old["net"]:
            raise ValueError(f"Right-side paired result differs: {role}")
    coverage = {}
    for pose, mask in masks.items():
        is_pose = actual == pose
        coverage[pose] = {"actual": int(is_pose.sum()), "hits": int((is_pose & mask).sum()),
                          "leaks": int((~is_pose & mask).sum())}
        if coverage[pose] != decision["coverage"][pose]:
            raise ValueError(f"Routing coverage differs: {pose}")
    candidate = scores["candidate"]
    functional = (candidate["overall"]["successes"] >= candidate["overall"]["required"]
                  and all(row["passed"] for row in candidate["requested"].values())
                  and all(row["passed"] for row in candidate["actual"].values()))
    routed = all(row["hits"] >= row["actual"] * gate["minimum_supine_and_right_latch_coverage"]
                 and row["leaks"] <= gate["maximum_latch_leaks"] for row in coverage.values())
    right_gain = all(row["net"] >= gate["minimum_right_net_gain_per_reference"] for row in paired.values())
    passed = functional and routed and right_gain
    if passed != decision["final_holdout_passed"]:
        raise ValueError("Recount disagrees with the original holdout decision")
    result = {"status": "passed-raw-signal-recount", "checked_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": plan_hash, "analysis_sha256": score_hash,
              "script_sha256": sha(Path(__file__).resolve()), "prefix_sha256": prefix_hash,
              "trace_sha256": trace_hashes, "gate": gate, "scores": scores, "coverage": coverage,
              "paired_right": paired, "functional_gate_passed": functional,
              "routing_gate_passed": routed, "right_gain_gate_passed": right_gain,
              "original_holdout_gate_passed": passed, "physical_steps": 0, "learning_updates": 0,
              "hardware_io": False,
              "scope": "Recounts saved standing signals and pose/routing masks, without new physics, policy inference or tuning."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "scores": {role: scores[role]["overall"] for role in ROLES},
                      "paired_right_net": {role: row["net"] for role, row in paired.items()},
                      "original_holdout_gate_passed": passed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
