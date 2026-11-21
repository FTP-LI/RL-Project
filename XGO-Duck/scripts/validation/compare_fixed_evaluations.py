"""Summarize two verified fixed-command policy evaluations.

This is an effect reference, not a reward ablation: the pretrained model's
training budget is unknown. Require matching saved evaluation conditions, bind
all source traces to their independent verification, and recompute signed
responses on the first episodes. No simulation, inference or training runs.
CSV model labels use the actual evaluation run IDs: later training stages must
not inherit a hard-coded 5000-update label. An explicit output directory keeps
new analyses separate from already reported comparisons.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path, verification_path=None):
    report = json.loads((path / "evaluation.json").read_text())
    verified = json.loads((verification_path or path / "verification.json").read_text())
    assert report["status"] == "completed" and verified["status"] == "passed"
    assert report["run_id"] == verified["run_id"]
    assert sha(path / "evaluation.json") == verified["report_sha256"]
    assert sha(path / "protocol.json") == report["protocol_sha256"]
    assert sha(path / "evaluate_fixed_policy.py") == report["evaluator_sha256"]
    assert sha(path / "evaluation_metrics.py") == report["metric_code_sha256"]
    protocol = json.loads((path / "protocol.json").read_text())["evaluation"]
    assert report["linear_xy_error_definition"] == "time_mean_planar_l2_then_episode_mean"
    rows = []
    for condition, check in zip(report["conditions"], verified["checks"], strict=True):
        name = condition["condition"]
        assert check["condition"] == name
        assert condition["command"] == protocol["commands"][name]
        trace = path / f"{name}-trace.npz"
        video = path / condition["video"]["path"]
        assert sha(trace) == check["trace_sha256"]
        assert sha(video) == check["video_sha256"] == condition["video"]["sha256"]
        with np.load(trace, allow_pickle=False) as data:
            mask = data["active"]
            counts = mask.sum(axis=0)
            command = np.asarray(condition["command"])
            xy = (np.linalg.norm(data["velocity"]-command[:2], axis=2)*mask).sum(axis=0)/counts
            yaw = (np.abs(data["yaw_rate"]-command[2])*mask).sum(axis=0)/counts
            vx = (data["velocity"][:, :, 0]*mask).sum(axis=0)/counts
            signed_yaw = (data["yaw_rate"]*mask).sum(axis=0)/counts
            assert np.all(counts > 0)
            np.testing.assert_allclose(xy.mean(), condition["linear_xy_mae_m_s"], atol=1e-6)
            np.testing.assert_allclose(yaw.mean(), condition["yaw_mae_rad_s"], atol=1e-6)
        rows.append(dict(
            condition=name, command_vx=float(command[0]), command_yaw=float(command[2]),
            episodes=len(counts), full_episode_survival=condition["full_episode_survival"],
            falls=condition["fall_count_excluding_timeouts"],
            linear_xy_mae_m_s=float(xy.mean()), yaw_mae_rad_s=float(yaw.mean()),
            measured_vx_m_s=float(vx.mean()), measured_yaw_rad_s=float(signed_yaw.mean()),
            stop_drift_m_median=condition["final_displacement_m_median"] if name == "stop" else None,
            action_delta_rms=condition["action_delta_rms"],
            contact_foot_slip_m_s=condition["contact_foot_slip_m_s"],
            meets_numeric_criteria=condition["meets_numeric_criteria"],
            trace_sha256=check["trace_sha256"], video_sha256=check["video_sha256"],
        ))
    return report, verified, protocol, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("trained", type=Path)
    parser.add_argument("--reference-verification", type=Path,
                        help="Existing hash-bound verification of the unchanged reference run")
    parser.add_argument("--output", type=Path,
                        help="New analysis directory; must not already exist")
    args = parser.parse_args()
    reference, trained = args.reference.resolve(), args.trained.resolve()
    destination = args.output.resolve() if args.output else trained / "reference-comparison"
    destination.mkdir(parents=True, exist_ok=args.output is None)
    # Reuse a completed, hash-bound check when the early run's original checker
    # predates report hashing. read_run still validates report and every trace.
    reference_verification = (args.reference_verification.resolve()
                              if args.reference_verification else reference / "verification.json")
    if "report_sha256" not in json.loads(reference_verification.read_text()):
        # The early reference checker did not bind the report hash. Recompute
        # with today's checker in an isolated copy; preserve the original run.
        verifier = Path(__file__).with_name("verify_fixed_evaluation.py")
        with tempfile.TemporaryDirectory(prefix="reference-check-", dir=destination) as name:
            temporary = Path(name).resolve()
            assert temporary.is_relative_to(destination)
            shutil.copytree(reference, temporary, dirs_exist_ok=True)
            result = subprocess.run([sys.executable, "-B", str(verifier), str(temporary)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            reference_verification = destination / "reference-verification.json"
            shutil.copy2(temporary / "verification.json", reference_verification)
        shutil.copy2(verifier, destination / verifier.name)
    ref, ref_check, ref_protocol, ref_rows = read_run(reference, reference_verification)
    new, new_check, new_protocol, new_rows = read_run(trained)
    assert ref_protocol == new_protocol
    for field in ("phase", "seed", "num_envs", "episode_seconds", "step_dt", "physics_dt"):
        assert ref[field] == new[field], field
    assert ref["phase"] == "reference" and ref["seed"] == 1001
    assert sha(reference / "env.yaml") == sha(trained / "env.yaml")
    result = dict(
        status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], output_directory=str(destination),
        reference_run=ref["run_id"], trained_run=new["run_id"],
        reference_model_sha256=ref["model_sha256"], trained_model_sha256=new["model_sha256"],
        matching_environment_sha256=sha(trained / "env.yaml"),
        protocol_sha256=ref["protocol_sha256"],
        evaluator_source_identical=ref["evaluator_sha256"] == new["evaluator_sha256"],
        metric_source_identical=ref["metric_code_sha256"] == new["metric_code_sha256"],
        source_verifications={ref["run_id"]: sha(reference_verification),
                              new["run_id"]: sha(trained / "verification.json")},
        reference=ref_rows, trained=new_rows,
        scope="Same saved evaluation configuration, commands and development seed; model histories and budgets differ; not a controlled reward ablation",
        heldout_evaluation_completed=False, hardware_verified=False,
    )
    (destination / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    with (destination / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *new_rows[0]])
        writer.writeheader()
        for label, rows in ((ref["run_id"], ref_rows), (new["run_id"], new_rows)):
            writer.writerows(dict(model=label, **row) for row in rows)
    shutil.copy2(__file__, destination / Path(__file__).name)
    print(json.dumps({k:v for k,v in result.items() if k not in ("reference", "trained")}, indent=2))
    print(json.dumps(new_rows, indent=2))


if __name__ == "__main__":
    main()
