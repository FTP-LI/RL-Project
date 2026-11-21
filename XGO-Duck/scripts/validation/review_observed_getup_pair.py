"""Summarize the closed observed PPO pair from saved records only.

No trainer, MuJoCo, optimizer step, CUDA context or policy evaluation is used.
Keep loop timing separate from supervisor time, which includes viewing waits.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "records/reports/getup-observed-throughput-20261116"
PLAN = ROOT / "configs/experiments/observed_getup_json_throughput_20261116.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, value):
    with (REPORT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    plan = read(PLAN)
    now = datetime.now(timezone.utc).isoformat()
    source_checks = []
    for filename in ("observed_getup_throughput_20261116.json", PLAN.name):
        source_plan = read(ROOT / "configs/experiments" / filename)
        bad = [name for name, expected in source_plan["source_sha256"].items()
               if sha(ROOT / name) != expected]
        assert not bad, bad
        source_checks.append({"protocol": filename, "files": len(source_plan["source_sha256"]),
                              "unchanged": True})
    arms = {}
    for role, settings in plan["arms"].items():
        run = ROOT / "records/experiments" / settings["run_id"]
        view = ROOT / plan["coordinator"] / role
        receipt = read(view / "independent-verification.json")
        result = read(run / "online-result.json")
        supervisor = read(view / "supervisor-result.json")
        assert receipt["status"] == "passed-record-and-export-checks"
        assert supervisor["exit_code"] == 0
        assert receipt["transitions"] == result["actual_transitions"] == plan["limits"][role]["transitions"]
        assert receipt["updates"] == result["actual_updates"] == settings["training_updates"]
        assert sha(view / "execution.log") == supervisor["log_sha256"]
        assert sha(run / "model-final.pt") == receipt["checkpoint_sha256"]
        assert sha(ROOT / "artifacts" / settings["run_id"] / "policy.onnx") == receipt["onnx_sha256"]
        rows = [json.loads(line) for line in (run / "learning-metrics.jsonl").read_text().splitlines()]
        assert abs(sum(row["ppo_seconds"] for row in rows)-receipt["ppo_seconds"]) < 1e-6
        telemetry = read(run / "gpu-telemetry.json")["samples"]
        assert statistics.median(x["gpu_percent"] for x in telemetry) == receipt["gpu_percent_median"]
        arms[role] = dict(receipt, controls=plan["limits"][role]["controls"],
                         physical_substeps=result["actual_physical_substeps"],
                         supervisor_wall_seconds=supervisor["wall_seconds"],
                         supervisor_time_outside_loop_seconds=supervisor["wall_seconds"]-receipt["elapsed_seconds"],
                         loop_time_outside_ppo_seconds=receipt["elapsed_seconds"]-receipt["ppo_seconds"],
                         ppo_fraction_of_loop=receipt["ppo_seconds"]/receipt["elapsed_seconds"],
                         video_capture_fraction_of_loop=receipt["video_capture_seconds"]/receipt["elapsed_seconds"],
                         gpu_telemetry_samples=len(telemetry),
                         peak_gpu_fraction=receipt["peak_gpu_mib"]/16384,
                         evidence=(view / "independent-verification.json").relative_to(ROOT).as_posix(),
                         evidence_sha256=sha(view / "independent-verification.json"))
    old_verifier = REPORT / "verification-source-before-episode-index.zip"
    with zipfile.ZipFile(old_verifier) as archive:
        assert len(archive.namelist()) == 1
        assert hashlib.sha256(archive.read(archive.namelist()[0])).hexdigest() == arms["throughput"]["source_sha256"]

    # Verify the real failed checkpoint instead of inferring its content from logs.
    import torch
    failed = ROOT / "records/experiments/duck_getup_observed-e3072_s3263_20261116T011622"
    initial = torch.load(failed / "model-init.pt", map_location="cpu", weights_only=True)
    partial = torch.load(failed / "stopped-partial.pt", map_location="cpu", weights_only=True)
    state = read(failed / "state.json")
    assert partial["completed_updates"] == state["completed_updates"] == 1
    assert partial["control_steps"] == state["controls"] == 24
    assert partial["actor_steps_this_run"] == 0 and partial["critic_steps_this_run"] == 120
    assert all(torch.equal(initial["actor_state_dict"][k], v)
               for k, v in partial["actor_state_dict"].items())
    increments = {}
    for network in ("actor", "critic"):
        steps = []
        for point in (initial, partial):
            entries = point[network + "_optimizer_state_dict"]["state"].values()
            counts = {int(x["step"]) for x in entries}
            assert len(counts) == 1
            steps.append(next(iter(counts)))
            assert all(torch.isfinite(t).all() for x in entries for t in x.values() if isinstance(t, torch.Tensor))
        increments[network] = steps[1]-steps[0]
    assert increments == {"actor": 0, "critic": 120}
    partial_check = {"status": "passed-cpu-readback", "checked_at": now,
                     "checkpoint": (failed / "stopped-partial.pt").relative_to(ROOT).as_posix(),
                     "checkpoint_sha256": sha(failed / "stopped-partial.pt"),
                     "controls": 24, "physical_substeps": 120, "transitions": 73728,
                     "completed_ppo_updates": 1, "optimizer_step_increments": increments,
                     "actor_weights_unchanged": True, "exact_physical_resume_verified": False,
                     "new_physical_steps": 0, "new_learning_updates": 0}
    write("partial-checkpoint-verification.json", partial_check)

    control, throughput = arms["control"], arms["throughput"]
    comparison = {"status": "completed-pair-independently-reconciled", "checked_at": now,
                  "protocol": PLAN.relative_to(ROOT).as_posix(), "protocol_sha256": sha(PLAN),
                  "arms": arms,
                  "loop_throughput_gain_percent": (throughput["net_transitions_per_second"]/control["net_transitions_per_second"]-1)*100,
                  "loop_time_reduction_percent": (1-throughput["elapsed_seconds"]/control["elapsed_seconds"])*100,
                  "peak_gpu_increase_mib": throughput["peak_gpu_mib"]-control["peak_gpu_mib"],
                  "peak_gpu_increase_gib": (throughput["peak_gpu_mib"]-control["peak_gpu_mib"])/1024,
                  "total_real_training_transitions_including_failed": sum(x["transitions"] for x in arms.values())+73728,
                  "total_control_steps_including_failed": sum(x["controls"] for x in arms.values())+24,
                  "total_physical_substeps_including_failed": sum(x["physical_substeps"] for x in arms.values())+120,
                  "total_completed_ppo_updates_including_failed": sum(x["updates"] for x in arms.values())+1,
                  "total_actor_gradient_steps_including_failed": sum(x["last_actor_steps"] for x in arms.values()),
                  "total_critic_gradient_steps_including_failed": sum(x["last_critic_steps"] for x in arms.values())+120,
                  "source_checks": source_checks,
                  "interpretation": [
                      "One run per size, equal transitions and common parent; unequal actor step counts and update boundaries.",
                      "Supervisor time includes viewing wait, imports, initialization and finalization; it is not a clean GPU benchmark.",
                      "PPO timer includes synchronized host and device work; it is not pure kernel duration.",
                      "GPU telemetry is device-wide and includes non-loop intervals; no per-process CPU or kernel timeline was captured.",
                      "Outside-PPO time is a remainder, not a direct simulation-only measurement.",
                      "Complete training clips show changing policies; final-policy action evaluation has not been performed."
                  ], "cpu_bottleneck_proven": False, "local_compute_maximized": False,
                  "gait_improvement_verified": False, "model_promoted": False,
                  "new_physical_steps": 0, "new_learning_updates": 0, "hardware_io": False,
                  "review_source_sha256": sha(Path(__file__))}
    write("comparison.json", comparison)

    ps = subprocess.run(["ps", "-eo", "pid,ppid,stat,etimes,args"], check=True, text=True, capture_output=True)
    known_names = ("observed_getup", "native_ppo_viewer", "supervised_ppo_process", "run_local_getup_ppo", "getup_ppo", "train.py")
    relevant = [line for line in ps.stdout.splitlines()[1:]
                if line.strip() and int(line.split()[0]) != os.getpid()
                and "XGO-Duck" in line and any(name in line for name in known_names)]
    lock = Path("/home/atlas/XGO-Duck/training.lock")
    lock_free = None
    if lock.exists():
        with lock.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_free = True
                fcntl.flock(stream, fcntl.LOCK_UN)
            except BlockingIOError:
                lock_free = False
    process = {"checked_at": datetime.now(timezone.utc).isoformat(),
               "scope": "Ubuntu-24.04 project worker and observer command lines; not a Windows-wide process audit",
               "ps_exit_code": ps.returncode, "matching_processes": relevant,
               "training_lock_exists": lock.exists(), "training_lock_free": lock_free,
               "status": "no-project-workers-found" if not relevant and lock_free is not False else "review-required"}
    write("process-check.json", process)
    assert not relevant and lock_free is not False, process
    print(json.dumps({k: comparison[k] for k in ("status", "loop_throughput_gain_percent", "loop_time_reduction_percent",
                     "peak_gpu_increase_gib", "total_real_training_transitions_including_failed")}, ensure_ascii=False))
    print(json.dumps(process, ensure_ascii=False))


if __name__ == "__main__":
    main()
