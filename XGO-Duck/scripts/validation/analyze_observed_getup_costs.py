"""Audit saved PPO clock scopes and measure CPU-only authorization overhead.

The benchmark calls only the authorization predicate against an isolated copy
of project metadata. It never launches a worker or admits a real training run.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter, process_time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/getup-observed-throughput-20261116/performance-review"
PLAN = ROOT / "configs/experiments/observed_getup_json_throughput_20261116.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def benchmark(function, calls=100):
    for _ in range(3):
        function()
    passes = []
    for _ in range(3):
        wall, cpu = perf_counter(), process_time()
        for _ in range(calls):
            function()
        passes.append({"calls": calls, "wall_seconds": perf_counter()-wall,
                       "cpu_seconds": process_time()-cpu})
    return {"passes": passes,
            "median_wall_ms_per_call": statistics.median(x["wall_seconds"]/calls*1000 for x in passes),
            "median_cpu_ms_per_call": statistics.median(x["cpu_seconds"]/calls*1000 for x in passes)}


def main():
    assert sys.platform.startswith("linux"), "Measure on the actual WSL path"
    assert not (OUT / "analysis.json").exists(), "Preserve the completed measurement"
    plan = read(PLAN)
    sources = {}
    for name in ("scripts/training/run_getup_supine_training.py",
                 "scripts/validation/run_local_getup_ppo.py",
                 "scripts/training/getup_adaptive_ppo.py",
                 "scripts/training/getup_recovery_retention_ppo.py",
                 "scripts/training/native_ppo_observation.py",
                 "scripts/training/supervised_ppo_process.py"):
        path = ROOT / name
        assert sha(path) == plan["source_sha256"][name]
        sources[name] = sha(path)
    source = (ROOT / "scripts/training/run_getup_supine_training.py").read_text()
    markers = ("began = time.monotonic()", "env, obs, algorithm, meta = initialize(p, record, scene)",
               "publish(status='training', paired_zero_start_verified=True)",
               "export(actor, meta, probe, p, record)", "publish(elapsed_seconds=time.monotonic()-began)")
    lines = {m: source[:source.index(m, source.index("def train("))].count("\n")+1 for m in markers}
    assert list(lines.values()) == sorted(lines.values())
    arms = {}
    for role, settings in plan["arms"].items():
        run = ROOT / "records/experiments" / settings["run_id"]
        result, observation = read(run / "online-result.json"), read(run / "observation.json")
        rows = [json.loads(line) for line in (run / "learning-metrics.jsonl").read_text().splitlines()]
        live = observation["rows"]
        assert len(rows) == len(live) == settings["training_updates"]
        assert sha(run / "learning-metrics.jsonl") == result["samples_sha256"]
        assert sha(run / "observation.json") == result["observation_sha256"]
        for raw, shown in zip(rows, live, strict=True):
            assert all(shown[k] == value for k, value in raw.items())
            assert abs(shown["net_transitions_per_second"] - shown["controls"]*settings["num_envs"]/shown["seconds_since_checked_start"]) < 1e-7
        operation = result["elapsed_seconds"]
        observed_window = live[-1]["seconds_since_checked_start"]
        ppo = sum(row["ppo_seconds"] for row in rows)
        offsets = [row["elapsed"]-shown["seconds_since_checked_start"] for row, shown in zip(rows, live)]
        arms[role] = {
            "run_id": settings["run_id"], "transitions": result["actual_transitions"],
            "training_operation_seconds": operation,
            "operation_transitions_per_second": result["actual_transitions"]/operation,
            "observed_sampling_update_window_seconds": observed_window,
            "observed_window_transitions_per_second": result["actual_transitions"]/observed_window,
            "ppo_seconds": ppo, "ppo_fraction_of_observed_window": ppo/observed_window,
            "outside_ppo_within_observed_window_seconds": observed_window-ppo,
            "video_sampling_seconds": live[-1]["video_sample_seconds"],
            "estimated_before_training_offset_range_seconds": [min(offsets), max(offsets)],
            "after_last_update_metric_seconds": operation-rows[-1]["elapsed"],
            "between_updates_seconds": sum(row["between_updates_seconds"] or 0 for row in rows),
            "critic_minibatches": rows[-1]["critic_steps_this_run"],
            "adaptive_guard_calls_in_ppo": sum(row["anchor_batches"] for row in rows),
            "actual_release_and_reset_gaps": [row["update"] for row in rows if row["completed_episodes"]],
            "raw_metrics_sha256": sha(run / "learning-metrics.jsonl"),
            "observed_metrics_sha256": sha(run / "observation.json"),
        }
    # Reproduce the active predicate using copied metadata, never changing project.json.
    sys.path.insert(0, str(ROOT / "scripts/training"))
    import supervised_ppo_process as supervisor
    project_before = sha(ROOT / "project.json")
    project = read(ROOT / "project.json")
    project["staged_training_request"].update(current_stage=plan["stage"], current_protocol_frozen=True,
                                             current_protocol=PLAN.relative_to(ROOT).as_posix(),
                                             pause_requested=False, learning_running=False, simulation_running=False)
    project["goal_execution"].update(status="active", pause_requested_by_user=False)
    run_id = plan["arms"]["throughput"]["run_id"]
    command = plan["supervisor_workers"][run_id]
    protocol_hash = sha(PLAN)
    measurements = {}
    with tempfile.TemporaryDirectory(prefix="cpu-cost-", dir=OUT) as directory:
        fixture = Path(directory).resolve()
        assert fixture.is_relative_to(OUT.resolve())
        copied_project = fixture / "project.json"
        copied_project.write_text(json.dumps(project, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        with patch.object(supervisor, "PROJECT", copied_project):
            def authorized():
                assert supervisor._authorized(PLAN, protocol_hash, plan["stage"], run_id, command, before_start=False)
            measurements["full_authorization_predicate"] = benchmark(authorized)
            measurements["project_json_read"] = benchmark(lambda: supervisor._read(copied_project))
            measurements["protocol_read_and_parse"] = benchmark(lambda: json.loads(PLAN.read_bytes()))
            measurements["resolve_constant_paths"] = benchmark(lambda: (PLAN.resolve(), Path(supervisor.__file__).resolve()))
            measurements["hash_supervisor_and_parent_guard"] = benchmark(
                lambda: (sha(Path(supervisor.__file__)), sha(supervisor.PARENT_EXIT_GUARD)))
        assert fixture.is_relative_to(OUT.resolve())
    assert sha(ROOT / "project.json") == project_before
    a, b = arms["control"], arms["throughput"]
    result = {
        "status": "saved-timing-and-cpu-costs-verified", "at": datetime.now(timezone.utc).isoformat(),
        "arms": arms, "clock_source_lines": lines, "source_sha256": sources,
        "operation_throughput_gain_percent": (b["operation_transitions_per_second"]/a["operation_transitions_per_second"]-1)*100,
        "observed_window_throughput_gain_percent": (b["observed_window_transitions_per_second"]/a["observed_window_transitions_per_second"]-1)*100,
        "cpu_measurements": measurements,
        "nominal_2040_predicate_seconds_at_measured_median": 2040*measurements["full_authorization_predicate"]["median_wall_ms_per_call"]/1000,
        "limitations": [
            "elapsed_seconds includes initialization, checked-start comparison, sampling, PPO, export and cleanup.",
            "seconds_since_checked_start ends at metric publication; it includes video setup, rollout, PPO, checks and publication overhead.",
            "The difference of asynchronous timestamp sites estimates initialization; it is not an exact separate timer.",
            "Read-only CPU replay is not a live worker attribution and may have different filesystem cache or load.",
            "Summed CPU sub-benchmarks overlap and must not be summed as causal runtime fractions.",
            "CUDA synchronization and PPO arithmetic still need actual bounded profiling."
        ],
        "project_file_unchanged_by_benchmark": True, "cpu_only": True,
        "new_physical_steps": 0, "learning_updates": 0, "hardware_io": False,
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "operation_throughput_gain_percent", "observed_window_throughput_gain_percent",
                                           "cpu_measurements", "nominal_2040_predicate_seconds_at_measured_median")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
