"""Compare pause/digest decisions and WSL CPU cost without launching a worker."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import supervised_ppo_process as original
from canonical_ppo_authorization import CanonicalAuthorization
from analyze_observed_getup_costs import benchmark, read, sha

OUT = ROOT / "records/reports/getup-observed-throughput-20261116/performance-review"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    output = OUT / "canonical-check.json"
    assert not output.exists()
    project_hash = sha(ROOT / "project.json")
    tests = []
    with tempfile.TemporaryDirectory(prefix="canonical-check-", dir=OUT) as directory:
        fixture = Path(directory).resolve()
        assert fixture.is_relative_to(OUT.resolve())
        protocol, project_path, guard_path = [fixture / name for name in ("plan.json", "project.json", "parent.py")]
        guard_path.write_bytes(original.PARENT_EXIT_GUARD.read_bytes())
        command = [sys.executable, "worker.py", "--arm", "throughput"]
        plan = {"supervisor_workers": {"fixture": command}, "supervisor_sha256": sha(Path(original.__file__)),
                "parent_exit_guard_sha256": sha(guard_path)}
        base = {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                "staged_training_request": {"pause_requested": False, "current_stage": "fixture",
                    "current_protocol_frozen": True, "current_protocol": "plan.json",
                    "learning_running": False, "simulation_running": False}}
        write(protocol, plan)
        digest = sha(protocol)
        candidate = CanonicalAuthorization(fixture, project_path, protocol, digest, "fixture", "fixture",
                                           command, original.__file__, guard_path, original._read, original._sha)

        def compare(label, project, expected, *, start=False):
            write(project_path, project)
            a = original._authorized(protocol, digest, "fixture", "fixture", command, before_start=start)
            b = candidate(before_start=start)
            assert a is b is expected, (label, a, b, expected)
            tests.append({"case": label, "original": a, "candidate": b})

        with patch.object(original, "ROOT", fixture), patch.object(original, "PROJECT", project_path), \
                patch.object(original, "PARENT_EXIT_GUARD", guard_path):
            compare("active-start", base, True, start=True)
            compare("active-callback", base, True)
            for group, field, value in (
                ("goal_execution", "status", "paused"), ("goal_execution", "pause_requested_by_user", True),
                ("staged_training_request", "pause_requested", True),
                ("staged_training_request", "current_stage", "another"),
                ("staged_training_request", "current_protocol_frozen", False),
                ("staged_training_request", "current_protocol", "another.json")):
                changed = deepcopy(base); changed[group][field] = value
                compare(field, changed, False)
                compare(field+"-cleared", base, True)
            for field in ("learning_running", "simulation_running"):
                changed = deepcopy(base); changed["staged_training_request"][field] = True
                compare(field+"-before-start", changed, False, start=True)
                compare(field+"-during-run", changed, True)
            protocol.write_text(json.dumps(plan)+"\n", encoding="utf-8")
            compare("protocol-bytes-changed", base, False)
            write(protocol, plan); compare("protocol-restored", base, True)
            guard_path.write_text("changed\n", encoding="utf-8")
            compare("parent-source-changed", base, False)
            guard_path.write_bytes((ROOT / "scripts/training/parent_exit_exec.py").read_bytes())
            compare("parent-source-restored", base, True)
            alias = deepcopy(base); alias["staged_training_request"]["current_protocol"] = "./plan.json"
            write(project_path, alias)
            assert original._authorized(protocol, digest, "fixture", "fixture", command, before_start=False)
            assert not candidate(before_start=False)
            tests.append({"case": "noncanonical-spelling", "original": True, "candidate": False,
                          "meaning": "New protocols require the canonical project-relative spelling"})
        assert fixture.is_relative_to(OUT.resolve())

    # Time the actual archived protocol and code through /mnt/c, with a private
    # active-state copy. The real project remains unadmitted for learning.
    protocol = ROOT / "configs/experiments/observed_getup_json_throughput_20261116.json"
    plan = read(protocol); run_id = plan["arms"]["throughput"]["run_id"]
    command = plan["supervisor_workers"][run_id]; digest = sha(protocol)
    project = read(ROOT / "project.json")
    project["staged_training_request"].update(current_stage=plan["stage"], current_protocol_frozen=True,
        current_protocol=protocol.relative_to(ROOT).as_posix(), pause_requested=False)
    with tempfile.TemporaryDirectory(prefix="canonical-cost-", dir=OUT) as directory:
        fixture = Path(directory).resolve(); assert fixture.is_relative_to(OUT.resolve())
        project_path = fixture / "project.json"; write(project_path, project)
        candidate = CanonicalAuthorization(ROOT, project_path, protocol, digest, plan["stage"], run_id,
            command, original.__file__, original.PARENT_EXIT_GUARD, original._read, original._sha)
        with patch.object(original, "PROJECT", project_path):
            def old_call():
                assert original._authorized(protocol, digest, plan["stage"], run_id, command, before_start=False)
            def new_call():
                assert candidate(before_start=False)
            old_cost, new_cost = benchmark(old_call), benchmark(new_call)
        assert fixture.is_relative_to(OUT.resolve())
    assert sha(ROOT / "project.json") == project_hash
    result = {"status": "canonical-decisions-and-cpu-benchmark-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "tests": tests, "original_cost": old_cost, "canonical_cost": new_cost,
              "wall_cost_reduction_percent": (1-new_cost["median_wall_ms_per_call"]/old_cost["median_wall_ms_per_call"])*100,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                    (Path(__file__), ROOT / "scripts/training/canonical_ppo_authorization.py",
                     ROOT / "scripts/training/observed_getup_path_entry.py")},
              "real_project_unchanged": True, "actual_training_speedup_verified": False,
              "physical_substeps": 0, "learning_updates": 0, "hardware_io": False}
    write(output, result)
    print(json.dumps({k: result[k] for k in ("status", "original_cost", "canonical_cost", "wall_cost_reduction_percent")}, ensure_ascii=False))
    print("DECISION_CASES", len(tests))


if __name__ == "__main__":
    main()
