"""Check the actual runner's source hashing with synthetic plans, without a scene."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import guarded_factor_zero_entry as entry
from verify_guarded_factor_zero_entry import fixture_plan, write

OUT = ROOT / "records/reports/force-limit-guard"
PROTOCOL = ROOT / "configs/experiments/force_limit_guard_cpu_preflight.json"


def main():
    project, protocol = entry.read(ROOT / "project.json"), entry.read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] is True
    assert project["staged_training_request"]["pause_requested"] is True
    assert project["staged_training_request"]["learning_running"] is False
    assert protocol["status"] == "cpu-fixture-only-force-limit-guard" and protocol["physical_steps_authorized"] == 0
    for path, expected in protocol["source_sha256"].items():
        assert entry.sha(entry.inside(path)) == expected, path
    prior = entry.read(OUT / "original-guard-entry.json")
    assert prior["status"] == "passed-original-guard-fake-force-limit-entry"
    for path, expected in prior["source_sha256"].items():
        assert entry.sha(entry.inside(path)) == expected, path

    import torch
    assert not torch.cuda.is_initialized()
    cases = []
    with patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden")), \
         patch.object(torch.cuda, "init", side_effect=RuntimeError("CUDA forbidden")):
        import mjlab.scene
        with patch.object(mjlab.scene.Scene, "__init__", side_effect=RuntimeError("Real scene forbidden")), \
             tempfile.TemporaryDirectory(prefix="source-preflight-", dir=OUT) as temporary:
            import run_continuous_roll_stand_walk as base
            plan, path, mock_project, output, page, observer, admission, sync = fixture_plan(Path(temporary), "nominal")
            assert entry.preflight(mock_project, plan, path, "zero")["allowed"]
            with patch.object(base, "PLAN", path):
                result = base.checked()
                assert result["training_source_sha256"]["vendor/bam/bam/mjlab.py"] == plan["training_source_sha256"]["vendor/bam/bam/mjlab.py"]
                cases.append("actual-runner-hashes-current-extended-sources")
                for label, group, name in (
                    ("bam-source-mismatch", "training_source_sha256", "vendor/bam/bam/mjlab.py"),
                    ("delay-source-mismatch", "library_source_sha256", next(
                        n for n in plan["library_source_sha256"] if n.endswith("/buffers/delay_buffer.py"))),
                ):
                    altered = json.loads(json.dumps(plan))
                    altered[group][name] = "0" * 64
                    write(path, altered)
                    try:
                        base.checked()
                    except AssertionError:
                        cases.append("actual-runner-rejects-" + label)
                    else:
                        raise AssertionError("source mismatch was accepted: " + label)
    assert not torch.cuda.is_initialized()
    inputs = [Path(__file__), Path(entry.__file__), PROTOCOL, OUT / "original-guard-entry.json",
              ROOT / "scripts/validation/run_continuous_roll_stand_walk.py",
              ROOT / "scripts/validation/verify_guarded_factor_zero_entry.py"]
    result = {"status": "passed-actual-source-check-synthetic-plan", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "real_runner_checked_function_exercised": True, "real_environment_constructed": False,
              "torch_cuda_initialized": False, "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
              "source_sha256": {p.relative_to(ROOT).as_posix(): entry.sha(p) for p in inputs},
              "scope": "Actual frozen-source checker with synthetic plans; no real observer, CUDA zero or physical run."}
    target = OUT / "source-check.json"
    if target.exists():
        previous = entry.read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(cases), "physics_steps": 0,
                      "torch_cuda_initialized": False}, indent=2))


if __name__ == "__main__":
    main()
