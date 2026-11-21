"""Rehearse coordinator lifecycle with a fake zero callback and no simulator."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import native_zero_coordinator as coordinator


def main() -> None:
    base = ROOT / "records/reports/native-observer-preflight"
    temporary = Path(tempfile.mkdtemp(prefix="_zero-lifecycle-check-", dir=base)).resolve()
    if not temporary.is_relative_to(base.resolve()):
        raise RuntimeError("unexpected temporary path")
    real_read, real_write = coordinator.old.read, coordinator.write
    real_sha = coordinator.old.sha
    results = []
    try:
        for case in ("success", "wrong-ui", "pause-during-zero", "observer-exit-during-zero"):
            folder = temporary / case
            folder.mkdir()
            output = folder / "output"
            output.mkdir()
            plan_path, wrapper_path = folder / "upstream.json", folder / "wrapper.json"
            project_path, draft_path = folder / "project.json", folder / "draft.json"
            run_id = "synthetic-zero-no-training"
            upstream_files = ("scripts/validation/guarded_factor_zero_entry.py",
                              "scripts/validation/factor_scene_guard_adapter.py",
                              "scripts/validation/model_force_limit_guard.py")
            plan = {"run_id": run_id, "stage": "synthetic-zero-stage", "guarded_output": output.relative_to(ROOT).as_posix(),
                    "page": (folder / "unused.html").relative_to(ROOT).as_posix(), "factor_arm": "nominal",
                    "frozen_sha256": {name: real_sha(ROOT / name) for name in upstream_files}}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            wrapper_ref = wrapper_path.relative_to(ROOT).as_posix()
            critical = ("scripts/validation/native_zero_coordinator.py", "scripts/training/native_run_observer.py",
                        "scripts/training/native_observer_guard.py")
            # Short waits only in this synthetic lifecycle fixture.
            wrapper = {"heartbeat_seconds": 0.2, "max_heartbeat_seconds": 5, "max_source_seconds": 10,
                       "ui_wait_seconds": 10, "max_ui_age_seconds": 20,
                       "source_sha256": {name: real_sha(ROOT / name) for name in critical}}
            wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
            wrapper_sha, plan_sha = real_sha(wrapper_path), real_sha(plan_path)
            project = {"staged_training_request": {"pause_requested": False, "learning_running": False,
                       "simulation_running": False, "current_protocol_frozen": True,
                       "current_protocol": plan_path.relative_to(ROOT).as_posix(), "current_stage": plan["stage"]},
                       "goal_execution": {"pause_requested_by_user": False, "native_file_observer_evaluation":
                       {"native_zero_plan": wrapper_ref, "native_zero_plan_sha256": wrapper_sha}}}
            project_path.write_text(json.dumps(project), encoding="utf-8")
            draft_path.write_text(json.dumps({"arms": [{"name": "nominal"}]}), encoding="utf-8")
            session = output / "native-zero-run"
            proof_errors: list[str] = []
            called: list[bool] = []

            def fixture_read(path):
                return real_read(project_path) if Path(path).resolve() == (ROOT / "project.json").resolve() else real_read(path)

            def fixture_write(path, value):
                if Path(path).resolve() == (ROOT / "project.json").resolve():
                    return real_write(project_path, value)
                return real_write(path, value)

            def write_synthetic_proof():
                deadline = time.monotonic() + 8
                try:
                    health_file = session / "view/native-health.json"
                    while time.monotonic() < deadline:
                        if health_file.is_file():
                            health = real_read(health_file)
                            if health.get("health") == "fresh":
                                screenshot = session / "ui-check.jpg"
                                shutil.copyfile(base / "status-check.jpg", screenshot)
                                proof = {"status": "native-zero-view-observed", "run_id": "wrong" if case == "wrong-ui" else run_id,
                                         "observer_pid": health["observer_pid"], "wrapper_sha256": wrapper_sha,
                                         "screenshot": screenshot.relative_to(ROOT).as_posix(),
                                         "screenshot_sha256": real_sha(screenshot),
                                         "observed_at": datetime.now(timezone.utc).isoformat()}
                                real_write(session / "ui-proof.json", proof)
                                return
                        time.sleep(0.1)
                    proof_errors.append("observer did not become ready")
                except Exception as exc:
                    proof_errors.append(type(exc).__name__ + ": " + str(exc))

            def fake_zero(_path, _plan, _decision, _runtime, may_continue):
                called.append(may_continue())
                assert called[-1] is True
                if case == "pause-during-zero":
                    state = real_read(project_path)
                    state["staged_training_request"]["pause_requested"] = True
                    real_write(project_path, state)
                    assert may_continue() is False
                    raise RuntimeError("synthetic pause")
                if case == "observer-exit-during-zero":
                    pid = real_read(session / "view/native-health.json")["observer_pid"]
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while may_continue() and time.monotonic() < deadline:
                        time.sleep(0.1)
                    assert may_continue() is False
                    raise RuntimeError("synthetic observer exit")
                return {"status": "synthetic-zero-not-runtime", "physics_substeps": 0,
                        "learning_updates": 0, "hardware_io": False}

            fake_torch = ModuleType("torch")
            fake_mjlab = ModuleType("mjlab")
            fake_envs = ModuleType("mjlab.envs")
            fake_mjlab.envs = fake_envs
            fake_modules = {"torch": fake_torch, "mjlab": fake_mjlab, "mjlab.envs": fake_envs,
                            "common_scene_contract": ModuleType("common_scene_contract"),
                            "run_continuous_roll_stand_walk": ModuleType("run_continuous_roll_stand_walk"),
                            "factor_scene_guard_adapter": ModuleType("factor_scene_guard_adapter")}
            helper = threading.Thread(target=write_synthetic_proof, daemon=True)
            with patch.object(coordinator, "WRAPPER", wrapper_ref), patch.object(coordinator, "verify_wrapper", return_value=wrapper_sha), \
                    patch.object(coordinator.provenance, "PLAN", plan_path), patch.object(coordinator.provenance, "PLAN_SHA", plan_sha), \
                    patch.object(coordinator.provenance, "audit", return_value={"scope": "synthetic-only"}), \
                    patch.object(coordinator.old, "DRAFT", draft_path.relative_to(ROOT).as_posix()), \
                    patch.object(coordinator.old, "read", side_effect=fixture_read), patch.object(coordinator, "write", side_effect=fixture_write), \
                    patch.object(coordinator.old, "execute_zero", side_effect=fake_zero), patch.dict(sys.modules, fake_modules):
                helper.start()
                exit_code = coordinator.run(wrapper_path)
                helper.join(timeout=3)
            assert not proof_errors and not helper.is_alive(), proof_errors
            result, final_project = real_read(session / "result.json"), real_read(project_path)
            final_health = real_read(session / "view/native-health.json")
            final_text = (session / "view/native-status.txt").read_text(encoding="utf-8")
            assert final_project["staged_training_request"]["simulation_running"] is False
            assert final_health["health"] == "stopped" and "本次进程已经停止" in final_text
            if case == "success":
                assert exit_code == 0 and called == [True] and result["status"] == "synthetic-zero-not-runtime"
            elif case == "wrong-ui":
                assert exit_code == 2 and not called and "UI proof" in result["error"]
            else:
                assert exit_code == 2 and called == [True] and result["status"] == "stopped-review-required"
                if case == "pause-during-zero":
                    assert final_project["staged_training_request"]["pause_requested"] is True
            results.append({"case": case, "exit_code": exit_code, "fake_zero_calls": len(called),
                            "result_status": result["status"], "observer_final_health": final_health["health"]})

        record = {"status": "synthetic-lifecycle-passed-no-runtime", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "cases": results, "coordinator_sha256": real_sha(Path(coordinator.__file__)),
                  "test_sha256": real_sha(Path(__file__)), "scope": "Synthetic UI proof and mocked zero callback only",
                  "physics_steps": 0, "learning_updates": 0, "hardware_io": False, "real_cuda_initialized": False}
        (base / "native-zero-lifecycle-check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "cases": len(results), "physics_steps": 0}, ensure_ascii=False))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
