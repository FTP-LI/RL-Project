"""Exercise native zero coordinator admission checks without starting CUDA."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import native_zero_coordinator as coordinator


def main() -> None:
    out = ROOT / "records/reports/native-observer-preflight"
    temporary = Path(tempfile.mkdtemp(prefix="_zero-coordinator-check-", dir=out)).resolve()
    if not temporary.is_relative_to(out.resolve()):
        raise RuntimeError("unexpected temporary path")
    cases = []
    try:
        screenshot = temporary / "fixture.jpg"
        shutil.copyfile(out / "status-check.jpg", screenshot)
        now = datetime.now(timezone.utc)
        proof = {"status": "native-zero-view-observed", "run_id": "synthetic-run", "observer_pid": 4242,
                 "wrapper_sha256": "a" * 64, "screenshot": screenshot.relative_to(ROOT).as_posix(),
                 "screenshot_sha256": coordinator.old.sha(screenshot), "observed_at": now.isoformat()}
        args = {"run_id": "synthetic-run", "observer_pid": 4242, "wrapper_sha": "a" * 64,
                "screenshot": screenshot, "now": now, "max_age": 45}
        coordinator.verify_ui(proof, **args)
        cases.append("ui-matching-fixture")
        ui_cases = {
            "ui-run": dict(proof, run_id="other"),
            "ui-pid": dict(proof, observer_pid=4243),
            "ui-wrapper": dict(proof, wrapper_sha256="b" * 64),
            "ui-file": dict(proof, screenshot="records/other.jpg"),
            "ui-digest": dict(proof, screenshot_sha256="0" * 64),
            "ui-stale": dict(proof, observed_at=(now - timedelta(seconds=46)).isoformat()),
            "ui-future": dict(proof, observed_at=(now + timedelta(seconds=6)).isoformat()),
            "ui-no-timezone": dict(proof, observed_at=now.replace(tzinfo=None).isoformat()),
        }
        for name, item in ui_cases.items():
            try:
                coordinator.verify_ui(item, **args)
            except ValueError:
                cases.append(name)
            else:
                raise AssertionError("expected UI refusal: " + name)

        wrapper_path = temporary / "wrapper.json"
        wrapper_ref = wrapper_path.relative_to(ROOT).as_posix()
        source_ref = "scripts/training/native_run_observer.py"
        wrapper = {"status": "frozen-zero-only", "upstream_plan": coordinator.provenance.PLAN.relative_to(ROOT).as_posix(),
                   "upstream_plan_sha256": coordinator.provenance.PLAN_SHA, "dynamics_authorized": False,
                   "learning_updates": 0, "physics_substeps": 0, "hardware_io": False, "automatic_retry": False,
                   "heartbeat_seconds": 2, "max_heartbeat_seconds": 15, "max_source_seconds": 30,
                   "ui_wait_seconds": 180, "max_ui_age_seconds": 45,
                   "source_sha256": {source_ref: coordinator.old.sha(ROOT / source_ref)}}
        project = {"goal_execution": {"native_file_observer_evaluation": {}}}
        wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
        execution = project["goal_execution"]["native_file_observer_evaluation"]
        execution.update(native_zero_plan=wrapper_ref, native_zero_plan_sha256=coordinator.old.sha(wrapper_path))
        with patch.object(coordinator, "WRAPPER", wrapper_ref), patch.object(coordinator, "REQUIRED_SOURCES", {source_ref}):
            coordinator.verify_wrapper(wrapper_path, wrapper, project)
            cases.append("wrapper-matching-fixture")
            changes = {
                "wrapper-dynamics": {"dynamics_authorized": True},
                "wrapper-hardware": {"hardware_io": True},
                "wrapper-learning": {"learning_updates": 1},
                "wrapper-false-steps": {"physics_substeps": False},
                "wrapper-retry": {"automatic_retry": True},
                "wrapper-timing": {"max_source_seconds": 300},
                "wrapper-source": {"source_sha256": {source_ref: "0" * 64}},
                "wrapper-missing-source": {"source_sha256": {}},
            }
            for name, changed in changes.items():
                altered = deepcopy(wrapper)
                altered.update(changed)
                wrapper_path.write_text(json.dumps(altered), encoding="utf-8")
                execution["native_zero_plan_sha256"] = coordinator.old.sha(wrapper_path)
                try:
                    coordinator.verify_wrapper(wrapper_path, altered, project)
                except ValueError:
                    cases.append(name)
                else:
                    raise AssertionError("expected wrapper refusal: " + name)
            wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
            execution["native_zero_plan_sha256"] = "0" * 64
            try:
                coordinator.verify_wrapper(wrapper_path, wrapper, project)
            except ValueError:
                cases.append("wrapper-not-registered")
            else:
                raise AssertionError("unregistered wrapper accepted")

        # Synthetic project state: exercise pause, ownership and source checks
        # without writing the real project file or starting a runtime.
        critical = ("scripts/validation/native_zero_coordinator.py", "scripts/training/native_run_observer.py",
                    "scripts/training/native_observer_guard.py")
        wrapper["source_sha256"] = {name: coordinator.old.sha(ROOT / name) for name in critical}
        wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
        frozen_sha = coordinator.old.sha(wrapper_path)
        real_read = coordinator.old.read
        project_state = deepcopy(real_read(ROOT / "project.json"))
        project_state["goal_execution"]["native_file_observer_evaluation"].update(
            native_zero_plan=wrapper_ref, native_zero_plan_sha256=frozen_sha)
        upstream = real_read(coordinator.provenance.PLAN)

        def fixture_read(path):
            return deepcopy(project_state) if Path(path).resolve() == (ROOT / "project.json").resolve() else real_read(path)

        def fixture_write(path, value):
            assert Path(path).resolve() == (ROOT / "project.json").resolve()
            project_state.clear()
            project_state.update(deepcopy(value))

        with patch.object(coordinator, "WRAPPER", wrapper_ref), patch.object(coordinator.old, "read", side_effect=fixture_read), \
                patch.object(coordinator, "write", side_effect=fixture_write):
            assert coordinator.project_allows(upstream, frozen_sha, running=False)
            cases.append("project-idle-allowed")
            project_state["staged_training_request"]["pause_requested"] = True
            assert not coordinator.project_allows(upstream, frozen_sha, running=False)
            cases.append("project-pause-refused")
            project_state["staged_training_request"]["pause_requested"] = False
            project_state["staged_training_request"]["learning_running"] = True
            assert not coordinator.project_allows(upstream, frozen_sha, running=False)
            cases.append("project-learning-refused")
            project_state["staged_training_request"]["learning_running"] = False
            coordinator.set_running(4242, True, "synthetic-running")
            assert coordinator.project_allows(upstream, frozen_sha, running=True)
            cases.append("project-owned-start")
            try:
                coordinator.set_running(4243, False, "synthetic-stop")
            except RuntimeError:
                cases.append("project-wrong-owner-refused")
            else:
                raise AssertionError("wrong owner stopped project state")
            coordinator.set_running(4242, False, "synthetic-stop")
            assert coordinator.project_allows(upstream, frozen_sha, running=False)
            cases.append("project-owned-stop")
            real_sha = coordinator.old.sha

            def changed_sha(path):
                return "0" * 64 if Path(path).resolve() == (ROOT / critical[1]).resolve() else real_sha(path)

            with patch.object(coordinator.old, "sha", side_effect=changed_sha):
                assert not coordinator.project_allows(upstream, frozen_sha, running=False)
            cases.append("project-source-change-refused")

        record = {"status": "offline-admission-checks-passed-no-runtime", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "cases": cases, "coordinator_sha256": coordinator.old.sha(Path(coordinator.__file__)),
                  "test_sha256": coordinator.old.sha(Path(__file__)),
                  "scope": "Synthetic wrapper and UI fixtures; no real UI admission or simulator",
                  "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                  "torch_imported": "torch" in sys.modules, "real_runtime_tested": False}
        assert record["torch_imported"] is False
        (out / "native-zero-coordinator-check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "cases": len(cases), "physics_steps": 0}, ensure_ascii=False))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
