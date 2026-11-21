"""Check that the frozen baseline refuses changed project and UI inputs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import native_nominal_baseline as baseline


def main():
    plan, out = baseline.validate()
    project = baseline.read(ROOT / "project.json")
    original_read = baseline.read
    cases = {"frozen_plan_accepted": True}
    for label, key, value in (("pause_refused", "pause_requested", True),
                              ("other_protocol_refused", "current_protocol", "not-this-plan"),
                              ("active_training_refused", "learning_running", True)):
        altered = deepcopy(project)
        altered["staged_training_request"][key] = value
        def substitute(path, data=altered):
            return data if Path(path).resolve() == (ROOT / "project.json").resolve() else original_read(path)
        with patch.object(baseline, "read", side_effect=substitute):
            try:
                baseline.validate()
                cases[label] = False
            except ValueError:
                cases[label] = True
    altered_plan = deepcopy(plan)
    altered_plan["dynamics_authorized"] = False
    def changed_plan(path):
        return altered_plan if Path(path).resolve() == baseline.PLAN.resolve() else original_read(path)
    with patch.object(baseline, "read", side_effect=changed_plan):
        try:
            baseline.validate()
            cases["no_dynamic_authorization_refused"] = False
        except ValueError:
            cases["no_dynamic_authorization_refused"] = True

    temp = out / ("admission-fixture-" + uuid.uuid4().hex[:10])
    assert temp.resolve().parent == out.resolve()
    temp.mkdir()
    try:
        image = temp / "fixture.jpg"
        shutil.copyfile(ROOT / "records/reports/native-observer-preflight/status-check.jpg", image)
        proof = {"status": "native-baseline-view-observed", "run_id": plan["run_id"], "observer_pid": 123,
                 "protocol_sha256": baseline.previous.sha(baseline.PLAN), "screenshot": image.relative_to(ROOT).as_posix(),
                 "screenshot_sha256": baseline.previous.sha(image), "observed_at": datetime.now(timezone.utc).isoformat()}
        proof_path = temp / "proof.json"
        def try_proof(value):
            proof_path.write_text(json.dumps(value), encoding="utf-8")
            try:
                baseline.verify_ui(proof_path, image, plan["run_id"], 123, baseline.previous.sha(baseline.PLAN))
                return True
            except ValueError:
                return False
        cases["matching_fixture_accepted"] = try_proof(proof)
        cases["wrong_process_refused"] = not try_proof(dict(proof, observer_pid=124))
        cases["stale_view_refused"] = not try_proof(dict(proof, observed_at=(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat()))
        cases["changed_image_hash_refused"] = not try_proof(dict(proof, screenshot_sha256="0"*64))
    finally:
        shutil.rmtree(temp)
    cases["no_runtime_or_output_started"] = not (out / "native-view").exists() and not (ROOT / "records/evaluations" / plan["run_id"]).exists()
    if not all(cases.values()): raise AssertionError(cases)
    result = {"status": "admission-fixtures-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": baseline.previous.sha(baseline.PLAN), "cases": cases,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "physics_steps": 0, "learning_updates": 0, "scope": "Fixture checks; no real UI or simulator."}
    destination = out / "admission-check.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
