"""Check boolean force-limit compatibility and original guard invariants on CPU."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import model_force_limit_guard as original
import model_force_limit_guard_bool as candidate

PARAMS = ROOT / "records/reports/matched-start-source-audit/sources/07-1910_m6.json.txt"
PLAN = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
OUT = ROOT / "records/reports/native-observer-preflight/force-bool-candidate-check.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parameters = json.loads(PARAMS.read_text(encoding="utf-8"))
    expected = 7.7 * float(parameters["kt"]) / float(parameters["R"])
    xml = f'''<mujoco><worldbody><body><joint name="robot/hinge" type="hinge"/>
      <geom type="sphere" size="0.05" mass="1"/></body></worldbody>
      <actuator><motor joint="robot/hinge" forcelimited="true" forcerange="{-expected:.17g} {expected:.17g}"/></actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    names = ("actuator_trnid", "actuator_trntype", "actuator_forcelimited", "actuator_forcerange")
    before = {name: np.asarray(getattr(model, name)).copy() for name in names}
    assert before["actuator_forcelimited"].dtype == np.bool_
    metadata, arrays = candidate.check(model, ["hinge"], 7.7, parameters)
    assert metadata["status"] == "passed" and metadata["force_limit_flag_original_dtype"] == "bool"
    assert arrays["actuator_forcelimited"].dtype == np.bool_ and arrays["actuator_forcelimited"].tolist() == [True]
    assert all(np.array_equal(before[name], np.asarray(getattr(model, name))) for name in names)
    assert all(np.array_equal(before[name], arrays[name]) and before[name].dtype == arrays[name].dtype for name in names)

    class FakeModel:
        nu = 1

        def __init__(self):
            self.actuator_trnid = np.array([[0, -1]], dtype=np.int32)
            self.actuator_trntype = np.array([0], dtype=np.int32)
            self.actuator_forcelimited = np.array([True], dtype=bool)
            self.actuator_forcerange = np.array([[-expected, expected]], dtype=np.float64)

        def joint(self, name):
            if name != "robot/hinge":
                raise KeyError(name)
            return SimpleNamespace(id=0)

    cases = []
    for name, mutate, message in (
        ("bool-disabled", lambda x: setattr(x, "actuator_forcelimited", np.array([False])), "actuator force limit must be enabled"),
        ("integer-disabled", lambda x: setattr(x, "actuator_forcelimited", np.array([0], dtype=np.uint8)), "actuator force limit must be enabled"),
        ("integer-invalid", lambda x: setattr(x, "actuator_forcelimited", np.array([2], dtype=np.uint8)), "actuator force limit must be enabled"),
        ("float-flag", lambda x: setattr(x, "actuator_forcelimited", np.array([1.0])), "actuator mapping and flags must be integer arrays"),
        ("float-mapping", lambda x: setattr(x, "actuator_trnid", np.array([[0.0, -1.0]])), "actuator mapping and flags must be integer arrays"),
        ("float-transmission", lambda x: setattr(x, "actuator_trntype", np.array([0.0])), "actuator mapping and flags must be integer arrays"),
        ("wrong-transmission", lambda x: setattr(x, "actuator_trntype", np.array([1], dtype=np.int32)), "actuator transmission does not match controlled joints"),
        ("wrong-joint", lambda x: setattr(x, "actuator_trnid", np.array([[1, -1]], dtype=np.int32)), "actuator transmission does not match controlled joints"),
        ("wrong-force", lambda x: setattr(x, "actuator_forcerange", np.array([[-expected, expected + 0.1]])), "compiled actuator force range differs from expected voltage limit"),
        ("nonfinite-force", lambda x: setattr(x, "actuator_forcerange", np.array([[-expected, np.nan]])), "actuator force ranges must be finite floats"),
        ("bad-shape", lambda x: setattr(x, "actuator_forcelimited", np.array([True, True])), "actuator field shape mismatch"),
    ):
        fixture = FakeModel()
        mutate(fixture)
        try:
            candidate.check(fixture, ["hinge"], 7.7, parameters)
        except ValueError as exc:
            assert str(exc) == message, (name, str(exc))
            cases.append({"case": name, "refused": True, "reason": str(exc)})
        else:
            raise AssertionError("invalid fixture accepted: " + name)
    integer = FakeModel()
    integer.actuator_forcelimited = np.array([1], dtype=np.uint8)
    old_meta, old_arrays = original.check(integer, ["hinge"], 7.7, parameters)
    new_meta, new_arrays = candidate.check(integer, ["hinge"], 7.7, parameters)
    assert new_meta["force_limit_flag_validation"] == "unchanged"
    assert all(np.array_equal(old_arrays[name], new_arrays[name]) for name in names)
    assert old_meta["expected_force_limit_nm"] == new_meta["expected_force_limit_nm"]
    frozen = json.loads(PLAN.read_text(encoding="utf-8"))
    assert sha(Path(original.__file__)) == frozen["frozen_sha256"]["scripts/validation/model_force_limit_guard.py"]
    record = {"status": "cpu-bool-compatibility-passed-real-xgo-readback-pending", "checked_at": datetime.now(timezone.utc).isoformat(),
              "mujoco_version": mujoco.__version__, "real_cpu_model_bool_flag_passed": True,
              "original_arrays_and_dtypes_preserved": True, "integer_fixture_matches_original": True,
              "invalid_cases": cases, "source_sha256": {str(path): sha(path) for path in
                  (Path(__file__), Path(candidate.__file__), Path(original.__file__), PARAMS, PLAN)},
              "scope": "CPU minimal MuJoCo model and fake invalid inputs; actual XGO flag and force limits remain unverified.",
              "physics_steps": 0, "learning_updates": 0, "hardware_io": False, "cuda_initialized": False}
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "invalid_cases": len(cases), "physics_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
