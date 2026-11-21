"""Verify the bool-guard adapter, raw readback and mandatory OS log redirection."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_zero_bool_repair as repair
from recorded_process import run_recorded


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = ROOT / "records/reports/native-observer-preflight"
    temporary = Path(tempfile.mkdtemp(prefix="_bool-repair-check-", dir=base)).resolve()
    if not temporary.is_relative_to(base.resolve()):
        raise RuntimeError("unexpected temporary path")
    try:
        params = json.loads((ROOT / "records/reports/matched-start-source-audit/sources/07-1910_m6.json.txt").read_text())
        limit = 7.7 * float(params["kt"]) / float(params["R"])
        xml = f'''<mujoco><worldbody><body><joint name="robot/hinge" type="hinge"/>
          <geom type="sphere" size="0.05" mass="1"/></body></worldbody>
          <actuator><motor joint="robot/hinge" forcelimited="true" forcerange="{-limit:.17g} {limit:.17g}"/></actuator></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        original_check, original_module = repair.original_guard.check, repair.bool_guard.original
        zero = temporary / "valid-zero"
        zero.mkdir()
        with repair.bool_guard_context(lambda value: repair.capture_force_fields(value, zero, "a" * 64)):
            metadata, arrays = repair.original_guard.check(model, ["hinge"], 7.7, params)
        assert repair.original_guard.check is original_check and repair.bool_guard.original is original_module
        assert metadata["status"] == "passed" and arrays["actuator_forcelimited"].dtype == np.bool_
        saved = json.loads((zero / "actuator-observed-before-check.json").read_text(encoding="utf-8"))
        with np.load(zero / "actuator-observed-before-check.npz", allow_pickle=False) as archive:
            assert archive["actuator_forcelimited"].dtype == np.bool_ and archive["actuator_forcelimited"].tolist() == [True]
        assert saved["arrays_sha256"] == sha(zero / "actuator-observed-before-check.npz")
        assert saved["fields"]["actuator_forcelimited"]["dtype"] == "bool"

        model.actuator_forcelimited[0] = False
        invalid = temporary / "invalid-zero"
        invalid.mkdir()
        try:
            with repair.bool_guard_context(lambda value: repair.capture_force_fields(value, invalid, "b" * 64)):
                repair.original_guard.check(model, ["hinge"], 7.7, params)
        except ValueError as exc:
            assert str(exc) == "actuator force limit must be enabled"
        else:
            raise AssertionError("disabled force limit accepted")
        with np.load(invalid / "actuator-observed-before-check.npz", allow_pickle=False) as archive:
            assert archive["actuator_forcelimited"].tolist() == [False]
        assert repair.original_guard.check is original_check and repair.bool_guard.original is original_module

        fake_upstream = temporary / "upstream.json"
        fake_upstream.write_text("{}\n", encoding="utf-8")
        fake_ref, wrapper_ref = fake_upstream.relative_to(ROOT).as_posix(), (temporary / "wrapper.json").relative_to(ROOT).as_posix()
        wrapper = {"upstream_plan": fake_ref, "upstream_plan_sha256": sha(fake_upstream), "max_wall_seconds": 300}
        saved_wrapper, saved_plan = repair.coordinator.WRAPPER, repair.coordinator.provenance.PLAN
        with patch.object(repair, "UPSTREAM", fake_ref), patch.object(repair, "WRAPPER", wrapper_ref):
            with repair.repair_context(wrapper):
                assert repair.coordinator.WRAPPER == wrapper_ref and repair.coordinator.provenance.PLAN == fake_upstream
            assert repair.coordinator.WRAPPER == saved_wrapper and repair.coordinator.provenance.PLAN == saved_plan
            try:
                with repair.repair_context(dict(wrapper, upstream_plan_sha256="0" * 64)):
                    pass
            except ValueError:
                pass
            else:
                raise AssertionError("wrong upstream hash accepted")

        dummy = temporary / "unredirected.log"
        dummy.write_bytes(b"placeholder")
        try:
            repair.verify_log_descriptors(dummy)
        except ValueError:
            pass
        else:
            raise AssertionError("unrecorded descriptors accepted")
        log, receipt = temporary / "redirected.log", temporary / "redirected.json"
        code = ("import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
                "import native_zero_bool_repair as r;r.verify_log_descriptors(Path(sys.argv[2]));print('fd-ok')")
        command = [sys.executable, "-u", "-c", code, str(ROOT / "scripts/validation"), str(log)]
        result = run_recorded(command, cwd=ROOT, log_path=log, receipt_path=receipt, max_wall_seconds=15)
        assert result["status"] == "completed" and result["exit_code"] == 0 and b"fd-ok" in log.read_bytes()
        assert result["log_sha256"] == sha(log)
        record = {"status": "offline-repair-adapter-check-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "bool_adapter_passed": True, "disabled_flag_rejected_after_raw_capture": True,
                  "raw_arrays_and_dtypes_preserved": True, "original_functions_restored": True,
                  "wrong_upstream_hash_refused": True, "unrecorded_descriptors_refused": True,
                  "recorded_descriptors_accepted": True,
                  "source_sha256": {str(path): sha(path) for path in (Path(__file__), Path(repair.__file__),
                      Path(repair.bool_guard.__file__), Path(repair.original_guard.__file__))},
                  "scope": "CPU minimal MuJoCo model and short test child; no actual XGO runtime",
                  "physics_steps": 0, "learning_updates": 0, "hardware_io": False, "cuda_initialized": False}
        (base / "native-zero-bool-repair-check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "raw_readback": True, "logging_gate": True,
                          "physics_steps": 0}, ensure_ascii=False))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
