"""Inspect the zero-step actuator dtype failure using saved data and CPU MuJoCo.

The small model isolates Python binding dtypes; it is not the XGO-Duck scene.
No simulator step, CUDA operation, learning, or hardware access is performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import model_force_limit_guard as guard

RUN = ROOT / "records/reports/duck_roll-factor_nominal_s1241_20261115T214019"
PARAMS = ROOT / "records/reports/matched-start-source-audit/sources/07-1910_m6.json.txt"
SIM_SOURCE = Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/lib/python3.12/site-packages/mjlab/sim/sim.py")
OUT = ROOT / "records/reports/native-observer-preflight/force-dtype-diagnosis.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract_path = RUN / "zero/runtime-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (contract.get("status") != "closed" or contract.get("control_steps") != 0
            or contract.get("physics_substeps") != 0 or contract.get("interface_checks") != 0):
        raise ValueError("unexpected recorded zero-run contract")
    parameters = json.loads(PARAMS.read_text(encoding="utf-8"))
    limit = 7.7 * float(parameters["kt"]) / float(parameters["R"])
    xml = f'''<mujoco><worldbody><body><joint name="robot/hinge" type="hinge"/>
      <geom type="sphere" size="0.05" mass="1"/></body></worldbody>
      <actuator><motor joint="robot/hinge" forcelimited="true" forcerange="{-limit:.17g} {limit:.17g}"/></actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    fields = {}
    for name in ("actuator_trnid", "actuator_trntype", "actuator_forcelimited", "actuator_forcerange"):
        array = np.asarray(getattr(model, name))
        fields[name] = {"dtype": str(array.dtype), "shape": list(array.shape), "values": array.tolist(),
                        "numpy_integer_subtype": bool(np.issubdtype(array.dtype, np.integer))}
    try:
        guard.check(model, ["hinge"], 7.7, parameters)
    except ValueError as exc:
        minimal_guard = {"status": "rejected", "error": str(exc)}
    else:
        minimal_guard = {"status": "passed", "error": None}

    saved_path = RUN / "zero/scene-static-fields.npz"
    with np.load(saved_path, allow_pickle=False) as archive:
        actual_mapping = np.asarray(archive["actuator_trnid"])
        saved = {"actuator_trnid_dtype": str(actual_mapping.dtype), "actuator_trnid_shape": list(actual_mapping.shape),
                 "flag_field_saved": "actuator_forcelimited" in archive.files,
                 "transmission_type_saved": "actuator_trntype" in archive.files}
    sim_text = SIM_SOURCE.read_text(encoding="utf-8")
    if "def mj_model(self) -> mujoco.MjModel:" not in sim_text or "self._mj_model = model" not in sim_text:
        raise ValueError("installed simulation model binding differs from the inspected code")
    result = {"status": "cpu-dtype-diagnosis-completed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "mujoco_version": mujoco.__version__, "minimal_model_fields": fields, "minimal_model_guard": minimal_guard,
              "saved_xgo_fields": saved, "installed_sim_model_binding": "mujoco.MjModel",
              "recorded_control_steps": contract["control_steps"], "recorded_physics_substeps": contract["physics_substeps"],
              "recorded_interface_checks": contract["interface_checks"],
              "source_sha256": {str(path): sha(path) for path in (Path(__file__), Path(guard.__file__),
                                PARAMS, SIM_SOURCE, contract_path, saved_path)},
              "scope": "Actual saved XGO mapping and a separate CPU-only MuJoCo model; actual XGO flag array was not saved.",
              "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "fields": fields, "minimal_model_guard": minimal_guard,
                      "saved_xgo_fields": saved}, ensure_ascii=False))


if __name__ == "__main__":
    main()
