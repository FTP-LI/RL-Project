"""Exercise factor declaration and mutation with isolated CPU-only fixtures."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_factor_configuration import apply_to_nominal_config, factor_configuration, validate_arm

PROTOCOL = ROOT / "configs/experiments/roll_factor_configuration_cpu_preflight.json"
OUT = ROOT / "records/reports/roll-factor-configuration"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = read(ROOT / "project.json")
    assert project["goal_execution"]["pause_requested_by_user"]
    assert project["staged_training_request"]["pause_requested"]
    preflight = read(PROTOCOL)
    for name, expected in preflight["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    draft = read(ROOT / preflight["arms_source"])
    recorded = read(ROOT / preflight["range_source"])["skills"]["roll"]
    nominal = preflight["nominal"]
    ranges = {"foot_friction": recorded["training_foot_friction"], "vin_range_v": recorded["training_voltage"]}
    calls = []

    def fresh(_):
        calls.append(1)
        actuator = SimpleNamespace(vin_range=(7.7, 7.7), untouched="preserved")
        robot = SimpleNamespace(articulation=SimpleNamespace(actuators=[actuator]))
        event = SimpleNamespace(params={"ranges": (1.0, 1.0), "other": "preserved"})
        return SimpleNamespace(events={"foot_friction": event}, scene=SimpleNamespace(entities={"robot": robot}))

    cases = []
    for arm in draft["arms"]:
        cfg = factor_configuration(fresh, arm, nominal, ranges)({})
        assert cfg.events["foot_friction"].params == {"ranges": (arm["foot_friction"],) * 2, "other": "preserved"}
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        assert actuator.vin_range == (arm["vin_range_v"],) * 2 and actuator.untouched == "preserved"
        cases.append("apply-" + arm["name"])
    assert len(calls) == len(draft["arms"])

    mutable_arm = deepcopy(draft["arms"][1])
    mutable_nominal = dict(nominal)
    mutable_ranges = {key: list(value) for key, value in ranges.items()}
    callback = factor_configuration(fresh, mutable_arm, mutable_nominal, mutable_ranges)
    mutable_arm.update(draft["arms"][2])
    mutable_nominal["foot_friction"] = 1.2
    mutable_ranges["foot_friction"][:] = [1.0, 1.0]
    cfg = callback({})
    assert cfg.events["foot_friction"].params["ranges"] == (0.8, 0.8)
    assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (7.7, 7.7)
    cases.append("callback-keeps-validated-factors-after-caller-mutation")

    reference = draft["arms"][0]
    invalid = [
        ("two-factors-mislabeled", {**reference, "foot_friction": 0.8, "vin_range_v": 8.0, "changed_factors": []}),
        ("two-factors-declared", {**reference, "foot_friction": 0.8, "vin_range_v": 8.0, "changed_factors": list(ranges)}),
        ("factor-not-declared", {**reference, "foot_friction": 0.8}),
        ("factor-declared-without-change", {**reference, "changed_factors": ["foot_friction"]}),
        ("duplicate-factor", {**reference, "foot_friction": 0.8, "changed_factors": ["foot_friction", "foot_friction"]}),
        ("unknown-factor", {**reference, "foot_friction": 0.8, "changed_factors": ["unknown"]}),
        ("nonnumeric-factor-label", {**reference, "changed_factors": [[]]}),
        ("nan", {**reference, "foot_friction": float("nan")}),
        ("infinity", {**reference, "vin_range_v": float("inf")}),
        ("boolean-value", {**reference, "foot_friction": True}),
        ("friction-out-of-range", {**reference, "foot_friction": 0.69, "changed_factors": ["foot_friction"]}),
        ("voltage-out-of-range", {**reference, "vin_range_v": 8.01, "changed_factors": ["vin_range_v"]}),
        ("unsafe-name", {**reference, "name": "../outside"}),
        ("unexpected-field", {**reference, "extra": 1}),
    ]
    for label, arm in invalid:
        before = len(calls)
        try:
            factor_configuration(fresh, arm, nominal, ranges)
        except ValueError:
            pass
        else:
            raise AssertionError(label)
        assert len(calls) == before, label
        cases.append("reject-" + label)
    cfg = fresh({})
    cfg.events["foot_friction"].params["ranges"] = (0.8, 0.8)
    original = deepcopy(cfg)
    try:
        apply_to_nominal_config(cfg, draft["arms"][2], nominal, ranges)
    except ValueError:
        pass
    else:
        raise AssertionError("nonnominal input must fail")
    assert cfg.events["foot_friction"].params == original.events["foot_friction"].params
    assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (7.7, 7.7)
    cases.append("reject-nonnominal-input-without-mutation")
    assert "torch" not in sys.modules
    paths = [Path(__file__), ROOT / "scripts/validation/roll_factor_configuration.py", PROTOCOL]
    result = {"status": "passed-isolated-cpu-fixtures", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
              "torch_imported": False, "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              "scope": "Factor declaration and configuration-object mutations on fixtures; not a simulation or runtime-integration test."}
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "adapter-fixture-verification-fixed-callback.json"
    if receipt.exists():
        previous = read(receipt)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}, "Existing evidence differs; retain it and create a separate protocol."
    else:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
