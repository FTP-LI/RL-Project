"""Exercise selective friction comparison using saved arrays and CPU fixtures."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from selective_friction_guard import STATIC_FIELDS, RESET_FIELDS, compare_reset_fields, compare_static, resolve_foot_geometry_ids

PROTOCOL = ROOT / "configs/experiments/roll_selective_friction_guard_preflight.json"
OUT = ROOT / "records/reports/roll-selective-friction-guard"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    project, plan = read(ROOT / "project.json"), read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    assert plan["status"] == "offline-only-not-runtime-authorized" and plan["physics_steps_authorized"] == 0
    for name, expected in plan["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    prior = read(ROOT / "records/reports/roll-s20-saved-startup-review/rollout-start-verification.json")
    assert prior["status"] == "passed-independent-saved-rollout-start-verification"
    assert prior["geometry_index_to_name_mapping_checked"] is False
    tree = ast.parse((ROOT / "scripts/validation/inspect_common_skill_plant.py").read_text(encoding="utf-8"))
    canonical = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "FIELDS" for target in node.targets)]
    assert len(canonical) == 1 and set(canonical[0]) == STATIC_FIELDS and len(STATIC_FIELDS) == 33
    assert RESET_FIELDS == {"dof_damping", "dof_frictionloss", "geom_friction"}
    ids = tuple(plan["assumed_fixture_geom_indices"])
    reference_dir = ROOT / "records/reports/roll-domain-parameter-random-s20/low-low/zero"
    reference_expanded = arrays(reference_dir / "expanded-fields-before-physics.npz")
    reference_primary = arrays(reference_dir / "nominal-primary-fields.npz")
    reference_static = arrays(reference_dir / "scene-static-fields.npz")
    rows = []
    for arm in plan["arms"]:
        source_plan = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{arm}.json")
        friction = source_plan["parameter_arm"]["factors"]["foot_friction"]
        for label, folder in (("zero", ROOT / f"records/reports/roll-domain-parameter-random-s20/{arm}/zero"),
                              ("sample-start", ROOT / "records/evaluations" / source_plan["run_id"])):
            expanded = arrays(folder / "expanded-fields-before-physics.npz")
            primary = arrays(folder / "nominal-primary-fields.npz")
            static = arrays(folder / "scene-static-fields.npz")
            compare_static(static, reference_static)
            compare_reset_fields(expanded, reference_expanded, foot_geom_ids=ids, friction=friction, worlds=128)
            compare_reset_fields(primary, reference_primary, foot_geom_ids=ids, friction=friction)
            rows.append({"arm": arm, "source": label, "friction": friction, "static_fields": len(static),
                         "expanded_fields": len(expanded), "primary_fields": len(primary)})

    cases = []

    def reject(name, action):
        try:
            action()
        except (ValueError, KeyError, TypeError, AssertionError):
            cases.append(name)
            return
        raise AssertionError(f"Unexpected acceptance: {name}")

    base = reference_expanded

    def altered(field, index, value):
        result = {key: item.copy() for key, item in base.items()}
        result[field][index] = value
        return result

    kwargs = {"foot_geom_ids": ids, "friction": 0.8, "worlds": 128}
    reject("non-target-geometry", lambda: compare_reset_fields(altered("geom_friction", (0, 0, 0), 2.0), base, **kwargs))
    reject("foot-other-friction-component", lambda: compare_reset_fields(altered("geom_friction", (0, ids[0], 1), 2.0), base, **kwargs))
    reject("foot-one-world-wrong", lambda: compare_reset_fields(altered("geom_friction", (0, ids[0], 0), 1.0), base, **kwargs))
    reject("damping-changed", lambda: compare_reset_fields(altered("dof_damping", (0, 0), base["dof_damping"][0, 0] + 1), base, **kwargs))
    reject("frictionloss-changed", lambda: compare_reset_fields(altered("dof_frictionloss", (0, 0), base["dof_frictionloss"][0, 0] + 1), base, **kwargs))
    reject("wrong-requested-friction", lambda: compare_reset_fields(base, base, foot_geom_ids=ids, friction=1.3, worlds=128))
    reject("missing-field", lambda: compare_reset_fields({k: v for k, v in base.items() if k != "dof_damping"}, base, **kwargs))
    reject("extra-field", lambda: compare_reset_fields({**base, "unexpected": np.zeros(1)}, base, **kwargs))
    reject("nan-field", lambda: compare_reset_fields(altered("geom_friction", (0, ids[0], 0), np.nan), base, **kwargs))
    reject("wrong-dtype", lambda: compare_reset_fields({**base, "geom_friction": base["geom_friction"].astype(np.float64)}, base, **kwargs))
    reject("wrong-shape", lambda: compare_reset_fields({**base, "geom_friction": base["geom_friction"][:127]}, base, **kwargs))
    reject("wrong-world-count", lambda: compare_reset_fields(base, base, foot_geom_ids=ids, friction=0.8, worlds=127))
    reject("duplicate-foot-indices", lambda: compare_reset_fields(base, base, foot_geom_ids=(ids[0], ids[0]), friction=0.8, worlds=128))
    reject("out-of-range-foot-index", lambda: compare_reset_fields(base, base, foot_geom_ids=(ids[0], 31), friction=0.8, worlds=128))
    reject("boolean-foot-index", lambda: compare_reset_fields(base, base, foot_geom_ids=(False, ids[1]), friction=0.8, worlds=128))
    reject("invalid-friction", lambda: compare_reset_fields(base, base, foot_geom_ids=ids, friction=float("nan"), worlds=128))
    reject("negative-friction", lambda: compare_reset_fields(base, base, foot_geom_ids=ids, friction=-0.1, worlds=128))
    changed_static = {key: item.copy() for key, item in reference_static.items()}
    changed_static["geom_friction"][ids[0], 0] += 1
    reject("static-foot-friction-changed", lambda: compare_static(changed_static, reference_static))
    reject("static-missing-field", lambda: compare_static({k: v for k, v in reference_static.items() if k != "geom_friction"}, reference_static))
    reject("static-both-empty", lambda: compare_static({}, {}))
    without_mass = {k: v for k, v in reference_static.items() if k != "body_mass"}
    reject("static-both-missing-mass", lambda: compare_static(without_mass, without_mass))
    without_static_friction = {k: v for k, v in reference_static.items() if k != "geom_friction"}
    reject("static-both-missing-friction", lambda: compare_static(without_static_friction, without_static_friction))
    reset_only_geom = {"geom_friction": base["geom_friction"]}
    reject("reset-both-only-friction", lambda: compare_reset_fields(reset_only_geom, reset_only_geom, **kwargs))
    without_damping = {k: v for k, v in base.items() if k != "dof_damping"}
    reject("reset-both-missing-damping", lambda: compare_reset_fields(without_damping, without_damping, **kwargs))
    without_frictionloss = {k: v for k, v in base.items() if k != "dof_frictionloss"}
    reject("reset-both-missing-frictionloss", lambda: compare_reset_fields(without_frictionloss, without_frictionloss, **kwargs))
    extra_both = {**base, "unexpected": np.zeros((128, 1))}
    reject("reset-both-extra-field", lambda: compare_reset_fields(extra_both, extra_both, **kwargs))

    class FakeModel:
        ngeom = 31

        def __init__(self, mapping):
            self.mapping = mapping

        def geom(self, name):
            return SimpleNamespace(id=self.mapping[name])

    mapping = {"robot/left_foot_collision": ids[0], "robot/right_foot_collision": ids[1]}
    assert resolve_foot_geometry_ids(FakeModel(mapping)) == ids
    cases.append("fixture-name-resolution")
    reject("missing-foot-name", lambda: resolve_foot_geometry_ids(FakeModel({"robot/left_foot_collision": ids[0]})))
    reject("duplicate-resolved-indices", lambda: resolve_foot_geometry_ids(FakeModel({key: ids[0] for key in mapping})))
    reject("invalid-resolved-index", lambda: resolve_foot_geometry_ids(FakeModel({**mapping, "robot/right_foot_collision": 31})))
    reject("fractional-resolved-index", lambda: resolve_foot_geometry_ids(FakeModel({**mapping, "robot/right_foot_collision": 30.5})))
    before = deepcopy(base)
    compare_reset_fields(base, base, **kwargs)
    assert all(np.array_equal(base[key], before[key]) for key in base)
    cases.append("comparison-does-not-mutate-inputs")
    assert "torch" not in sys.modules

    paths = [Path(__file__), ROOT / "scripts/validation/selective_friction_guard.py", PROTOCOL,
             ROOT / "scripts/validation/inspect_common_skill_plant.py", OUT / "verification.json", OUT / "schema-first-failure.json",
             OUT / "sources-before-schema-guard/selective_friction_guard.py.txt",
             OUT / "sources-before-schema-guard/verify_selective_friction_guard.py.txt",
             ROOT / "records/reports/roll-s20-saved-startup-review/verification.json",
             ROOT / "records/reports/roll-s20-saved-startup-review/rollout-start-verification.json"]
    result = {"status": "passed-saved-data-and-cpu-fixtures", "checked_at": datetime.now(timezone.utc).isoformat(),
              "saved_array_comparisons": rows, "fault_cases": cases, "source_files_checked": len(plan["source_sha256"]),
              "fixture_geom_indices": list(ids), "actual_model_geometry_mapping_verified": False,
              "static_required_fields": len(STATIC_FIELDS), "reset_required_fields": sorted(RESET_FIELDS),
              "runtime_guard_integrated": False, "physics_steps": 0, "learning_updates": 0,
              "hardware_io": False, "torch_imported": False,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              "scope": "Saved arrays and a fake model only; actual model name mapping, runtime integration, voltage state and CUDA zero remain unverified."}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "verification-schema-guard.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "saved_configurations_checked": len(rows),
                      "fault_and_binding_cases": len(cases), "source_files_checked": result["source_files_checked"]}, indent=2))


if __name__ == "__main__":
    main()
