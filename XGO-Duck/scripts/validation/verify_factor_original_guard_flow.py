"""Exercise the original scene guard through a fake base environment, CPU only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
from factor_scene_guard_adapter import factor_guard_bindings, require_complete

PROTOCOL = ROOT / "configs/experiments/roll_factor_guard_adapter_cpu_preflight.json"
OUT = ROOT / "records/reports/roll-factor-guard-adapter"


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
    project, protocol = read(ROOT / "project.json"), read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    assert protocol["status"] == "offline-adapter-test-only" and protocol["physics_steps_authorized"] == 0
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    fixture = read(OUT / "fake-bindings-verification-reset-enforced.json")
    assert fixture["status"] == "passed-fake-scene-bindings"
    for name, expected in fixture["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    draft = read(ROOT / "configs/experiments/roll_parameter_matched_review_draft.json")
    nominal = {"foot_friction": 1.0, "vin_range_v": 7.7}
    ranges = {"foot_friction": (0.7, 1.3), "vin_range_v": (7.4, 8.0)}
    original = ROOT / "records/reports/roll-domain-parameter-random-s20/low-low/zero"
    names = ("scene-static-fields.npz", "nominal-primary-fields.npz")
    references = {name: sha(original / name) for name in names}
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    import torch
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden")), \
         patch.object(torch.cuda, "init", side_effect=RuntimeError("CUDA forbidden")):
        import mjlab.envs
        import mjlab.scene
        import common_scene_contract as scene

        real_environment = mjlab.envs.ManagerBasedRlEnv
        original_compare = scene.compare_arrays
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        with patch.object(mjlab.scene.Scene, "__init__", side_effect=RuntimeError("Real scene forbidden")), \
             tempfile.TemporaryDirectory(prefix="factor-original-guard-", dir=OUT) as temp:
            root = Path(temp) / "startup"
            (root / "roll").mkdir(parents=True)
            for name in names:
                shutil.copyfile(original / name, root / "roll" / name)
                assert sha(root / "roll" / name) == references[name]
            static_reference, primary_reference = (arrays(root / "roll" / name) for name in names)

            def configuration(_):
                actuator = NS(vin_range=(7.7, 7.7))
                robot = NS(articulation=NS(actuators=[actuator]))
                return NS(events={"foot_friction": NS(params={"ranges": (1.0, 1.0)})},
                          scene=NS(entities={"robot": robot}))

            def fake_environment(fault=None):
                counts = {"created": 0, "closed": 0, "steps": 0}

                class FakeModel:
                    ngeom = 31
                    def __init__(self):
                        for key, value in static_reference.items(): setattr(self, key, value.copy())
                        if fault == "bad-static": self.geom_friction[0, 0] += 1
                        self.opt = NS(timestep=0.004, iterations=30, ls_iterations=50)
                    def geom(self, name):
                        mapping = {"robot/left_foot_collision": 12, "robot/right_foot_collision": 30}
                        if fault == "missing-geometry": mapping.pop("robot/right_foot_collision")
                        return NS(id=mapping[name])

                class FakeBase:
                    def __init__(self, cfg, **kwargs):
                        counts["created"] += 1
                        self.common_step_counter, self.num_envs = 0, 128
                        self.physics_dt, self.step_dt = 0.004, 0.02
                        primary = {key: value.copy() for key, value in primary_reference.items()}
                        primary["geom_friction"][[12, 30], 0] = np.float32(cfg.events["foot_friction"].params["ranges"][0])
                        expanded = {key: np.broadcast_to(value, (128,) + value.shape).copy() for key, value in primary.items()}
                        if fault == "single-world-change": expanded["geom_friction"][7, 0, 0] += 1
                        if fault == "all-world-nontarget": expanded["geom_friction"][:, 0, 0] += 1
                        if fault == "missing-damping": expanded.pop("dof_damping")
                        model = NS(**{key: Tensor(value) for key, value in expanded.items()})
                        self.sim = NS(mj_model=FakeModel(), model=model, expanded_fields=set(expanded),
                                      wp_data=NS(contact=NS(dist=np.zeros(25600, dtype=np.float32))))
                    def reset(self, *args, **kwargs): return ({"actor": None}, {})
                    def step(self, *args, **kwargs):
                        counts["steps"] += 1
                        raise AssertionError("Physical step forbidden in fake test")
                    def close(self): counts["closed"] += 1
                return FakeBase, counts

            def exercise(arm, fault=None):
                fake, counts = fake_environment(fault)
                folder = Path(temp) / f"run-{len(cases)}-{fault or arm['name']}"
                folder.mkdir()
                base = NS(configuration=configuration, install_guard=scene.install_guard)
                original_configuration, original_install = base.configuration, base.install_guard
                with patch.object(mjlab.envs, "ManagerBasedRlEnv", fake), \
                     patch.object(scene, "interface_check", return_value={"status": "fake-interface"}):
                    try:
                        with factor_guard_bindings(base, scene, mjlab.envs, arm=arm, nominal=nominal, ranges=ranges,
                                                   startup_root=root, reference_sha256=references) as state:
                            cfg = base.configuration({})
                            assert cfg.events["foot_friction"].params["ranges"] == (arm["foot_friction"],) * 2
                            assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (arm["vin_range_v"],) * 2
                            cls = base.install_guard({"skill": "roll"}, folder, root, "rollout")
                            env = cls(cfg=cfg)
                            try:
                                env.reset()
                                checked = require_complete(state)
                                assert checked["foot_geom_ids"] == [12, 30]
                                assert len(state["static"]) == len(state["primary"]) == len(state["expanded"]) == 1
                            finally:
                                env.close()
                    finally:
                        assert mjlab.envs.ManagerBasedRlEnv is fake and scene.compare_arrays is original_compare
                        assert base.configuration is original_configuration and base.install_guard is original_install
                        assert counts["created"] == counts["closed"] == 1 and counts["steps"] == 0
                assert mjlab.envs.ManagerBasedRlEnv is real_environment
                return folder, counts

            for arm in draft["arms"]:
                folder, counts = exercise(arm)
                runtime = read(folder / "runtime-contract.json")
                assert runtime["control_steps"] == runtime["physics_substeps"] == 0
                cases.append("original-guard-valid-" + arm["name"])
            faults = {
                "bad-static": (ValueError, "static physical field changed: geom_friction"),
                "single-world-change": (AssertionError, "geom_friction"),
                "all-world-nontarget": (ValueError, "unexpected or incorrect geom_friction change"),
                "missing-damping": (ValueError, "physical-field set mismatch"),
                "missing-geometry": (KeyError, "robot/right_foot_collision"),
            }
            for fault, (error_type, message) in faults.items():
                try:
                    exercise(draft["arms"][1], fault)
                except error_type as exc:
                    assert message in str(exc), (fault, str(exc))
                    cases.append("original-guard-stopped-" + fault)
                else:
                    raise AssertionError(f"Original guard flow accepted {fault}")
            assert scene.compare_arrays is original_compare and mjlab.envs.ManagerBasedRlEnv is real_environment
    assert not torch.cuda.is_initialized()
    p = read(ROOT / "project.json")
    assert p["goal_execution"]["pause_requested_by_user"] and p["staged_training_request"]["pause_requested"]
    paths = [Path(__file__), ROOT / "scripts/validation/factor_scene_guard_adapter.py",
             ROOT / "scripts/validation/common_scene_contract.py", ROOT / "scripts/validation/selective_friction_guard.py",
             OUT / "fake-bindings-verification-reset-enforced.json", OUT / "original-guard-flow-verification-exception-matched.json",
             OUT / "sources-before-reset-enforcement/verify_factor_original_guard_flow.py.txt", PROTOCOL]
    result = {"status": "passed-original-guard-flow-with-fake-environment", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "source_files_checked": len(protocol["source_sha256"]),
              "original_guard_function_exercised": True, "real_environment_constructed": False,
              "fake_model_only": True, "interface_check_stubbed": True, "real_model_mapping_verified": False,
              "checks_enforced_before_reset_return": True,
              "observer_verified": False, "torch_cuda_initialized": False, "physics_steps": 0,
              "learning_updates": 0, "hardware_io": False,
              "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
              "scope": "Original guard control flow with fake base environment and fake geometry map; no real scene, CUDA zero or runtime validation."}
    target = OUT / "original-guard-flow-verification-reset-enforced.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(cases),
                      "physics_steps": 0, "torch_cuda_initialized": False}, indent=2))


if __name__ == "__main__":
    main()
