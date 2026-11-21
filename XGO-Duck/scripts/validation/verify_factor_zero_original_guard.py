"""Exercise the new entry through the original guard with fake environment data."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import guarded_factor_zero_entry as entry
import factor_scene_guard_adapter as bindings
from verify_guarded_factor_zero_entry import arrays, fixture_plan, write

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
    prior = entry.read(OUT / "entry-fixtures.json")
    assert prior["status"] == "passed-cpu-fake-force-limit-entry-real-zero-pending"
    for path, expected in prior["source_sha256"].items():
        assert entry.sha(entry.inside(path)) == expected, path
    original = entry.inside(entry.REFERENCE_ROOT) / "roll"
    static, primary = (arrays(original / name) for name in entry.REFERENCE_FILES)
    cases = []
    import torch
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden")), \
         patch.object(torch.cuda, "init", side_effect=RuntimeError("CUDA forbidden")):
        import mjlab.envs
        import mjlab.scene
        import common_scene_contract as scene

        original_env, original_compare = mjlab.envs.ManagerBasedRlEnv, scene.compare_arrays

        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        def configuration(_):
            actuator = NS(vin_range=(7.7, 7.7))
            return NS(events={"foot_friction": NS(params={"ranges": (1.0, 1.0)})},
                      scene=NS(entities={"robot": NS(articulation=NS(actuators=[actuator]))}))

        def make_fake(fault, plan):
            counts = {"created": 0, "closed": 0, "physical_steps": 0, "resets": 0}

            class FakeModel:
                ngeom = 31
                def __init__(self, cfg):
                    for key, value in static.items(): setattr(self, key, value.copy())
                    if fault == "bad-static": self.geom_friction[0, 0] += 1
                    self.nu = len(plan["walking"]["joint_names"])
                    self.actuator_trntype = np.zeros(self.nu, dtype=np.int32)
                    self.actuator_forcelimited = np.ones(self.nu, dtype=np.uint8)
                    motor = entry.read(entry.inside(entry.MOTOR_PARAMS))
                    voltage = cfg.scene.entities["robot"].articulation.actuators[0].vin_range[0]
                    limit = voltage * motor["kt"] / motor["R"]
                    self.actuator_forcerange = np.tile(np.array([-limit, limit], np.float64), (self.nu, 1))
                    if fault == "bad-force-limit": self.actuator_forcerange[0, 1] += 0.1
                    if fault == "force-limit-disabled": self.actuator_forcelimited[0] = 0
                    self.opt = NS(timestep=0.004, iterations=30, ls_iterations=50)
                def joint(self, name):
                    return NS(id=plan["walking"]["joint_names"].index(name.removeprefix("robot/")) + 1)
                def geom(self, name):
                    return NS(id={"robot/left_foot_collision": 12, "robot/right_foot_collision": 30}[name])

            class FakeBase:
                def __init__(self, cfg, **kwargs):
                    counts["created"] += 1
                    self.common_step_counter, self.num_envs = 0, 128
                    self.physics_dt, self.step_dt = 0.004, 0.02
                    fields = {key: value.copy() for key, value in primary.items()}
                    fields["geom_friction"][[12, 30], 0] = np.float32(cfg.events["foot_friction"].params["ranges"][0])
                    expanded = {key: np.broadcast_to(value, (128,) + value.shape).copy() for key, value in fields.items()}
                    if fault == "all-world-nontarget": expanded["geom_friction"][:, 0, 0] += 1
                    self.sim = NS(mj_model=FakeModel(cfg), model=NS(**{k: Tensor(v) for k, v in expanded.items()}),
                                  expanded_fields=set(expanded), wp_data=NS(contact=NS(dist=np.zeros(25600, np.float32))))
                def reset(self, *args, **kwargs):
                    counts["resets"] += 1
                    return ({"actor": None}, {})
                def step(self, *args, **kwargs):
                    counts["physical_steps"] += 1
                    raise AssertionError("physical step forbidden")
                def close(self): counts["closed"] += 1
            return FakeBase, counts

        def exercise(temp, arm, fault=None):
            folder = temp / (arm + "-" + (fault or "valid"))
            folder.mkdir()
            plan, path, project, output, page, observer, admission, sync = fixture_plan(folder, arm)
            decision = entry.preflight(project, plan, path, "zero")
            assert decision["allowed"], decision
            fake, counts = make_fake(fault, plan)
            base = NS(configuration=configuration, install_guard=scene.install_guard, PLAN=None, OUT=None, PAGE=None,
                      PROJECT=None, STARTUP=None, collect=lambda *args: None)

            def run(zero):
                assert zero is True
                target = base.OUT / "zero"
                target.mkdir()
                cfg = base.configuration(plan)
                cls = base.install_guard(plan["roll"], target, base.STARTUP, "rollout")
                env = cls(cfg=cfg)
                try:
                    env.reset()
                    if fault == "attempt-step": env.step(None)
                    (target / "startup.npz").write_bytes(b"synthetic startup")
                    write(target / "startup.json", {"status": "passed", "physical_substeps": 0,
                                                    "startup_sha256": entry.sha(target / "startup.npz")})
                finally:
                    env.close()
            base.run = run
            # The facade is a fake CUDA indicator: it does not initialize CUDA.
            simulated_torch = NS(no_grad=nullcontext, cuda=NS(is_initialized=lambda: True))
            with patch.object(mjlab.envs, "ManagerBasedRlEnv", fake), \
                 patch.object(scene, "interface_check", return_value={"status": "fake-interface"}):
                try:
                    result = entry.execute_zero(path, plan, decision,
                                                (base, scene, mjlab.envs, simulated_torch, bindings), lambda: True)
                finally:
                    assert mjlab.envs.ManagerBasedRlEnv is fake and scene.compare_arrays is original_compare
                    assert counts["created"] == counts["closed"] == 1 and counts["physical_steps"] == 0
                    expected_resets = 0 if fault in {"bad-static", "bad-force-limit", "force-limit-disabled"} else 1
                    assert counts["resets"] == expected_resets
            assert mjlab.envs.ManagerBasedRlEnv is original_env
            assert result["foot_geom_ids"] == [12, 30]
            runtime = entry.read(output / "zero/runtime-contract.json")
            assert runtime["control_steps"] == runtime["physics_substeps"] == 0
            force = entry.read(output / "zero/actuator-force-limits.json")
            assert force["checked_before_reset"] and force["actuator_count"] == 14

        with patch.object(mjlab.scene.Scene, "__init__", side_effect=RuntimeError("Real scene forbidden")), \
             tempfile.TemporaryDirectory(prefix="original-zero-guard-", dir=OUT) as temporary:
            temp = Path(temporary)
            for arm in ("nominal", "friction-low", "voltage-high"):
                exercise(temp, arm)
                cases.append("original-guard-zero-" + arm)
            for fault, kind, message in (("bad-static", ValueError, "static physical field changed"),
                                         ("all-world-nontarget", ValueError, "unexpected or incorrect geom_friction change"),
                                         ("bad-force-limit", ValueError, "compiled actuator force range differs"),
                                         ("force-limit-disabled", ValueError, "actuator force limit must be enabled"),
                                         ("attempt-step", RuntimeError, "zero-entry-forbids-physics-step")):
                try:
                    exercise(temp, "nominal", fault)
                except kind as exc:
                    assert message in str(exc), (fault, str(exc))
                else:
                    raise AssertionError("fault was not rejected: " + fault)
                cases.append("original-guard-stop-" + fault)
    assert not torch.cuda.is_initialized()
    assert mjlab.envs.ManagerBasedRlEnv is original_env and scene.compare_arrays is original_compare
    paths = [Path(__file__), Path(entry.__file__), PROTOCOL, OUT / "entry-fixtures.json",
             ROOT / "scripts/validation/verify_guarded_factor_zero_entry.py",
             ROOT / "scripts/validation/model_force_limit_guard.py",
             ROOT / "scripts/validation/factor_scene_guard_adapter.py",
             ROOT / "scripts/validation/common_scene_contract.py"]
    result = {"status": "passed-original-guard-fake-force-limit-entry", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "original_guard_function_exercised": True, "interface_check_stubbed": True,
              "fake_model_only": True, "cuda_indicator_simulated": True, "torch_cuda_initialized": False,
              "real_environment_constructed": False, "observer_verified": False,
              "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
              "source_sha256": {path.relative_to(ROOT).as_posix(): entry.sha(path) for path in paths},
              "scope": "Real original guard and entry wiring with fake simulator, fake model and fake policy interface; no CUDA zero run."}
    record = OUT / "original-guard-entry.json"
    if record.exists():
        previous = entry.read(record)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}, "source or result changed; keep existing evidence and create a separate review"
    else:
        with record.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(cases), "physics_steps": 0,
                      "torch_cuda_initialized": False}, indent=2))


if __name__ == "__main__":
    main()
