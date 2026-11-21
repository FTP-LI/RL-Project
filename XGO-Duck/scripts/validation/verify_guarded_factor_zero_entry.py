"""Verify paused admission and zero-only wiring without a real simulator."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import guarded_factor_zero_entry as entry
import factor_scene_guard_adapter as bindings

OUT = ROOT / "records/reports/force-limit-guard"
PROTOCOL = ROOT / "configs/experiments/force_limit_guard_cpu_preflight.json"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def fixture_plan(temp, arm_name):
    parent = entry.read(entry.inside(entry.PARENT))
    draft = entry.read(entry.inside(entry.DRAFT))
    arm = next(a for a in draft["arms"] if a["name"] == arm_name)
    plan = deepcopy(parent)
    seed = 91241
    plan.update(stage="S20-roll-factor-zero-" + arm_name, seed=seed,
                run_id=f"duck_roll-factor_{arm_name}_s{seed}_20990101T000000")
    plan["roll"]["seed"] = seed
    output = temp / plan["run_id"]
    output.mkdir()
    page, admission = temp / (arm_name + "-observer.html"), temp / (arm_name + "-admission.json")
    page.write_text("<p>synthetic observer only</p>", encoding="utf-8")
    plan.update(zero_authorized=True, zero_physics_substeps_authorized=0, dynamics_authorized=False,
                factor_arm=arm_name, parameter_factors={k: arm[k] for k in ("foot_friction", "vin_range_v")},
                guarded_output=output.relative_to(ROOT).as_posix(), page=page.relative_to(ROOT).as_posix(),
                admission_record=admission.relative_to(ROOT).as_posix(), reference_root=entry.REFERENCE_ROOT,
                reference_sha256={n: parent["frozen_sha256"][f"{entry.REFERENCE_ROOT}/roll/{n}"] for n in entry.REFERENCE_FILES})
    plan["frozen_sha256"].update({name: entry.sha(entry.inside(name)) for name in entry.REQUIRED_SOURCES})
    audit = entry.read(entry.inside(entry.SOURCE_AUDIT))
    for name in entry.EXTRA_TRAINING_SOURCES:
        plan["training_source_sha256"][name] = audit["external_source_sha256"][name]
    for name in entry.EXTRA_LIBRARY_SOURCES:
        plan["library_source_sha256"][audit["work_root"] + "/" + name] = audit["external_source_sha256"][name]
    plan_path = temp / (arm_name + "-plan.json")
    observer = temp / (arm_name + "-observer.json")
    project = {"staged_training_request": {"pause_requested": False, "current_protocol_frozen": True,
               "current_protocol": plan_path.relative_to(ROOT).as_posix(), "current_stage": plan["stage"]},
               "goal_execution": {"pause_requested_by_user": False}}

    def sync(value):
        write(plan_path, value)
        protocol_sha, page_sha = entry.sha(plan_path), entry.sha(page)
        write(observer, {"status": "passed-in-run-view-validation", "protocol_sha256": protocol_sha,
                         "page_sha256": page_sha, "visible_from_windows": True})
        write(admission, {"protocol_sha256": protocol_sha,
                          "observer": {"path": observer.relative_to(ROOT).as_posix(), "sha256": entry.sha(observer)}})
        write(output / "windows-entry.json", {"protocol_sha256": protocol_sha, "page_sha256": page_sha})
    sync(plan)
    return plan, plan_path, project, output, page, observer, admission, sync


def fake_runtime(plan, fault=None):
    """Exercise the actual adapter using saved arrays and fake model objects."""
    root = entry.inside(entry.REFERENCE_ROOT) / "roll"
    static, primary = arrays(root / entry.REFERENCE_FILES[0]), arrays(root / entry.REFERENCE_FILES[1])
    flags = {"may_continue": True, "physical_steps": 0, "closed": 0}

    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    class Model:
        ngeom = 31
        def __init__(self, cfg):
            self.nu = len(plan["walking"]["joint_names"])
            self.actuator_trnid = static["actuator_trnid"].copy()
            self.actuator_trntype = np.zeros(self.nu, dtype=np.int32)
            self.actuator_forcelimited = np.ones(self.nu, dtype=np.uint8)
            motor = entry.read(entry.inside(entry.MOTOR_PARAMS))
            voltage = cfg.scene.entities["robot"].articulation.actuators[0].vin_range[0]
            limit = voltage * motor["kt"] / motor["R"]
            self.actuator_forcerange = np.tile(np.array([-limit, limit], np.float64), (self.nu, 1))
            if fault == "bad-force-limit": self.actuator_forcerange[0, 1] += 0.1
            if fault == "nan-force-limit": self.actuator_forcerange[0, 1] = np.nan
            if fault == "force-limit-disabled": self.actuator_forcelimited[1] = 0
            if fault == "wrong-transmission": self.actuator_trntype[0] = 1
            if fault == "duplicate-joint": self.actuator_trnid[0, 0] = self.actuator_trnid[1, 0]
            if fault == "missing-force-field": del self.actuator_forcerange
        def joint(self, name):
            return NS(id=plan["walking"]["joint_names"].index(name.removeprefix("robot/")) + 1)
        def geom(self, name):
            return NS(id={"robot/left_foot_collision": 12, "robot/right_foot_collision": 30}[name])

    def original_compare(*args):
        raise AssertionError("comparison was not wrapped")

    scene = NS(compare_arrays=original_compare)
    class PreviousEnv: pass
    environment = NS(ManagerBasedRlEnv=PreviousEnv)

    def configuration(_):
        actuator = NS(vin_range=(7.7, 7.7))
        return NS(events={"foot_friction": NS(params={"ranges": (1.0, 1.0)})},
                  scene=NS(entities={"robot": NS(articulation=NS(actuators=[actuator]))}))

    def install(_plan, folder, _root, _mode):
        class Audited:
            def __init__(self, cfg):
                self._common_ready, self.common_step_counter, self.num_envs = True, 0, 128
                self.primary = {key: value.copy() for key, value in primary.items()}
                self.primary["geom_friction"][[12, 30], 0] = np.float32(cfg.events["foot_friction"].params["ranges"][0])
                expanded = {key: np.broadcast_to(value, (128,) + value.shape).copy() for key, value in self.primary.items()}
                self.sim = NS(mj_model=Model(cfg), expanded_fields=set(expanded), model=NS())
                for key, value in expanded.items(): setattr(self.sim.model, key, Tensor(value))
                if fault != "skip-static": scene.compare_arrays(static, static)
            def reset(self, *args, **kwargs):
                scene.compare_arrays(self.primary, primary)
                return "synthetic-observation"
            def step(self, *args):
                flags["physical_steps"] += 1
            def close(self):
                flags["closed"] += 1
        environment.ManagerBasedRlEnv = Audited
        return Audited

    base = NS(configuration=configuration, install_guard=install, PLAN=None, OUT=None, PAGE=None, PROJECT=None,
              STARTUP=None, collect=lambda *a: None)

    def run(zero):
        assert zero is True
        folder = base.OUT / "zero"
        folder.mkdir()
        cfg = base.configuration(plan)
        assert cfg.events["foot_friction"].params["ranges"] == (plan["parameter_factors"]["foot_friction"],) * 2
        assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (plan["parameter_factors"]["vin_range_v"],) * 2
        cls = base.install_guard(plan["roll"], folder, base.STARTUP, "rollout")
        env = cls(cfg)
        try:
            if fault == "pause-before-reset": flags["may_continue"] = False
            assert env.reset() == "synthetic-observation"
            if fault == "attempt-step": env.step("synthetic-action")
            if fault == "attempt-collect": base.collect()
            (folder / "startup.npz").write_bytes(b"synthetic startup, not a real npz")
            write(folder / "startup.json", {"status": "passed", "physical_substeps": 0,
                                           "startup_sha256": entry.sha(folder / "startup.npz")})
        finally:
            env.close()
        if fault != "missing-contract":
            write(folder / "runtime-contract.json", {"status": "closed", "control_steps": 0,
                                                     "physics_substeps": 0, "interface_checks": 1})

    base.run = run
    torch = NS(no_grad=nullcontext, cuda=NS(is_initialized=lambda: True))
    return (base, scene, environment, torch, bindings), flags, PreviousEnv, original_compare


def main():
    project, protocol = entry.read(ROOT / "project.json"), entry.read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] is True
    assert project["staged_training_request"]["pause_requested"] is True
    assert protocol["status"] == "cpu-fixture-only-force-limit-guard" and protocol["physical_steps_authorized"] == 0
    for name, digest in protocol["source_sha256"].items():
        assert entry.sha(entry.inside(name)) == digest, name
    before = {p.name for p in (ROOT / "records/evaluations").iterdir()}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cases = []
    for mode in ("zero", "sample"):
        result = subprocess.run([sys.executable, "-X", "importtime", str(Path(entry.__file__)),
                                 "--plan", entry.DRAFT, "--mode", mode], cwd=ROOT, env=env,
                                text=True, capture_output=True, check=False)
        assert result.returncode == 2 and json.loads(result.stdout) == {"status": "refused", "reason": "pause-requested"}
        assert not any(line.rstrip().endswith("| torch") or line.rstrip().endswith("| mujoco") for line in result.stderr.splitlines())
        cases.append("actual-paused-" + mode + "-before-simulator-import")

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpu-entry-fixture-", dir=OUT) as temporary:
        temp = Path(temporary)
        for arm_name in ("nominal", "friction-low", "voltage-high"):
            plan, path, mock, output, page, observer, admission, sync = fixture_plan(temp, arm_name)
            decision = entry.preflight(mock, plan, path, "zero")
            assert decision["allowed"], decision
            assert entry.preflight(mock, plan, path, "sample")["reason"] == "dynamic-sampling-not-implemented"
            runtime, flags, previous, compare_before = fake_runtime(plan)
            result = entry.execute_zero(path, plan, decision, runtime, lambda: flags["may_continue"])
            assert result["foot_geom_ids"] == [12, 30] and flags["physical_steps"] == 0 and flags["closed"] == 1
            assert runtime[2].ManagerBasedRlEnv is previous and runtime[1].compare_arrays is compare_before
            assert result["status"] == "zero-completed-awaiting-independent-verification"
            force = entry.read(output / "zero/actuator-force-limits.json")
            assert force["checked_before_reset"] and force["actuator_count"] == 14
            assert force["arrays_sha256"] == entry.sha(output / "zero/actuator-force-limits.npz")
            cases.append("fake-zero-wiring-" + arm_name)
            cases.append("sample-refused-" + arm_name)

        admission_root = temp / "admission-fixtures"
        admission_root.mkdir()
        plan, path, mock, output, page, observer, admission, sync = fixture_plan(admission_root, "friction-low")

        def mutate_case(label, mutate, expected):
            changed = deepcopy(plan)
            mutate(changed)
            sync(changed)
            result = entry.preflight(mock, changed, path, "zero")
            assert result["reason"] == expected, (label, result)
            cases.append(label)

        mutate_case("reject-dynamic-authorization", lambda p: p.update(dynamics_authorized=True), "zero-only-authorization-required")
        mutate_case("reject-unbounded-zero", lambda p: p.update(zero_physics_substeps_authorized=1), "zero-only-authorization-required")
        mutate_case("reject-boolean-zero", lambda p: p.update(zero_physics_substeps_authorized=False), "zero-only-authorization-required")
        mutate_case("reject-bridge-change", lambda p: p.update(cross_model_bridge_controls=25), "parent-controller-or-source-mismatch")
        mutate_case("reject-sample-budget-change", lambda p: p.update(maximum_controls=2000), "parent-controller-or-source-mismatch")
        mutate_case("reject-roll-settings-change", lambda p: p["roll"].update(episodes=64), "roll-controller-mismatch")
        mutate_case("reject-roll-seed-mismatch", lambda p: p["roll"].update(seed=1), "invalid-seed")
        mutate_case("reject-double-factor", lambda p: p["parameter_factors"].update(vin_range_v=8.0), "one-factor-arm-mismatch")
        mutate_case("reject-unknown-factor", lambda p: p.update(factor_arm="unlisted"), "one-factor-arm-mismatch")
        mutate_case("reject-extra-plan-field", lambda p: p.update(unreviewed=True), "parent-plan-fields-mismatch")
        mutate_case("reject-invalid-run-id", lambda p: p.update(run_id="../outside"), "invalid-run-id")
        mutate_case("reject-parent-source-edit", lambda p: p["frozen_sha256"].update({"scripts/validation/run_continuous_roll_stand_walk.py": "0"*64}), "parent-frozen-hashes-mismatch")
        mutate_case("reject-adapter-source-edit", lambda p: p["frozen_sha256"].update({"scripts/validation/factor_scene_guard_adapter.py": "0"*64}), "source-hash-mismatch")
        mutate_case("reject-parent-training-source-edit", lambda p: p["training_source_sha256"].update({"uv.lock": "0"*64}), "parent-runtime-sources-mismatch")
        mutate_case("reject-missing-bam-source", lambda p: p["training_source_sha256"].pop("vendor/bam/bam/mjlab.py"), "actuator-runtime-sources-not-frozen")
        mutate_case("reject-missing-delay-source", lambda p: p["library_source_sha256"].pop(next(k for k in p["library_source_sha256"] if k.endswith("/buffers/delay_buffer.py"))), "buffer-runtime-sources-not-frozen")
        mutate_case("reject-reference-root", lambda p: p.update(reference_root="records/reports"), "reference-root-mismatch")
        mutate_case("reject-reference-hash", lambda p: p["reference_sha256"].update({entry.REFERENCE_FILES[0]: "0"*64}), "reference-hash-mismatch")
        sync(plan)
        changed = deepcopy(plan)
        changed["seed"] += 1
        assert entry.preflight(mock, changed, path, "zero")["reason"] == "plan-path-or-content-mismatch"
        cases.append("reject-memory-only-plan-change")
        paused = deepcopy(mock)
        paused["goal_execution"]["pause_requested_by_user"] = True
        assert entry.preflight(paused, plan, path, "zero")["reason"] == "pause-requested"
        cases.append("reject-new-pause")
        unfrozen = deepcopy(mock)
        unfrozen["staged_training_request"]["current_protocol_frozen"] = False
        assert entry.preflight(unfrozen, plan, path, "zero")["reason"] == "protocol-not-frozen-or-matched"
        cases.append("reject-unfrozen-plan")
        page.write_text("changed page", encoding="utf-8")
        assert entry.preflight(mock, plan, path, "zero")["reason"] == "observer-not-verified"
        cases.append("reject-changed-observer-page")
        sync(plan)
        write(output / "windows-entry.json", {"protocol_sha256": "0"*64, "page_sha256": entry.sha(page)})
        assert entry.preflight(mock, plan, path, "zero")["reason"] == "observer-entry-mismatch"
        cases.append("reject-windows-entry-mismatch")
        sync(plan)
        (output / "zero").mkdir()
        assert entry.preflight(mock, plan, path, "zero")["reason"] == "output-already-exists"
        (output / "zero").rmdir()
        cases.append("reject-existing-zero-output")

        faults = {"skip-static": "static comparison missing or repeated", "pause-before-reset": "pause-requested-before-reset",
                  "attempt-step": "zero-entry-forbids-physics-step", "attempt-collect": "zero-entry-forbids-sampling",
                  "missing-contract": "runtime-contract.json", "bad-force-limit": "compiled actuator force range differs",
                  "nan-force-limit": "actuator force ranges must be finite", "force-limit-disabled": "actuator force limit must be enabled",
                  "wrong-transmission": "actuator transmission does not match", "duplicate-joint": "actuator transmission does not match",
                  "missing-force-field": "missing compiled actuator field"}
        for fault, expected in faults.items():
            folder = temp / ("fault-" + fault)
            folder.mkdir()
            p, candidate, active, output, page, observer, admission, sync = fixture_plan(folder, "nominal")
            decision = entry.preflight(active, p, candidate, "zero")
            assert decision["allowed"], decision
            runtime, flags, previous, compare_before = fake_runtime(p, fault)
            try:
                entry.execute_zero(candidate, p, decision, runtime, lambda: flags["may_continue"])
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                assert expected in str(exc), (fault, str(exc))
            else:
                raise AssertionError("fault was not rejected: " + fault)
            assert flags["physical_steps"] == 0 and flags["closed"] == 1
            assert runtime[2].ManagerBasedRlEnv is previous and runtime[1].compare_arrays is compare_before
            assert not (output / "zero/factor-zero-wiring.json").exists()
            cases.append("stop-" + fault)

    assert {p.name for p in (ROOT / "records/evaluations").iterdir()} == before
    assert "torch" not in sys.modules and "mujoco" not in sys.modules
    result = {"status": "passed-cpu-fake-force-limit-entry-real-zero-pending", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
              "real_environment_constructed": False, "actual_cuda_zero_verified": False,
              "observer_verified": False, "scope": "Actual entry and adapter with synthetic project, observer, model and CUDA indicator; no real runtime.",
              "source_sha256": {p.relative_to(ROOT).as_posix(): entry.sha(p) for p in
                               (Path(__file__), Path(entry.__file__), PROTOCOL,
                                ROOT / "scripts/validation/model_force_limit_guard.py",
                                ROOT / "scripts/validation/factor_scene_guard_adapter.py")}}
    record = OUT / "entry-fixtures.json"
    if record.exists():
        previous = entry.read(record)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}, "source or result changed; keep existing evidence and create a separate review"
    else:
        with record.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(cases), "physics_steps": 0}, indent=2))


if __name__ == "__main__":
    main()
