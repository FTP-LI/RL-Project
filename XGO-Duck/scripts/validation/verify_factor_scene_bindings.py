"""Exercise the factor-scene adapter using saved fields and fake environments."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
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
    draft = read(ROOT / "configs/experiments/roll_parameter_matched_review_draft.json")
    nominal = {"foot_friction": 1.0, "vin_range_v": 7.7}
    ranges = {"foot_friction": (0.7, 1.3), "vin_range_v": (7.4, 8.0)}
    original = ROOT / "records/reports/roll-domain-parameter-random-s20/low-low/zero"
    names = ("scene-static-fields.npz", "nominal-primary-fields.npz")
    references = {name: sha(original / name) for name in names}
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []

    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    class Model:
        ngeom = 31
        def __init__(self, mapping=None, flip_after_first=False):
            self.mapping = mapping or {"robot/left_foot_collision": 12, "robot/right_foot_collision": 30}
            self.flip_after_first, self.calls = flip_after_first, 0
        def geom(self, name):
            self.calls += 1
            if self.flip_after_first and self.calls > 2:
                name = "robot/right_foot_collision" if name == "robot/left_foot_collision" else "robot/left_foot_collision"
            return NS(id=self.mapping[name])

    with tempfile.TemporaryDirectory(prefix="factor-bindings-", dir=OUT) as temp:
        root = Path(temp) / "startup"
        (root / "roll").mkdir(parents=True)
        for name in names:
            shutil.copyfile(original / name, root / "roll" / name)
            assert sha(root / "roll" / name) == references[name]
        ref_static = arrays(root / "roll/scene-static-fields.npz")
        ref_primary = arrays(root / "roll/nominal-primary-fields.npz")

        def fixture(fault=None):
            counts = {"closed": 0}
            def unchanged_compare(actual, expected):
                raise AssertionError("unwrapped comparison should not be called")
            scene = NS(compare_arrays=unchanged_compare)
            class PreviousEnv: pass
            env_module = NS(ManagerBasedRlEnv=PreviousEnv)

            def configuration(_):
                actuator = NS(vin_range=(7.7, 7.7))
                robot = NS(articulation=NS(actuators=[actuator]))
                return NS(events={"foot_friction": NS(params={"ranges": (1.0, 1.0)})},
                          scene=NS(entities={"robot": robot}))

            def install(plan, folder, startup_root, mode):
                class Audited:
                    def __init__(self, cfg):
                        self._common_ready = True
                        self.common_step_counter, self.num_envs = 0, 128
                        self.sim = NS(mj_model=Model(flip_after_first=fault == "mapping-switch"), expanded_fields=set(ref_primary), model=NS())
                        primary = {key: value.copy() for key, value in ref_primary.items()}
                        primary["geom_friction"][[12, 30], 0] = np.float32(cfg.events["foot_friction"].params["ranges"][0])
                        self.primary = primary
                        expanded = {key: np.broadcast_to(value, (128,) + value.shape).copy() for key, value in primary.items()}
                        if fault == "non-target-world": expanded["geom_friction"][7, 0, 0] += 1
                        if fault == "foot-other-component": expanded["geom_friction"][7, 12, 1] += 1
                        if fault == "extra-expanded-field": expanded["unexpected"] = np.zeros((128, 1), np.float32)
                        self.sim.expanded_fields = set(expanded)
                        for key, value in expanded.items(): setattr(self.sim.model, key, Tensor(value))
                        static = {key: value.copy() for key, value in ref_static.items()}
                        if fault == "static-changed": static["geom_friction"][12, 0] += 1
                        if fault != "skip-static": scene.compare_arrays(static, ref_static)
                        if fault == "repeat-static": scene.compare_arrays(static, ref_static)
                    def reset(self, *args, **kwargs):
                        if fault != "skip-primary": scene.compare_arrays(self.primary, ref_primary)
                        if fault == "repeat-primary": scene.compare_arrays(self.primary, ref_primary)
                        return "fake-observation"
                    def close(self): counts["closed"] += 1
                env_module.ManagerBasedRlEnv = Audited
                return Audited

            base = NS(configuration=configuration, install_guard=install)
            return base, scene, env_module, PreviousEnv, unchanged_compare, counts

        def run_valid(arm):
            base, scene, env, previous, compare_before, counts = fixture()
            cfg_before, install_before = base.configuration, base.install_guard
            with factor_guard_bindings(base, scene, env, arm=arm, nominal=nominal, ranges=ranges,
                                       startup_root=root, reference_sha256=references) as state:
                cfg = base.configuration({})
                assert cfg.events["foot_friction"].params["ranges"] == (arm["foot_friction"],) * 2
                assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (arm["vin_range_v"],) * 2
                cls = base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")
                instance = cls(cfg)
                assert instance.reset() == "fake-observation"
                assert require_complete(state)["foot_geom_ids"] == [12, 30]
                instance.close()
            assert env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before
            assert base.configuration is cfg_before and base.install_guard is install_before and counts["closed"] == 1

        for arm in draft["arms"]:
            run_valid(arm)
            cases.append("valid-" + arm["name"])

        faults = {
            "skip-static": (RuntimeError, "static comparison missing or repeated"),
            "repeat-static": (RuntimeError, "static comparison missing or repeated"),
            "static-changed": (ValueError, "static physical field changed: geom_friction"),
            "skip-primary": (RuntimeError, "primary comparison missing or repeated"),
            "repeat-primary": (RuntimeError, "primary comparison missing or repeated"),
            "non-target-world": (ValueError, "unexpected or incorrect geom_friction change"),
            "foot-other-component": (ValueError, "unexpected or incorrect geom_friction change"),
            "extra-expanded-field": (ValueError, "physical-field set mismatch"),
            "mapping-switch": (RuntimeError, "foot geometry mapping changed during reset"),
        }
        for fault, (error_type, message) in faults.items():
            base, scene, env, previous, compare_before, counts = fixture(fault)
            cfg_before, install_before = base.configuration, base.install_guard
            try:
                with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                           startup_root=root, reference_sha256=references):
                    cfg = base.configuration({})
                    cls = base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")
                    instance = cls(cfg)
                    try:
                        instance.reset()
                    finally:
                        instance.close()
            except error_type as exc:
                assert message in str(exc), (fault, str(exc))
            else:
                raise AssertionError(f"Fault not stopped: {fault}")
            assert env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before
            assert base.configuration is cfg_before and base.install_guard is install_before
            assert counts["closed"] == 1, fault
            cases.append("stop-" + fault)

        base, scene, env, previous, compare_before, _ = fixture()
        with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                   startup_root=root, reference_sha256=references) as state:
            try:
                require_complete(state)
            except RuntimeError as exc:
                assert "selective guard checks incomplete or repeated" in str(exc)
            else:
                raise AssertionError("incomplete checks were accepted")
            cases.append("incomplete-checks-refused")
            try:
                with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                           startup_root=root, reference_sha256=references):
                    pass
            except RuntimeError as exc:
                assert "factor guard requires an exclusive process" in str(exc)
            else:
                raise AssertionError("concurrent binding accepted")
            cases.append("nested-binding-refused")
            cfg = base.configuration({})
            cls = base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")
            instance = cls(cfg)
            try:
                cls(cfg)
            except RuntimeError as exc:
                assert "guard environment already attempted" in str(exc)
                cases.append("second-environment-refused")
            else:
                raise AssertionError("second environment accepted")
            instance.reset()
            for label, args, kwargs in (("repeat-reset", (), {}), ("positional-reset", ([],), {}),
                                        ("partial-reset", (), {"env_ids": [0]})):
                try:
                    instance.reset(*args, **kwargs)
                except RuntimeError as exc:
                    assert "unexpected partial, repeated or post-step reset" in str(exc)
                    cases.append(label + "-refused")
                else:
                    raise AssertionError(label)
            instance.close()
        assert env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before

        base, scene, env, previous, compare_before, counts = fixture("skip-static")
        with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                   startup_root=root, reference_sha256=references):
            cfg = base.configuration({})
            cls = base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")
            try:
                cls(cfg)
            except RuntimeError as exc:
                assert "static comparison missing or repeated" in str(exc)
            else:
                raise AssertionError("missing static check accepted")
            try:
                cls(cfg)
            except RuntimeError as exc:
                assert "guard environment already attempted" in str(exc)
                cases.append("retry-after-failed-init-refused")
            else:
                raise AssertionError("failed initialization retried")
        assert counts["closed"] == 1 and env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before

        base, scene, env, previous, compare_before, _ = fixture()
        path = root / "roll/nominal-primary-fields.npz"
        original_bytes = path.read_bytes()
        try:
            with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                       startup_root=root, reference_sha256=references):
                path.write_bytes(original_bytes + b"changed")
                try:
                    base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")
                except ValueError as exc:
                    assert "frozen reference mismatch: nominal-primary-fields.npz" in str(exc)
                    cases.append("reference-change-refused")
                else:
                    raise AssertionError("changed reference accepted")
        finally:
            path.write_bytes(original_bytes)
        assert env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before

        base, scene, env, previous, compare_before, _ = fixture()
        with factor_guard_bindings(base, scene, env, arm=draft["arms"][1], nominal=nominal, ranges=ranges,
                                   startup_root=root, reference_sha256=references) as state:
            instance = base.install_guard({"skill": "roll"}, Path(temp), root, "rollout")(base.configuration({}))
            try:
                instance.reset()
                require_complete(state)
                path.write_bytes(original_bytes + b"changed")
                try:
                    require_complete(state)
                except RuntimeError as exc:
                    assert "frozen reference changed after check: nominal-primary-fields.npz" in str(exc)
                    cases.append("reference-change-after-reset-refused")
                else:
                    raise AssertionError("changed final reference accepted")
            finally:
                path.write_bytes(original_bytes)
                instance.close()
        assert env.ManagerBasedRlEnv is previous and scene.compare_arrays is compare_before
    assert "torch" not in sys.modules
    paths = [Path(__file__), ROOT / "scripts/validation/factor_scene_guard_adapter.py",
             ROOT / "scripts/validation/roll_factor_configuration.py", ROOT / "scripts/validation/selective_friction_guard.py", PROTOCOL,
             OUT / "fake-bindings-verification-exception-matched.json", OUT / "sources-before-reset-enforcement/factor_scene_guard_adapter.py.txt",
             OUT / "sources-before-reset-enforcement/verify_factor_scene_bindings.py.txt"]
    result = {"status": "passed-fake-scene-bindings", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "reference_source_files_checked": len(protocol["source_sha256"]),
              "fake_model_only": True, "real_environment_constructed": False, "physics_steps": 0,
              "learning_updates": 0, "hardware_io": False, "torch_imported": False,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              "scope": "Fake installer, fake model and saved fields only; real guard flow and CUDA remain separate checks."}
    target = OUT / "fake-bindings-verification-reset-enforced.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(cases), "source_files_checked": len(protocol["source_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
