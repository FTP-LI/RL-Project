"""Regression gate for the two short parameter-randomized getup policies.

This script freezes one evaluation protocol and then runs the original four
release strata against each final ONNX export. It reuses the official fixed
getup evaluator without changing its physics, observation adapter, renderer,
or metrics. No optimizer, hardware IO, retry, or favorable reset is allowed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from record_io import write

DRAFT = ROOT / "configs" / "experiments" / "parameter_randomized_getup_regression_draft.json"
PLAN = ROOT / "configs" / "experiments" / "parameter_randomized_getup_regression.json"
REPORT = ROOT / "records" / "reports" / "parameter-randomized-regression"
RUNS = ROOT / "records" / "evaluations"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def model_info(arm: dict) -> dict:
    model = ROOT / arm["onnx"]
    return dict(name=arm["name"], model=relative(model), model_sha256=sha(model),
                checkpoint=arm["checkpoint"])


def freeze() -> None:
    assert not PLAN.exists(), "Frozen regression protocol already exists; retain it"
    draft = read(DRAFT)
    assert draft["status"] == "reserved-no-physical-execution"
    arms = [model_info(arm) for arm in draft["arms"]]
    assert {arm["name"] for arm in arms} == {"candidate", "control"}
    source_paths = [
        Path(__file__),
        ROOT / "scripts" / "validation" / "evaluate_standup_reference.py",
        ROOT / "scripts" / "validation" / "standup_metrics.py",
        ROOT / "scripts" / "validation" / "evaluate_fixed_policy.py",
        ROOT / "scripts" / "training" / "record_io.py",
    ]
    protocol = dict(
        status="frozen-before-evaluation",
        frozen_at=datetime.now(timezone.utc).isoformat(),
        stage=draft["stage"],
        parent_training_protocol=draft["parent_training_protocol"],
        evaluation_seed=draft["evaluation_seed"],
        arms=arms,
        getup=copy.deepcopy(draft["getup"]),
        roll=copy.deepcopy(draft["roll"]),
        continuous_seven_command=copy.deepcopy(draft["continuous_seven_command"]),
        stop_policy=draft["stop_policy"],
        learning_updates=0,
        physical_steps_authorized_by_protocol=0,
        hardware_io=False,
        automatic_retry=False,
        automatic_extension=False,
        executed_gates=[],
        source_sha256={relative(path): sha(path) for path in source_paths},
        scope=("Development regression of the two final short-budget exports; "
               "getup is executed first, while roll and seven-command gates remain reserved."),
    )
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write(PLAN, protocol)
    write(REPORT / "freeze.json", dict(status="passed-cpu-freeze", protocol_sha256=sha(PLAN),
                                       physical_steps=0, learning_updates=0,
                                       hardware_io=False, source_sha256=protocol["source_sha256"]))
    (REPORT / "README.md").write_text(
        "# 参数随机化模型回归\n\n"
        "本协议固定 seed、四姿态起身门槛和两个末次 ONNX 模型。执行顺序为候选模型、对照模型；"
        "滚翻和七指令仍保留在同一冻结协议中，只有在起身回归完成后继续。\n\n"
        "- [冻结协议](../../configs/experiments/parameter_randomized_getup_regression.json)\n"
        "- [冻结核验](freeze.json)\n",
        encoding="utf-8",
    )
    print("PARAMETER_RANDOMIZED_REGRESSION_FROZEN", relative(PLAN), sha(PLAN), flush=True)


def run_getup() -> None:
    """Run both arms through the official evaluator in one visible process."""
    import importlib
    import torch

    assert PLAN.exists(), "Freeze the protocol first"
    protocol = read(PLAN)
    assert protocol["status"] == "frozen-before-evaluation"
    for path, digest in protocol["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    assert not (REPORT / "getup-completed.json").exists(), "Getup regression already completed"

    evaluator = importlib.import_module("evaluate_standup_reference")
    results = []
    for arm in protocol["arms"]:
        model = ROOT / arm["model"]
        run_id = (f"duck_getup_parameter-randomized-regression_{arm['name']}"
                  f"_s{protocol['evaluation_seed']}_"
                  f"{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        output = RUNS / run_id
        report = REPORT / arm["name"]
        output.mkdir(parents=True, exist_ok=False)
        report.mkdir(parents=True, exist_ok=False)
        (output / "videos").mkdir(parents=True, exist_ok=False)
        plan = read(ROOT / "configs" / "experiments" / "standup_reference.json")
        plan.update(
            status="frozen-regression-arm",
            frozen_at=protocol["frozen_at"],
            seed=protocol["evaluation_seed"],
            episodes_per_pose=protocol["getup"]["episodes_per_pose"],
            poses=protocol["getup"]["poses"],
            model=arm["model"],
            model_sha256=arm["model_sha256"],
            condition_success_rate_min=protocol["getup"]["success_fraction_min"],
            learning_updates=0,
            automatic_retry=False,
            scope="Regression arm; four requested release strata only; no hardware acceptance",
            regression_protocol_sha256=sha(PLAN),
        )
        arm_plan = report / "protocol.json"
        write(arm_plan, plan)
        state = dict(status="running", arm=arm["name"], run_id=run_id,
                     path=str(output), conditions=[], learning_updates=0,
                     model=arm["model"], model_sha256=arm["model_sha256"],
                     evaluation_seed=protocol["evaluation_seed"],
                     regression_protocol_sha256=sha(PLAN), started_at=datetime.now(timezone.utc).isoformat())
        write(report / "evaluation.json", state)
        write(output / "protocol.json", plan)
        write(output / "regression-arm.json", arm)
        try:
            # The official coordinator requires a foreground TTY and its own
            # frozen model. Calling its evaluator directly keeps the same
            # physics/metrics while allowing each final ONNX export.
            evaluator.MODEL = model
            evaluator.PLAN = arm_plan
            evaluator.OUT = report
            evaluator.evaluate(plan, output, state)
            state["status"] = "completed"
            state["ended_at"] = datetime.now(timezone.utc).isoformat()
            write(output / "evaluation.json", state)
            write(report / "evaluation.json", state)
            rows = []
            for pose in plan["poses"]:
                rows.append(read(output / f"{pose}.json"))
            summary = dict(arm=arm["name"], run_id=run_id, model=arm["model"],
                           model_sha256=arm["model_sha256"], poses=rows,
                           all_conditions_passed=bool(state.get("all_conditions_passed", False)),
                           protocol_sha256=sha(PLAN), completed_at=state["ended_at"])
            write(report / "summary.json", summary)
            results.append(summary)
        except BaseException as error:
            state.update(status="failed", error=repr(error), ended_at=datetime.now(timezone.utc).isoformat())
            write(output / "evaluation.json", state)
            write(report / "evaluation.json", state)
            raise
        finally:
            # Keep each arm isolated in the same process and release cached GPU memory.
            evaluator.MODEL = ROOT / "official_resources/xgoduck_runtime_arduino/python/xgoduck_getup.onnx"
            evaluator.PLAN = ROOT / "configs/experiments/standup_reference.json"
            evaluator.OUT = ROOT / "records/reports/standup-reference"
            torch.cuda.empty_cache()
    combined = dict(status="completed", protocol_sha256=sha(PLAN), arms=results,
                    all_arms_passed=all(x["all_conditions_passed"] for x in results),
                    completed_at=datetime.now(timezone.utc).isoformat())
    write(REPORT / "getup-completed.json", combined)
    protocol["executed_gates"] = ["getup"]
    protocol["getup_result"] = relative(REPORT / "getup-completed.json")
    protocol["status"] = "getup-completed-roll-and-seven-command-reserved"
    write(PLAN, protocol)
    print("PARAMETER_RANDOMIZED_GETUP_REGRESSION", json.dumps(combined, ensure_ascii=False), flush=True)


def run_roll() -> None:
    """Run the official forward-roll gate for both final exports."""
    import importlib
    import torch

    assert PLAN.exists() and (REPORT / "getup-completed.json").exists()
    protocol = read(PLAN)
    assert protocol["status"] == "getup-completed-roll-and-seven-command-reserved"
    for path, digest in protocol["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    assert not (REPORT / "roll-completed.json").exists(), "Roll regression already completed"

    evaluator = importlib.import_module("evaluate_roll_reference")
    results = []
    for arm in protocol["arms"]:
        model = ROOT / arm["model"]
        run_id = (f"duck_roll_parameter-randomized-regression_{arm['name']}"
                  f"_s{protocol['evaluation_seed']}_"
                  f"{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        output = RUNS / run_id
        report = REPORT / "roll" / arm["name"]
        output.mkdir(parents=True, exist_ok=False)
        (output / "videos").mkdir(parents=True, exist_ok=False)
        report.mkdir(parents=True, exist_ok=False)
        plan = read(ROOT / "configs" / "experiments" / "roll_reference.json")
        plan.update(
            status="frozen-regression-arm",
            frozen_at=protocol["frozen_at"],
            seed=protocol["evaluation_seed"],
            episodes=protocol["roll"]["episodes"],
            model=arm["model"],
            model_sha256=arm["model_sha256"],
            success_fraction_min=protocol["roll"]["success_fraction_min"],
            learning_updates=0,
            automatic_retry=False,
            scope="Regression arm; official forward roll only; no hardware acceptance",
            regression_protocol_sha256=sha(PLAN),
        )
        arm_plan = report / "protocol.json"
        write(arm_plan, plan)
        state = dict(status="running", arm=arm["name"], run_id=run_id,
                     path=str(output), conditions=[], learning_updates=0,
                     model=arm["model"], model_sha256=arm["model_sha256"],
                     evaluation_seed=protocol["evaluation_seed"],
                     regression_protocol_sha256=sha(PLAN), started_at=datetime.now(timezone.utc).isoformat())
        write(report / "evaluation.json", state)
        write(output / "protocol.json", plan)
        write(output / "regression-arm.json", arm)
        try:
            evaluator.MODEL = model
            evaluator.PLAN = arm_plan
            evaluator.OUT = report
            evaluator.evaluate(plan, output, state)
            state["status"] = "completed"
            state["ended_at"] = datetime.now(timezone.utc).isoformat()
            write(output / "evaluation.json", state)
            write(report / "evaluation.json", state)
            result = dict(arm=arm["name"], run_id=run_id, model=arm["model"],
                          model_sha256=arm["model_sha256"],
                          condition=state["conditions"][0],
                          all_conditions_passed=bool(state.get("all_conditions_passed", False)),
                          protocol_sha256=sha(PLAN), completed_at=state["ended_at"])
            write(report / "summary.json", result)
            results.append(result)
        except BaseException as error:
            state.update(status="failed", error=repr(error), ended_at=datetime.now(timezone.utc).isoformat())
            write(output / "evaluation.json", state)
            write(report / "evaluation.json", state)
            raise
        finally:
            evaluator.MODEL = ROOT / "official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx"
            evaluator.PLAN = ROOT / "configs/experiments/roll_reference.json"
            evaluator.OUT = ROOT / "records/reports/roll-reference"
            torch.cuda.empty_cache()
    combined = dict(status="completed", protocol_sha256=sha(PLAN), arms=results,
                    all_arms_passed=all(x["all_conditions_passed"] for x in results),
                    completed_at=datetime.now(timezone.utc).isoformat())
    write(REPORT / "roll-completed.json", combined)
    protocol["executed_gates"] = [*protocol.get("executed_gates", []), "roll"]
    protocol["roll_result"] = relative(REPORT / "roll-completed.json")
    protocol["status"] = "getup-roll-completed-seven-command-reserved"
    write(PLAN, protocol)
    print("PARAMETER_RANDOMIZED_ROLL_REGRESSION", json.dumps(combined, ensure_ascii=False), flush=True)


def run_walk() -> None:
    """Run the seven-command functional screen using both getup exports."""
    import importlib
    import numpy as np

    assert PLAN.exists() and (REPORT / "roll-completed.json").exists()
    protocol = read(PLAN)
    assert protocol["status"] == "getup-roll-completed-seven-command-reserved"
    for path, digest in protocol["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    assert not (REPORT / "seven-command-completed.json").exists(), "Seven-command regression already completed"

    base = read(ROOT / "configs" / "experiments" / "seven_command" / "s1007.json")
    collector = importlib.import_module("grouped_motion_collection")
    results = []
    for arm in protocol["arms"]:
        model = ROOT / arm["model"]
        run_id = (f"duck_seven-command_parameter-randomized-regression_{arm['name']}"
                  f"_s{protocol['evaluation_seed']}_"
                  f"{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        output = RUNS / run_id
        report = REPORT / "seven-command" / arm["name"]
        output.mkdir(parents=True, exist_ok=False)
        (output / "videos").mkdir(parents=True, exist_ok=False)
        report.mkdir(parents=True, exist_ok=False)
        plan = copy.deepcopy(base)
        plan.update(
            status="frozen-regression-arm",
            frozen_at=protocol["frozen_at"],
            seed=protocol["evaluation_seed"],
            model=arm["model"],
            model_sha256=arm["model_sha256"],
            controller_module="static_command_controller",
            heldout_used=False,
            learning_updates=0,
            automatic_retry=False,
            automatic_extension=False,
            scope="Regression arm; seven command functional screen only; no hardware acceptance",
            regression_protocol_sha256=sha(PLAN),
        )
        arm_plan = report / "protocol.json"
        write(arm_plan, plan)
        write(output / "protocol.json", plan)
        write(output / "regression-arm.json", arm)
        n = int(plan["parallel_worlds"])
        h = int(plan["maximum_control_steps"])
        groups = {name: np.asarray(ids, dtype=int) for name, ids in plan["groups"].items()}
        desired = np.zeros((n, 3), dtype=np.float32)
        for name, ids in groups.items():
            desired[ids] = np.asarray(plan["commands"][name], dtype=np.float32)
        state = dict(status="running", arm=arm["name"], run_id=run_id,
                     path=str(output), seed=plan["seed"], num_envs=n,
                     parallel_worlds=n, episode_seconds=plan["seconds"], conditions=[],
                     model=arm["model"], model_sha256=arm["model_sha256"],
                     evaluation_seed=protocol["evaluation_seed"], learning_updates=0,
                     hardware_io=False, protocol_sha256=sha(PLAN),
                     started_at=datetime.now(timezone.utc).isoformat())
        def save():
            write(output / "evaluation.json", state)
            write(report / "evaluation.json", state)
        save()
        try:
            collector.collect(plan, {}, output, state, save, desired, groups, n, h)
            state["status"] = state.get("status", "completed")
            state["ended_at"] = datetime.now(timezone.utc).isoformat()
            save()
            assessments = []
            for condition in state.get("conditions", []):
                name = condition["condition"]
                criteria = plan["criteria"]
                checks = {
                    "survival": condition["full_episode_survival"] >= criteria["survival_min"],
                    "linear_error": condition["linear_xy_mae_m_s"] <= criteria["linear_error"][name],
                    "yaw_error": condition["yaw_mae_rad_s"] <= criteria["yaw_error_max"],
                    "stand_drift": (name != "stand" or
                                    condition["final_displacement_m_median"] <= criteria["stand_drift_median_max"]),
                }
                assessments.append(dict(
                    condition=name,
                    checks=checks,
                    passed=all(checks.values()),
                    blocked_by=[key for key, value in checks.items() if not value],
                    scope="Stored-metric screen; signed-speed/block/inward checks remain in the independent walking verifier",
                ))
            summary = dict(arm=arm["name"], run_id=run_id, model=arm["model"],
                           model_sha256=arm["model_sha256"], conditions=state.get("conditions", []),
                           gate_assessment=assessments,
                           functional_development_passed=all(row["passed"] for row in assessments),
                           formal_acceptance_eligible=False,
                           protocol_sha256=sha(PLAN), completed_at=state["ended_at"])
            write(report / "summary.json", summary)
            results.append(summary)
        except BaseException as error:
            state.update(status="failed", error=repr(error), ended_at=datetime.now(timezone.utc).isoformat())
            save()
            raise
    combined = dict(status="completed", protocol_sha256=sha(PLAN), arms=results,
                    all_arms_passed=all(x["functional_development_passed"] for x in results),
                    completed_at=datetime.now(timezone.utc).isoformat())
    write(REPORT / "seven-command-completed.json", combined)
    protocol["executed_gates"] = [*protocol.get("executed_gates", []), "continuous_seven_command"]
    protocol["continuous_seven_command_result"] = relative(REPORT / "seven-command-completed.json")
    protocol["status"] = "all-regression-gates-completed-review-required"
    write(PLAN, protocol)
    print("PARAMETER_RANDOMIZED_SEVEN_COMMAND_REGRESSION", json.dumps(combined, ensure_ascii=False), flush=True)


def refresh_freeze() -> None:
    """Record a pre-gate executor fix without changing any gate."""
    assert PLAN.exists()
    protocol = read(PLAN)
    assert protocol["status"] in (
        "frozen-before-evaluation",
        "getup-completed-roll-and-seven-command-reserved",
        "getup-roll-completed-seven-command-reserved",
        "all-regression-gates-completed-review-required",
    )
    protocol["source_sha256"][relative(Path(__file__))] = sha(Path(__file__))
    protocol["freeze_revision_at"] = datetime.now(timezone.utc).isoformat()
    protocol["freeze_revision_reason"] = "Update the frozen executor hash after adding the next regression gate"
    write(PLAN, protocol)
    write(REPORT / "freeze-revision.json", dict(status="passed-pre-evaluation-revision",
                                                protocol_sha256=sha(PLAN),
                                                reason=protocol["freeze_revision_reason"],
                                                source_sha256=protocol["source_sha256"]))
    print("PARAMETER_RANDOMIZED_REGRESSION_FREEZE_REFRESHED", relative(PLAN), sha(PLAN), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "refresh", "getup", "roll", "walk"))
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "refresh":
        refresh_freeze()
    elif args.mode == "roll":
        run_roll()
    elif args.mode == "walk":
        run_walk()
    else:
        run_getup()


if __name__ == "__main__":
    main()
