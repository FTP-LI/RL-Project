"""Read frozen sources to check the adapter's reset-order assumptions."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/roll-factor-guard-adapter"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def main():
    project = read(ROOT / "project.json")
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    draft = read(ROOT / "configs/experiments/roll_parameter_matched_review_draft.json")
    parent_path = ROOT / draft["base"]["protocol"]
    assert sha(parent_path) == draft["base"]["protocol_sha256"]
    parent = read(parent_path)
    roots = {"frozen_sha256": ROOT, "training_source_sha256": Path.home() / "XGO-Duck/workspace/xgoduck_rl",
             "library_source_sha256": Path("/")}
    counts = {}
    for group, base in roots.items():
        for name, expected in parent[group].items():
            assert sha(base / name) == expected, (group, name)
        counts[group] = len(parent[group])
    env_path = Path.home() / "XGO-Duck/workspace/xgoduck_rl/.venv/lib/python3.12/site-packages/mjlab/envs/manager_based_rl_env.py"
    assert str(env_path) in parent["library_source_sha256"]
    env_tree = ast.parse(env_path.read_text(encoding="utf-8"))
    env_class = next(node for node in env_tree.body if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRlEnv")
    initializer, reset = function(env_class, "__init__"), function(env_class, "reset")
    init_self_reset = [node.lineno for node in ast.walk(initializer) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == "self" and node.func.attr == "reset"]
    reset_keyword_only = [argument.arg for argument in reset.args.kwonlyargs]
    assert not init_self_reset and reset_keyword_only == ["seed", "env_ids", "options"]

    baseline_path = ROOT / "scripts/validation/run_continuous_roll_stand_walk.py"
    run = function(ast.parse(baseline_path.read_text(encoding="utf-8")), "run")
    resets = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "env" and node.func.attr == "reset"]
    collects = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "collect"]
    assert len(resets) == len(collects) == 1 and resets[0] < collects[0]

    guard_path = ROOT / "scripts/validation/common_scene_contract.py"
    install = function(ast.parse(guard_path.read_text(encoding="utf-8")), "install_guard")
    audited = next(node for node in install.body if isinstance(node, ast.ClassDef) and node.name == "AuditedEnv")
    guard_init, guard_reset = function(audited, "__init__"), function(audited, "reset")
    comparisons = {}
    for name, method in (("initialization", guard_init), ("reset", guard_reset)):
        comparisons[name] = [node.lineno for node in ast.walk(method) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Name) and node.func.id == "compare_arrays"]
        assert not any(isinstance(node, ast.Name) and node.id == "compare_arrays" and isinstance(node.ctx, ast.Store)
                       for node in ast.walk(method))
    assert len(comparisons["initialization"]) == 2 and len(comparisons["reset"]) == 1
    fixture_path = OUT / "original-guard-flow-verification-reset-enforced.json"
    fixture = read(fixture_path)
    assert fixture["status"] == "passed-original-guard-flow-with-fake-environment" and fixture["interface_check_stubbed"]
    for name, expected in fixture["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    result = {"status": "passed-source-order-audit", "checked_at": datetime.now(timezone.utc).isoformat(),
              "s17_source_counts": counts, "installed_environment_source_sha256": sha(env_path),
              "environment_init_self_reset_calls": init_self_reset, "reset_keyword_only_parameters": reset_keyword_only,
              "baseline_env_reset_line": resets[0], "baseline_collect_line": collects[0],
              "guard_compare_call_lines": comparisons, "fixture_interface_check_stubbed": True,
              "real_environment_constructed": False, "physics_steps": 0, "learning_updates": 0,
              "hardware_io": False, "torch_imported": "torch" in sys.modules,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__), parent_path,
                                 baseline_path, guard_path, fixture_path)},
              "scope": "Static source and hash audit; not a real environment or runtime verification."}
    assert not result["torch_imported"]
    target = OUT / "source-call-order-verification.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "s17_source_counts": counts, "reset_line": resets[0],
                      "collect_line": collects[0], "reset_keyword_only_parameters": reset_keyword_only}, indent=2))


if __name__ == "__main__":
    main()
