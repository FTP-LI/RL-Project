"""Freeze a nominal zero-step plan after source checks, without launching simulation.

The earlier parameter screen mixed control modes and lacked a matched nominal
arm. This prepares the unchanged S17 controller with a scoped fresh seed and
records missing observer admission; it never fabricates observer evidence.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import guarded_factor_zero_entry as entry

OUT = ROOT / "records/reports/resume-20261115"
SEED_AUDIT = OUT / "seed-audit.json"
SOURCE_AUDIT = OUT / "source-audit.json"
PLAN = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
WSL_LOGS = Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl")
SEED = 1241


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def check_file(path: Path, expected: str, group: str, rows: list[dict]) -> None:
    actual = entry.sha(path) if path.is_file() else None
    rows.append({"group": group, "path": str(path), "expected_sha256": expected,
                 "actual_sha256": actual, "matches": actual == expected})


def main() -> None:
    if SOURCE_AUDIT.exists() or PLAN.exists():
        raise FileExistsError("Do not overwrite a prior audit or frozen plan")
    state = entry.read(ROOT / "project.json")
    task, goal = state["staged_training_request"], state["goal_execution"]
    if (task["pause_requested"] or goal["pause_requested_by_user"] or
            task["learning_running"] or task["simulation_running"] or
            task["current_stage"] != "S20-matched-comparison-preparation" or task["current_protocol_frozen"]):
        raise RuntimeError("Resume preparation is not in the expected idle state")
    seed_check = entry.read(SEED_AUDIT)
    if seed_check["status"] != "no-matches-in-scoped-windows-records" or seed_check["seed"] != SEED:
        raise ValueError("Windows seed audit is not clear")

    search = ["rg", "-l", "--hidden"]
    for glob in ("*.json", "*.yaml", "*.log", "*.txt"):
        search += ["-g", glob]
    search += [r'(_s1241_|seed[ =:_-]+1241\b|"seed"\s*:\s*1241\b)', str(WSL_LOGS)]
    wsl_search = subprocess.run(search, text=True, capture_output=True)
    if wsl_search.returncode not in (0, 1) or not WSL_LOGS.is_dir():
        raise RuntimeError("WSL seed search failed: " + wsl_search.stderr)
    files = subprocess.run(["rg", "--files", "--hidden", str(WSL_LOGS)], text=True, capture_output=True)
    if files.returncode != 0:
        raise RuntimeError("WSL filename enumeration failed: " + files.stderr)
    seed_names = [name for name in files.stdout.splitlines() if re.search(r"_s1241(?:_|/|\.|$)", name)]

    parent = entry.read(entry.inside(entry.PARENT))
    draft = entry.read(entry.inside(entry.DRAFT))
    audit = entry.read(entry.inside(entry.SOURCE_AUDIT))
    work = Path(audit["work_root"])
    checks: list[dict] = []
    for name, expected in parent["frozen_sha256"].items():
        check_file(ROOT / name, expected, "s17-workspace", checks)
    for name, expected in parent["training_source_sha256"].items():
        check_file(work / name, expected, "s17-training-source", checks)
    for name, expected in parent["library_source_sha256"].items():
        check_file(Path(name), expected, "s17-library-source", checks)
    for name in sorted(entry.EXTRA_TRAINING_SOURCES | entry.EXTRA_LIBRARY_SOURCES):
        check_file(work / name, audit["external_source_sha256"][name], "additional-runtime-source", checks)
    for name, expected in ((entry.PARENT, entry.PARENT_SHA), (entry.DRAFT, entry.DRAFT_SHA),
                           (entry.RANGES, entry.RANGES_SHA), (entry.SOURCE_AUDIT, entry.SOURCE_AUDIT_SHA)):
        check_file(ROOT / name, expected, "fixed-reference", checks)
    evidence_errors = []
    for name, expected in entry.EVIDENCE.items():
        try:
            entry._check_evidence(name, expected)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            evidence_errors.append({"path": name, "error": type(exc).__name__, "message": str(exc)})
    problems = [row for row in checks if not row["matches"]]
    text_matches = wsl_search.stdout.splitlines()
    passed = not (problems or evidence_errors or text_matches or seed_names)
    result = {"status": "passed-scoped-source-and-seed-audit" if passed else "failed-review-required",
              "checked_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
              "windows_seed_audit": SEED_AUDIT.relative_to(ROOT).as_posix(),
              "windows_seed_audit_sha256": entry.sha(SEED_AUDIT),
              "wsl_text_search_command": search, "wsl_text_search_exit_code": wsl_search.returncode,
              "wsl_text_matches": text_matches, "wsl_filename_matches": seed_names,
              "source_checks": checks, "evidence_errors": evidence_errors,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "scope": "File hashes and scoped seed search only; real observer and CUDA zero remain unverified."}
    write_new(SOURCE_AUDIT, result)
    if not passed:
        print(json.dumps({"status": result["status"], "source_mismatches": problems,
                          "evidence_errors": evidence_errors, "wsl_matches": text_matches + seed_names}, ensure_ascii=False))
        raise SystemExit(2)

    plan = deepcopy(parent)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_id = f"duck_roll-factor_nominal_s{SEED}_{stamp}"
    output = ROOT / "records/reports" / run_id
    page = ROOT / "records" / (run_id + "-live.html")
    admission = output / "admission.json"
    if output.exists() or page.exists():
        raise FileExistsError("Run output or page already exists")
    arm = next(row for row in draft["arms"] if row["name"] == "nominal")
    plan.update(stage="S20-roll-factor-zero-nominal", seed=SEED, run_id=run_id,
                zero_authorized=True, zero_physics_substeps_authorized=0, dynamics_authorized=False,
                factor_arm="nominal", parameter_factors={k: arm[k] for k in ("foot_friction", "vin_range_v")},
                guarded_output=output.relative_to(ROOT).as_posix(), page=page.relative_to(ROOT).as_posix(),
                admission_record=admission.relative_to(ROOT).as_posix(), reference_root=entry.REFERENCE_ROOT,
                reference_sha256={name: parent["frozen_sha256"][f"{entry.REFERENCE_ROOT}/roll/{name}"]
                                  for name in entry.REFERENCE_FILES})
    plan["roll"]["seed"] = SEED
    plan["frozen_sha256"].update({name: entry.sha(entry.inside(name)) for name in entry.REQUIRED_SOURCES})
    for path in (SEED_AUDIT, SOURCE_AUDIT, Path(__file__), ROOT / "scripts/validation/audit_resumed_seed.py"):
        plan["frozen_sha256"][path.relative_to(ROOT).as_posix()] = entry.sha(path)
    for name in entry.EXTRA_TRAINING_SOURCES:
        plan["training_source_sha256"][name] = audit["external_source_sha256"][name]
    for name in entry.EXTRA_LIBRARY_SOURCES:
        plan["library_source_sha256"][audit["work_root"] + "/" + name] = audit["external_source_sha256"][name]
    if set(plan) != set(parent) | entry.EXTRA:
        raise RuntimeError("Plan no longer matches the zero-entry schema")
    write_new(PLAN, plan)
    output.mkdir(exist_ok=False)
    title = html.escape(run_id)
    page.write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>XGO-Duck 观察准备</title>'
                    f'<body><h1>XGO-Duck 观察准备</h1><p>状态：准备中；尚未执行仿真或训练。</p>'
                    f'<p>运行：<code>{title}</code></p><p>真实观察入口尚未验证。</p>'
                    f'<p><a href="reports/{title}/README.md">本轮记录</a></p></body></html>\n', encoding="utf-8")
    (output / "README.md").write_text(
        "# 名义参数零步准备\n\n已冻结零物理步方案；真实观察入口尚未通过，未运行仿真或训练。\n\n"
        "候选种子及源码检查见 [恢复准备](../resume-20261115/README.md)。动态采样不在此方案授权范围内。\n",
        encoding="utf-8")
    plan_hash, page_hash = entry.sha(PLAN), entry.sha(page)
    write_new(output / "windows-entry.json", {"status": "prepared-observer-entry", "protocol_sha256": plan_hash,
              "page": page.relative_to(ROOT).as_posix(), "page_sha256": page_hash,
              "browser_visual_verified": False, "real_observer_verified": False})
    task.update(current_stage=plan["stage"], current_protocol=PLAN.relative_to(ROOT).as_posix(),
                current_protocol_frozen=True, current_protocol_execution="zero-only-awaiting-observer",
                learning_running=False, simulation_running=False)
    goal.update(current_phase=plan["stage"], current_protocol=PLAN.relative_to(ROOT).as_posix())
    state["current_phase"] = plan["stage"]
    (ROOT / "project.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision = entry.preflight(state, plan, PLAN, "zero")
    write_new(output / "entry-readiness.json", {"status": "awaiting-real-observer" if not decision["allowed"] else "unexpected-admission",
              "decision": decision, "protocol_sha256": plan_hash, "page_sha256": page_hash,
              "real_observer_verified": False, "cuda_zero_verified": False,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False})
    if decision["allowed"] or decision.get("reason") != "incomplete-or-invalid-admission":
        raise RuntimeError("Unexpected zero-entry preflight decision: " + str(decision))
    print(json.dumps({"status": "zero-plan-frozen-observer-pending", "plan": PLAN.relative_to(ROOT).as_posix(),
                      "protocol_sha256": plan_hash, "run_id": run_id,
                      "workspace_checks": len(parent["frozen_sha256"]), "wsl_training_checks": len(parent["training_source_sha256"]),
                      "wsl_library_checks": len(parent["library_source_sha256"]), "extra_runtime_checks": len(entry.EXTRA_TRAINING_SOURCES | entry.EXTRA_LIBRARY_SOURCES),
                      "observer_verified": False, "cuda_zero_verified": False, "physics_steps": 0,
                      "learning_updates": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
