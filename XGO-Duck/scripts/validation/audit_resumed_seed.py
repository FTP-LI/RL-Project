"""Check the next development seed against local saved run records.

Only existing text and filenames are examined; binary traces are not parsed.
The result is not proof that the seed was never used outside these scopes.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/resume-20261115/seed-audit.json"
SEED = 1241
SCOPES = ("configs/experiments", "records/training-archive", "records/reports")
GLOBS = ("*.json", "*.jsonl", "*.yaml", "*.yml", "*.md", "*.log", "*.sh")


def invoke(args: list[str], allowed: tuple[int, ...]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode not in allowed:
        raise RuntimeError(f"Search failed ({result.returncode}): {result.stderr}")
    return result


def main() -> None:
    project = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    task, goal = project["staged_training_request"], project["goal_execution"]
    if (task["pause_requested"] or goal["pause_requested_by_user"] or
            task["learning_running"] or task["simulation_running"]):
        raise RuntimeError("Resume preparation must be idle and explicitly resumed")
    if OUT.exists():
        raise FileExistsError("Do not overwrite a prior seed audit")
    pattern = r'(_s1241_|seed[ =:_-]+1241\b|"seed"\s*:\s*1241\b)'
    search = ["rg", "-l", "--no-messages"]
    for glob in GLOBS:
        search.extend(("-g", glob))
    search.extend((pattern, *SCOPES))
    found = invoke(search, (0, 1))
    text_matches = sorted(found.stdout.splitlines())
    listing = invoke(["rg", "--files", *SCOPES], (0,))
    name_matches = sorted(line for line in listing.stdout.splitlines()
                          if re.search(r"(?:^|[\\/])[^\\/]*_s1241(?:_|[\\/.]|$)", line))
    wsl = json.loads((ROOT / "records/reports/progress-archive-20261114/wsl-training-logs/manifest.json").read_text(encoding="utf-8"))
    wsl_names = sorted(row["path"] for row in wsl["files"] if re.search(r"_s1241(?:_|/|\.|$)", row["path"]))
    result = {"status": "no-matches-in-scoped-windows-records" if not (text_matches or name_matches or wsl_names)
              else "matches-require-review", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "seed": SEED, "windows_scopes": list(SCOPES), "text_globs": list(GLOBS),
              "text_search_command": search, "text_search_exit_code": found.returncode,
              "text_matches": text_matches, "filename_matches": name_matches,
              "wsl_manifest_filename_matches": wsl_names,
              "limitations": "Binary traces, other directories and historical runs outside the recorded scopes are not searched; WSL text is checked separately.",
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "seed", "text_search_exit_code", "text_matches", "filename_matches", "wsl_manifest_filename_matches")}, ensure_ascii=False))
    if result["status"] != "no-matches-in-scoped-windows-records":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
