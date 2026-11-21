"""Verify combined logs, exit codes and bounded child cleanup without simulation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import recorded_process


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = ROOT / "records/reports/native-observer-preflight"
    temporary = Path(tempfile.mkdtemp(prefix="_recorded-process-check-", dir=out)).resolve()
    if not temporary.is_relative_to(out.resolve()):
        raise RuntimeError("unexpected temporary path")
    try:
        log, receipt = temporary / "failure.log", temporary / "failure.json"
        command = [sys.executable, "-u", "-c", "import os,sys; os.write(1,b'OUT\\n'); os.write(2,b'ERR\\n'); sys.exit(7)"]
        failure = recorded_process.run_recorded(command, cwd=ROOT, log_path=log, receipt_path=receipt, max_wall_seconds=10)
        assert failure["status"] == "child-failed" and failure["exit_code"] == 7
        assert log.read_bytes() == b"OUT\nERR\n" and failure["log_sha256"] == sha(log)
        try:
            recorded_process.run_recorded(command, cwd=ROOT, log_path=log, receipt_path=receipt, max_wall_seconds=10)
        except ValueError:
            pass
        else:
            raise AssertionError("existing recording was overwritten")

        timeout_log, timeout_receipt = temporary / "timeout.log", temporary / "timeout.json"
        script = ("import subprocess,sys,time; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                  "print('grandchild='+str(p.pid),flush=True); time.sleep(30)")
        timeout = recorded_process.run_recorded([sys.executable, "-u", "-c", script], cwd=ROOT,
                    log_path=timeout_log, receipt_path=timeout_receipt, max_wall_seconds=1.5)
        assert timeout["status"] == "wall-timeout" and timeout["exit_code"] is not None
        line = timeout_log.read_text(encoding="utf-8").strip()
        assert line.startswith("grandchild=")
        grandchild_pid = int(line.split("=", 1)[1])
        if os.name == "posix":
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                proc = Path(f"/proc/{grandchild_pid}/stat")
                if not proc.exists() or proc.read_text().split()[2] == "Z":
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("timed-out descendant still executing")
        assert timeout["log_sha256"] == sha(timeout_log)
        record = {"status": "recorded-process-check-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "stdout_and_stderr_exact": True, "nonzero_exit_recorded": True,
                  "existing_log_not_overwritten": True, "timeout_recorded": True,
                  "timed_out_descendant_not_executing": os.name == "posix",
                  "logger_sha256": sha(Path(recorded_process.__file__)), "test_sha256": sha(Path(__file__)),
                  "scope": "Synthetic short-lived child processes only; no simulator or training",
                  "physics_steps": 0, "learning_updates": 0, "hardware_io": False}
        (out / "recorded-process-check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "stdout_and_stderr_exact": True,
                          "timeout_recorded": True, "physics_steps": 0}, ensure_ascii=False))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
