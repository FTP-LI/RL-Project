"""Recover the frozen shared-value run after its zero-step launcher failure."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_shared_value as trial


ENTRY = Path(__file__).resolve()
PLAN = trial.PLAN.with_stem(trial.PLAN.stem+"_recovery")
PAGE = ROOT/"records/walk-right-support-shared-value-recovery-live.html"
OUT = trial.OUT/"value-zero-step-recovery"
base = trial.base


def dispatch(spec):
    with (
        patch.multiple(base, PLAN=PLAN, PAGE=PAGE, OUT=OUT, collect=trial.collect, targets=trial.targets),
        patch.object(base.shared_fit, "fit", trial.fit_checked),
    ):
        base.run(spec)


def freeze():
    base.check_project()
    assert not any(path.exists() for path in (PLAN, PAGE, OUT))
    spec = deepcopy(trial.checked())
    old = ROOT/spec["record"]
    log = trial.OUT/"value-coordinator.log"
    assert "bad operand type for unary +: '_patch'" in log.read_text(encoding="utf-8")
    state = read(old/"state.json")
    assert state["status"] == "starting" and state["controls"] == state["transitions"] == state["learning_updates"] == 0
    assert not any((old/name).exists() for name in ("startup.json", "first-episodes.npz", "execution-exit.txt"))
    witness = read(old/"windows-observer-ready.json")
    state_path, page_path = old/"state-before-launch-stop.json", old/"observer-page-before-launch-stop.html"
    copyfile(old/"state.json", state_path)
    copyfile(trial.PAGE, page_path)
    assert witness["initial_state_sha256"] == sha(state_path) and witness["page_sha256"] == sha(page_path)
    (old/"execution-exit.txt").write_text("1\n", encoding="ascii")
    state.update(status="stopped-review-required", reason="Launcher context fault before CUDA sampling")
    with patch.object(base, "PAGE", trial.PAGE):
        base.publish(old, state)
    OUT.mkdir()
    receipt = OUT/"launch-stop.json"
    write(receipt, dict(status="verified-zero-step-launch-stop", original_protocol_sha256=sha(trial.PLAN),
        original_record=spec["record"], coordinator_log_sha256=sha(log), original_observer_state_sha256=sha(state_path),
        original_observer_page_sha256=sha(page_path), exit_code=1, physical_substeps=0, critic_updates=0, actor_updates=0))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT/"records/experiments"/f"duck_walk_right-support-shared-value_reference_s{spec['seed']}_{stamp}"
    spec.update(record=base.relative(folder), page=base.relative(PAGE), zero_step_recovery=base.relative(receipt))
    paths = (ENTRY, ENTRY.with_name("observe_walk_right_support_shared_value_recovery.py"), trial.PLAN,
             log, state_path, page_path, old/"execution-exit.txt", receipt)
    spec["source_sha256"].update({base.relative(path): sha(path) for path in paths})
    with patch.object(base, "run") as target:
        dispatch(spec)
        target.assert_called_once_with(spec)
    folder.mkdir()
    write(PLAN, spec)
    write(folder/"protocol.json", spec)
    (folder/"README.md").write_text(f"# 右转支撑奖励共享价值\n\n[恢复方案](../../../configs/experiments/{PLAN.name}) · "
                                   f"[运行观察](../../{PAGE.name})\n", encoding="utf-8")
    with patch.object(base, "PAGE", PAGE):
        base.publish(folder, dict(status="starting", run_id=folder.name, purpose="右转支撑奖励 · 共享价值",
            controls=0, maximum_controls=1000, transitions=0, learning_updates=0, updates=[]))
    print("RIGHT_SUPPORT_SHARED_VALUE_RECOVERY_FROZEN", sha(PLAN), base.relative(folder), flush=True)


def checked():
    with patch.object(trial, "PLAN", PLAN):
        return trial.checked()


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        import verify_walk_right_support_shared_value as readback

        with patch.multiple(readback.verifier, PLAN=PLAN, PAGE=PAGE, OUT=OUT, checked=checked,
                            read_targets=readback.read_targets):
            readback.verifier.main()
    else:
        with (Path.home()/"XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dispatch(checked())


if __name__ == "__main__":
    main()
