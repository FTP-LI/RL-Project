"""Recover the early official trial after a zero-physics buffer restoration fault."""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

import run_walk_early_official_right as trial
from record_io import write
from s1_reference_inputs import ROOT, read, sha


ENTRY = Path(__file__).resolve()
PLAN = trial.PLAN.with_stem(trial.PLAN.stem+"_recovery")
OUT = trial.OUT/"zero-physics-recovery"
INDEX = ROOT/"records/walk-early-official-right-recovery-live.html"
native = trial.native
PROCESSOR_FIELDS = ("_raw_actions", "_processed_actions", "requested_target", "projected_destination")
original_cuda_zero = trial.cuda_zero.__wrapped__


def check_mutable(term):
    for name in PROCESSOR_FIELDS:
        value = getattr(term, name)
        assert not value.is_inference(), name
        value.copy_(value)


@torch.no_grad()
def cuda_zero(env, actor, spec, record, desired):
    term = env.action_manager.get_term("joint_pos")
    before = {name: getattr(term, name).clone() for name in PROCESSOR_FIELDS}
    check_mutable(term)
    # The fixture inference may be optimized, but its restored live buffers must be ordinary tensors.
    original_cuda_zero(env, actor, spec, record, desired)
    check_mutable(term)
    for name, value in before.items():
        torch.testing.assert_close(getattr(term, name), value, atol=0, rtol=0)
    path = record/"early-route-zero.json"
    write(path, dict(read(path), live_buffer_mutability_and_values_verified=True))


@contextmanager
def recovered():
    with patch.multiple(trial, PLAN=PLAN, OUT=OUT, cuda_zero=cuda_zero):
        yield


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    spec = deepcopy(trial.checked())
    original_cases = spec["cases"]
    first = ROOT/next(iter(original_cases.values()))["record"]
    execution = trial.OUT/"execution.json"
    zero = read(first/"early-route-zero.json")
    state = read(first/"evaluation.json")
    assert len(read(execution)) == 1 and read(execution)[0]["returncode"] == 1
    assert (first/"execution-exit.txt").read_text().strip() == "1"
    assert state["actual"] == dict(attempted_controls=1, completed_controls=0,
        attempted_substeps=0, completed_substeps=0, initial_resets=1)
    assert state["status"] == "stopped-review-required" and "Inplace update to inference tensor" in state["error"]
    assert zero["physical_substeps"] == 0 and not list(first.glob("trace-*.npz"))
    for item in list(original_cases.values())[1:]:
        folder = ROOT/item["record"]
        assert read(folder/"state.json")["status"] == "starting" and not (folder/"execution.log").exists()
    sample_buffers = SimpleNamespace(**{name: torch.zeros(2, 14) for name in PROCESSOR_FIELDS})
    check_mutable(sample_buffers)
    with torch.inference_mode():
        bad_buffer = sample_buffers._raw_actions.clone()
    assert bad_buffer.is_inference() and not bad_buffer.clone().is_inference()
    assert not torch.cuda.is_initialized()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    for key, item in original_cases.items():
        name = f"duck_walk_early-official-right_{item['arm']}_s{item['seed']}_{stamp}"
        env = dict(item["environment"], run_id=name)
        assert dict(env, run_id=item["environment"]["run_id"]) == item["environment"]
        cases[key] = dict(item, environment=env, record=f"records/evaluations/{name}",
                          page=f"records/walk-early-official-right-recovery-{item['arm']}-live.html")
    spec.update(cases=cases, index=native.rel(INDEX), recovery=dict(original_protocol=native.rel(trial.PLAN),
        original_protocol_sha256=sha(trial.PLAN), failed_case=native.rel(first),
        previous_completed_physical_substeps=0, previous_attempted_controls=1,
        scope="Restore ordinary mutable processor tensors after the zero-step probe. Keep all physical settings and gates."))
    for path in (trial.PLAN, trial.OUT/"preflight.json", execution, first/"execution.log", first/"execution-exit.txt",
                 first/"evaluation.json", first/"early-route-zero.json", first/"windows-view-verification.json"):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY, ENTRY.with_name("observe_walk_early_official_right_recovery.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    with recovered(), patch.object(trial.recovery.routing, "ready"), patch.object(native, "sample") as sampler:
        trial.sample(spec, next(iter(cases)))
        sampler.assert_called_once()
    OUT.mkdir()
    fixture = OUT/"archived-input-fixture.npz"
    copyfile(trial.OUT/"archived-input-fixture.npz", fixture)
    spec["reference_sha256"][native.rel(fixture)] = sha(fixture)
    for item in cases.values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    write(PLAN, spec)
    write(OUT/"preflight.json", dict(status="passed-early-official-right-cpu", protocol_sha256=sha(PLAN),
        original_protocol_sha256=sha(trial.PLAN), physical_substeps=0, original_physical_budget_unchanged=True,
        fixture_sha256=sha(fixture), runtime_dispatch_exercised=True, cpu_buffer_mutability_exercised=True))
    links = []
    for label, item in zip(trial.LABELS, cases.values(), strict=True):
        record, page = (ROOT/item[name] for name in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record/"protocol.json")
        (record/"README.md").write_text(f"# 右转早期切换 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"七指令首回合 · {label}",
            controls=0, maximum_controls=spec["maximum_controls_per_case"], transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转早期切换</h1><ul>'+''.join(links)+'</ul>', encoding="utf-8")
    print("EARLY_OFFICIAL_RIGHT_RECOVERY_FROZEN", sha(PLAN), list(cases), flush=True)


def verify():
    import verify_walk_early_official_right as scorer

    previous = scorer.await_case

    def completed(folder):
        previous(folder)
        assert (folder/"execution-exit.txt").read_text().strip() == "0", folder
        assert read(folder/"early-route-zero.json")["live_buffer_mutability_and_values_verified"], folder

    with patch.object(scorer, "await_case", completed):
        scorer.main()


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--replay", action="store_true")
    action.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    with recovered():
        if args.run:
            with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
                native.run(trial.checked())
        elif args.verify:
            verify()
        elif args.replay:
            from replay_walk_right_startup_seven_command import render

            spec = trial.checked()
            render(spec, spec["known_case_seed"], PLAN, OUT, trial.ARMS, trial.LABELS)
        else:
            trial.sample(trial.checked(), args.case)


if __name__ == "__main__":
    main()
