"""Evaluate a fixed forward feedback margin with the trained right-only route."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_walk_right_support_forward_retention as repeat
import run_walk_right_support_seven_command as previous
from record_io import write
from s1_reference_inputs import ROOT, read, sha


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_support_forward_margin.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = previous.OUT.parent / "forward-margin"
INDEX = ROOT / "records/walk-right-support-forward-margin-live.html"
COHORTS = ("known", "fresh")
ARMS = previous.ARMS
native, routing, recovery = previous.native, previous.routing, previous.recovered
LABELS = dict(zip(ARMS, ("父策略 · 0.12", "支撑奖励 · 0.12", "支撑奖励 · 0.15"), strict=True))
NAMES = dict(zip(ARMS, ("parent012", "support012", "support015"), strict=True))


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    spec, draft = deepcopy(previous.checked()), read(DRAFT)
    assert draft["status"] == "draft-not-executable" and tuple(draft["arms"]) == ARMS
    assert not any(draft[k] for k in ("heldout_used", "hardware_io", "learning_updates",
                                      "automatic_retry", "physical_substeps_authorized"))
    evidence = previous.OUT / "analysis.json"
    projection = previous.OUT.parent / "forward-feedback-margin-cpu.json"
    physical = repeat.OUT / "analysis.json"
    screen, short = read(projection), read(physical)
    assert screen["status"] == "passed-archived-forward-feedback-projection"
    assert screen["seven_command_result_sha256"] == sha(evidence) and screen["checked_forward_rows"] == 121600
    assert screen["original_joint_envelope_preserved"] and not screen["physical_substeps"]
    assert short["status"] == "passed-independent-forward-repeatability-readback"
    assert short["protocol_sha256"] == sha(repeat.PLAN) and short["first_forward_targets_exact"]
    assert not short["learning_updates"] and not short["full_episode_scoring_performed"]
    assert draft["maximum_joint_correction_rad"] == screen["maximum_joint_correction_rad"] == spec["feedback"]["maximum_joint_correction_rad"]
    caps = [row["forward_cap_rad_s"] for row in draft["arms"].values()]
    assert caps == [screen["original_feedback_cap_rad_s"]]*2+[screen["candidate_feedback_cap_rad_s"]]
    known, fresh = (draft[f"{cohort}_development_seed"] for cohort in COHORTS)
    reserved = (fresh, draft["reserved_training_replication_seed"], draft["sealed_holdout_seed"])
    assert len(set(reserved)) == 3 and not set(reserved) & set(spec["development_seeds"])
    for root in ("experiments", "evaluations", "benchmarks"):
        assert not any(any(f"_s{seed}_" in path.name for seed in reserved) for path in (ROOT/"records"/root).iterdir())
    source_case = spec["cases"][f"{known}-routed-parent"]
    common = source_case["environment"]
    n, count, horizon = (common[k] for k in ("parallel_worlds", "episodes_per_command", "maximum_control_steps"))
    assert (n, count, horizon) == (224, draft["worlds_per_command"], draft["controls_per_case"])
    maximum = n*horizon*len(ARMS)
    assert maximum == draft["maximum_transitions_per_seed"]
    cases, components = {}, {}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    for cohort, seed in zip(COHORTS, (known, fresh), strict=True):
        env = deepcopy(common)
        env.update(seed=seed, assignment_seed=70000+seed, stage=draft["stage"])
        if cohort == "fresh":
            order = np.random.default_rng(env["assignment_seed"]).permutation(n)
            back = list(env["commands"]).index("backward")*count
            zero = np.flatnonzero(order == 0).item()
            order[back], order[zero] = order[zero], order[back]
            env["groups"] = {name: order[i*count:(i+1)*count].tolist() for i, name in enumerate(env["commands"])}
        _, desired = native.rollout.groups_and_commands(env)
        components[cohort] = native.feedback.component_probe(
            env, spec["feedback"], desired, read(ROOT/spec["feedback_screen"]))
        for arm, changes in draft["arms"].items():
            name = f"duck_walk_forward-margin_{NAMES[arm]}_s{seed}_{stamp}"
            cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, cohort=cohort, **changes,
                environment=dict(env, run_id=name), record=f"records/evaluations/{name}",
                page=f"records/walk-forward-margin-{NAMES[arm]}-s{seed}-live.html")
    spec.update(stage=draft["stage"], cases=cases, development_seeds=[known, fresh], index=native.rel(INDEX),
        maximum_transitions=2*maximum, maximum_batched_substeps=2*len(ARMS)*horizon*5,
        physical_substeps_authorized=2*len(ARMS)*horizon*5, known_case_seed=known, fresh_case_seed=fresh,
        maximum_speed_loss_m_s=draft["maximum_forward_speed_change_m_s"],
        minimum_yaw_improvement={cohort: draft[f"minimum_mean_forward_yaw_improvement_on_{cohort}_rad_s"]
                                 for cohort in COHORTS},
        heldout_seed=draft["sealed_holdout_seed"], known_must_pass_before_fresh=True)
    baseline = ROOT/source_case["record"]/"verification.json"
    assert sha(baseline) == read(evidence)["records"][f"{known}-routed-parent"]["verification_sha256"]
    spec["known_baseline_receipt"] = native.rel(baseline)
    refs = (DRAFT, previous.PLAN, repeat.PLAN, evidence, projection, physical, baseline)
    scripts = (ENTRY, ENTRY.with_name("observe_walk_right_support_forward_margin.py"),
               ENTRY.with_name("verify_walk_right_support_forward_margin.py"))
    spec["reference_sha256"].update({native.rel(path): sha(path) for path in refs})
    spec["source_sha256"].update({native.rel(path): sha(path) for path in scripts})
    previous.check_cases(spec)
    with patch.object(recovery, "sample") as fixture:
        sample(spec, f"{known}-ppo-candidate")
        actual, key = fixture.call_args.args
        assert actual["feedback"]["maximum_yaw_input_rad_s"] == .15 and key == f"{known}-ppo-candidate"
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    for cohort in COHORTS:
        (OUT/cohort).mkdir()
    write(PLAN, spec)
    write(OUT/"preflight.json", dict(status="passed-frozen-forward-feedback-margin",
        protocol_sha256=sha(PLAN), controller_components=components, physical_substeps=0, learning_updates=0))
    links = []
    for item in cases.values():
        record, page = (ROOT/item[k] for k in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record/"protocol.json")
        label = f"{item['seed']} · {LABELS[item['arm']]}"
        (record/"README.md").write_text(f"# 前进反馈对照 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name,
            purpose=f"前进反馈七指令 · {label}", controls=0, maximum_controls=horizon,
            transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>前进反馈七指令对照</h1>'
        '<p>新种子在已知种子过线后开始。</p><ul>'+"".join(links)+"</ul>", encoding="utf-8")
    print("FORWARD_MARGIN_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT/"preflight.json")
    assert proof["status"] == "passed-frozen-forward-feedback-margin" and proof["protocol_sha256"] == sha(PLAN)
    assert spec["known_must_pass_before_fresh"] and spec["stage"] == "S2-forward-feedback-margin"
    previous.check_cases(spec)
    return spec


def scoped(spec, cohort):
    cases = {key: item for key, item in spec["cases"].items() if item["cohort"] == cohort}
    assert len(cases) == len(ARMS)
    return dict(spec, cases=cases)


def admit(spec, cohort):
    if cohort == "fresh":
        proof = read(OUT/"known/analysis.json")
        assert proof["protocol_sha256"] == sha(PLAN) and proof["promoted_to_fresh_development"]


def sample(spec, key):
    item = spec["cases"][key]
    local = dict(spec, feedback=dict(spec["feedback"], maximum_yaw_input_rad_s=item["forward_cap_rad_s"]))
    with patch.multiple(recovery, PLAN=PLAN, OUT=OUT/item["cohort"]):
        recovery.sample(local, key)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", choices=COHORTS)
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    spec = checked()
    cohort = args.run or spec["cases"][args.case]["cohort"]
    admit(spec, cohort)
    if args.run:
        with patch.multiple(native, OUT=OUT/cohort, ENTRY=ENTRY):
            native.run(scoped(spec, cohort))
    else:
        sample(spec, args.case)


if __name__ == "__main__":
    main()
