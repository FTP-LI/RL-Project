"""Freeze and sample the observed early right-turn official fallback."""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_walk_right_startup_seven_command_recovery as recovery
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walk_early_official_right import EarlyRightActor


native = recovery.native
ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_early_official_right.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = ROOT / "records/reports/common-walk-early-official-right"
INDEX = ROOT / "records/walk-early-official-right-live.html"
ARMS = ("routed-parent", "shadow-official", "early-official")
LABELS = ("父策略", "影子计算", "实际切换")


@contextmanager
def observed_controller(route):
    phase = native.rollout.ForwardPhase

    class ObservedPhase(phase):
        def map(self, observed, step, candidate, active):
            route.observe(observed, step, active)
            return super().map(observed, step, candidate, active)

    with patch.object(native.rollout, "ForwardPhase", ObservedPhase):
        yield


def fixture_from_archive(spec, prior, score):
    record = ROOT / prior["record"]
    receipt = record/"verification.json"
    assert sha(receipt) == score["verification_sha256"]
    p, proof = prior["environment"], read(receipt)
    last, decision = (spec["detector"][key] for key in ("last_observation_control", "decision_control"))
    segments = []
    for kind, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        path = record/f"trace-{kind}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as f:
            segments.append({name: f[name][:decision] for name in ("observations", "active", "applied_joint_targets")})
    obs, active, target = (np.concatenate([row[key] for row in segments])[:decision] for key in segments[0])
    ids = sorted(p["groups"]["turn-right"])
    assert active.shape == (decision, spec["worlds"])
    return dict(observations=obs, active=active, previous_target=target[last-1, ids])


@torch.inference_mode()
def exercise(spec, p, official, base, desired, device, fixture):
    args = (official, base, desired, p, device, spec["detector"], spec["target_step_cap_rad"])
    as_input = lambda obs: native.rollout.td(torch.as_tensor(obs, device=device))
    frames, active = fixture["observations"], fixture["active"]
    first, current = as_input(frames[0]), as_input(frames[-1])
    reference = EarlyRightActor(*args, ARMS[0])
    before = reference(first)
    right = reference.right_worlds
    other = torch.ones(len(desired), dtype=torch.bool, device=device)
    other[right] = False
    raw = {}
    for arm in ARMS:
        route = EarlyRightActor(*args, arm)
        route.previous_target = lambda: torch.as_tensor(fixture["previous_target"], device=device)
        torch.testing.assert_close(route(first), before, atol=0, rtol=0)
        # Reconstruct the original turn feedback from the raw IMU prefix before issuing a fallback.
        with patch.object(native.rollout, "ForwardPhase", native.feedback.Controller), observed_controller(route):
            controller = native.rollout.ForwardPhase(p, spec["phase"], desired)
            for i, obs in enumerate(frames):
                mapped = controller.map(obs, i+1, False, active[i])
                np.testing.assert_array_equal(mapped, obs[:, 48:51])
                if i+1 < len(active):
                    controller.reset(np.flatnonzero(active[i] & ~active[i+1]))
        raw[arm] = route(current)
        decision = route.receipt
        assert decision["triggered_worlds"] == spec["expected_triggered_worlds"]
        assert decision["guarded_worlds"] == ([] if arm == ARMS[0] else spec["expected_triggered_worlds"])
        assert decision["selected_worlds"] == (spec["expected_triggered_worlds"] if arm == ARMS[-1] else [])
        if arm == ARMS[-1]:
            expected = official(current[route.other_ids])[route.right_rows_in_other_batch]
            changed = torch.as_tensor(spec["expected_triggered_worlds"], device=device)
            expected_rows = [route.ids.tolist().index(world) for world in spec["expected_triggered_worlds"]]
            torch.testing.assert_close(raw[arm][changed], expected[expected_rows], atol=0, rtol=0)
            torch.testing.assert_close(raw[arm][other], raw[ARMS[0]][other], atol=0, rtol=0)
            assert torch.any(raw[arm][changed] != raw[ARMS[0]][changed])
            ended = active[-1].copy()
            ended[spec["expected_triggered_worlds"]] = False
            route.observe(obs, route.decision_control+1, ended)
            torch.testing.assert_close(route(current), raw[ARMS[0]], atol=0, rtol=0)
        subset = current[route.forward_ids]
        torch.testing.assert_close(route(subset), official(subset), atol=0, rtol=0)
    torch.testing.assert_close(raw[ARMS[1]], raw[ARMS[0]], atol=0, rtol=0)
    return raw[ARMS[-1]]


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    draft = read(DRAFT)
    assert draft["status"] == "draft-early-official-right-screen-no-execution"
    assert tuple(draft["arms"]) == ARMS and ROOT/draft["reference_protocol"] == recovery.PLAN
    assert not any(draft[key] for key in
                   ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io", "automatic_retry"))
    detection_path, target_path = (ROOT/draft[key] for key in ("detection_readback", "target_readback"))
    target, detection = read(target_path), read(detection_path)
    aggregate = recovery.OUT/"analysis.json"
    assert target["gyro_review_sha256"] == sha(detection_path) and detection["source_result_sha256"] == sha(aggregate)
    assert target["protocol_sha256"] == sha(recovery.PLAN) and target["abrupt_route_passed_target_screen"]
    spec = deepcopy(recovery.checked())
    prior = spec["cases"][draft["reference_case"]]
    p = prior["environment"]
    seed = draft["development_seed"]
    n, h = draft["worlds"], draft["controls"]
    assert (p["seed"], p["parallel_worlds"], p["maximum_control_steps"]) == (seed, n, h)
    assert p["joint_names"] == target["joint_names"] and n == 224 and p["episodes_per_command"] == 32
    detector = draft["right_only_detector"]
    assert detector["decision_control"] == target["first_selected_control"] == 76
    assert detector["absolute_mean_observed_gyro_max_rad_s"] == target["limit_rad_s"]
    expected = sorted(case["world"] for case in target["cases"])
    assert expected == [13, 198] and all(not case["parent_success"] for case in target["cases"])
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    for arm in ARMS:
        name = f"duck_walk_early-official-right_{arm}_s{seed}_{stamp}"
        cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, right="base", environment=dict(p, run_id=name, stage=draft["stage"]),
            record=f"records/evaluations/{name}", page=f"records/walk-early-official-right-{arm}-live.html")
    spec.update(stage=draft["stage"], cases=cases, arms=list(ARMS), development_seeds=[seed],
                index=native.rel(INDEX), detector=detector, worlds=n, expected_triggered_worlds=expected,
                target_step_cap_rad=target["ordinary_per_joint_max_rad"], known_case_seed=seed,
                maximum_transitions=len(cases)*n*h, maximum_batched_substeps=len(cases)*h*5,
                physical_substeps_authorized=len(cases)*h*5)
    assert spec["maximum_transitions"] == draft["maximum_transitions"]
    assert spec["maximum_batched_substeps"] == draft["maximum_batched_substeps"]
    archived = read(aggregate)["records"][draft["reference_case"]]
    fixture = fixture_from_archive(spec, prior, archived)
    _, desired = native.rollout.groups_and_commands(p)
    torch.set_num_threads(4)
    models = [native.rollout.make_actor(dict(model=path, model_sha256=checksum), "cpu")
              for path, checksum in (spec["models"][name] for name in ("official", "base"))]
    exercise(spec, p, *models, desired, "cpu", fixture)
    assert not torch.cuda.is_initialized()
    for path in (DRAFT, recovery.PLAN, detection_path, target_path, aggregate, ROOT/archived["path"]/"verification.json"):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for name in (ENTRY.name, "walk_early_official_right.py", "observe_walk_early_official_right.py",
                 "verify_walk_early_official_right.py", "replay_walk_right_startup_seven_command.py"):
        path = ENTRY.with_name(name)
        spec["source_sha256"][native.rel(path)] = sha(path)
    with patch.object(recovery.routing, "ready"), patch.object(native, "sample") as sampler:
        sample(spec, next(iter(cases)))
        sampler.assert_called_once()
    OUT.mkdir()
    fixture_path = OUT/"archived-input-fixture.npz"
    np.savez_compressed(fixture_path, **fixture)
    spec["reference_sha256"][native.rel(fixture_path)] = sha(fixture_path)
    for item in cases.values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    write(PLAN, spec)
    write(OUT/"preflight.json", dict(status="passed-early-official-right-cpu", protocol_sha256=sha(PLAN),
        expected_triggered_worlds=expected, first_non_right_exact=True, shadow_outputs_exact=True,
        terminal_and_forward_paths_checked=True, entry_context_exercised=True, physical_substeps=0, learning_updates=0))
    links = []
    for label, item in zip(LABELS, cases.values(), strict=True):
        record, page = (ROOT/item[name] for name in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record/"protocol.json")
        (record/"README.md").write_text(f"# 右转早期切换 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"七指令首回合 · {label}",
            controls=0, maximum_controls=h, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转早期切换</h1><ul>'+''.join(links)+'</ul>',
                     encoding="utf-8")
    print("EARLY_OFFICIAL_RIGHT_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT/"preflight.json")
    assert proof["status"] == "passed-early-official-right-cpu" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[key] for key in ("learning_updates", "heldout_used", "hardware_io", "automatic_retry"))
    assert tuple(spec["arms"]) == ARMS and spec["physical_substeps_authorized"] == 15000
    for item in spec["cases"].values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


@torch.inference_mode()
def cuda_zero(env, actor, spec, record, desired):
    p = spec["environment"]
    assert torch.device(env.device).type == "cuda" and env.common_step_counter == 0
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(env.device).clone()
    term = env.action_manager.get_term("joint_pos")
    with np.load(OUT/"archived-input-fixture.npz", allow_pickle=False) as f:
        sent = exercise(spec, p, actor.official, actor.learned, desired, env.device, f)
    fields = ("_raw_actions", "_processed_actions", "requested_target", "projected_destination")
    saved = {name: getattr(term, name).clone() for name in fields}
    try:
        term.process_actions(sent)
        requested = sent + actor.home
        torch.testing.assert_close(term.requested_target, requested, atol=0, rtol=0)
        torch.testing.assert_close(term._processed_actions, requested.clamp(actor.low, actor.high), atol=0, rtol=0)
    finally:
        for name, value in saved.items():
            setattr(term, name, value)
    assert env.common_step_counter == 0 and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(env.device))
    write(record/"early-route-zero.json", dict(status="passed-actual-cuda-early-route", physical_substeps=0,
        protocol_sha256=sha(PLAN), native_start_sha256=read(record/"native-start.json")["matched_state_sha256"],
        fixture_sha256=sha(OUT/"archived-input-fixture.npz"), actual_projector_checked=True,
        matched_first_routes=True, reset_and_random_state_preserved=True))
    print("EARLY_OFFICIAL_RIGHT_CUDA_ZERO", actor.arm, flush=True)


def sample(spec, key):
    item = spec["cases"][key]
    p, arm = item["environment"], item["arm"]
    feedback = native.feedback
    original_feedback, original_rollout = feedback.collect, feedback.original_collect

    def make(official, base, old_right, desired, commands, device):
        assert old_right is base and commands == p["commands"]
        return EarlyRightActor(official, base, desired, p, device, spec["detector"], spec["target_step_cap_rad"], arm)

    def attach(env, actor, local, record, state, desired):
        cuda_zero(env, actor, local, record, desired)
        term = env.action_manager.get_term("joint_pos")
        actor.previous_target = lambda: term._processed_actions[actor.right_worlds]

        def read_controls(*args):
            with observed_controller(actor):
                original_rollout(*args)

        with patch.object(feedback, "original_collect", read_controls):
            original_feedback(env, actor, local, record, state, desired)
        assert actor.receipt is not None
        write(record/"right-decision.json", actor.receipt)

    with (
        patch.multiple(recovery, PLAN=PLAN, OUT=OUT, ARMS=ARMS, RightCommandActor=make),
        patch.object(feedback, "collect", attach),
    ):
        recovery.sample(spec, key)


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
