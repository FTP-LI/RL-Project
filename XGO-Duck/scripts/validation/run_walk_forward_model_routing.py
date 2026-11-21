"""Freeze and sample an external-command model choice on the existing walking plant."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
from time import monotonic, sleep
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

import run_walk_backward_paired_eval as native
from check_walk_forward_model_routing import DRAFT, OUT, official_rows
from record_io import write
from s1_reference_inputs import ROOT, read, sha


ENTRY = Path(__file__).resolve()
PLAN = DRAFT.with_stem(DRAFT.stem.removesuffix("_draft"))
INDEX = ROOT / "records/walk-forward-model-routing-live.html"
STARTUP = native.OUT / "startup"


class FixedCommandActor(nn.Module):
    """The collector calls the full batch for nominal actions and a forward-only subset for feedback."""

    def __init__(self, official, learned, desired, commands, device):
        super().__init__()
        self.official = official
        self.learned = learned
        mask = official_rows(desired, commands)
        self.worlds = len(mask)
        self.forward_ids = torch.as_tensor(np.flatnonzero(mask), device=device)
        self.other_ids = torch.as_tensor(np.flatnonzero(~mask), device=device)

    def forward(self, inputs):
        count = inputs["actor"].shape[0]
        if count != self.worlds:
            assert 0 < count <= len(self.forward_ids)
            return self.official(inputs)
        forward = self.official(inputs[self.forward_ids])
        other = self.learned(inputs[self.other_ids])
        result = forward.new_empty((count, forward.shape[-1]))
        result.index_copy_(0, self.forward_ids, forward)
        return result.index_copy_(0, self.other_ids, other)


def freeze():
    native.check_project()
    assert not PLAN.exists() and not INDEX.exists()
    draft = read(DRAFT)
    assert draft["status"] == "draft-forward-routing-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    source = native.pinned(draft["reference_protocol"])
    prior = native.pinned(draft["previous_scores"])
    spec = deepcopy(native.checked())
    assert source == native.PLAN and not any(read(prior)["physically_eligible_for_confirmation"].values())
    proof_path = OUT / "cpu-components.json"
    proof = read(proof_path)
    assert proof["status"] == "passed-cpu-routing-components" and proof["draft_sha256"] == sha(DRAFT)
    assert proof["source_sha256"] == sha(ENTRY.with_name("check_walk_forward_model_routing.py"))
    official, learned = (native.pinned(proof[name]) for name in ("official", "learned"))
    assert official == ROOT / draft["official_model"] and learned == ROOT / draft["equal_weight_model"]
    receipt = ROOT / draft["equal_weight_readback"]
    assert sha(learned) == read(receipt)["onnx_sha256"]
    models = {"official": (native.rel(official), sha(official)), "control": (native.rel(learned), sha(learned))}
    seeds = draft["development_seeds"]
    arms = list(draft["comparison"])
    assert arms == ["official", "control", "routed"]
    cases, components = {}, {}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    common = spec["cases"]["1139-official"]["environment"]
    n, count, controls = (common[k] for k in ("parallel_worlds", "episodes_per_command", "maximum_control_steps"))
    assert count == draft["possible_fixed_command_budget"]["worlds_per_command"] and controls == 1000
    for seed in seeds:
        assert seed not in spec["development_seeds"]
        assert not any(f"_s{seed}_" in p.name for p in (ROOT / "records/evaluations").iterdir())
        permutation = np.random.default_rng(70000 + seed).permutation(n)
        camera = list(common["commands"]).index("backward") * count
        initial = np.flatnonzero(permutation == 0).item()
        permutation[camera], permutation[initial] = permutation[initial], permutation[camera]
        groups = {name: permutation[i*count:(i+1)*count].tolist() for i, name in enumerate(common["commands"])}
        base = dict(common, seed=seed, assignment_seed=70000+seed, groups=groups, stage=draft["stage"])
        _, desired = native.rollout.groups_and_commands(base)
        components[str(seed)] = native.feedback.component_probe(
            base, spec["feedback"], desired, read(ROOT / spec["feedback_screen"]))
        assert np.flatnonzero(official_rows(desired, common["commands"])).tolist() == sorted(groups["forward"])
        for arm in arms:
            primary = "control" if arm == "control" else "official"
            path, digest = models[primary]
            name = f"duck_walk_forward-routing_{arm}_s{seed}_{stamp}"
            env = dict(base, model=path, model_sha256=digest, run_id=name)
            cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, primary=primary, environment=env,
                record=f"records/evaluations/{name}", page=f"records/walk-forward-routing-{seed}-{arm}-live.html")
    budget = draft["possible_fixed_command_budget"]
    assert len(cases)*n*controls == budget["maximum_transitions"]
    assert len(cases)*controls*5 == budget["maximum_batched_substeps"]
    spec.update(status="frozen-native-forward-model-routing", cases=cases, arms=arms, development_seeds=seeds,
                index=native.rel(INDEX), stage=draft["stage"], models=models,
                backward_preservation="No negative matched net change versus the equal-weight arm in either seed.")
    for path in (source, prior, DRAFT, proof_path, official, learned, receipt):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY, ENTRY.with_name("verify_walk_forward_model_routing.py"),
                 ENTRY.with_name("observe_walk_forward_model_routing.py"),
                 ENTRY.with_name("check_walk_forward_model_routing.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    for item in cases.values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    write(OUT / "frozen-preflight.json", dict(status="passed-frozen-routing-preflight", protocol_sha256=sha(PLAN),
        fixed_component_proof_sha256=sha(proof_path), controller_components=components, physical_substeps=0,
        learning_updates=0))
    links = []
    for key, item in cases.items():
        folder, page = (ROOT / item[name] for name in ("record", "page"))
        folder.mkdir()
        shutil.copyfile(PLAN, folder / "protocol.json")
        (folder / "README.md").write_text(
            f"# 七指令模型选择 {key}\n\n[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        native.publish(folder, page, dict(status="starting", run_id=folder.name, purpose=f"七指令模型选择 · {key}",
            controls=0, maximum_controls=controls, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{key}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>七指令模型选择对照</h1><ul>'
                     + "".join(links) + '</ul>', encoding="utf-8")
    print("FORWARD_ROUTING_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec = read(PLAN)
    preflight = read(OUT / "frozen-preflight.json")
    assert spec["status"] == "frozen-native-forward-model-routing"
    assert preflight["status"] == "passed-frozen-routing-preflight" and preflight["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for case in spec["cases"].values():
        native.rollout.check_sources(dict(spec, environment=case["environment"]))
    return spec


def ready(item):
    record = ROOT / item["record"]
    path = record / "windows-observer-ready.json"
    deadline = monotonic() + 60
    while not path.is_file() and monotonic() < deadline:
        sleep(1)
    proof = read(path)
    assert proof["observer_host"] == "Windows" and proof["status"] == "starting-page-observed"
    assert proof["run_id"] == record.name and proof["protocol_sha256"] == sha(PLAN)
    assert proof["page_sha256"] == sha(ROOT / item["page"])
    assert proof["initial_state_sha256"] == sha(record / "state.json")


def sample(spec, key):
    item = spec["cases"][key]
    ready(item)
    original_start, original_collect = native.rollout.startup, native.feedback.collect
    learned = None

    def startup(env, obs, actor, p, folder, unused_probes, desired):
        nonlocal learned
        original_start(env, obs, actor, p, folder, STARTUP / item["primary"], desired)
        if item["arm"] == "routed":
            model, fingerprint = spec["models"]["control"]
            secondary = dict(p, model=model, model_sha256=fingerprint)
            learned = native.rollout.make_actor(secondary, env.device)
            secondary_folder = folder / "learned-startup"
            secondary_folder.mkdir()
            original_start(env, obs, learned, secondary, secondary_folder, STARTUP / "control", desired)
            write(folder / "model-routing.json", dict(status="passed-two-model-startup", external_commands=desired.tolist(),
                official_worlds=np.flatnonzero(official_rows(desired, p["commands"])).tolist(),
                official_sha256=p["model_sha256"], learned_sha256=fingerprint, physical_substeps=0,
                observer_ready_sha256=sha(folder / "windows-observer-ready.json")))

    def collect(env, actor, local, folder, state, desired):
        if item["arm"] == "routed":
            assert learned is not None
            actor = FixedCommandActor(actor, learned, desired, item["environment"]["commands"], env.device)
        original_collect(env, actor, local, folder, state, desired)

    with patch.multiple(native, PLAN=PLAN, OUT=OUT), patch.object(native.rollout, "startup", startup), \
            patch.object(native.feedback, "collect", collect):
        native.sample(spec, key)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        spec = checked()
        ready(next(iter(spec["cases"].values())))
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(spec)
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
