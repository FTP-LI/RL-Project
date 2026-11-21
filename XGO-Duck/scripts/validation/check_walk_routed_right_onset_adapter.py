"""Exercise the nested target processors on saved observations without a simulator."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from run_walk_routed_right_onset import DRAFT, collect_right, feedback, native, routing
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_walk_routed_right_onset import right_readback


def main():
    parent, draft = routing.checked(), read(DRAFT)
    p = parent["cases"]["1141-routed"]["environment"]
    _, desired = native.rollout.groups_and_commands(p)
    data = ROOT / parent["cases"]["1141-routed"]["record"]
    checked = read(data / "verification.json")
    paths = (data / "trace-shared-prefix.npz", data / f"trace-{native.ROLE}.npz")
    assert [sha(path) for path in paths] == checked["trace_hashes"]
    fields = ("observations", "active", "actions")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as item:
            parts.append({k: item[k][:250] for k in fields})
    trace = {key: np.concatenate([part[key] for part in parts])[:250] for key in fields}
    assert trace["active"].all()
    torch.set_num_threads(3)
    actors = [native.rollout.make_actor(dict(model=path, model_sha256=digest), "cpu")
              for path, digest in parent["models"].values()]
    wrapped = routing.FixedCommandActor(*actors, desired, p["commands"], "cpu")
    check = dict(draft["component_screen"], maximum_correction_controls=50)
    local = dict(parent, environment=p, right_onset=check)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort_model = ort.InferenceSession(str(ROOT / parent["models"]["control"][0]), sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    outputs, receipts = {}, {}
    with TemporaryDirectory(dir=routing.OUT) as temporary:
        for arm in ("reference", "candidate"):
            term = feedback.ProjectedWalkAction.__new__(feedback.ProjectedWalkAction)
            term.cfg = native.rollout.make_config(p)[0].actions["joint_pos"]
            term._env = SimpleNamespace(device="cpu")
            term._raw_actions = torch.zeros((224, 14))
            term._processed_actions = torch.zeros_like(term._raw_actions)
            term._scale, term._offset, term._clip = 1., torch.tensor(p["model_home"]), None
            env = SimpleNamespace(device="cpu", action_manager=SimpleNamespace(get_term=lambda _: term))
            folder = Path(temporary) / arm
            folder.mkdir()
            targets = []

            def archived_collect(*args):
                controller = native.rollout.ForwardPhase(p, parent["phase"], desired)
                with torch.no_grad():
                    for index, source in enumerate(trace["observations"]):
                        command = controller.map(source.copy(), index+1, index >= parent["phase"]["warmup_controls"],
                                                 trace["active"][index])
                        np.testing.assert_allclose(command, source[:, 48:51], atol=1e-6, rtol=0)
                        sent = torch.from_numpy(trace["actions"][index])
                        term.process_actions(sent)
                        torch.testing.assert_close(term._raw_actions, sent, atol=0, rtol=0)
                        targets.append(native.rollout.cpu(term._processed_actions))

            apply = arm == "candidate"
            with patch.object(feedback, "original_collect", archived_collect):
                collect_right(env, wrapped, local, folder, {}, desired, apply)
            outputs[arm] = np.stack(targets)
            _, receipts[arm] = right_readback(dict(trace, applied_joint_targets=outputs[arm]), p, local, folder,
                                              ort_model, apply)
    delta = abs(outputs["candidate"]-outputs["reference"]).max(axis=2)
    assert not delta[:200].any()
    assert np.flatnonzero(delta.any(axis=0)).tolist() == [13, 198]
    assert receipts["candidate"]["triggered_worlds"] == receipts["reference"]["triggered_worlds"] == [13, 198]
    assert not torch.cuda.is_initialized()
    write(routing.OUT / "right-onset-adapter-cpu.json", dict(status="passed-right-onset-live-adapter-on-archive",
        draft_sha256=sha(DRAFT), sampler_sha256=sha(Path(__file__).with_name("run_walk_routed_right_onset.py")),
        verifier_sha256=sha(Path(__file__).with_name("verify_walk_routed_right_onset.py")), source_sha256=sha(Path(__file__)),
        controls=250, worlds=224, triggered_worlds=[13, 198], maximum_target_difference_rad=float(delta.max()),
        exact_prefix=True, physical_substeps=0, learning_updates=0))
    print("RIGHT_ONSET_ADAPTER_CPU", receipts["candidate"]["verified_alternate_inputs"], float(delta.max()), flush=True)


if __name__ == "__main__":
    main()
