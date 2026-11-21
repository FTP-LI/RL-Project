"""Check right-only model routing without changing the other inference batches."""

from pathlib import Path

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_native import OUT, load_actor, source_plan
from common_walk_startup import make_actor, td
from record_io import write
from run_walk_forward_model_routing import FixedCommandActor


DRAFT = ROOT / "configs/experiments/common_walk_right_startup_seven_command_draft.json"


class RightCommandActor(FixedCommandActor):
    def __init__(self, official, learned, right, desired, commands, device):
        super().__init__(official, learned, desired, commands, device)
        self.right = right
        right_mask = np.all(desired == np.asarray(commands["turn-right"], np.float32), axis=1)
        right_worlds = np.flatnonzero(right_mask)
        other = self.other_ids.cpu().numpy()
        self.right_worlds = torch.as_tensor(right_worlds, device=device)
        self.right_rows_in_other_batch = torch.as_tensor(np.flatnonzero(right_mask[other]), device=device)
        assert len(right_worlds) == len(self.right_rows_in_other_batch) == 32
        assert len(other) == 192 and len(self.forward_ids) == 32

    def forward(self, inputs):
        nominal = super().forward(inputs)
        if len(inputs["actor"]) != self.worlds:
            return nominal
        alternate = self.right(inputs[self.other_ids])
        return nominal.index_copy_(0, self.right_worlds, alternate[self.right_rows_in_other_batch])


def main():
    draft, parent = source_plan()
    plan = read(DRAFT)
    assert not any(plan[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    case = parent["cases"][str(plan["known_case_seed"])]
    p, record = case["environment"], ROOT / case["record"]
    trace = record / "trace-shared-prefix.npz"
    proof = read(record / "verification.json")
    assert sha(trace) == proof["shared_prefix_sha256"]
    desired = np.empty((224, 3), np.float32)
    for name, ids in p["groups"].items():
        desired[ids] = p["commands"][name]
    torch.set_num_threads(4)
    official = make_actor(p, "cpu")
    learned = load_actor(draft, p, "cpu")
    nominal = FixedCommandActor(official, learned, desired, p["commands"], "cpu")
    parent_actor = RightCommandActor(official, learned, learned, desired, p["commands"], "cpu")
    # Use an already verified, different actor to exercise row replacement before new ONNX exports exist.
    fixture = RightCommandActor(official, learned, official, desired, p["commands"], "cpu")
    right, other = fixture.right_worlds, fixture.other_ids
    kept = torch.ones(224, dtype=torch.bool)
    kept[right] = False
    recorded_steps = [0, 49, 99, 149, 199]
    with np.load(trace, allow_pickle=False) as saved:
        observations = saved["observations"][recorded_steps]
    changed = 0
    with torch.no_grad():
        for obs in observations:
            inputs = td(torch.from_numpy(obs))
            baseline, alternate = nominal(inputs), fixture(inputs)
            torch.testing.assert_close(parent_actor(inputs), baseline, atol=0, rtol=0)
            torch.testing.assert_close(alternate[kept], baseline[kept], atol=0, rtol=0)
            target = official(inputs[other])[fixture.right_rows_in_other_batch]
            torch.testing.assert_close(alternate[right], target, atol=0, rtol=0)
            subset = inputs[fixture.forward_ids]
            torch.testing.assert_close(fixture(subset), official(subset), atol=0, rtol=0)
            changed += int(((alternate-baseline).abs().max(dim=1).values > 0).sum())
    assert changed > 0 and not torch.cuda.is_initialized()
    result = dict(status="passed-right-only-archive-routing", draft_sha256=sha(DRAFT), source_sha256=sha(Path(__file__)),
        archive_sha256=sha(trace), matched_non_right_rows=int(kept.sum())*len(recorded_steps),
        checked_right_rows=len(right)*len(recorded_steps), changed_right_fixture_rows=changed,
        checked_forward_feedback_rows=32*len(recorded_steps), nominal_parent_matches_exactly=True,
        fixture_model="official used only as a differing CPU test input; trained exports require their own readback",
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT / "right-route-components.json", result)
    print("RIGHT_STARTUP_ROUTE_COMPONENTS", result, flush=True)


if __name__ == "__main__":
    main()
