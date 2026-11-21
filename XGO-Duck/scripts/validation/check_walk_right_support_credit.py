"""Replay the online startup credit against actual native captures and resets."""

from pathlib import Path

import numpy as np
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walk_right_support_credit import RightSupportCredit


OUT = ROOT/"records/reports/common-walk-right-support-yaw-probe"
PLAN = ROOT/"configs/experiments/common_walk_right_support_yaw_probe.json"


@torch.no_grad()
def boundary_fixture(settings, value):
    n = 7
    counter = RightSupportCredit(n, "cpu", settings, value)
    events = []
    for control in range(1, 205):
        age = torch.full((n,), control, dtype=torch.int32)
        if control >= 95:
            age[2] = control-94
        right = torch.ones(n, dtype=torch.bool)
        right[3] = False
        done = torch.zeros_like(right)
        done[2], done[4] = control == 94, control == 95
        feet = torch.ones((n,2), dtype=torch.bool)
        during = (age >= 76) & (age <= 80)
        feet[during,0] = False
        if during[1] and control % 2:
            feet[1] = torch.tensor([True, False])
        moving = (age >= 81) & (age <= 95)
        actual = torch.where(moving, -.45, 0.)
        observed = actual.clone()
        observed[5] *= -1
        if control == 35:
            observed[6] = float("nan")
        reward = counter.advance(age, right, done, observed, actual, feet)
        assert torch.isfinite(reward).all()
        events.extend((control, i, int(age[i])) for i in torch.nonzero(reward).flatten().tolist())
    assert events == [(95, 0, 95), (189, 2, 95)]
    return [list(row) for row in events]


@torch.no_grad()
def replay(folder, p, settings, value):
    contact_path, native_path = folder/"pre-reset-foot-contact.npz", folder/"native-rollout.npz"
    assert sha(contact_path) == p["contact_sha256"] and sha(native_path) == p["native_rollout_sha256"]
    with np.load(native_path, allow_pickle=False) as data, np.load(contact_path, allow_pickle=False) as contact_data:
        stop = settings["decision_deadline_control"]
        age, gyro, yaw = [torch.from_numpy(data[key][:stop]) for key in ("age", "gyro", "yaw")]
        terminal = torch.from_numpy(data["terminated"][:stop] | data["timed_out"][:stop])
        contact = torch.from_numpy(contact_data["contact"][:stop])
    n = age.shape[1]
    counter = RightSupportCredit(n, "cpu", settings, value)
    right = torch.ones(n, dtype=torch.bool)
    matrix = torch.stack([counter.advance(age[i], right, terminal[i], gyro[i], yaw[i], contact[i])
                          for i in range(len(age))])
    valid = (age == torch.arange(1, len(age)+1)[:, None]).all(dim=0) & ~terminal.any(dim=0)
    expected = {(event["control"], event["world"]) for event in p["windows"]["3s"]["events"]}
    hits = {(k+1, world) for k,world in torch.nonzero((matrix > 0) & valid[None]).tolist()}
    assert hits == expected
    assert ((matrix == 0.) | (matrix == value)).all() and not (matrix[terminal] != 0.).any()
    assert not ((matrix > 0).sum(dim=0) > 1).any()
    return dict(checked_native_controls=int(matrix.numel()), matched_credit_events=len(hits),
                credit_controls=sorted(k for k,_ in hits), terminal_credit_count=0)


def main():
    torch.set_num_threads(3)
    probe_path, scale_path = OUT/"analysis.json", OUT/"reward-scale.json"
    spec, proof, scale = (read(path) for path in (PLAN, probe_path, scale_path))
    assert proof["ready_for_bounded_reward_preflight"] and scale["source_probe_sha256"] == sha(probe_path)
    assert proof["protocol_sha256"] == sha(PLAN)
    settings = spec["capture"]
    value = scale["maximum_once_per_episode_credit"]
    fixture = boundary_fixture(settings, value)
    actual = {name: replay(ROOT/case["record"], case, settings, value) for name,case in proof["cases"].items()}
    assert not torch.cuda.is_initialized()
    source = ROOT/"scripts/training/walk_right_support_credit.py"
    result = dict(status="passed-online-support-credit-cpu", source_sha256=sha(Path(__file__)), reward_source_sha256=sha(source),
        source_probe_sha256=sha(probe_path), reward_scale_sha256=sha(scale_path), checked_reset_fixture=fixture,
        actual_native_replays=actual, reward_per_episode=value, physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"reward-cpu.json", result)
    print("RIGHT_SUPPORT_CREDIT_CPU", value, {name: row["matched_credit_events"] for name,row in actual.items()}, fixture, flush=True)


if __name__ == "__main__":
    main()
