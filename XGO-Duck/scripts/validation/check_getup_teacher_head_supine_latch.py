"""Verify the native latched actor on two saved controls without simulation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import getup_shared_release as shared
from getup_supine_release_actor import SupineReleaseActor
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT / 'configs/experiments/getup_teacher_head_supine_latch_component.json'
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/supine-latch-component.json'


def pinned(item):
    name, checksum = item
    path = ROOT / name
    assert sha(path) == checksum
    return read(path)


def run_device(device, spec, release, later, mask, plan):
    parent, candidate = (shared.actor_model(spec['policies'][role], device)
                         for role in ('reference-a', 'candidate'))
    clock = SimpleNamespace(value=plan['release_counter'] - 1)
    gate = SupineReleaseActor(parent, candidate, lambda: clock.value,
                             plan['release_counter'], plan['normalized_negative_gravity_x_threshold']).eval()
    cases = [shared.td('actor', torch.from_numpy(x).to(device)) for x in (release, later)]
    truth = torch.from_numpy(mask).to(device)
    try:
        gate(cases[0])
    except ValueError:
        assert gate.supine_at_release is None
    else:
        raise AssertionError('The actor latched before release')
    clock.value = plan['release_counter']
    with torch.inference_mode():
        before = shared.signature(gate.state_dict())
        selected = gate(cases[0])
        assert torch.equal(gate.supine_at_release, truth)
        assert torch.equal(selected[~truth], parent(cases[0])[~truth])
        assert torch.equal(selected[truth], candidate(cases[0])[truth])
        clock.value = plan['later_control'] - 1
        follow = gate(cases[1])
        assert torch.equal(gate.supine_at_release, truth)
        assert torch.equal(follow[~truth], parent(cases[1])[~truth])
        assert torch.equal(follow[truth], candidate(cases[1])[truth])
        assert before == shared.signature(gate.state_dict())
    return dict(device=device, inputs=len(mask) * len(cases), selected=int(mask.sum()),
                preserved=int((~mask).sum()), before_release_guard=True, saved_latch_retained=True,
                branch_outputs_byte_exact=True, actor_weights_unchanged=True)


def main():
    assert not OUT.exists()
    plan = read(PLAN)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == plan['stage'] and ROOT / project['goal_execution']['current_protocol'] == PLAN
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    assert pinned(plan['preflight'])['admitted_to_native_preparation']
    score = pinned(plan['evidence'])
    spec = pinned(plan['source_protocol'])
    assert score['protocol_sha256'] == plan['source_protocol'][1] and spec['prefix_controls'] == plan['release_counter']
    record = ROOT / score['record']
    trace = record / 'trace-reference-a.npz'
    assert sha(trace) == score['arms']['reference-a']['trace_sha256']
    with np.load(trace, allow_pickle=False) as saved:
        later_idx = plan['later_control'] - spec['prefix_controls'] - 1
        release, later = saved['actor_input'][[0, later_idx]]
    actual = np.array([row['release_pose'] for row in score['arms']['reference-a']['timely']['episodes']])
    mask = actual == 'supine'
    g = later[:, 3:6].astype(np.float64)
    later_mask = -g[:, 0] / np.linalg.norm(g, axis=1) >= plan['normalized_negative_gravity_x_threshold']
    assert (later_mask != mask).any()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    with torch.random.fork_rng(devices=[0]):
        results = {device: run_device(device, spec, release, later, mask, plan) for device in ('cpu', 'cuda:0')}
    write(OUT, dict(status='passed-native-supine-latch-zero-step', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        component_sha256=sha(ROOT / 'scripts/training/getup_supine_release_actor.py'), trace_sha256=sha(trace),
        later_observation_would_reclassify=int((later_mask != mask).sum()), results=results,
        physical_substeps=0, learning_updates=0, heldout_used=False))
    print('GETUP_SUPINE_LATCH_NATIVE', {device: (row['selected'], row['preserved']) for device, row in results.items()},
          'later relatch avoided', int((later_mask != mask).sum()), flush=True)


if __name__ == '__main__':
    main()
