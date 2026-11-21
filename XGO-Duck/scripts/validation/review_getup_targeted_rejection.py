"""Locate the rows rejecting the combined replay term, without new physical data."""
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import screen_getup_targeted_replay as screen
from getup_success_step_control import SuccessStepControlledPPO
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    spec = read(screen.PLAN)
    source = ROOT/spec['result']
    saved = read(source)
    assert saved['status'] == 'below-primary-cpu-admission' and sha(screen.PLAN) == saved['protocol_sha256']
    for path, checksum in spec['source_sha256'].items():
        assert sha(ROOT/path) == checksum, path
    prior = read(ROOT/spec['prior_result'])
    frozen = read(screen.sequence.PLAN)
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    parent = frozen['training']['control']
    cells = read(ROOT/parent['anchor_cells'])
    original = SuccessStepControlledPPO.replay_drift
    rejections = {0.: [], .01: []}

    @torch.no_grad()
    def locate(learner):
        rms, maximum = original(learner)
        if maximum > learner.replay_limits[1]:
            diff = learner.actor(td('actor', learner.anchor_inputs))-learner.anchor_targets
            row, col = np.unravel_index(int(diff.abs().argmax()), tuple(diff.shape))
            rejections[learner.retention_weight].append(dict(rms_rad=rms, max_rad=maximum,
                row=int(row), joint=joints[col], signed_rad=float(diff[row, col]),
                cells=[key for key, ids in cells.items() if row in ids],
                exceeded_values=int((diff.abs() > learner.replay_limits[1]).sum())))
        return rms, maximum

    expected = dict(prior=prior, limits_rad=prior['limits_rad'], anchor_weight=parent['anchor_weight'])
    with patch.object(SuccessStepControlledPPO, 'replay_drift', locate):
        reproduced = screen.replay_variant('combined', spec, joints, expected)
    evidence = {}
    for name, arm in reproduced['arms'].items():
        old = saved['variants']['combined']['arms'][name]
        assert arm['accepted_updates'] == old['accepted_updates'] and arm['first_rejection'] == old['first_rejection']
        assert arm['steps'] == old['steps']
        failures = ast.literal_eval(arm['first_rejection']['cause'].split(' after ', 1)[1])
        terminal = rejections[arm['retention_weight']][-len(failures):]
        for raw, detail in zip(failures, terminal, strict=True):
            assert detail['rms_rad'] == raw['rms_rad'] and detail['max_rad'] == raw['max_rad']
            detail['scale'] = raw['scale']
        evidence[name] = dict(update=arm['first_rejection']['update'], proposals=terminal)
    write(source.with_name('targeted-replay-rejection.json'), dict(status='reproduced-primary-rejections',
          source_sha256=sha(source), script_sha256=sha(Path(__file__)), arms=evidence,
          physical_substeps=0, optimizer_updates_saved=0, heldout_used=False))
    print('TARGETED_REJECTIONS', {name: [(r['row'], r['joint'], r['cells']) for r in arm['proposals']]
                                  for name, arm in evidence.items()})


if __name__ == '__main__':
    main()
