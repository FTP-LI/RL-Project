"""Find the input and joint behind each rejected saved CPU PPO update."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import check_getup_success_step_control_sequence as sequence
from getup_success_step_control import SuccessStepControlledPPO
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    old_path = sequence.OUT/'sequence-preflight.json'
    old = read(old_path)
    assert old['status'] == 'cpu-sequence-below-admission'
    assert old['script_sha256'] == sha(Path(sequence.__file__))
    frozen = read(sequence.PLAN)
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    cells = read(ROOT/frozen['training']['control']['anchor_cells'])
    original_drift = SuccessStepControlledPPO.replay_drift
    original_update = SuccessStepControlledPPO.update
    active_proposals, runs = [], []

    @torch.no_grad()
    def observed_drift(learner):
        drift = original_drift(learner)
        if drift[1] > learner.replay_limits[1]:
            errors = learner.actor(td('actor', learner.anchor_inputs))-learner.anchor_targets
            index, joint = np.unravel_index(int(errors.abs().argmax()), tuple(errors.shape))
            value = float(errors[index, joint])
            np.testing.assert_allclose(abs(value), drift[1], atol=1e-7, rtol=0)
            active_proposals.append(dict(rms_rad=drift[0], max_abs_rad=drift[1],
                signed_rad=value, input_index=int(index), joint=joints[joint],
                cells=[key for key, indices in cells.items() if index in indices]))
        return drift

    def observed_update(learner, batch, values):
        active_proposals.clear()
        try:
            return original_update(learner, batch, values)
        except sequence.ReplayStepRejected:
            assert len(active_proposals) == len(learner.step_scales)
            runs.append([dict(row, step_scale=scale) for row, scale in
                         zip(active_proposals, learner.step_scales, strict=True)])
            raise

    with patch.object(SuccessStepControlledPPO, 'replay_drift', observed_drift), \
         patch.object(SuccessStepControlledPPO, 'update', observed_update), \
         patch.object(sequence, 'write') as captured:
        sequence.main()
    captured.assert_called_once()
    _, replay = captured.call_args.args
    assert replay['status'] == old['status'] and len(runs) == len(old['arms'])
    for name, arm in replay['arms'].items():
        expected = old['arms'][name]
        for key in ('accepted_updates', 'reduced_rate_updates'):
            assert arm[key] == expected[key]
        for key in ('update', 'window_start'):
            assert arm['first_rejection'][key] == expected['first_rejection'][key]
        np.testing.assert_allclose(arm['final_rad'], expected['final_rad'], atol=2e-6, rtol=0)
    evidence = {name: dict(rejected_update=replay['arms'][name]['first_rejection']['update'], proposals=rows)
                for name, rows in zip(replay['arms'], runs, strict=True)}
    write(sequence.OUT/'sequence-drift-localization.json', dict(status='reproduced-saved-rejections',
          source_sha256=sha(Path(__file__)), original_result_sha256=sha(old_path), arms=evidence,
          physical_substeps=0, optimizer_updates_saved=0, heldout_used=False))
    print('GETUP_REJECTIONS_LOCALIZED', {key: [row['joint'] for row in arm['proposals']]
                                       for key, arm in evidence.items()})


if __name__ == '__main__':
    main()
