"""Verify the online learner uses the admitted adaptive update path."""
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

import check_getup_success_step_control_sequence as sequence
from getup_adaptive_ppo import AdaptiveGetupPPO
from getup_supine_learning import load_parent
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    spec = read(ROOT/'configs/experiments/getup_adaptive_replay_archives.json')
    baseline_path = ROOT/spec['result']
    baseline = read(baseline_path)
    assert baseline['status'] == 'passed-both-cpu-archives'
    expected = baseline['archives']['original-43']['arms']

    def make(path, plan, anchor, retention, limits, weight, bounded, scales):
        assert bounded
        constructor = partial(AdaptiveGetupPPO, replay_limits=limits, step_scales=scales,
                              adaptive_settings=spec['terms'], reward_scale=plan['reward_scale'],
                              check_active=lambda: None, focus_weight=plan['focus_weight'], retention_replay=retention,
                              retention_weight=weight, retention_seed=plan['retention_seed'],
                              retention_samples_per_cell=plan['retention_samples_per_cell'])
        return load_parent(path, plan, anchor, 'cpu', constructor)[0]

    with patch.object(sequence, 'learner_factory', make), patch.object(sequence, 'write') as capture:
        sequence.main()
    capture.assert_called_once()
    _, replay = capture.call_args.args
    for role, arm in replay['arms'].items():
        assert arm['passed'] and arm['accepted_updates'] == expected[role]['accepted_updates'] == 32
        for actual, prior in zip(arm['steps'], expected[role]['steps'], strict=True):
            assert (actual['update'], actual['actor_steps'], actual['scale']) == (
                prior['update'], prior['actor_steps'], prior['scale'])
            np.testing.assert_allclose([actual['rms_rad'], actual['max_rad']],
                                       [prior['rms_rad'], prior['max_rad']], atol=1e-7, rtol=0)
    path = sequence.OUT/'adaptive-ppo-entry-parity.json'
    write(path, dict(status='passed-online-learner-cpu-parity', comparison_sha256=sha(baseline_path),
                     script_sha256=sha(Path(__file__)), learner_sha256=sha(ROOT/'scripts/training/getup_adaptive_ppo.py'),
                     arms={key: value['accepted_updates'] for key, value in replay['arms'].items()},
                     physical_substeps=0, optimizer_updates_saved=0, heldout_used=False))
    print('ADAPTIVE_PPO_ENTRY_PARITY', path, flush=True)


if __name__ == '__main__':
    main()
