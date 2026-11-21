"""Test adaptive replay on the original and a disjoint saved-world block."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import screen_getup_targeted_replay as screen
from getup_adaptive_replay import adaptive_replay
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_adaptive_replay.json'


def main():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-cpu-adaptive-screen'
    assert not spec['dynamic_execution_authorized'] and not spec['heldout_used']
    for path, checksum in spec['source_sha256'].items():
        assert sha(ROOT/path) == checksum, path
    fixed = read(screen.PLAN)
    prior = read(ROOT/fixed['prior_result'])
    frozen = read(screen.sequence.PLAN)
    parent = frozen['training']['control']
    expected = dict(prior=prior, limits_rad=prior['limits_rad'], anchor_weight=parent['anchor_weight'])
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    source = ROOT/parent['probe_dataset']
    assert sha(source) == parent['probe_sha256']
    saved_batch = screen.sequence.saved_batch
    results = {}
    for label, (first, stop) in spec['world_blocks'].items():
        names = ('actor_obs_sample', 'critic_obs', 'actions', 'rewards', 'terminal')
        assert stop-first == 32
        with np.load(source) as data:
            assert stop <= data['actions'].shape[1]
            saved = {key: torch.from_numpy(data[key][:, first:stop].copy()) for key in names}
            q = torch.from_numpy(data['qpos'][parent['focus_interval'][0]-1, first:stop, 3:7].copy())
        supine = screen.sequence.actual_supine_at_release(q, parent['release_pose_minimum_alignment'])

        def use_block(learner, original_data, original_mask, begin, length, plan):
            if first == 0:
                assert all(torch.equal(saved[key], original_data[key]) for key in saved)
                assert torch.equal(supine, original_mask)
            return saved_batch(learner, saved, supine, begin, length, plan)

        def make(learner, joint_order, terms, variant):
            assert variant == 'adaptive' and len(joint_order) == learner.anchor_targets.shape[1]
            return adaptive_replay(learner, terms)

        with patch.object(screen, 'side_replay', make), patch.object(screen.sequence, 'saved_batch', use_block):
            results[label] = screen.replay_variant('adaptive', spec, joints, expected)
        print('ADAPTIVE_REPLAY', label,
              {k: (v['accepted_updates'], v['final_rad'], v['passed']) for k, v in results[label]['arms'].items()},
              flush=True)
    admitted = all(arm['passed'] for block in results.values() for arm in block['arms'].values())
    write(ROOT/spec['result'], dict(status='passed-both-cpu-blocks' if admitted else 'below-cpu-admission',
          protocol_sha256=sha(PLAN), source_sha256=sha(Path(__file__)), blocks=results,
          physical_substeps=0, optimizer_updates_saved=0, heldout_used=False, dynamic_execution_authorized=False))


if __name__ == '__main__':
    main()
