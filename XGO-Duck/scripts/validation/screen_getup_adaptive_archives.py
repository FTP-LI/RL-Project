"""Check both adaptive replay arms on two complete, separately seeded archives."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import screen_getup_targeted_replay as screen
from getup_adaptive_replay import adaptive_replay
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_adaptive_replay_archives.json'
SHAPES = dict(actor_obs_sample=(61,), critic_obs=(75,), actions=(14,), rewards=(), terminal=(), qpos=(21,))


def load_fixture(record, parent):
    with np.load(record/'value-data.npz') as source:
        batch = {key: torch.from_numpy(source[key][:, :32].copy()) for key in SHAPES}
    for name, features in SHAPES.items():
        assert batch[name].shape == (400, 32, *features), (record, name, batch[name].shape)
        assert torch.isfinite(batch[name]).all(), name
    q = batch.pop('qpos')[parent['focus_interval'][0]-1, :, 3:7]
    return batch, screen.sequence.actual_supine_at_release(q, parent['release_pose_minimum_alignment'])


def main():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-cpu-archive-screen' and not spec['dynamic_execution_authorized']
    out = ROOT/spec['result']
    assert not out.exists()
    for path, checksum in spec['source_sha256'].items():
        assert sha(ROOT/path) == checksum, path
    prior = read(ROOT/read(screen.PLAN)['prior_result'])
    frozen = read(screen.sequence.PLAN)
    parent = frozen['training']['control']
    expected = dict(prior=prior, limits_rad=prior['limits_rad'], anchor_weight=parent['anchor_weight'])
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    saved_batch = screen.sequence.saved_batch
    records = {label: ROOT/path for label, path in spec['archives'].items()}
    original, independent = (read(p/'protocol.json') for p in records.values())
    for field in spec['matched_contract']:
        assert original[field] == independent[field], field
    assert original['seed'] != independent['seed'] and not original['heldout_used'] and not independent['heldout_used']
    fixtures = {label: load_fixture(record, parent) for label, record in records.items()}
    results = {}
    for label, (saved, supine) in fixtures.items():
        def use_archive(learner, original_data, original_mask, begin, length, plan):
            if label == 'original-43':
                assert all(torch.equal(saved[key], original_data[key]) for key in saved)
                assert torch.equal(supine, original_mask)
            return saved_batch(learner, saved, supine, begin, length, plan)

        def make(learner, joint_order, terms, variant):
            assert variant == 'adaptive' and len(joint_order) == learner.anchor_targets.shape[1]
            return adaptive_replay(learner, terms)

        with patch.object(screen, 'side_replay', make), patch.object(screen.sequence, 'saved_batch', use_archive):
            results[label] = screen.replay_variant('adaptive', spec, joints, expected)
        print('ADAPTIVE_ARCHIVE', label,
              {key: (arm['accepted_updates'], arm['final_rad'], arm['passed'])
               for key, arm in results[label]['arms'].items()}, flush=True)
    admitted = all(arm['passed'] for block in results.values() for arm in block['arms'].values())
    write(out, dict(status='passed-both-cpu-archives' if admitted else 'below-cpu-admission',
          protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)), archives=results,
          archive_sha256={name: sha(path/'value-data.npz') for name, path in records.items()},
          physical_substeps=0, optimizer_updates_saved=0, heldout_used=False, dynamic_execution_authorized=False))


if __name__ == '__main__':
    main()
