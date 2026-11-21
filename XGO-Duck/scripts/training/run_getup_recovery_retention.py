"""Run the frozen supine rollout path with the new optimizer factory."""
from functools import partial
import hashlib
import json
from unittest.mock import patch

import numpy as np
import torch

import run_getup_supine_training as supine
from getup_recovery_retention_ppo import RecoveryRetentionPPO, retention_indices
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def run_with_retention(operation, p, record, *args):
    """Inject only the learner factory for this synchronous foreground call."""
    original_initialize = supine.initialize
    original_load = supine.load_parent
    path, cells = ROOT/p['retention_data'], ROOT/p['retention_cells']
    with np.load(path) as data:
        replay = data['observations'].copy(), data['targets'].copy(), read(cells)
    optimizer = partial(RecoveryRetentionPPO, focus_weight=p['focus_weight'], retention_replay=replay,
                        retention_weight=p['retention_weight'], retention_seed=p['retention_seed'],
                        retention_samples_per_cell=p['retention_samples_per_cell'])

    def load(checkpoint, plan, anchor_replay, device, _inherited_optimizer):
        return original_load(checkpoint, plan, anchor_replay, device, optimizer)

    def initialize(plan, output, scene):
        with patch.object(supine, 'load_parent', load):
            result = original_initialize(plan, output, scene)
        env, _, algorithm, _ = result
        try:
            # The saved run counter counts only the immediately preceding
            # run. Read total Adam steps from the live inherited moments.
            receipt = read(output/'parent-readback.json')
            for name in ('actor', 'critic'):
                state = getattr(algorithm, f'{name}_optimizer').state.values()
                counts = {int(item['step']) for item in state}
                assert len(counts) == 1
                receipt[f'{name}_adam_steps'] = counts.pop()
            assert receipt['actor_adam_steps'] == p['inherited_adam_steps']['actor']
            assert receipt['critic_adam_steps'] == p['inherited_adam_steps']['critic']
            write(output/'parent-readback.json', receipt)
            return result
        except BaseException:
            env.close()
            raise

    with patch.object(supine, 'initialize', initialize):
        return operation(p, record, *args)


def verify_retention(p, proof):
    """Recompute every recorded replay draw from the frozen independent stream."""
    path = ROOT/proof['record']/'learning-metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cells = read(ROOT/p['retention_cells'])
    per_update = p['learner']['epochs']*p['learner']['minibatches']
    assert len(rows) == proof['updates']
    sample_hashes = []
    for k, row in enumerate(rows):
        assert row['retention_batches'] == per_update and row['retention_weight'] == p['retention_weight']
        assert np.isfinite(row['retention_loss']) and row['retention_loss'] >= 0.
        draws = torch.stack([retention_indices(cells, p['retention_seed'], k*per_update+i,
                                                p['retention_samples_per_cell']) for i in range(per_update)])
        digest = hashlib.sha256(draws.numpy().tobytes()).hexdigest()
        assert digest == row['retention_batch_sha256']
        sample_hashes.append(digest)
    assert any(row['retention_loss'] > 0. for row in rows)
    return dict(weight=p['retention_weight'], retained_samples_per_minibatch=len(cells)*p['retention_samples_per_cell'],
                retention_batches=len(rows)*per_update, sample_hashes=sample_hashes, metrics_sha256=sha(path))
