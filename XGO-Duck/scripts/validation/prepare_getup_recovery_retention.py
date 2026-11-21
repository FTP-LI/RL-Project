"""Prepare balanced parent-action replay from already verified development data."""
from pathlib import Path

import numpy as np
import torch

import getup_shared_release as shared
from getup_teacher_anchor import balanced_indices, mean_anchor_loss
from record_io import write
from review_getup_recovery_retention import AGE_STEPS, HISTORY, OUT as PAIR, evidence
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/'records/reports/getup-recovery-retention'
DRAFT = ROOT/'configs/experiments/getup_recovery_retention_draft.json'


def main():
    torch.set_num_threads(4)
    draft = read(DRAFT)
    audit_path = ROOT/draft['retention_audit']
    audit = read(audit_path)
    assert draft['status'] == 'draft-not-executable'
    assert audit['status'] == 'passed-saved-development-only' and not audit['heldout_used']
    assert not torch.cuda.is_initialized() and not OUT.exists()
    assert sha(ROOT/draft['starting_checkpoint']) == draft['starting_checkpoint_sha256']
    archive = draft['parent_success_inputs']
    assert archive['actor_input_control_ages'] == AGE_STEPS
    groups = ('actual-right-side', 'requested-supine')
    fields = {name: [] for name in ('observations', 'targets', 'seed', 'control_age', 'group', 'world')}
    cells = {}
    offset = 0
    for label, folder, role in (('previous', HISTORY, 'success-anchor'), ('current', PAIR, 'fixed-parent')):
        _, spec, _, ids, observations, targets, digest = evidence(folder, role)
        seed = spec['environment']['seed']
        assert audit['data_audit'][label]['seed'] == seed
        assert audit['data_audit'][label]['evaluation_sha256'] == digest
        for group in groups:
            n = len(ids[group])
            assert audit['data_audit'][label]['groups'][group]['environment_indices'] == ids[group].tolist()
            for index, age in enumerate(AGE_STEPS):
                cells[f'{group}|{age}|{seed}'] = list(range(offset, offset+n))
                fields['observations'].append(observations[group][index])
                fields['targets'].append(targets[group][index])
                fields['seed'].append(np.full(n, seed, dtype=np.int64))
                fields['control_age'].append(np.full(n, age, dtype=np.int64))
                fields['group'].append(np.full(n, group))
                fields['world'].append(ids[group])
                offset += n
    data = {name: np.concatenate(values) for name, values in fields.items()}
    assert len(cells) == 24 and len(data['observations']) == archive['available_inputs'] == 900
    assert data['observations'].shape == (900, 61) and data['targets'].shape == (900, 14)
    assert np.isfinite(data['observations']).all() and np.isfinite(data['targets']).all()
    assert len(set(zip(data['seed'], data['group'], data['control_age'], data['world'], strict=True))) == 900
    rng = torch.Generator(device='cpu').manual_seed(54046)
    samples_per_cell = 8
    sample = balanced_indices(cells, rng, samples_per_cell).reshape(len(cells), samples_per_cell).numpy()
    for indices, name in zip(sample, sorted(cells), strict=True):
        assert set(indices) <= set(cells[name])
    counts = {name: int(np.count_nonzero(data['group'][sample] == name)) for name in groups}
    assert counts == {'actual-right-side': 96, 'requested-supine': 96}
    model = dict(model=draft['starting_model'], model_sha256=draft['starting_model_sha256'], seed=46)
    actor = shared.actor_model(model)
    actor.mlp.requires_grad_(True)
    means = actor(shared.td('actor', torch.from_numpy(data['observations'][sample.ravel()])))
    targets = torch.from_numpy(data['targets'][sample.ravel()])
    loss = mean_anchor_loss(means, targets, .10)
    grads = torch.autograd.grad(loss, tuple(actor.mlp.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in grads)
    norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
    assert norm > 0 and not torch.cuda.is_initialized()

    OUT.mkdir()
    np.savez_compressed(OUT/'parent-replay.npz', **data)
    write(OUT/'sampling-cells.json', cells)
    result = dict(status='passed-cpu-replay-preparation', draft_sha256=sha(DRAFT),
                  audit_sha256=sha(audit_path), script_sha256=sha(Path(__file__)),
                  data_sha256=sha(OUT/'parent-replay.npz'), cells_sha256=sha(OUT/'sampling-cells.json'),
                  observations=900, cells=24, samples_per_cell=samples_per_cell, draw_counts=counts,
                  unweighted_replay_loss=float(loss.detach()), raw_actor_gradient_norm=norm,
                  physical_substeps=0, optimizer_updates=0, cuda_initialized=False, heldout_used=False)
    write(OUT/'replay-preflight.json', result)
    print('GETUP_RETENTION_REPLAY_READY', result['observations'], result['cells'], 'gradient', norm, flush=True)


if __name__ == '__main__':
    main()
