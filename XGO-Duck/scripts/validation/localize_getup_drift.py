"""Attribute the stopped getup guard's squared error using saved inputs only."""
from pathlib import Path

import numpy as np
import torch

from getup_recovery_retention_pair import OUT, PLAN
from getup_supine_learning import load_parent
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def measure(errors):
    return dict(rms_rad=float(np.sqrt(np.mean(errors ** 2))),
                max_rad=float(np.max(np.abs(errors))),
                squared_error_rad2=float(np.sum(errors ** 2)))


def summarize(errors, cells, joints):
    pose_indices = {}
    for name, indices in cells.items():
        pose_indices.setdefault(name.split('|')[0], set()).update(indices)
    total = measure(errors)
    return dict(total=total,
                joints={name: measure(errors[:, i]) for i, name in enumerate(joints)},
                cells={name: measure(errors[indices]) for name, indices in cells.items()},
                poses={pose: measure(errors[sorted(indices)]) for pose, indices in pose_indices.items()})


def main():
    torch.set_num_threads(4)
    frozen, stop_path = read(PLAN), OUT/'stop-verification.json'
    stop = read(stop_path)
    assert stop['protocol_sha256'] == sha(PLAN)
    assert stop['control_updates'] == 32 and stop['candidate_updates'] == 0
    plan = frozen['training']['control']
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    record = ROOT/stop['control_record']
    paths = {key: ROOT/plan[key] for key in ('anchor_data', 'anchor_cells', 'retention_data', 'retention_cells')}
    anchor_cells, retention_cells = read(paths['anchor_cells']), read(paths['retention_cells'])
    with np.load(paths['anchor_data']) as data:
        anchor = data['observations'].copy(), data['teacher_targets'].copy(), anchor_cells
    with np.load(paths['retention_data']) as data:
        retention = data['observations'].copy(), data['targets'].copy()
    for dataset, cells in ((anchor, anchor_cells), (retention, retention_cells)):
        assert sorted({i for indices in cells.values() for i in indices}) == list(range(len(dataset[0])))
        assert dataset[1].shape == (len(dataset[0]), len(joints))

    points = {}
    for name, filename in (('initial', 'model-init.pt'), ('update_25', 'model-25.pt'),
                           ('stopped', 'stopped-partial.pt')):
        path = record/filename
        expected = stop['checkpoint_cpu_drift'][name]
        assert sha(path) == expected['checkpoint_sha256']
        learner, _ = load_parent(path, plan, anchor, 'cpu')
        with torch.inference_mode():
            a = learner.actor(td('actor', learner.anchor_inputs)).numpy() - anchor[1]
            r = learner.actor(td('actor', torch.from_numpy(retention[0]))).numpy() - retention[1]
        assert np.isfinite(a).all() and np.isfinite(r).all()
        anchor_summary = summarize(a.astype(np.float64), anchor_cells, joints)
        retention_summary = summarize(r.astype(np.float64), retention_cells, joints)
        np.testing.assert_allclose([anchor_summary['total']['rms_rad'], anchor_summary['total']['max_rad']],
                                   [expected['anchor_rms_rad'], expected['anchor_max_rad']], atol=2e-6, rtol=0)
        for pose, result in retention_summary['poses'].items():
            np.testing.assert_allclose(result['rms_rad'], expected['retention_rms_rad'][pose], atol=2e-6, rtol=0)
        points[name] = dict(anchor=anchor_summary, retention=retention_summary)
    assert not torch.cuda.is_initialized()
    write(OUT/'drift-localization.json', dict(status='verified-saved-checkpoints-only',
          source_sha256=sha(Path(__file__)), stop_verification_sha256=sha(stop_path), protocol_sha256=sha(PLAN),
          input_hashes={key: sha(path) for key, path in paths.items()}, joint_order=joints,
          anchor_inputs=len(anchor[0]), parent_retention_inputs=len(retention[0]), points=points,
          last_passing_update=stop['last_passing_update'], last_passing_checkpoint_audited=False,
          physical_substeps=0, learning_updates=0, heldout_used=False))
    print('GETUP_DRIFT_LOCALIZED', len(anchor[0]), len(retention[0]))


if __name__ == '__main__':
    main()
