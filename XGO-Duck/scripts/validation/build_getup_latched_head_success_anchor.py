"""Anchor four successful release poses to the exact warm-start actor."""

from pathlib import Path

import numpy as np

from build_getup_archived_policy_teacher_replay import POSES, pack, read_rows, save
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import actor, predict, success


PLAN = ROOT / 'configs/experiments/getup_latched_head_success_anchor.json'
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/corrective-replay'


def main():
    from record_io import write

    p = read(PLAN)
    project = read(ROOT / 'project.json')
    pinned(p['draft'])
    draft = ROOT / p['draft'][0]
    assert project['current_phase'] == p['stage'] and ROOT / project['goal_execution']['current_protocol'] == draft
    assert not any(p[key] for key in ('heldout_used', 'physical_substeps', 'learning_updates'))
    receipt = OUT / 'success-anchor.json'
    assert not receipt.exists()
    score = pinned(p['source'])
    assert score['seed'] == 1162 and not score['heldout_used']
    record = ROOT / score['record']
    spec = pinned((f"{score['record']}/protocol.json", score['protocol_sha256']))
    role = p['role']
    successful, poses = success(score['arms'][role])
    review = pinned(p['excluded_world_review'])['seeds'][str(score['seed'])]['groups']
    excluded = review['candidate_only_success']['worlds'] + review['candidate_only_failure']['worlds']
    eligible = successful & ~np.isin(np.arange(len(poses)), excluded)
    quota, controls = p['worlds_per_actual_pose'], p['controls']
    rng = np.random.default_rng(p['sample_seed'])
    groups = {pose: np.sort(rng.choice(np.flatnonzero(eligible & (poses == pose)), quota, replace=False))
              for pose in POSES}
    worlds = np.concatenate(list(groups.values()))
    assert np.intersect1d(worlds, excluded).size == 0 and len(np.unique(worlds)) == 4 * quota
    x, y, _, trace_sha = read_rows(record, role, score, worlds, controls)
    network, _ = actor(spec, role)
    targets = predict(network, x)
    error = float(np.abs(y - targets).max())
    assert error <= p['maximum_readback_error_rad'], error
    data = pack(x, controls, worlds, score['seed'], teacher_targets=targets,
                actual_pose=np.broadcast_to(poses[worlds], x.shape[:2]))
    cells = {f'{pose}|{control}': np.flatnonzero((data['actual_pose'] == pose) &
             (data['control'] == control)).tolist() for pose in POSES for control in controls}
    assert len(cells) == 32 and all(len(rows) == quota for rows in cells.values())
    replay_path, cells_path = OUT / 'success-anchor.npz', OUT / 'success-anchor-cells.json'
    assert not replay_path.exists() and not cells_path.exists()
    replay = save(replay_path, data)
    write(cells_path, cells)
    write(receipt, dict(status='passed-four-actual-pose-success-anchor', protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)), source_score_sha256=p['source'][1], trace_sha256=trace_sha,
        model_sha256=spec['policies'][role]['model_sha256'], replay=replay,
        cells=[cells_path.relative_to(ROOT).as_posix(), sha(cells_path)], cell_count=len(cells),
        successful_worlds={pose: ids.tolist() for pose, ids in groups.items()}, excluded_worlds=excluded,
        maximum_onnx_readback_error_rad=error, physical_substeps=0, learning_updates=0, heldout_used=False))
    print('GETUP_SUCCESS_ANCHOR', len(worlds), replay['rows'], len(cells), 'max error', error, flush=True)


if __name__ == '__main__':
    main()
