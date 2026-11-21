"""Build one balanced four-pose parent replay from completed development episodes."""
import numpy as np
import onnxruntime as ort

from record_io import write
from review_getup_side_to_supine_retention import AGE, PARENT, branch, predict
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/'records/reports/getup-right-upright-ppo-replay'
MODEL = ROOT/'artifacts/delivery/trained/getup-supine-weighted.onnx'
VALUE_PLAN = ROOT/'configs/experiments/getup_right_upright_value_coverage.json'
ARCHIVES = {
    1084: ROOT/'records/reports/getup-early-support-ppo-functional/analysis.json',
    1087: ROOT/'records/reports/getup-side-to-supine-replay/evaluation/analysis.json',
}
POSES = ('prone', 'supine', 'left-side', 'right-side')


def build():
    assert not OUT.exists()
    model_sha = sha(MODEL)
    assert model_sha == read(VALUE_PLAN)['environment']['model_sha256']
    rng = np.random.default_rng(110103)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
    groups = {pose: [] for pose in POSES}
    receipts = {}
    for seed, archive in ARCHIVES.items():
        spec, trace, requested, release, success = branch(archive, PARENT)
        assert spec['environment']['seed'] == seed
        assert spec['policies'][PARENT]['model_sha256'] == model_sha
        counts = {}
        steps = np.asarray(AGE)-spec['prefix_controls']
        with np.load(trace) as data:
            np.testing.assert_allclose(data['time'][steps], (np.asarray(AGE)+1)*.02, atol=1e-6, rtol=0)
            actor, action = data['actor_input'], data['raw_policy_actions']
            for pose in POSES:
                eligible = np.flatnonzero(success & (release == pose))
                quota = 18 if seed == 1087 and pose in ('left-side', 'right-side') else 19
                worlds = np.sort(rng.choice(eligible, quota, replace=False))
                groups[pose].append(dict(
                    observations=actor[steps][:, worlds], targets=action[steps][:, worlds],
                    seed=np.full(quota, seed), world=worlds, requested_pose=requested[worlds],
                    actual_pose=release[worlds],
                ))
                counts[pose] = dict(eligible=len(eligible), retained=quota)
        receipts[seed] = dict(counts=counts, source=archive.relative_to(ROOT).as_posix(),
                              source_sha256=sha(archive), trace=trace.relative_to(ROOT).as_posix(),
                              trace_sha256=sha(trace))
    rows, cells, cursor = [], {}, 0
    for pose, group in groups.items():
        for i, age in enumerate(AGE):
            cell = {key: np.concatenate([row[key][i] if key in ('observations', 'targets') else row[key]
                                         for row in group]) for key in group[0]}
            size = len(cell['world'])
            cell['control_age'] = np.full(size, age)
            cells[f'{pose}|{age}'] = list(range(cursor, cursor+size))
            cursor += size
            rows.append(cell)
    data = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    assert cursor == 900 and len(cells) == 24
    assert data['observations'].shape == (900,61) and data['targets'].shape == (900,14)
    assert len(set(zip(data['seed'], data['world'], data['control_age'], strict=True))) == 900
    assert np.isfinite(data['observations']).all() and np.isfinite(data['targets']).all()
    predicted = predict(session, data['observations'])
    np.testing.assert_allclose(predicted, data['targets'], atol=2e-5, rtol=2e-5)
    OUT.mkdir()
    replay, sampling = OUT/'parent-replay.npz', OUT/'sampling-cells.json'
    np.savez_compressed(replay, **data)
    write(sampling, cells)
    with np.load(replay) as saved:
        for key, value in data.items():
            np.testing.assert_array_equal(saved[key], value)
    write(OUT/'analysis.json', dict(status='passed-four-pose-parent-replay', rows=cursor, cells=len(cells),
          unique_worlds=150, actual_poses=list(POSES), control_ages=AGE, model_sha256=model_sha,
          selection_seed=110103, archives=receipts, maximum_onnx_error=float(abs(predicted-data['targets']).max()),
          replay_sha256=sha(replay), cells_sha256=sha(sampling), physical_substeps=0, learning_updates=0,
          heldout_used=False))
    print('RIGHT_UPRIGHT_RETENTION_VERIFIED', cursor, len(cells), flush=True)


if __name__ == '__main__':
    build()
