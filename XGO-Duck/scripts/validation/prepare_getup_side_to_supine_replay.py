"""Prepare matched replay coverage with byte-identical targets on unchanged rows."""
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

from review_getup_side_to_supine_retention import AGE, BASES, OUT as DIAGNOSIS, PARENT, branch, predict
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_side_to_supine_replay_draft.json'
OUT = ROOT/'records/reports/getup-side-to-supine-replay'


def build():
    assert not OUT.exists()
    spec = read(PLAN)
    source = ROOT/spec['diagnosis']
    assert source==DIAGNOSIS/'parent-retention-review.json'
    proof = read(source)
    assert proof['status']=='passed-saved-parent-retention-review' and not proof['heldout_used']
    assert spec['status']=='draft-not-executable' and list(AGE)==spec['control_ages']
    assert [x['seed'] for x in proof['independent_archives']]==spec['archive_build_seeds']+[spec['archive_check_seed']]
    rng = np.random.default_rng(spec['selection_seed'])
    arrays = {'control':[],'candidate':[]}
    inputs = {'control':[],'candidate':[]}
    targets = {'control':[],'candidate':[]}
    cells = {}
    cursor = 0
    model_sha = None
    source_receipts = []
    keep = spec['requested_supine_worlds_per_seed']-spec['replace_per_supine_cell']

    for source_path,archive in zip(BASES,proof['independent_archives'],strict=True):
        assert sha(source_path)==archive['source_sha256']
        run,trace,requested,release,success = branch(source_path,PARENT)
        seed = run['environment']['seed']
        assert archive['seed']==seed and archive['trace_sha256']==sha(trace)
        actor = run['policies'][PARENT]
        model = ROOT/actor['model']
        digest = sha(model)
        assert digest==actor['model_sha256']
        model_sha = digest if model_sha is None else model_sha
        assert digest==model_sha
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        if seed==spec['archive_check_seed']:
            ids = np.flatnonzero(success&(release=='supine')&np.isin(requested,('left-side','right-side')))
            steps = np.asarray(AGE)-run['prefix_controls']
            with np.load(trace) as data:
                x = data['actor_input'][steps][:,ids].reshape(-1,61)
                y = data['raw_policy_actions'][steps][:,ids].reshape(-1,14)
            error = abs(predict(session,x)-y)
            assert error.max()<2e-5
            source_receipts.append(dict(seed=seed,role='reference-only',worlds=len(ids),verified_inputs=len(x),
                                        maximum_onnx_error=float(error.max()),trace_sha256=sha(trace)))
            continue
        right = rng.choice(np.flatnonzero(success&(release=='right-side')),spec['right_worlds_per_seed'],replace=False)
        supine = rng.permutation(np.flatnonzero(success&(requested=='supine')))
        assert len(supine)==spec['requested_supine_worlds_per_seed']
        side = []
        for name,n in (('left-side',8 if seed==1061 else 7),('right-side',7 if seed==1061 else 8)):
            eligible = np.flatnonzero(success&(release=='supine')&(requested==name))
            side.extend(rng.choice(eligible,n,replace=False).tolist())
        mixed = np.concatenate((supine[:keep],side))
        assert len(mixed)==len(supine) and len(set(mixed))==len(mixed)
        assert not set(right)&set(np.concatenate((supine,mixed)))
        assert len(side)==spec['replace_per_supine_cell']
        with np.load(trace) as data:
            for age in AGE:
                step = age-run['prefix_controls']
                for cell,group in (('actual-right',{'control':right,'candidate':right}),
                                   ('supine-slot',{'control':supine,'candidate':mixed})):
                    size = len(group['control'])
                    cells[f'{seed}|{age}|{cell}'] = list(range(cursor,cursor+size))
                    cursor += size
                    for role,ids in group.items():
                        obs = data['actor_input'][step,ids].copy()
                        action = data['raw_policy_actions'][step,ids].copy()
                        predicted = predict(session,obs)
                        np.testing.assert_allclose(predicted,action,atol=2e-5,rtol=2e-5)
                        inputs[role].append(obs)
                        targets[role].append(action)
                        arrays[role].extend((seed,age,int(i),str(requested[i]),str(release[i])) for i in ids)
        source_receipts.append(dict(seed=seed,role='build',actual_right_worlds=len(right),
             requested_supine_control_worlds=len(supine),requested_supine_candidate_worlds=keep,
             side_to_supine_candidate_worlds=len(side),trace_sha256=sha(trace)))

    assert cursor==spec['rows_per_arm']==900 and len(cells)==spec['cells_per_arm']==24
    result = {}
    for role in arrays:
        record = arrays[role]
        assert len(record)==len(set(record))==cursor
        data = dict(observations=np.concatenate(inputs[role]),targets=np.concatenate(targets[role]),
            seed=np.array([row[0] for row in record]),control_age=np.array([row[1] for row in record]),
            world=np.array([row[2] for row in record]),requested_pose=np.array([row[3] for row in record]),
            actual_pose=np.array([row[4] for row in record]))
        assert data['observations'].shape==(900,61) and data['targets'].shape==(900,14)
        assert np.isfinite(data['observations']).all() and np.isfinite(data['targets']).all()
        result[role] = data
    c,t = (result[r] for r in ('control','candidate'))
    changed = (c['world']!=t['world'])
    assert changed.sum()==180
    np.testing.assert_array_equal(c['seed'],t['seed'])
    np.testing.assert_array_equal(c['control_age'],t['control_age'])
    assert np.all(c['requested_pose'][changed]=='supine') and np.all(t['actual_pose'][changed]=='supine')
    assert np.isin(t['requested_pose'][changed],('left-side','right-side')).all()
    for key in c:
        np.testing.assert_array_equal(c[key][~changed],t[key][~changed])
    OUT.mkdir()
    write(OUT/'sampling-cells.json',cells)
    files = {}
    for role,data in result.items():
        path = OUT/f'{role}-replay.npz'
        np.savez_compressed(path,**data)
        files[role] = dict(path=path.name,sha256=sha(path),actual_pose_counts=dict(Counter(data['actual_pose'].tolist())))
    write(OUT/'preflight.json',dict(status='passed-matched-cpu-replay-pools',plan_sha256=sha(PLAN),
        source_sha256=sha(source),script_sha256=sha(Path(__file__)),parent_model_sha256=model_sha,
        rows_per_arm=cursor,cells=len(cells),changed_rows=int(changed.sum()),unchanged_rows=int((~changed).sum()),
        sampling_cells_sha256=sha(OUT/'sampling-cells.json'),files=files,source_receipts=source_receipts,
        heldout_used=False,physical_substeps=0,learning_updates=0,diagnostic_seed_used_for_training=False))
    print('SIDE_TO_SUPINE_REPLAY_PREPARED',cursor,int(changed.sum()),source_receipts,flush=True)


if __name__=='__main__':
    build()
