"""Measure the actual supine coverage gap and available successful parent replay."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/getup-early-support-ppo-functional'
PLAN = ROOT/'configs/experiments/getup_early_support_ppo_evaluation.json'
BASES = (
    ROOT/'records/reports/getup-local-ppo-20261110T011523/evaluation-analysis.json',
    ROOT/'records/reports/getup-pose-routing/seed1062-analysis.json',
    ROOT/'records/reports/getup-pose-routing/seed1063-analysis.json',
)
RETENTION = ROOT/'records/reports/getup-recovery-retention'
PARENT = 'fixed-weighted'
AGE = (100,110,120,130,140,149)


def branch(path,role):
    proof = read(path)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used'] and proof['learning_updates']==0
    record = ROOT/proof['record']
    spec = read(record/'protocol.json')
    arm = proof['arms'][role]
    trace = record/f'trace-{role}.npz'
    assert sha(trace)==arm['trace_sha256']
    episodes = arm['timing']['episodes']
    assert [row['env_index'] for row in episodes]==list(range(len(episodes)))
    requested = np.array([row['requested_pose'] for row in episodes])
    actual = np.array([row['release_pose'] for row in episodes])
    success = np.array([row['hold_start_seconds'] is not None for row in episodes])&np.array(arm['end_hold']['stable_by_environment'])&(actual!='other')
    assert int(success.sum())==arm['combined_successes']
    return spec,trace,requested,actual,success


def predict(session,observations):
    return np.concatenate([session.run(None,{'obs':sample[None]})[0] for sample in observations])


def main():
    target = OUT/'parent-retention-review.json'
    assert not target.exists()
    source = OUT/'analysis.json'
    evaluation = read(source)
    assert evaluation['protocol_sha256']==sha(PLAN)
    spec,trace,requested,actual,parent_success = branch(source,PARENT)
    _,_,other_requested,other_actual,candidate_success = branch(source,'candidate')
    np.testing.assert_array_equal(other_requested,requested)
    np.testing.assert_array_equal(other_actual,actual)
    on_side = np.isin(requested,('left-side','right-side'))
    supine = actual=='supine'
    lost = parent_success&~candidate_success
    assert lost.sum()==5 and (on_side&supine)[lost].all()
    replay = read(RETENTION/'replay-preflight.json')
    cells = RETENTION/'sampling-cells.json'
    pool = RETENTION/'parent-replay.npz'
    assert sha(cells)==replay['cells_sha256'] and sha(pool)==replay['data_sha256']
    groups = {key.split('|')[0] for key in read(cells)}
    assert groups=={'actual-right-side','requested-supine'}
    names = spec['model_interface']['joint_names'].split(',')
    assert len(names)==14
    pre_steps = np.arange(100,114)-spec['prefix_controls']
    with np.load(trace) as data:
        observations = data['actor_input'][pre_steps][:,supine].copy()
        expected = data['raw_policy_actions'][pre_steps][:,supine].copy()
        times = data['time'][pre_steps]
    assert observations.shape==(14,180,61)
    np.testing.assert_allclose(times,np.arange(101,115)*.02,atol=1e-6,rtol=0)
    policies = spec['policies']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    models = {}
    flat = observations.reshape(-1,61)
    for role in (PARENT,'control','candidate'):
        path = ROOT/policies[role]['model']
        assert sha(path)==policies[role]['model_sha256']
        session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        models[role] = predict(session,flat).reshape(expected.shape)
    np.testing.assert_allclose(models[PARENT],expected,atol=2e-5,rtol=2e-5)
    archive = []
    reference_session = ort.InferenceSession(str(ROOT/policies[PARENT]['model']),sess_options=options,
                                             providers=['CPUExecutionProvider'])
    for path in BASES:
        prior,branch_trace,nominal,release,success = branch(path,PARENT)
        assert prior['policies'][PARENT]['model_sha256']==policies[PARENT]['model_sha256']
        selected = dict(side_to_supine=success&(release=='supine')&np.isin(nominal,('left-side','right-side')),
                        requested_supine=success&(nominal=='supine'),actual_right=success&(release=='right-side'))
        with np.load(branch_trace) as data:
            idx = np.asarray(AGE)-prior['prefix_controls']
            counts = {}
            for key,mask in selected.items():
                ids = np.flatnonzero(mask)
                assert len(ids)>=8
                x = data['actor_input'][idx][:,ids].reshape(-1,61)
                y = data['raw_policy_actions'][idx][:,ids].reshape(-1,14)
                np.testing.assert_allclose(predict(reference_session,x[::13]),y[::13],atol=2e-5,rtol=2e-5)
                counts[key] = dict(worlds=len(ids),available_inputs=len(x))
        archive.append(dict(seed=prior['environment']['seed'],source=path.relative_to(ROOT).as_posix(),
                            source_sha256=sha(path),trace_sha256=sha(branch_trace),counts=counts))
    assert [row['seed'] for row in archive]==[1061,1062,1063]
    effects = {}
    for label,mask in dict(lost=lost,both_pass=parent_success&candidate_success,
                           new_success=~parent_success&candidate_success).items():
        local = mask[supine]
        if not local.any():
            continue
        group = dict(worlds=np.flatnonzero(mask&supine).tolist(),models={})
        for role in ('control','candidate'):
            delta = models[role][:,local]-models[PARENT][:,local]
            rms = np.sqrt((delta**2).mean(axis=(0,2)))
            axes = (delta.astype(float)**2).sum(axis=(0,1))
            order = np.argsort(-axes)[:3]
            group['models'][role] = dict(median_world_rms_rad=float(np.median(rms)),
                 maximum_abs_rad=float(abs(delta).max()),dominant_joints=[names[i] for i in order])
        effects[label] = group
    result = dict(status='passed-saved-parent-retention-review',evaluation_sha256=sha(source),source_sha256=sha(Path(__file__)),
        old_replay_groups=sorted(groups),old_side_to_supine_inputs=0,
        actual_supine=int(supine.sum()),side_to_supine=int((supine&on_side).sum()),
        parent_successes_lost_to_candidate=np.flatnonzero(lost).tolist(),
        same_parent_input_before_reward=effects,independent_archives=archive,
        evaluation_used_for_diagnosis_only=True,physical_substeps=0,learning_updates=0,heldout_used=False)
    write(target,result)
    print('PARENT_RETENTION_REVIEW',result['side_to_supine'],result['parent_successes_lost_to_candidate'],
          [(row['seed'],row['counts']['side_to_supine']['worlds']) for row in archive],effects,flush=True)


if __name__=='__main__':
    main()
