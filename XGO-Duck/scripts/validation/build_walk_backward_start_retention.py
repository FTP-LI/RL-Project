"""Read balanced six-command teacher anchors from verified development traces."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from s1_reference_inputs import ROOT, read, sha
from record_io import write


PLAN = ROOT/'configs/experiments/common_walk_backward_start_retention.json'
OUT = ROOT/'records/reports/common-walk-backward-start-curriculum/retention'
KEYS = ('observations','actions','active','joint_envelope_ok')


def fields(path, digest):
    assert sha(path)==digest
    with np.load(path) as source:
        return {k:source[k] for k in KEYS}


def cohort(plan, relative, digest):
    path = ROOT/relative
    assert sha(path)==digest
    evidence = read(path)
    assert evidence['status']=='passed-integrity' and not evidence['heldout_used'] and not evidence['learning_updates']
    record = ROOT/evidence['record']
    if not record.is_dir():
        record = ROOT/'records/training-archive/evaluations'/record.name
    protocol = record/'protocol.json'
    assert sha(protocol)==evidence['protocol_sha256']
    options = read(protocol)
    p = options['environment']
    model = ROOT/p['model']
    assert sha(model)==p['model_sha256']
    role = plan['reference_role']
    prefix = fields(record/'trace-shared-prefix.npz',evidence['shared_prefix_sha256'])
    suffix = fields(record/f'trace-{role}.npz',evidence['trace_sha256'][role])
    trace = {k:np.concatenate((prefix[k],suffix[k])) for k in KEYS}
    h,n = trace['active'].shape
    assert (h,n)==(p['maximum_control_steps'],p['parallel_worlds'])
    assert trace['observations'].shape==(h,n,61) and trace['actions'].shape==(h,n,14)
    assert set(plan['retained_commands'])==set(p['groups'])-{'backward'}
    steps = np.asarray(plan['control_steps'],dtype=np.int64)
    rows = []
    for index,command in enumerate(plan['retained_commands']):
        group = np.asarray(p['groups'][command],dtype=np.int64)
        score = evidence['scores'][role][command]
        assert score['successes']==score['survivors']==score['episodes']==len(group)==plan['worlds_per_command_seed']
        idx = np.ix_(steps-1,group)
        assert trace['active'][idx].all() and trace['joint_envelope_ok'][idx].all()
        count = len(steps)*len(group)
        rows.append(dict(observations=trace['observations'][idx].reshape(count,61),
                         recorded_action=trace['actions'][idx].reshape(count,14),
                         command=np.full(count,index,dtype=np.int64),
                         seed=np.full(count,p['seed'],dtype=np.int64),
                         control=np.repeat(steps,len(group)),world=np.tile(group,len(steps))))
    source = dict(path=relative,sha256=digest,seed=p['seed'],branch=role,model_sha256=p['model_sha256'],
                  trace_sha256=evidence['trace_sha256'][role],prefix_sha256=evidence['shared_prefix_sha256'])
    return model,source,rows


def main():
    assert not OUT.exists()
    plan = read(PLAN)
    assert plan['status']=='frozen-offline-retention-only'
    assert not any(plan[key] for key in ('physical_substeps','learning_updates','heldout_used'))
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==plan['stage'] and not project['learning_running']
    selected = [cohort(plan,path,digest) for path,digest in plan['development_sources']]
    assert len({model for model,_,_ in selected})==1
    metadata = [proof for _,proof,_ in selected]
    assert len({row['seed'] for row in metadata})==len(metadata)
    rows = [row for _,_,groups in selected for row in groups]
    arrays = {key:np.concatenate([row[key] for row in rows]) for key in rows[0]}
    assert all(np.isfinite(value).all() for value in arrays.values())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    onnx = ort.InferenceSession(str(selected[0][0]),sess_options=options,providers=['CPUExecutionProvider'])
    target = np.concatenate([onnx.run(None,{'obs':row[None]})[0] for row in arrays['observations']])
    raw = arrays.pop('recorded_action')
    np.testing.assert_allclose(target,raw,atol=plan['onnx_atol'],rtol=plan['onnx_rtol'])
    counts = np.bincount(arrays['command'],minlength=len(plan['retained_commands']))
    assert np.all(counts==len(selected)*len(plan['control_steps'])*plan['worlds_per_command_seed'])
    OUT.mkdir()
    bank = OUT/'anchors.npz'
    np.savez_compressed(bank,**arrays,teacher_action=target)
    write(OUT/'analysis.json',dict(status='passed-six-command-retention-integrity',plan_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),bank_sha256=sha(bank),sources=metadata,
          samples_per_command=dict(zip(plan['retained_commands'],counts.tolist(),strict=True)),total_samples=len(target),
          maximum_onnx_replay_error=float(abs(target-raw).max()),physical_substeps=0,learning_updates=0,
          heldout_used=False,scope='Balanced official-policy targets for both PPO arms. Dynamic seven-command validation remains separate.'))
    print('BACKWARD_SIX_COMMAND_ANCHORS',len(target),counts.tolist(),float(abs(target-raw).max()),flush=True)


if __name__=='__main__':
    main()
