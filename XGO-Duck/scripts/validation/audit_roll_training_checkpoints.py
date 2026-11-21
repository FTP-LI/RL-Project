"""Verify the earlier low-friction checkpoint without repeating training."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_roll_friction_pair import export
from review_roll_forward_input_stop import ROOT, read, sha
from record_io import write


def main():
    torch.set_num_threads(2)
    p = read(ROOT/'project.json')['goal_execution']['roll_friction_paired']
    protocol = ROOT/p['protocol']
    assert sha(protocol)==p['sha256']
    training = read(protocol)
    role = 'low-friction'
    arm = training['arms'][role]
    ident = arm['run_id']
    record = ROOT/'records/experiments'/ident
    final_proof = ROOT/p['reviews'][role]
    final_export = final_proof.with_name('export.json')
    proof, deployed = read(final_proof),read(final_export)
    actor_file = ROOT/deployed['model']
    assert proof['status']=='passed-paired-training-readback' and arm['extra_checkpoints']==[15,30]
    assert sha(actor_file)==deployed['model_sha256'] and sha(ROOT/deployed['checkpoint'])==proof['checkpoint_sha256']
    warm_path, candidate = (record/name for name in ('checkpoint-update-15.pt','checkpoint-update-30.pt'))
    parent_file = ROOT/training['recipe']['parent_checkpoint']
    warm, early, parent = (torch.load(item,map_location='cpu',weights_only=True) for item in (warm_path,candidate,parent_file))
    assert warm['iter']==14 and early['iter']==29 and parent['iter']==-1
    assert all(torch.equal(value,parent['actor_state_dict'][name]) for name,value in warm['actor_state_dict'].items())
    for epoch, checkpoint in ((15,warm),(30,early)):
        identity = checkpoint['infos']['roll_capacity']
        assert identity['run_id']==ident and identity['completed_updates']==epoch and identity['plan_scope']==training['scope']
    stats_path = record/'policy-drift.jsonl'
    import json

    logged = {row['update']:row for row in map(json.loads,stats_path.read_text(encoding='utf-8').splitlines())}
    expected = training['training_checks']
    data_path = ROOT/expected['drift_trace']
    assert sha(data_path)==expected['drift_trace_sha256']
    with np.load(data_path,allow_pickle=False) as data:
        observations = data['observations'][expected['drift_frames']].reshape(-1,61).copy()
    report = ROOT/p['report']/role/'checkpoint-comparison'
    assert not report.exists()
    report.mkdir()
    temporary = actor_file.with_name('policy-after-30-updates.partial.onnx')
    destination = actor_file.with_name('policy-after-30-updates.onnx')
    assert not (temporary.exists() or destination.exists())
    receipt = export(training,dict(arm,updates=30),candidate,temporary)
    temporary.replace(destination)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1

    def read_output(path):
        session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        return np.concatenate([session.run(None,{'obs':row[None]})[0] for row in observations])

    difference = read_output(destination)-read_output(actor_file)
    assert np.isfinite(difference).all() and abs(difference).max()>0 and not torch.cuda.is_initialized()
    result = report/'assessment.json'
    write(result,dict(training_protocol_sha256=sha(protocol),source_sha256={item.relative_to(ROOT).as_posix():sha(item)
        for item in (protocol,warm_path,candidate,parent_file,stats_path,final_proof,final_export,data_path,Path(__file__))},
        training_run_id=ident,earlier_checkpoint=candidate.relative_to(ROOT).as_posix(),
        exported_model=destination.relative_to(ROOT).as_posix(),export=receipt,final_model=deployed['model'],
        warmup_actor_equals_parent=True,old_action_drift={str(epoch):logged[epoch] for epoch in (15,30,50)},
        observations=len(observations),between_checkpoints_output_rms_rad=float(np.sqrt(np.mean(difference**2))),
        between_checkpoints_output_max_abs_rad=float(abs(difference).max()),
        physical_steps_added=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print('ROLL_EARLIER_CHECKPOINT_VERIFIED',sha(destination),len(observations),flush=True)


if __name__=='__main__':
    main()
