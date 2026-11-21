"""CPU numerical admission for future batched official-roll evaluation.

The old collector makes one ONNX call per world. Check a frozen imported
actor on512existing model-relative inputs before implementing the batched
collector. This is not a GPU speed benchmark, new episode or motion pass.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_release_coverage import actor_model
from getup_value_bootstrap import td


def main():
    draft=read(ROOT/'configs/experiments/roll_common_confirmation_draft.json')
    parent=ROOT/draft['parent_protocol'];assert sha(parent)==draft['parent_sha256']
    p=read(parent);assert sha(ROOT/p['model'])==p['model_sha256']==draft['model_sha256']
    state=read(ROOT/'records/reports/common-plant-screen/roll/reference-progress.json')
    assert state['status']=='completed'
    folder=Path(state['path']);source=folder/'forward-roll-trace.npz'
    original=read(ROOT/'records/reports/common-plant-screen/roll/reference-analysis.json')
    assert original['status']=='passed' and sha(source)==original['trace_sha256']
    assert original['model_sha256']==draft['model_sha256']
    with np.load(source) as f:
        observed=f['observations'];assert observed.shape==(451,32,61)
        selected=observed[[0,1,10,25,50,75,100,125,150,200,250,300,350,400,425,450]].reshape(512,61).copy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    torch.set_num_threads(2);actor=actor_model(dict(p,seed=1020),'cpu')
    before={k:v.clone() for k,v in actor.state_dict().items()}
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in selected])
    with torch.no_grad():batch=actor(td('actor',torch.from_numpy(selected))).numpy()
    np.testing.assert_allclose(batch,expected,atol=2e-5,rtol=2e-5)
    for key,v in before.items():torch.testing.assert_close(v,actor.state_dict()[key],atol=0,rtol=0)
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/roll-common-confirmation';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'cpu-import-inputs.npz',observations=selected,official=expected,batched=batch)
    result=dict(status='passed-cpu-only',actual_inputs=512,source=str(source.relative_to(ROOT)),
        source_sha256=sha(source),source_analysis_sha256=sha(ROOT/'records/reports/common-plant-screen/roll/reference-analysis.json'),
        draft_sha256=sha(ROOT/'configs/experiments/roll_common_confirmation_draft.json'),
        model_sha256=p['model_sha256'],maximum_error=float(abs(batch-expected).max()),
        actor_and_normalization_unchanged=True,cuda_initialized=False,physics_steps=0,optimizer_steps=0,
        gpu_parity_pending=True,dynamic_collector_pending=True,source_freeze_pending=True,
        script_sha256=sha(Path(__file__)),scope='CPU import numerical admission only; no speed or roll-function claim')
    write(out/'cpu-import-check.json',result);print('ROLL_BATCHED_CPU_IMPORT',result['maximum_error'],'512inputs; no physics')


if __name__=='__main__':main()
