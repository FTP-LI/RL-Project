"""CPU timing of real saved observations: serial ONNX versus batched ONNX.

The original model stays byte-identical. Only batch dimension declarations
are changed in a separate diagnostic copy, never selected for deployment.
This isolates inference cost; it does not measure rendering or simulation.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    output=ROOT/'records/reports/evaluation-efficiency';output.mkdir(exist_ok=True)
    target=output/'inference-benchmark.json';assert not target.exists(),'Retain the recorded diagnostic'
    checked=read(ROOT/'records/reports/planar-filter/control/training-verification.json')
    progress=read(ROOT/'records/reports/planar-filter/control/evaluation-progress.json')
    assert progress['status']=='collected-and-verified'
    folder=Path(progress['evaluation']);verify=read(folder/'verification.json')
    assert verify['status']=='passed' and verify['report_sha256']==sha(folder/'evaluation.json')
    model=Path(checked['model']);original_hash=sha(model);assert original_hash==checked['model_sha256']
    source=folder/'forward-trace.npz'
    with np.load(source) as trace:obs=np.ascontiguousarray(trace['observations'][::10])
    assert obs.shape==(100,32,61) and np.isfinite(obs).all()
    derived=onnx.load(model)
    for value in (*derived.graph.input,*derived.graph.output):
        dimension=value.type.tensor_type.shape.dim[0]
        dimension.ClearField('dim_value');dimension.dim_param='diagnostic_batch'
    onnx.checker.check_model(derived)
    batch_model=output/'policy-batch-diagnostic.onnx';onnx.save(derived,batch_model)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    single=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    batch=ort.InferenceSession(str(batch_model),sess_options=options,providers=['CPUExecutionProvider'])
    infer=lambda session,x:session.run(None,{'obs':x})[0]
    for _ in range(3):infer(single,obs[0,:1]);infer(batch,obs[0])
    serial_times=[];batch_times=[];largest=0.
    # Alternate order across ten pairs to reduce monotonic thermal/order bias.
    for block in range(10):
        inputs=obs[block*10:(block+1)*10];results={}
        for mode in (('serial','batch') if block%2==0 else ('batch','serial')):
            begin=time.perf_counter()
            if mode=='serial':values=np.stack([np.concatenate([infer(single,row[None]) for row in step]) for step in inputs])
            else:values=np.stack([infer(batch,step) for step in inputs])
            elapsed=time.perf_counter()-begin
            (serial_times if mode=='serial' else batch_times).append(elapsed)
            results[mode]=values
        np.testing.assert_allclose(results['serial'],results['batch'],atol=2e-5,rtol=2e-5)
        largest=max(largest,float(abs(results['serial']-results['batch']).max()))
    assert sha(model)==original_hash
    original=read(folder/'evaluation.json')
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        original_model=str(model.relative_to(ROOT)),original_model_sha256=original_hash,original_unchanged=True,
        diagnostic_copy=str(batch_model.relative_to(ROOT)),diagnostic_sha256=sha(batch_model),
        trace=str(source.relative_to(ROOT)),trace_sha256=sha(source),frames=100,environments_per_frame=32,
        inputs=3200,parity_max_abs_error=largest,serial_seconds=serial_times,batch_seconds=batch_times,
        serial_seconds_per_1000_frames=float(sum(serial_times)*10),
        batch_seconds_per_1000_frames=float(sum(batch_times)*10),
        inference_speed_ratio=float(sum(serial_times)/sum(batch_times)),
        actual_original_evaluation_seconds={row['condition']:row['wall_seconds'] for row in original['conditions']},
        training_started=False,simulation_stepped=False,promoted_model=False,
        scope='One CPU replay under concurrent visible training; isolated inference timings cannot establish total evaluation speedup or rendering cost')
    write(target,result)
    print(json.dumps({k:result[k] for k in ('status','inputs','parity_max_abs_error','serial_seconds_per_1000_frames',
        'batch_seconds_per_1000_frames','inference_speed_ratio','actual_original_evaluation_seconds','scope')},indent=2))


if __name__=='__main__':main()
