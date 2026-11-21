"""CPU counterfactual input sensitivity, separate from physical success.

Replay the fixed saved real probe, then zero only its angle feature. This
measures whether the exported actor uses phase at all. It does not predict
what a zero-angle controller would do in a fresh physical rollout.
"""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/roll-angle-feature/startup-recovery'


def main():
    rows={}
    for arm in ('control','candidate'):
        result=read(OUT/arm/'training-verification.json');assert result['status']=='passed'
        record=ROOT/'records/experiments'/result['run_id'];model=ROOT/result['model']
        assert sha(model)==result['model_sha256']
        with np.load(record/'phase-drift-probe.npz') as f:obs=f['observations'].copy()
        masked=obs.copy();masked[:,55]=0
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        predict=lambda a:np.concatenate([session.run(None,{'obs':x[None]})[0] for x in a])
        actual=predict(obs);zero=predict(masked);delta=actual-zero
        if arm=='control':np.testing.assert_array_equal(actual,zero)
        np.savez_compressed(OUT/arm/'phase-sensitivity.npz',observations=obs,actual_phase_actions=actual,zero_phase_actions=zero)
        rows[arm]=dict(model_sha256=sha(model),probe_sha256=sha(record/'phase-drift-probe.npz'),
            inputs=len(obs),angle_min_rad=float(obs[:,55].min()),angle_max_rad=float(obs[:,55].max()),
            action_difference_rms_rad=float(np.sqrt(np.mean(delta.astype(float)**2))),
            action_difference_max_rad=float(abs(delta).max()),
            archive_sha256=sha(OUT/arm/'phase-sensitivity.npz'))
    assert rows['control']['probe_sha256']==rows['candidate']['probe_sha256']
    result=dict(status='passed',arms=rows,physics_steps=0,optimizer_steps=0,
        scope='Counterfactual inference on fixed real observations only; no extra gait evaluation or improvement claim',
        script_sha256=sha(Path(__file__)))
    write(OUT/'phase-sensitivity.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
