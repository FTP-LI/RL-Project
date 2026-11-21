"""Replay stored ONNX inputs and perturb commands at fixed observed states.

This diagnoses interface delivery and immediate network sensitivity only. It
does not predict the closed-loop result at larger commands, identify a training
cause, or alter any policy weights. No GPU simulation or robot connection.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--onnx', type=Path, help='Explicit model for non-default records; must match the saved SHA-256')
    args=p.parse_args()
    root=args.run.resolve()
    report=json.loads((root/'evaluation.json').read_text())
    project=Path(__file__).resolve().parents[2]
    # Use the copied project's model after migration, not the old WSL mount.
    # The recorded hash remains mandatory, including for an explicit override.
    model=(args.onnx or project/'official_resources/xgoduck_runtime_arduino/python/xgoduck_walk.onnx').resolve()
    assert hashlib.sha256(model.read_bytes()).hexdigest()==report['model_sha256'], 'Model hash does not match saved evaluation'
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    sess=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    name=sess.get_inputs()[0].name
    def infer(obs): return sess.run(None,{name:obs.astype(np.float32)})[0]
    data=np.load(root/'forward-trace.npz')
    probes=[]
    for step in (0,100,500):
        observation=data['observations'][step,0:1].copy()
        assert data['active'][step,0]
        assert np.allclose(observation[0,48:51],[0.15,0,0])
        replay=infer(observation)
        difference=float(np.max(np.abs(replay[0]-data['actions'][step,0])))
        assert difference<1e-6
        neutral=observation.copy()
        neutral[:,48:51]=0
        neutral_action=infer(neutral)
        responses=[]
        for command in ([0.15,0,0],[-0.1,0,0],[0.4,0,0],[0,0,0.4],[0,0,-0.4]):
            changed=neutral.copy()
            changed[:,48:51]=command
            action=infer(changed)
            responses.append(dict(command=command,max_action_delta_vs_stop=float(np.max(np.abs(action-neutral_action))),
                                  rms_action_delta_vs_stop=float(np.sqrt(np.mean(np.square(action-neutral_action))))))
        probes.append(dict(step_index=step,environment=0,replay_max_absolute_error=difference,responses=responses))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),status='passed',
                scope='Stored-input ONNX replay and fixed-state command sensitivity; no new simulation',
                model_sha256=report['model_sha256'],model_used=str(model),recorded_model_path=report['model'],source_trace_sha256=hashlib.sha256((root/'forward-trace.npz').read_bytes()).hexdigest(),
                probes=probes)
    (root/'command-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
