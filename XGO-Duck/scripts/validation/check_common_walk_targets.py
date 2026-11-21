"""CPU real-class admission and saved-request projection decision, no rollout."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from common_walk_targets import processor_fixture

OUT=ROOT/'records/reports/common-walk-confirmation'


def main():
    assert not (OUT/'target-decision.json').exists() and not torch.cuda.is_initialized()
    audit=read(OUT/'interface-audit.json');assert audit['status']=='passed-cpu-saved-data'
    p=read(ROOT/'configs/experiments/common_plant/walk.json');p['model_home']=audit['model_home']
    fixture=processor_fixture(p);limits=np.asarray(p['joint_limits'],np.float32);rows={}
    for source,digest in audit['source_sha256'].items():
        path=ROOT/source;assert sha(path)==digest
        with np.load(path) as f:requested=f['requested_joint_targets']
        legal=np.clip(requested,limits[:,0],limits[:,1]);delta=legal-requested
        rows[path.name.replace('-trace.npz','')]=dict(
            joint_samples=int(requested.size),modified_joint_samples=int((abs(delta)>1e-6).sum()),
            modified_joint_fraction=float((abs(delta)>1e-6).mean()),
            target_change_rms_rad=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
            target_change_abs_max_rad=float(abs(delta).max()))
    write(OUT/'target-decision.json',dict(status='passed-cpu-only',interface_audit_sha256=sha(OUT/'interface-audit.json'),
        real_processor=fixture,steady_rule='Project modelHOME+raw to joint limits, without sustained recovery2rad/s cap',
        transition_rule='Actual model-change entry rate control is separate and not accepted by this test',
        preserve='Raw caller actions, action history, policy weights, controller, common noise and full physical plant',
        saved_request_review=rows,cuda_initialized=False,new_physics_steps=0,new_learning_steps=0,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'scripts/validation/common_walk_targets.py':sha(ROOT/'scripts/validation/common_walk_targets.py')},
        scope='Target arithmetic and processor contract; no assertion of changed physical outcomes'))
    print('COMMON_WALK_PROJECTION_CPU_PASSED',fixture,rows,flush=True)


if __name__=='__main__':main()
