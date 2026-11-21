"""Describe which initial-state and first-action features the reset bank changes."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/common-walk-backward-start-curriculum/reset-readback'
PLAN = ROOT/'configs/experiments/common_walk_backward_start_reset_readback.json'


def main():
    result = OUT/'release-variation.json'
    assert not result.exists()
    inputs,initial = {},None
    for device in ('cpu','cuda'):
        receipt = OUT/f'{device}.json'
        proof = read(receipt)
        assert proof['status']=='passed-zero-step-reset-readback' and proof['protocol_sha256']==sha(PLAN)
        assert not any(proof[k] for k in ('physical_substeps','learning_updates','heldout_used'))
        data = OUT/f'{device}-readback.npz'
        assert sha(data)==proof['readback_sha256']
        with np.load(data) as archive:
            qpos,qvel = archive['source_local_qpos'],archive['source_qvel']
            count = proof['source_case_count']
            assert len(qpos)==len(qvel)==count
            if initial is None:
                initial = (qpos,qvel)
            else:
                np.testing.assert_array_equal(qpos,initial[0])
                np.testing.assert_array_equal(qvel,initial[1])
            record = {}
            for field in ('observation','adapted','policy'):
                a,b = (archive[f'{role}_full_{field}'][:count] for role in ('control','candidate'))
                record[field] = dict(exact_rows=int((a==b).all(axis=1).sum()),
                                     maximum_difference=float(abs(a-b).max()))
            inputs[device] = dict(receipt_sha256=sha(receipt),differences=record)
    qpos,qvel = initial
    yaw = 2*np.arctan2(qpos[:,6],qpos[:,3])
    summary = dict(count=len(qpos),local_xy_span_m=np.ptp(qpos[:,:2],axis=0).tolist(),
                   height_span_m=float(np.ptp(qpos[:,2])),quaternion_roll_pitch_components_max=float(abs(qpos[:,4:6]).max()),
                   unique_joint_configurations=len(np.unique(qpos[:,7:],axis=0)),
                   maximum_absolute_generalized_velocity=float(abs(qvel).max()),unwrapped_yaw_span_rad=float(np.ptp(yaw)))
    write(result,dict(status='passed-reset-variation-review',protocol_sha256=sha(PLAN),
                      script_sha256=sha(Path(__file__)),source_initial=summary,first_frame=inputs,
                      physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_RESET_VARIATION',summary,inputs,flush=True)


if __name__=='__main__':
    main()
