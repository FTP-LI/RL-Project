"""CPU-only mapping preparation from pinned runtime source and model metadata.

Exercise permutation invariance, position/velocity sign and quantization with
synthetic data. No network, serial port, calibration write or motor enable.
Physical identity, mechanical zero and real IMU axes remain unverified.
"""
from pathlib import Path
import csv
import json
import re
import sys
import numpy as np
import onnx
from s1_reference_inputs import ROOT,read,sha

RUNTIME=ROOT/'workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2'
OUT=ROOT/'records/hardware/deployment-preparation'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(RUNTIME/'python'))
    import rl_core as runtime
    assert Path(runtime.__file__).resolve()==RUNTIME/'python/rl_core.py'
    model=RUNTIME/'python/xgoduck_walk.onnx'
    metadata={p.key:p.value for p in onnx.load(model).metadata_props}
    names=metadata['joint_names'].split(',');home=np.fromstring(metadata['default_joint_pos'],sep=',')
    assert len(names)==14 and len(set(runtime.MOTOR_IDS))==15 and len(set(runtime.POLICY_IDS))==14
    assert set(runtime.MOTOR_IDS)-set(runtime.POLICY_IDS)=={34}
    header=RUNTIME/'sketch/duck_config.h';text=header.read_text()
    signs=[float(v) for v in re.search(r'MOTOR_SIGN\[NUM_MOTORS\]\s*=\s*\{([^}]+)',text).group(1).split(',') if v.strip()]
    assert signs==[-1.]*15
    assert 'M_N = 4095.0f' in text and 'SERVO_CENTER_POS = 2047' in text and 'CAL_HOLD_POS = 2048' in text
    rows=[]
    for index,(name,id,default) in enumerate(zip(names,runtime.POLICY_IDS,home)):
        bus=runtime.MOTOR_IDS.index(id)
        rows.append(dict(policy_index=index,joint=name,expected_id=id,bus_index=bus,
            firmware_sign=signs[bus],policy_home_rad=default,policy_home_deg=float(np.rad2deg(default)),
            physical_id_verified='pending',mechanical_zero_verified='pending',direction_verified='pending'))
    target=ROOT/'configs/hardware/policy_joint_mapping.csv'
    with target.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rng=np.random.default_rng(413);max_error=0.;checks=0
    for _ in range(100):
        angles=rng.uniform(-45,45,size=15);velocities=rng.uniform(-100,100,size=15)
        order=rng.permutation(15)
        packet={runtime.MOTOR_IDS[i]:(angles[i],velocities[i]) for i in order}
        recovered=np.asarray([packet[id] for id in runtime.POLICY_IDS])
        state=runtime.State(acc=[0,0,0],gyro=[0,0,0],angles=angles.tolist(),vels=velocities.tolist())
        q,dq=runtime.policy_from_state(state,home_deg=np.rad2deg(home))
        np.testing.assert_allclose(q,np.deg2rad(recovered[:,0])-home,atol=2e-7,rtol=1e-6)
        np.testing.assert_allclose(dq,np.deg2rad(recovered[:,1]),atol=2e-7,rtol=1e-6)
        targets=np.asarray(runtime.action_to_deg(q,mouth_deg=12,home_deg=np.rad2deg(home)))
        np.testing.assert_allclose(targets[runtime.POLICY_INDEX],angles[runtime.POLICY_INDEX],atol=1e-5)
        assert targets[runtime.MOUTH_INDEX]==12
        # C++ conversion casts toward zero before adding the fixed center.
        raw=2047+np.trunc(targets/360*4095*np.asarray(signs)).astype(int)
        decoded=(raw-2047)*360/4095*np.asarray(signs)
        error=float(np.abs(decoded-targets).max());max_error=max(max_error,error)
        assert error<=360/4095+1e-8
        checks+=1
    imu=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    np.testing.assert_array_equal(imu@imu.T,np.eye(3));assert np.linalg.det(imu)==1
    result=dict(status='passed-offline-only',source_commit=RUNTIME.name,permuted_packets_checked=checks,
        joints=rows,mouth_id=34,mouth_is_policy_output=False,
        position_quantization_max_deg=max_error,position_steps_per_revolution=4095,
        velocity_steps_per_revolution=4096,velocity_source_unit_steps_per_second=50,
        center_conversion=2047,center_hold=2048,imu_source_to_body=imu.tolist(),
        physical_calibration_verified=False,physical_deployment_started=False,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (model,header,RUNTIME/'python/rl_core.py',RUNTIME/'sketch/scs_bus.h')},
        mapping_sha256=sha(target),script_sha256=sha(Path(__file__)))
    (OUT/'mapping-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('joints','input_sha256')},indent=2))


if __name__=='__main__':main()
