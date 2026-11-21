"""Find a causal support window with both geometric and modeled motor leverage."""

from pathlib import Path

import numpy as np
from mjlab.scene import Scene

from record_io import write
from review_roll_active_lateral_signal import OUT, PLAN, ROOT, arrays, read, sha
from review_roll_head_motor_envelope import reconstruct_internal_target
from review_roll_head_support_geometry import Geometry
from run_continuous_roll_stand_walk import configuration


DIRECTIONS = dict(head_pitch=-.02, head_roll=-.02, neck_pitch=.02)
ENTRY_DEG = 120.


def angle_from_gravity(g):
    x,y,z = np.moveaxis(g,-1,0)
    lateral = np.sqrt(np.maximum(1.-y*y,0.))
    angle = np.rad2deg(np.unwrap(np.arctan2(x,-z/np.maximum(lateral,1e-6)),axis=0))
    return angle-angle[0]


def driver(motor, voltage, q, dq, initial, target):
    motor.vin = voltage
    motor.q_target_smooth = initial.copy()
    cmd = motor.compute_control(target,q,dq,.004)
    tau = motor.compute_torque(cmd,True,q,dq)
    saturated = np.isclose(abs(cmd),voltage*motor.max_pwm,atol=1e-9,rtol=0.)
    return tau, motor.q_target_smooth.copy(), saturated


def main():
    plan, analysis = read(PLAN), read(OUT/'analysis.json')
    source = OUT/'paired-head-pitch-authority.json'
    prior, signals = read(source), read(OUT/'support-signal-review.json')
    assert prior['protocol_sha256'] == analysis['protocol_sha256'] == signals['protocol_sha256'] == sha(PLAN)
    assert not prior['heldout_used'] and prior['physical_substeps'] == 0
    cfg = configuration(plan)
    scene = Scene(cfg.scene,device='cpu')
    (drive,) = scene['robot'].actuators
    (servo,) = cfg.scene.entities['robot'].articulation.actuators
    motor, model = drive._bam_model.actuator, scene.compile()
    geometry = Geometry(model)
    joints = [model.joint('robot/'+key).id for key in DIRECTIONS]
    addrs, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    lower, upper = model.jnt_range[joints].T
    cols = [plan['walking']['joint_names'].index(name) for name in DIRECTIONS]
    delta = np.asarray(list(DIRECTIONS.values()))
    split, start = analysis['shared_controls'], plan['initial_stand_controls']
    folder, hashes = ROOT/analysis['record'], analysis['source_sha256']
    prefix = arrays(folder/'shared-commands.npz',('observed','applied_targets'),split,hashes)
    results = {}
    for role, group in signals['groups'].items():
        ids = np.asarray([row['world'] for row in group['events']])
        gaps = start+np.asarray([row['first_airborne_control'] for row in group['events']])
        count = int(gaps.max())-split+1
        branch = folder/'clips'/role
        commands = arrays(branch/'commands.npz',('observed','applied_targets'),count,hashes)
        states = arrays(branch/'trace.npz',('qpos','qvel'),count,hashes)
        actor = np.concatenate((prefix['observed'][start:],commands['observed']))[...,3:6]
        angle = angle_from_gravity(actor)
        assert (angle[:,ids] >= ENTRY_DEG).any(axis=0).all()
        entry = start+np.argmax(angle[:,ids] >= ENTRY_DEG,axis=0)
        assert (entry >= split).all() and (gaps-entry >= 2).all()
        targets = np.concatenate((np.zeros_like(prefix['applied_targets'][:1]),prefix['applied_targets'],
            commands['applied_targets']))[...,cols].astype(np.float64)
        smooth = reconstruct_internal_target(targets,motor.model.max_velocity.value*.004)
        output = {}
        for offset in (0,1):
            k = entry+offset
            qpos, qvel = states['qpos'][k-split,ids], states['qvel'][k-split,ids]
            q, dq = qpos[:,addrs].astype(np.float64), qvel[:,dofs].astype(np.float64)
            original = targets[k+1,ids]
            request = np.clip(original+delta,lower,upper)
            gain_mm = np.empty_like(q)
            for i,(pose,velocity) in enumerate(zip(qpos,qvel,strict=True)):
                before = geometry.measure(pose,velocity)['clearance_mm']
                for j,addr in enumerate(addrs):
                    new = pose.copy()
                    new[addr] = np.clip(float(pose[addr])+delta[j],lower[j],upper[j])
                    gain_mm[i,j] = before-geometry.measure(new,velocity)['clearance_mm']
            voltages = {}
            for voltage in (servo.vin_min,servo.vin_range[0]):
                _, first, _ = driver(motor,voltage,q,dq,smooth[k,ids],targets[k,ids])
                a,smooth_a,saturation = driver(motor,voltage,q,dq,first,original)
                b,smooth_b,_ = driver(motor,voltage,q,dq,first,request)
                drive = (b-a)*np.sign(delta)
                assert (drive >= -1e-12).all()
                voltages[str(voltage)] = {name:dict(
                    geometric_lowered=int((gain_mm[:,j] > 0).sum()),
                    motor_authority=int((drive[:,j] > 1e-6).sum()),
                    both=int(((gain_mm[:,j] > 0) & (drive[:,j] > 1e-6)).sum()),
                    pwm_saturated=int(saturation[:,j].sum()),
                    unchanged_firmware_target=int((smooth_a[:,j] == smooth_b[:,j]).sum()),
                    median_clearance_reduction_mm=float(np.median(gain_mm[:,j])),
                    median_motor_change_nm=float(np.median(drive[:,j])))
                    for j,name in enumerate(DIRECTIONS)}
            output[str(offset)] = voltages
        results[role] = dict(worlds=ids.tolist(),minimum_lead_controls=int((gaps-entry).min()),
                             maximum_lead_controls=int((gaps-entry).max()),after_entry=output)
    target = OUT/'paired-support-actuator-window.json'
    assert not target.exists()
    write(target,dict(status='verified-causal-observation-and-fixed-state-leverage',protocol_sha256=sha(PLAN),
        source_sha256=sha(source),script_sha256=sha(Path(__file__)),heldout_used=False,physical_substeps=0,
        causal_projected_gravity_angle_deg=ENTRY_DEG,target_offsets_rad=DIRECTIONS,groups=results))
    print('ROLL_SUPPORT_ACTUATOR_WINDOW',{key:{name:row['both'] for name,row in value['after_entry']['0']['7.0'].items()}
        for key,value in results.items()},flush=True)


if __name__ == '__main__':
    main()
