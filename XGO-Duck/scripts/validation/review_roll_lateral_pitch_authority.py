"""Check whether the recorded head pitch actuator can deliver a small braking request."""

from pathlib import Path

import numpy as np
from mjlab.scene import Scene

from record_io import write
from review_roll_active_lateral_signal import OUT, PLAN, ROOT, arrays, read, sha
from review_roll_head_motor_envelope import check_target_replay, reconstruct_internal_target
from run_continuous_roll_stand_walk import configuration


PITCH_DECREMENT_RAD = .02


def torque(motor, voltage, smooth, target, q, dq, dt):
    motor.vin = voltage
    motor.q_target_smooth = smooth.copy()
    output = motor.compute_control(target,q,dq,dt)
    return motor.compute_torque(output,True,q,dq), motor.q_target_smooth.copy()


def main():
    from bam.feetech.actuator import STS3215Actuator

    plan, analysis = read(PLAN), read(OUT/'analysis.json')
    geometry_path, signals_path = (OUT/name for name in ('paired-head-clearance.json','support-signal-review.json'))
    geometry, signals = read(geometry_path), read(signals_path)
    assert geometry['protocol_sha256'] == analysis['protocol_sha256'] == signals['protocol_sha256'] == sha(PLAN)
    assert geometry['physical_substeps'] == 0 and not analysis['heldout_used']
    folder = ROOT/analysis['record']
    cfg = configuration(plan)
    scene = Scene(cfg.scene,device='cpu')
    (actuator,) = scene['robot'].actuators
    motor = actuator._bam_model.actuator
    assert isinstance(motor,STS3215Actuator)
    (servo,) = cfg.scene.entities['robot'].articulation.actuators
    dt = cfg.sim.mujoco.timestep
    assert dt == .004 and cfg.decimation == 5 and servo.delay_min_lag == servo.delay_max_lag == 1
    model = scene.compile()
    joint = model.joint('robot/head_pitch')
    address, = joint.qposadr
    dof, = joint.dofadr
    column = plan['walking']['joint_names'].index('head_pitch')
    split = analysis['shared_controls']
    roll_start = plan['initial_stand_controls']
    sha_entries = analysis['source_sha256']
    prefix = arrays(folder/'shared-commands.npz',('applied_targets',),split,sha_entries)['applied_targets'][...,column:column+1]
    answers = {}
    for role, group in signals['groups'].items():
        events = group['events']
        last = max(event['first_airborne_control'] for event in events)+roll_start
        branch = folder/'clips'/role
        targets = arrays(branch/'commands.npz',('applied_targets',),last-split+1,sha_entries)['applied_targets'][...,column:column+1]
        # Index k represents the plant state just after k controls. The
        # first substep sees the preceding request; the second sees the new one.
        targets = np.concatenate((np.zeros_like(prefix[:1]),prefix,targets)).astype(np.float64)
        smooth = reconstruct_internal_target(targets, motor.model.max_velocity.value*dt)
        check_target_replay(targets,smooth,motor,dt)
        states = arrays(branch/'trace.npz',('qpos','qvel'),last-split+1,sha_entries)
        worlds = np.asarray([event['world'] for event in events])
        gap = np.asarray([event['first_airborne_control']+roll_start for event in events])
        by_lead = {}
        for lead in (3,1):
            k = gap-lead
            q = states['qpos'][k-split,worlds,address,None].astype(np.float64)
            dq = states['qvel'][k-split,worlds,dof,None].astype(np.float64)
            original = targets[k+1,worlds]
            adjusted = np.maximum(original-PITCH_DECREMENT_RAD,joint.range[0])
            assert np.all(adjusted <= original+1e-8)
            by_voltage = {}
            for voltage in (servo.vin_min,servo.vin_range[0]):
                _, first = torque(motor,voltage,smooth[k,worlds],targets[k,worlds],q,dq,dt)
                before,_ = torque(motor,voltage,first,original,q,dq,dt)
                after,_ = torque(motor,voltage,first,adjusted,q,dq,dt)
                gain = (before-after).ravel()
                assert (gain >= -1e-12).all()
                by_voltage[str(voltage)] = dict(cases=len(gain), increased_braking=int((gain > 1e-6).sum()),
                    median_change_nm=float(np.median(gain)), minimum_change_nm=float(gain.min()), changes_nm=gain.tolist())
            by_lead[str(lead)] = by_voltage
        answers[role] = by_lead
    target = OUT/'paired-head-pitch-authority.json'
    assert not target.exists()
    write(target,dict(status='verified-fixed-state-electrical-readback', protocol_sha256=sha(PLAN),
        geometry_sha256=sha(geometry_path),signal_sha256=sha(signals_path),script_sha256=sha(Path(__file__)),
        physical_substeps=0,heldout_used=False,head_pitch_target_decrement_rad=PITCH_DECREMENT_RAD,
        scope='Counterfactual motor torque at unchanged archived q and dq, after the one-substep modeled delay.',
        groups=answers))
    print('ROLL_PITCH_AUTHORITY', {role:{lead:{voltage:(row['increased_braking'],row['cases'],row['median_change_nm'])
        for voltage,row in values.items()} for lead,values in leads.items()} for role,leads in answers.items()},flush=True)


if __name__ == '__main__':
    main()
