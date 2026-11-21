"""Bind saved model interfaces and unchanged simulation settings on CPU.

No policy inference, simulation integration or hardware access. Foot collision
masks are taken from an actual compiled prior diagnostic with identical source
hashes, not inferred from a zero collision counter in a rollout.
"""
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import yaml

from s1_reference_inputs import ROOT, WORK, OUT, read, save, sha


def main():
    exported = read(OUT/'progress.json'); assert exported['status'] == 'passed'
    original = ROOT/'records/evaluations/custom_onnx_reference_20261101T195044_s1001'
    diagnostic = ROOT/'records/evaluations/custom_onnx_mechanism_reference_20261101T171443_s1001'
    report, masks = read(original/'evaluation.json'), read(diagnostic/'evaluation.json')
    cfg = yaml.load((original/'env.yaml').read_text(), Loader=yaml.BaseLoader)
    assert masks['source_sha256'] == report['source_sha256']
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in [
        original/'evaluation.json', original/'env.yaml', diagnostic/'evaluation.json',
        OUT/'progress.json', Path(__file__)]}
    for name, expected in report['source_sha256'].items():
        assert sha(WORK/name) == expected, name
    actuator_file = WORK/'.venv/lib/python3.12/site-packages/mjlab/actuator/actuator.py'
    assert 'Minimum command delay in physics timesteps' in actuator_file.read_text()
    physics_dt = float(cfg['sim']['mujoco']['timestep']); decimation = int(cfg['decimation'])
    assert physics_dt*decimation == report['step_dt'] == .02
    robot = cfg['scene']['entities']['robot']
    act = robot['articulation']['actuators']; assert len(act)==1
    actuator = act[0]
    parameter_path = Path(actuator['json_path'])
    assert parameter_path.is_relative_to(WORK) and parameter_path.name=='1910_m6.json'
    assert all((x['contype'],x['conaffinity'])==(1,0) for x in masks['foot_collision_masks'])
    homes=[]
    for item in exported['models']:
        names=item['metadata']['joint_names'].split(',')
        nominal=[]
        for joint in names:
            matched=[float(v) for pattern,v in robot['init_state']['joint_pos'].items() if re.fullmatch(pattern,joint)]
            assert len(matched)==1;nominal.extend(matched)
        model_home=np.fromstring(item['metadata']['default_joint_pos'],sep=',')
        assert len(model_home)==len(nominal)==14
        homes.append(dict(role=item['role'],model_sha256=item['model_sha256'],joint_names=names,
            policy_home_rad=model_home.tolist(),simulation_home_rad=nominal,
            maximum_home_difference_deg=float(np.rad2deg(np.abs(model_home-nominal)).max())))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        training_started=False,simulation_started=False,hardware_started=False,
        models=homes,observation_layout=[['base_ang_vel',0,3],['projected_gravity',3,6],
          ['joint_pos_relative_to_policy_home',6,20],['joint_vel',20,34],['previous_raw_action',34,48],
          ['velocity_command',48,51],['head_command',51,55],['body_command',55,61]],
        action='14 raw network outputs; policy target = per-model home + action_scale * action; no runtime filter in fixed evaluation',
        timing=dict(physics_dt=physics_dt,decimation=decimation,control_dt=report['step_dt'],
          actuator_delay_physics_steps=[int(actuator['delay_min_lag']),int(actuator['delay_max_lag'])],
          actuator_delay_seconds=[physics_dt*int(actuator[k]) for k in ('delay_min_lag','delay_max_lag')]),
        actuator=dict(parameter_file=str(parameter_path),parameter_sha256=sha(parameter_path),kp_fw=float(actuator['kp_fw']),
          voltage_range=[float(x) for x in actuator['vin_range']],
          soft_joint_pos_limit_factor=float(robot['articulation']['soft_joint_pos_limit_factor']),
          delay_definition_source=str(actuator_file),delay_source_sha256=sha(actuator_file)),
        foot_collision_masks=masks['foot_collision_masks'],
        foot_pair_mask_contact_enabled=False,
        interpretation=[
          'Self-trained home differs from current official ONNX by about two degrees on hip pitch and ankles; evaluator converts both observations and targets explicitly.',
          'Metadata stiffness is not the BAM firmware gain; kp_fw=5 and the 1910 parameter model describe the simulated actuator.',
          'Foot-foot mask collision is disabled in this walk model; zero self-collision does not establish physically uncrossed feet.',
          'Physical dynamics, zero calibration and runtime filtering have not been validated on hardware.'],
        input_sha256=inputs)
    save('interface-contracts.json',result)
    print('S1A_CONTRACTS_PASSED: 61 inputs / 14 joints; control 20 ms; actuator delay 15-30 ms; foot-foot contact disabled')


if __name__=='__main__':main()
