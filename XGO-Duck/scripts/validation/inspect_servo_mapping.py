"""Audit the archived runtime mapping without serial, Bridge or robot access.

Run with the existing training Python (numpy and onnx already installed).
Checks execute the actual rl_core / wire / Policy math, not copied adapters:
signed one-joint observation/action probes, both home choices, packet slots,
mouth separation, and the existing filter slices. Firmware/JS arrays are parsed
but firmware is not compiled or executed. Results cannot certify physical IDs,
mechanical centers, directions, bus timing, or gait. No official files change.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

# Imported archive modules must not create cache files inside official_resources.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'official_resources/xgoduck_runtime_arduino'
OUTPUT = ROOT / 'records/hardware/servo-mapping'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def cpp_array(text, name):
    match = re.search(r'\b' + name + r'\[[^]]+\]\s*=\s*\{([^}]+)\}', text)
    if not match:
        raise ValueError(f'Cannot find {name}')
    return [int(v.strip()) for v in match[1].split(',') if v.strip()]


def main():
    core = module('rl_core', RUNTIME / 'python/rl_core.py')
    wire = module('wire', RUNTIME / 'python/wire.py')
    adapter = module('policy', RUNTIME / 'python/policy.py')
    header = (RUNTIME / 'sketch/duck_config.h').read_text()
    motors = cpp_array(header, 'MOTOR_IDS')
    signs = cpp_array(header, 'MOTOR_SIGN')
    js = (RUNTIME / 'assets/calibrate.js').read_text()
    ui_ids = json.loads(re.search(r'const MOTORS=(\[[^]]+\])', js)[1])
    assert motors == core.MOTOR_IDS == ui_ids, 'MCU / Python / UI slot mismatch'
    assert len(motors) == len(set(motors)) == 15
    assert len(core.POLICY_IDS) == len(set(core.POLICY_IDS)) == 14
    assert set(motors) - set(core.POLICY_IDS) == {34}
    assert len(signs) == 15 and all(v in (-1, 1) for v in signs)

    model_path = RUNTIME / 'python/xgoduck_walk.onnx'
    model = onnx.load(model_path)
    metadata = {m.key: m.value for m in model.metadata_props}
    names = metadata['joint_names'].split(',')
    model_home = np.rad2deg(np.array(metadata['default_joint_pos'].split(','), dtype=np.float32))
    assert len(names) == len(set(names)) == len(model_home) == 14
    assert np.isfinite(model_home).all()
    assert float(metadata['action_scale']) == core.ACTION_SCALE
    xml_path = ROOT / 'official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck/robot_walk.xml'
    xml_names = [element.attrib['joint'] for element in ET.parse(xml_path).getroot().find('actuator')]
    assert xml_names == names, 'ONNX joint names differ from archived walking actuator order'
    rows = []
    for policy_index, (servo_id, name) in enumerate(zip(core.POLICY_IDS, names)):
        slot = motors.index(servo_id)
        rows.append(dict(policy_index=policy_index, joint_name=name, servo_id=servo_id,
                         packet_slot=slot, firmware_sign=signs[slot],
                         runtime_home_deg=float(core.HOME_DEG[policy_index]),
                         model_home_deg=float(model_home[policy_index]),
                         actual_filter_alpha=None, physical_verification='pending'))

    # Both signs make a one-way reorder or accidental sign flip observable.
    # Encode through the real command packer, then inspect the 15 packet slots.
    probes = 0
    max_error = 0.0
    for home in (core.HOME_DEG, model_home):
        zero_targets = np.asarray(core.action_to_deg(np.zeros(14), home_deg=home))
        for i, slot in enumerate(core.POLICY_INDEX):
            for sign in (-1, 1):
                action = np.zeros(14, dtype=np.float32)
                action[i] = sign * 0.02
                targets = np.asarray(core.action_to_deg(action, home_deg=home))
                changed = np.flatnonzero(np.abs(targets - zero_targets) > 1e-5)
                assert changed.tolist() == [int(slot)]
                packet = wire.COMMAND.unpack(wire.encode_command(1, 0, wire.DISARM, targets.tolist()))
                np.testing.assert_allclose(packet[-15:], targets, atol=1e-6, rtol=0)
                velocities = np.zeros(15, dtype=np.float32)
                velocities[slot] = sign * 3.0
                state = core.State([0.0]*3, [0.0]*3, targets.tolist(), velocities.tolist())
                q, dq = core.policy_from_state(state, home_deg=home)
                expected_dq = np.zeros(14)
                expected_dq[i] = np.deg2rad(sign * 3.0)
                np.testing.assert_allclose(q, action, atol=2e-7, rtol=0)
                np.testing.assert_allclose(dq, expected_dq, atol=2e-7, rtol=0)
                obs = core.build_obs(state, action, np.zeros(3), np.zeros(4), home_deg=home)
                np.testing.assert_allclose(obs[6:20], action, atol=2e-7, rtol=0)
                np.testing.assert_allclose(obs[20:34], expected_dq, atol=2e-7, rtol=0)
                np.testing.assert_array_equal(obs[34:48], action)
                max_error = max(max_error, float(np.max(np.abs(q-action))))
                probes += 1
    for mouth in (-5.0, 12.0, 35.0):
        targets = np.asarray(core.action_to_deg(np.zeros(14), mouth_deg=mouth))
        np.testing.assert_allclose(targets[core.POLICY_INDEX], core.HOME_DEG, atol=1e-6)
        assert targets[core.MOUTH_INDEX] == float(np.clip(mouth, 0, core.MOUTH_MAX_DEG))

    # Isolate the real filter math from the neural net. This synthetic output is
    # not an ONNX or gait evaluation; it exposes each slot's actual filter alpha.
    class ProbeSession:
        def run(self, outputs, feed):
            return [np.full((1, 14), 0.02, dtype=np.float32)]

    policy = adapter.Policy()
    policy.session = ProbeSession()
    policy._outputs = ['actions']
    policy._feed = {'obs': policy._obs.reshape(1, 61)}
    filtered = policy.infer([0, 0, -1], [0, 0, 0], core.DEFAULT_DEG, [0]*15)
    for i, row in enumerate(rows):
        alpha = core.ACTION_ALPHA_HEAD if 5 <= i < 10 else core.ACTION_ALPHA
        np.testing.assert_allclose(filtered[i], 0.02*(1-alpha), atol=1e-7, rtol=0)
        row['actual_filter_alpha'] = alpha
    files = [RUNTIME / p for p in (
        'python/rl_core.py', 'python/policy.py', 'python/wire.py', 'python/controller.py',
        'sketch/duck_config.h', 'sketch/scs_bus.h', 'sketch/sketch.ino',
        'assets/calibrate.js', 'assets/servo.html')]
    files += [model_path, xml_path, Path(__file__).resolve()]
    result = dict(
        checked_at=datetime.now(timezone.utc).isoformat(), status='passed',
        scope='Offline archived software mapping checks only; no hardware or gait validation',
        hardware_verified=False, physical_changes_made=False,
        checks=dict(mcu_python_ui_order_equal=True, unique_ids=True,
                    onnx_xml_joint_order_equal=True, signed_joint_probes=probes,
                    home_choices_tested=2, max_action_roundtrip_error_rad=max_error,
                    mouth_probes=3, filter_slots_checked=14,
                    packet_target_slots_checked=True),
        findings=['ID 20 uses head alpha at policy index 9 in archived implementation',
                  'Walking runtime home differs from model metadata; keep both explicitly',
                  'MCU and calibration indices are packet slots, not policy indices'],
        motor_ids=motors, policy_ids=core.POLICY_IDS,
        mouth=dict(servo_id=34, packet_slot=core.MOUTH_INDEX, policy_index=None),
        mapping=rows,
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'mapping-check.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (OUTPUT / 'joint-mapping.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in ('status', 'scope', 'checks', 'findings')}, indent=2))


if __name__ == '__main__':
    main()
