"""Offline checks using the archived Controller.status schema and fake feedback.

No Arduino library, serial port, network connection, control loop or motor is
used. The actual Controller loads its ONNX sessions on CPU; all feedback and
timing are explicitly synthetic and cannot constitute board evidence.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT/'official_resources/xgoduck_runtime_arduino/python'
sys.path.insert(0, str(RUNTIME))
from controller import Controller
from wire import Feedback, ALL_MASK
from capture_board_status import EXPECTED_IDS, NoRedirect, check_sample, decode_status, request_status, status_url, summarize


class NoIOBridge:
    def __getattr__(self, method):
        raise AssertionError(f'Offline status inspection attempted Bridge I/O: {method}')


def main():
    source_files = [Path(__file__), Path(__file__).with_name('capture_board_status.py'),
                    RUNTIME/'controller.py', RUNTIME/'main.py', RUNTIME/'wire.py',
                    RUNTIME/'policy.py', RUNTIME/'rl_core.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    controller = Controller(NoIOBridge())
    assert controller.mode == 'shadow'
    no_feedback = controller.status()
    assert 'feedback_stale_or_missing' in check_sample(no_feedback)
    baseline = []
    for i in range(3):
        now = 100.0+i
        controller.started = 100.0
        controller.rx_time = now-0.005
        controller.received = 1+100*i
        controller.inferred = 1+50*i
        controller.sent = 1+50*i
        controller.rx_times = deque([now-0.01, now])
        controller.infer_times = deque([now-0.02, now])
        controller.infer_ms = deque([2.0, 2.1])
        controller.filtered = {'grav': [0.0, 0.0, -1.0]}
        controller.feedback = Feedback(1, ALL_MASK, 1+100*i, int(now*1e6), int(now*1e6),
                                       1+50*i, 10000, 10000, 6000, 0, 0, 0,
                                       (0.0, 0.0, 9.8), (0.0, 0.0, 0.0),
                                       (0.0,)*15, (0.0,)*15, (0.0,)*15)
        with patch('controller.time.monotonic', return_value=now):
            state = json.loads(json.dumps(controller.status()))
        baseline.append(dict(started_s=float(i), finished_s=i+0.01, state=state))
    assert EXPECTED_IDS == baseline[0]['state']['motors']
    cases = {}
    good = summarize(baseline, 2, source='synthetic-offline-fixture')
    assert good['telemetry_ready'] and good['median_counter_rates_hz'] == {'received': 100.0, 'inferred': 50.0, 'sent': 50.0}
    cases['complete_synthetic_stream'] = good
    # Reproduce the documented subtlety using the actual status implementation:
    # control_ready stays true even when one feedback-mask bit is missing.
    controller.feedback = replace(controller.feedback, mask=ALL_MASK & ~(1 << 2))
    with patch('controller.time.monotonic', return_value=102.0):
        partial = json.loads(json.dumps(controller.status()))
    assert partial['control_ready'] is True and partial['full_robot_ready'] is False
    assert partial['missing_joint_ids'] == [12]
    assert 'expected_servo_ids_not_all_reported' in check_sample(partial)
    changed = copy.deepcopy(baseline)
    changed[-1]['state'] = partial
    cases['missing_id_while_control_ready'] = summarize(changed, 2, source='synthetic-offline-fixture')
    variants = {
        'stale_feedback': ('feedback_age_ms', 200.0),
        'unexpected_motor_enable': ('enabled', True),
        'wrong_mode': ('mode', 'policy'),
        'imu_offline': ('imu_ok', False),
        'nonfinite_angle': ('angles', [float('nan')]+[0.0]*14),
        'duplicate_reported_id': ('servo_ids', [10]+EXPECTED_IDS[:-1]),
        'runtime_error': ('last_error', 'fixture fault'),
        'runtime_restart': ('uptime_s', 0.1),
        'mcu_restart': ('mcu_seq', 2),
        'new_bad_frame': ('bad_frames', 1),
        'malformed_ids': ('servo_ids', None),
    }
    for name, (field, value) in variants.items():
        changed = copy.deepcopy(baseline)
        changed[-1]['state'][field] = value
        result = summarize(changed, 2, source='synthetic-offline-fixture')
        assert not result['telemetry_ready'], name
        # Do not persist the injected NaN itself; record the resulting issues.
        cases[name] = result
    for name, field in (('stalled_inference', 'inferred'), ('stalled_mcu_sequence', 'mcu_seq')):
        changed = copy.deepcopy(baseline)
        for row in changed:
            row['state'][field] = 1
        cases[name] = summarize(changed, 2, source='synthetic-offline-fixture')
        assert not cases[name]['telemetry_ready']
    wrapped = copy.deepcopy(baseline)
    for row, seq in zip(wrapped, (2**32-101, 2**32-1, 99)):
        row['state']['mcu_seq'] = seq
    cases['normal_uint32_wrap'] = summarize(wrapped, 2, source='synthetic-offline-fixture')
    assert cases['normal_uint32_wrap']['telemetry_ready']
    cases['incomplete_capture'] = summarize(baseline[:1], 2, source='synthetic-offline-fixture')
    cases['http_failure'] = summarize([dict(started_s=0.0, finished_s=3.0, error='fixture timeout')], 2, source='synthetic-offline-fixture')
    assert not cases['incomplete_capture']['telemetry_ready'] and not cases['http_failure']['telemetry_ready']
    observed_requests = []
    body = json.dumps(baseline[0]['state']).encode()
    class FakeOpener:
        @contextmanager
        def open(self, request, timeout):
            observed_requests.append(dict(method=request.get_method(), url=request.full_url, body=request.data))
            yield io.BytesIO(body)
    endpoint = status_url('http://board.example:9527/')
    raw = request_status(FakeOpener(), endpoint, 3)
    assert raw == body and decode_status(raw) == baseline[0]['state']
    assert observed_requests == [dict(method='GET', url='http://board.example:9527/api/status', body=None)]
    bad_json = []
    for raw in (b'{"x": NaN}', b'{"x": 1e309}', b'[]', b'not json', b'\xff'):
        try:
            decode_status(raw)
        except (ValueError, UnicodeDecodeError):
            bad_json.append(raw.hex())
        else:
            raise AssertionError('Accepted a malformed status body')
    try:
        NoRedirect().redirect_request(Request(endpoint), None, 302, '', {}, 'http://elsewhere')
    except HTTPError:
        pass
    else:
        raise AssertionError('Unexpected redirect')
    for source in source_files:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[str(source.relative_to(ROOT))]
    output = ROOT/'records/hardware/status-tool'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  scope='Offline synthetic status-schema/parser checks only',
                  board_contacted=False, hardware_verified=False, calibration_verified=False,
                  network_requests_sent=0, bridge_requests_sent=0,
                  cases=cases, rejected_body_hex=bad_json,
                  observed_fake_request=observed_requests, source_sha256=hashes)
    (output/'software-check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], synthetic_cases=len(cases),
                         malformed_bodies_rejected=len(bad_json), board_contacted=False,
                         hardware_verified=False, control_ready_with_missing_id_reproduced=True), indent=2))
    print('Evidence:', output/'software-check.json')


if __name__ == '__main__':
    main()
