"""Read-only UNO Q readiness capture: GET /api/status, never a control API.

Run visibly from VS Code after App Lab identifies the board and the user leaves
it in Inference only. Save original HTTP bodies, host timings and decoded data.
This verifies observable feedback/inference readiness, not mechanical zero,
physical torque, calibration, servo uniqueness on the bus, or walking quality.
Only Python's standard library is needed. No package install or service starts.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IDS = list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35))
MAX_FEEDBACK_AGE_MS = 150.0  # Same freshness boundary as the pinned controller.


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, 'Status capture does not follow redirects', headers, fp)


def status_url(base_url):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('Use the board origin shown by App Lab, e.g. http://BOARD-IP:9527; no credentials or path')
    parsed.port  # Validate malformed ports before any request.
    return urlunsplit((parsed.scheme, parsed.netloc, '/api/status', '', ''))


def request_status(opener, endpoint, timeout):
    request = Request(endpoint, method='GET', headers={'Accept': 'application/json'})
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('Unexpected status response larger than 1 MiB')
        return raw


def decode_status(raw):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    def parse_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite JSON float')
        return number
    decoded = json.loads(raw.decode('utf-8'), parse_constant=reject_constant, parse_float=parse_float)
    if not isinstance(decoded, dict):
        raise ValueError('Expected one status object')
    return decoded


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def check_sample(state):
    issues = []
    if state.get('mode') != 'shadow':
        issues.append('mode_is_not_inference_only')
    if state.get('enabled') is not False:
        issues.append('motor_disable_not_reported')
    if state.get('motors') != EXPECTED_IDS:
        issues.append('motor_order_differs_from_archived_runtime')
    ids = state.get('servo_ids')
    if not isinstance(ids, list) or len(ids) != 15 or any(type(x) is not int for x in ids) or set(ids) != set(EXPECTED_IDS):
        issues.append('expected_servo_ids_not_all_reported')
    if state.get('missing_joint_ids') != []:
        issues.append('missing_joint_ids_not_empty')
    for key in ('imu_ok', 'full_robot_ready', 'control_ready'):
        if state.get(key) is not True:
            issues.append(f'{key}_not_true')
    for key in ('mcu_watchdog', 'mcu_calibrating', 'mcu_servo_debug'):
        if state.get(key) is not False:
            issues.append(f'{key}_not_false')
    age = state.get('feedback_age_ms')
    if not finite(age) or not 0 <= age < MAX_FEEDBACK_AGE_MS:
        issues.append('feedback_stale_or_missing')
    for key, size in (('angles', 15), ('velocities', 15), ('targets', 15),
                      ('policy_targets', 15), ('acc', 3), ('gyro', 3), ('gravity', 3)):
        values = state.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != size or not all(finite(x) for x in values):
            issues.append(f'{key}_invalid')
    for key in ('received', 'inferred', 'sent', 'mcu_seq', 'bad_frames', 'sequence_gaps', 'loop_missed'):
        if type(state.get(key)) is not int or state[key] < 0:
            issues.append(f'{key}_invalid')
    if type(state.get('mcu_seq')) is int and state['mcu_seq'] >= 2**32:
        issues.append('mcu_seq_out_of_range')
    for key in ('uptime_s', 'feedback_hz', 'inference_hz'):
        if not finite(state.get(key)) or state[key] < 0:
            issues.append(f'{key}_invalid')
    if state.get('last_error') != '':
        issues.append('runtime_reports_error')
    return issues


def summarize(rows, expected_duration, *, source):
    issues = set()
    good = []
    for row in rows:
        if 'error' in row:
            issues.add('http_or_decode_error')
            continue
        state = row['state']
        issues.update(check_sample(state))
        good.append(state)
    if len(good) < 2:
        issues.add('too_few_status_samples')
    span = rows[-1]['finished_s']-rows[0]['started_s'] if rows else 0.0
    if span < expected_duration:
        issues.add('capture_incomplete')
    restarts = []
    duplicates = 0
    rates = {'received': [], 'inferred': [], 'sent': []}
    counter_increases = {key: 0 for key in ('bad_frames', 'sequence_gaps', 'loop_missed')}
    for index, (a, b) in enumerate(zip(good, good[1:]), 1):
        if not all(finite(s.get('uptime_s')) for s in (a, b)):
            continue
        dt = b['uptime_s']-a['uptime_s']
        reset = dt <= 0
        for key in (*rates, *counter_increases):
            if not all(type(s.get(key)) is int for s in (a, b)):
                continue
            delta = b[key]-a[key]
            reset = reset or delta < 0
        if all(type(s.get('mcu_seq')) is int for s in (a, b)):
            seq_delta = (b['mcu_seq']-a['mcu_seq']) % 2**32
            reset = reset or seq_delta > 2**31
            duplicates += seq_delta == 0
        if reset:
            restarts.append(index)
            continue
        for key in rates:
            if all(type(s.get(key)) is int for s in (a, b)):
                rates[key].append((b[key]-a[key])/dt)
        for key in counter_increases:
            if all(type(s.get(key)) is int for s in (a, b)):
                counter_increases[key] += b[key]-a[key]
    if restarts:
        issues.add('restart_or_nonmonotonic_state')
    if counter_increases['bad_frames'] or counter_increases['sequence_gaps']:
        issues.add('feedback_frame_errors_during_capture')
    for key in ('received', 'inferred'):
        if not rates[key] or max(rates[key]) <= 0:
            issues.add(f'{key}_counter_not_advancing')
    if len(good) >= 2 and duplicates == len(good)-1:
        issues.add('mcu_sequence_not_advancing')
    ages = [s['feedback_age_ms'] for s in good if finite(s.get('feedback_age_ms'))]
    # /api/status exposes rolling inference summaries. Do not average their
    # p99 values and mislabel the result as a capture-wide latency percentile.
    latency = [s.get('inference_ms', {}) for s in good]
    inference_p99 = [v['p99'] for v in latency if isinstance(v, dict) and finite(v.get('p99'))]
    ready = not issues
    return dict(status='observed-readiness-passed' if ready else 'needs-attention',
                evidence_source=source, telemetry_ready=ready,
                calibration_verified=False, gait_verified=False, physical_torque_verified=False,
                http_samples=len(rows), decoded_samples=len(good), captured_span_s=span,
                requested_span_s=expected_duration, issues=sorted(issues),
                duplicate_mcu_snapshots=duplicates, restart_sample_indices=restarts,
                median_counter_rates_hz={k: statistics.median(v) if v else None for k, v in rates.items()},
                counter_increases=counter_increases,
                max_feedback_age_ms=max(ages) if ages else None,
                max_reported_rolling_inference_p99_ms=max(inference_p99) if inference_p99 else None,
                gyro_units='degree/s in status; policy uses filtered rad/s',
                limitations=['Positive counters do not certify a sustained 50 Hz control budget',
                    'GET snapshots do not prove physical motor torque or mechanical zero',
                    'Reported IDs do not prove there are no duplicate IDs on a shared bus',
                    'Counter rates are inferred between HTTP snapshots, not raw per-frame timing',
                    'No automatic mode change, disarm, calibration, or control request is sent'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--duration', type=float, default=15)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--timeout', type=float, default=3)
    args = parser.parse_args()
    endpoint = status_url(args.base_url)
    if not (math.isfinite(args.duration) and 2 <= args.duration <= 60 and
            math.isfinite(args.interval) and 0.1 <= args.interval <= 2 and
            math.isfinite(args.timeout) and 0.5 <= args.timeout <= 10):
        parser.error('duration: 2–60 s; interval: 0.1–2 s; timeout: 0.5–10 s')
    output = ROOT/'records/hardware'/f'status_{datetime.now().strftime("%Y%m%dT%H%M%S_%f")}'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/'capture_board_status.py')
    # LAN requests go directly to the explicitly supplied board. This changes
    # no system proxy setting and does not send the request through node proxies.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    rows = []
    started = time.monotonic()
    interrupted = False
    print(f'READ ONLY: {endpoint}\nRecords: {output}\nKeep Inference only; no control requests will be sent.', flush=True)
    with (output/'samples.jsonl').open('w', encoding='utf-8', buffering=1) as log:
        try:
            while True:
                row = dict(started_s=time.monotonic()-started, utc=datetime.now(timezone.utc).isoformat())
                try:
                    raw = request_status(opener, endpoint, args.timeout)
                    # Preserve even a malformed JSON/UTF-8 body before parsing.
                    row['raw_body_base64'] = base64.b64encode(raw).decode('ascii')
                    state = decode_status(raw)
                    row['state'] = state
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                row['finished_s'] = time.monotonic()-started
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                if 'state' in row:
                    s = row['state']
                    reported = s.get('servo_ids')
                    count = len(reported) if isinstance(reported, list) else '?'
                    print(f"{row['finished_s']:.1f}s | mode={s.get('mode')} enabled={s.get('enabled')} IDs={count}/15 age_ms={s.get('feedback_age_ms')} RX={s.get('received')} inference={s.get('inferred')}", flush=True)
                    if s.get('mode') != 'shadow' or s.get('enabled') is True:
                        print('Observed mode is outside this static capture. Stopping read-only capture; inspect the board UI.', flush=True)
                        break
                else:
                    print(row['error'], flush=True)
                if row['finished_s'] >= args.duration:
                    break
                time.sleep(min(args.interval, args.duration-row['finished_s']))
        except KeyboardInterrupt:
            interrupted = True
    result = summarize(rows, args.duration, source='live-board-http')
    if interrupted:
        result.update(status='interrupted', telemetry_ready=False)
        result['issues'].append('capture_interrupted')
    result.update(endpoint=endpoint, started_at=rows[0]['utc'] if rows else None,
                  finished_at=datetime.now(timezone.utc).isoformat(), interrupted=interrupted,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  samples_sha256=hashlib.sha256((output/'samples.jsonl').read_bytes()).hexdigest(),
                  control_requests_sent=0, proxy_used=False, redirects_followed=False,
                  parameters=vars(args), deployed_code_or_model_hash_verified=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['telemetry_ready'] and not interrupted else 2)


if __name__ == '__main__':
    main()
