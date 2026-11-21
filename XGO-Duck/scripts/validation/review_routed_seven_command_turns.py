"""Separate turning startup, sustained bias and variable yaw on saved traces."""

import json
from hashlib import file_digest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/continuous_routed_seven_command.json'
OUT = ROOT/'records/reports/continuous-routed-seven-command'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as data:
        return file_digest(data, 'sha256').hexdigest()


def summarize(full, command, criteria, outcome):
    expected = abs(command)
    direction = full*np.sign(command)
    absolute = abs(direction-expected)
    original = (absolute.mean(axis=0) <= criteria['yaw_error_max']) & (
        direction.mean(axis=0) >= criteria['signed_speed'][outcome['walking']['condition']])
    episodes = outcome['walking']['episode_scores']
    np.testing.assert_array_equal(original, [row['success'] for row in episodes])
    # The frozen scorer sums the original float32 signed velocity.
    np.testing.assert_allclose(absolute.mean(axis=0), [row['yaw_error'] for row in episodes], atol=2e-6, rtol=0.)
    np.testing.assert_allclose(direction.mean(axis=0), [row['signed_speed'] for row in episodes], atol=2e-6, rtol=0.)
    failed = ~original
    start, tail = absolute[:50], absolute[50:]
    tail_rate = direction[50:].mean(axis=0)
    tail_limit = criteria['signed_speed'][outcome['walking']['condition']]
    tail_error = tail.mean(axis=0)
    steady_low = tail_rate < tail_limit
    steady_high = tail_rate-expected > criteria['yaw_error_max']
    tail_bad = tail_error > criteria['yaw_error_max']
    rows = [dict(world=episode['world'], original_pass=episode['success'], reasons=episode['failed_checks'],
        first_second_yaw_mae=float(start[:, i].mean()), later_yaw_mae=float(tail_error[i]),
        later_signed_yaw_rate=float(tail_rate[i]), later_signed_yaw_bias=float(tail_rate[i]-expected),
        seconds_over_yaw_gate=int((absolute[:, i].reshape(-1,50).mean(axis=1) > criteria['yaw_error_max']).sum()))
        for i, episode in enumerate(episodes)]
    return dict(entered=len(episodes), original_pass=int(original.sum()), original_fail=int(failed.sum()),
        failure_counts=outcome['failure_counts'],
        original_fail_also_late_yaw_error=int((failed & tail_bad).sum()),
        original_fail_passing_both_after_first_second=int((failed & ~tail_bad & ~steady_low).sum()),
        original_fail_late_speed_below_gate=int((failed & steady_low).sum()),
        original_fail_late_bias_alone_over_gate=int((failed & steady_high).sum()),
        original_fail_late_yaw_error_within_speed_and_mean_bias=int((failed & tail_bad & ~steady_low & ~steady_high).sum()),
        median_failed_yaw_mae_by_second=np.median(absolute[:, failed].reshape(-1,50,failed.sum()).mean(axis=1), axis=1).tolist(),
        rows=rows)


def main():
    spec, analysis = read(PLAN), read(OUT/'analysis.json')
    record = ROOT/spec['record']
    state = read(record/'evaluation.json')
    assert analysis['protocol_sha256'] == state['protocol_sha256'] == sha(PLAN)
    assert not analysis['score']['passed_development_gate'] and not analysis['heldout_used']
    groups = ('turn-left', 'turn-right')
    rows = analysis['score']['commands']
    worlds = {name:[row['world'] for row in rows[name]['walking']['episode_scores']] for name in groups}
    collected, hashes = {name:[] for name in groups}, {}
    command_at = spec['sequence']['boundary_controls'][-1]
    last = command_at
    for chunk in state['record_chunks']:
        if chunk['last'] <= command_at:
            continue
        path = record/chunk['file']
        assert chunk['first'] == last+1 and sha(path) == chunk['sha256']
        with np.load(path, allow_pickle=False) as data:
            yaw, modes = data['angular_velocity'][..., 2], data['mode']
        for name in groups:
            assert (modes[:, worlds[name]] == 2).all()
            collected[name].append(yaw[:, worlds[name]].astype(np.float64))
        last = chunk['last']
        hashes[chunk['file']] = chunk['sha256']
    assert last == spec['maximum_unique_controls']
    data = {name: summarize(np.concatenate(collected[name]), spec['seven_command']['commands'][name][2],
              spec['walking']['criteria'], rows[name]) for name in groups}
    with (OUT/'turn-failure-review.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(status='verified-turn-failure-readback', protocol_sha256=sha(PLAN),
            analysis_sha256=sha(OUT/'analysis.json'), trace_sha256=hashes, original_gates_unchanged=True,
            heldout_used=False, physical_substeps=0, later_window_is_diagnostic=True, groups=data), stream, indent=2)
        stream.write('\n')
    for name, group in data.items():
        print('TURN_FAILURE_READBACK', name, {key:value for key,value in group.items()
            if key not in ('rows', 'median_failed_yaw_mae_by_second')}, flush=True)


if __name__ == '__main__':
    main()
