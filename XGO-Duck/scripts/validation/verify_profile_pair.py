"""Recompute a completed diagnostic pair from logs, checkpoints and raw trace.

CPU-only verification; no simulation or optimizer step. The observed overhead
of one ordered pair is descriptive and cannot establish a training speedup.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def verify(pair):
    import torch
    import yaml
    torch.set_num_threads(1)
    records, captures, configurations = {}, {}, {}
    original_exit_codes, result_paths = {}, {}
    recovered_modes = []
    for mode in ('reference', 'traced'):
        folder = pair/mode
        code = (folder/'exit-code.txt').read_text().strip()
        original_exit_codes[mode] = int(code)
        result_path = folder/'run-result.json'
        if code != '0':
            # Only the documented saved-capture repair is accepted. Never
            # relabel the original process as successful or rerun its training.
            assert mode == 'traced' and code == '1'
            recovery = read(pair/'recovery/capture-recovery.json')
            assert recovery['status'] == 'reconstructed-awaiting-independent-check'
            assert not recovery['training_rerun'] and recovery['original_files_unchanged']
            for name, expected in recovery['input_sha256'].items():
                assert sha(pair/name) == expected, name
            result_path = pair/'recovery/run-result.json'
            recovered_modes.append(mode)
        result_paths[mode] = str(result_path.relative_to(pair))
        request, result = read(folder/'profile-request.json'), read(result_path)
        expected_status = 'completed-updates-recovered' if mode in recovered_modes else 'completed'
        assert result['status'] == expected_status and result['mode'] == mode
        assert request['mode'] == mode and request['configuration_matches'] == {'env': True, 'agent': True}
        assert not result['gait_selection_allowed'] and not result['formal_baseline_changed']
        for key, expected in request['source_sha256'].items():
            assert sha(folder/'sources'/f'{key}.txt') == expected, (mode, key)
        assert sha(Path(request['plan']['checkpoint'])) == request['plan']['checkpoint_sha256']
        rows = read(folder/'iteration-timing.json')
        assert [row['iteration'] for row in rows] == list(range(5000, 5027))
        assert all(math.isfinite(v) for row in rows for v in row.values())
        assert all(row['wall_seconds'] > 0 for row in rows)
        for previous, current in zip(rows, rows[1:]):
            assert abs(current['wall_seconds']-(current['end_monotonic_seconds']-previous['end_monotonic_seconds'])) < 1e-8
        selected = rows[22:27]
        captures[mode] = sum(row['wall_seconds'] for row in selected)
        assert abs(captures[mode]-(rows[26]['end_monotonic_seconds']-rows[21]['end_monotonic_seconds'])) < 1e-7
        checkpoint = Path(result['checkpoint'])
        assert sha(checkpoint) == result['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['iter'] == 5026 and saved['infos']['env_state']['common_step_counter'] == 5027*24
        assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()} == {5027*20}
        for actor in ('actor', 'critic'):
            assert int(saved[f'{actor}_state_dict']['obs_normalizer.count'].item()) == 5027*24*1024
        assert all(torch.isfinite(t).all().item() for t in tensors(saved))
        params = Path(result['log_dir'])/'params'
        configs = {name: yaml.load((params/f'{name}.yaml').read_text(), Loader=yaml.BaseLoader) for name in ('env','agent')}
        assert int(configs['agent']['max_iterations']) == 27 and int(configs['agent']['seed']) == 42
        assert int(configs['env']['scene']['num_envs']) == 1024
        assert configs['agent']['run_name'] == request['run_name']
        configs['agent'].pop('run_name')
        configurations[mode] = configs
        records[mode] = request
    assert configurations['reference'] == configurations['traced']
    assert records['reference']['source_sha256'] == records['traced']['source_sha256']
    assert records['reference']['plan'] == records['traced']['plan']
    traced = pair/'traced'
    summary_path = pair/'recovery/profile-summary.json' if 'traced' in recovered_modes else traced/'profile-summary.json'
    summary = read(summary_path)
    assert summary['status'] == 'passed' and summary['require_cuda'] and summary['cuda_event_count'] > 0
    assert sha(traced/'trace.json') == summary['trace_sha256']
    trace = read(traced/'trace.json')
    events = trace['traceEvents']
    # CUDA annotations can repeat CPU scope names on more than one stream.
    # Count the actual host record_function calls, not all identically named X events.
    calls = Counter(event.get('name') for event in events
                    if event.get('ph') == 'X' and event.get('cat') == 'user_annotation')
    expected = records['traced']['protocol']['expected_captured_calls']
    assert {name: calls[name] for name in expected} == summary['actual_calls'] == expected
    for row in summary['operators']:
        if row['name'].startswith('xgo/') and row['cpu_total_us'] > 0:
            raw = [e for e in events if e.get('ph') == 'X'
                   and e.get('cat') == 'user_annotation' and e.get('name') == row['name']]
            assert len(raw) == row['count']
            assert abs(sum(e['dur'] for e in raw)-row['cpu_total_us']) < .01
    gpu_events = [event for event in events if event.get('ph') == 'X' and event.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    assert gpu_events, 'Raw trace contains no recognized GPU activity; inspect it before claiming CUDA attribution'
    assert not torch.cuda.is_initialized()
    output = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  captured_labels=[5022,5026], captured_updates_per_mode=5,
                  captured_wall_seconds=captures,
                  observed_traced_to_reference_ratio=captures['traced']/captures['reference'],
                  observed_extra_time_fraction=captures['traced']/captures['reference']-1,
                  scope_call_counts={name: calls[name] for name in expected},
                  raw_gpu_activity_events=len(gpu_events), trace_sha256=summary['trace_sha256'],
                  configuration_pair_matches=True, input_checkpoint_sha256=records['traced']['plan']['checkpoint_sha256'],
                  cuda_initialized=False, controlled_speedup_verified=False, gait_selection_allowed=False,
                  original_exit_codes=original_exit_codes, recovered_modes=recovered_modes,
                  result_paths=result_paths, summary_path=str(summary_path.relative_to(pair)),
                  interpretation=records['traced']['protocol']['interpretation'])
    write_paths = [pair/'reference/profile-request.json', pair/result_paths['reference'],
                   pair/'reference/iteration-timing.json', pair/'traced/profile-request.json',
                   pair/result_paths['traced'], pair/'traced/iteration-timing.json', summary_path]
    if recovered_modes:
        write_paths.append(pair/'recovery/capture-recovery.json')
    output['input_sha256'] = {str(path.relative_to(pair)): sha(path) for path in write_paths}
    (pair/'verification.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    shutil.copy2(__file__, pair/Path(__file__).name)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    print(json.dumps(verify(parser.parse_args().pair.resolve()), indent=2))
