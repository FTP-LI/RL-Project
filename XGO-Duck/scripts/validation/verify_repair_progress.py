"""Freeze and cross-check repair-pilot log, TensorBoard and timing prefixes.

The active writer can advance, so only complete updates common to all three
saved inputs are compared. This does not run simulation, learn, select a model,
or establish gait quality. Output also feeds the existing raw-curve plotter.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_matched_starts(folder):
    """Compare actual recorded starts before either gait result is selected.

    This is an additional CPU-only check. It does not alter the frozen pilot
    launcher or final verifier, and it never loads an environment or policy.
    Save the original YAML too: BaseLoader does not preserve function tags.
    """
    folder = folder.resolve()
    assert folder.parent == ROOT/'records/experiments'
    batch, protocol = read(folder/'batch.json'), read(folder/'protocol.json')
    assert [entry['arm'] for entry in batch['arms']] == ['control', 'target_envelope']
    snapshots = folder/'matched-start-inputs'
    snapshots.mkdir(exist_ok=True)
    evidence, comparisons, progress = {}, [], {}
    post = folder/'postprocessing.json'
    final = folder/'verification.json'
    completed = (post.is_file() and final.is_file()
                 and read(post)['status'] == 'completed' and read(final)['status'] == 'passed')
    # A finished experiment remains reproducible after new VS Code tasks or
    # observation code are added. Check its immutable source snapshots; report
    # working-tree drift separately. An active experiment still requires exact
    # working-tree hashes. Never overwrite the original start-check evidence.
    drift = {}
    original_check = folder/'matched-start-check.json'
    if completed and original_check.is_file():
        for name, digest in read(original_check)['inputs_sha256'].items():
            assert sha(ROOT/name) == digest, ('Historical input changed', name)
            evidence[name] = digest
    for entry in batch['arms']:
        arm, record = entry['arm'], Path(entry['record'])
        assert record.parent == ROOT/'records/experiments'
        request = read(record/'training-request.json')
        assert request['phase'] == 'repair-pilot' and request['arm'] == arm
        assert request['protocol_sha256'] == sha(folder/'protocol.json')
        assert (request['seed'], request['num_envs'], request['iterations']) == (42, 4096, 250)
        for key, digest in request['source_sha256'].items():
            assert sha(record/'sources'/f'{key}.txt') == digest, key
            current = Path(request['source_paths'][key])
            current_digest = sha(current) if current.is_file() else None
            if not completed:
                assert current_digest == digest, key
            if current_digest != digest:
                drift[key] = dict(path=str(current), recorded_sha256=digest,
                                  current_sha256=current_digest)
        location = read(record/'live-log-location.json')
        log_dir = Path(location['log_dir'])
        if not log_dir.is_absolute():
            log_dir = Path.home()/'XGO-Duck/workspace/xgoduck_rl'/log_dir
        paths = {name: record/name for name in
                 ('training-request.json', 'resume-plan.json', 'loaded-reset.json', 'first-action.json')}
        paths.update({f'{name}.yaml': log_dir/'params'/f'{name}.yaml' for name in ('env', 'agent')})
        for name, source in paths.items():
            target = snapshots/f'{arm}-{name}'
            raw = source.read_bytes()
            if target.exists():
                assert target.read_bytes() == raw, 'Immutable start input changed'
            else:
                target.write_bytes(raw)
            evidence[str(target.relative_to(ROOT))] = sha(target)
        config = {name: yaml.load(paths[f'{name}.yaml'].read_text(), Loader=yaml.BaseLoader)
                  for name in ('env', 'agent')}
        config['agent'].pop('run_name')
        functions = re.findall(r'!!python/name:([^\s]+)', paths['env.yaml'].read_text())
        if arm == 'target_envelope':
            term = config['env']['rewards'].pop('action_target_envelope')
            assert float(term['weight']) == protocol['target_envelope']['weight'] == -0.1
            assert float(term['params']['hip_roll_bound_deg']) == 10
            assert float(term['params']['hard_limit_margin_rad']) == 0.3
            function = 'gait_rewards.action_target_envelope_cost'
            assert functions.count(function) == 1
            functions.remove(function)
        else:
            assert 'action_target_envelope' not in config['env']['rewards']
        plan, loaded = read(paths['resume-plan.json']), read(paths['loaded-reset.json'])
        assert plan == request['resume_plan']
        assert sha(Path(plan['checkpoint'])) == plan['checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
        assert set(loaded['observation_sha256']) == {'actor', 'critic'}
        comparisons.append(dict(config=config, functions=functions, plan=plan, loaded=loaded,
                                first_action=read(paths['first-action.json']), sources=request['source_sha256']))
        progress[arm] = read(record/'progress.json')
    assert comparisons[0] == comparisons[1], 'Between-arm start/configuration mismatch'
    verifier = snapshots/'verify_repair_progress.py'
    if completed:
        verifier = snapshots/f'verify_repair_progress-{sha(Path(__file__))[:12]}.py'
    if verifier.exists():
        assert verifier.read_bytes() == Path(__file__).read_bytes(), 'Saved verifier differs'
    else:
        verifier.write_bytes(Path(__file__).read_bytes())
    evidence[str(verifier.relative_to(ROOT))] = sha(verifier)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(folder/'protocol.json'), source_sha256=sha(Path(__file__)),
                  matched_parent_and_restore_plan=True, matched_loaded_reset=True,
                  matched_initial_observations=True, matched_first_stochastic_action=True,
                  sole_reward_configuration_difference=True, frozen_source_hashes_match=True,
                  frozen_hash_scope='Archived sources; active runs additionally require unchanged working files',
                  completed_pilot=completed, current_sources_match_original=not drift,
                  current_source_differences=drift,
                  observation_sha256=comparisons[0]['loaded']['observation_sha256'],
                  first_action_sha256=comparisons[0]['first_action']['sha256'],
                  inputs_sha256=evidence, progress_snapshots=progress,
                  stable_gait_verified=False, final_budget_verified=False,
                  scope='Actual saved starts and configurations only; no simulation, learning or gait acceptance')
    output = folder/('matched-start-recheck.json' if completed else 'matched-start-check.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs_sha256'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', nargs='?')
    parser.add_argument('--require-finished', action='store_true')
    parser.add_argument('--compare-batch-starts', type=Path,
                        help='Verify recorded control/candidate starts without evaluating their policies')
    args = parser.parse_args()
    if bool(args.run_id) == bool(args.compare_batch_starts):
        parser.error('Provide one run_id or --compare-batch-starts, not both')
    if args.compare_batch_starts:
        if args.require_finished:
            parser.error('--require-finished applies to a single run_id')
        return verify_matched_starts(args.compare_batch_starts)
    record = ROOT/'records/experiments'/args.run_id
    assert record.parent == ROOT/'records/experiments'
    request, plan = read(record/'training-request.json'), read(record/'resume-plan.json')
    assert request['phase'] == 'repair-pilot' and request['iterations'] == 250
    assert request['resume_plan'] == plan
    assert request['protocol_sha256'] == sha(record/'protocol.json')
    for key, value in request['source_sha256'].items():
        assert sha(record/'sources'/f'{key}.txt') == value, key
    assert sha(Path(plan['checkpoint'])) == plan['checkpoint_sha256']
    correction = read(record/'loaded-reset.json')['correction']
    first_label=plan['next_iteration_label'];target_updates=plan['total_updates']
    assert correction['next_iteration_label'] == first_label == plan['completed_updates']
    assert correction['common_step_counter'] == plan['common_step_counter'] == first_label*plan['num_steps_per_env']
    assert correction['ppo_learning_rate_after'] == plan['restored_learning_rate']
    location = read(record/'live-log-location.json')
    log_dir = Path(location['log_dir'])
    if not log_dir.is_absolute():
        log_dir = Path.home()/'XGO-Duck/workspace/xgoduck_rl'/log_dir
    assert log_dir.name.endswith('_'+args.run_id)

    now = datetime.now(timezone.utc)
    out = record/'resume-verification'/now.strftime('%Y%m%dT%H%M%S%fZ')
    (out/'tensorboard').mkdir(parents=True)
    for event in log_dir.glob('events.out.tfevents.*'):
        (out/'tensorboard'/event.name).write_bytes(event.read_bytes())
    assert list((out/'tensorboard').iterdir())
    raw = (record/'training.log').read_bytes()
    (out/'training-prefix.log').write_bytes(raw)
    timing_bytes = (record/'iteration-timing.jsonl').read_bytes()
    (out/'iteration-timing.jsonl').write_bytes(timing_bytes[:timing_bytes.rfind(b'\n')+1])
    times = [json.loads(line) for line in (out/'iteration-timing.jsonl').read_text().splitlines()]
    timings = {item['iteration']: item for item in times}
    assert len(timings) == len(times)

    split = re.split(r'Learning iteration\s+(\d+)/(\d+)', raw.decode(errors='replace'))
    fields = {'collection': 'Collection time', 'learning': 'Learning time',
              'fps': 'Steps per second', 'reward': 'Mean reward',
              'length': 'Mean episode length', 'total_steps': 'Total steps'}
    parsed = {}
    for offset in range(1, len(split), 3):
        label, target, block = int(split[offset]), int(split[offset+1]), split[offset+2]
        if not re.search(r'ETA:\s+\S+', block):
            continue
        assert target == target_updates and label not in parsed
        values = {}
        for key, name in fields.items():
            match = re.search(re.escape(name)+r':\s+([-+\d.eE]+)', block)
            assert match, (label, name)
            values[key] = float(match[1])
            assert math.isfinite(values[key])
        assert values['total_steps'] == (label-first_label+1)*plan['num_envs']*plan['num_steps_per_env']
        parsed[label] = values
    assert parsed and sorted(parsed) == list(range(first_label, max(parsed)+1))

    ea = EventAccumulator(str(out/'tensorboard'), size_guidance={'scalars': 0}).Reload()
    tags = {'collection': 'Perf/collection_time', 'learning': 'Perf/learning_time',
            'fps': 'Perf/total_fps', 'reward': 'Train/mean_reward',
            'length': 'Train/mean_episode_length'}
    events = {key: {event.step: event.value for event in ea.Scalars(tag)} for key, tag in tags.items()}
    common = sorted(set(parsed).intersection(timings, *(set(items) for items in events.values())))
    assert common and common == list(range(first_label, common[-1]+1)) and len(common) <= plan['remaining_updates']
    for label in common:
        timing = timings[label]
        assert timing['new_updates'] == label-first_label+1
        assert all(math.isfinite(timing[key]) and timing[key] > 0 for key in
                   ('wall_seconds', 'collection_seconds', 'learning_seconds'))
        for key, values in events.items():
            tolerance = .00051 if key in ('collection', 'learning') else .01
            assert math.isfinite(values[label]) and abs(parsed[label][key]-values[label]) <= tolerance
            if key in ('collection', 'learning'):
                assert abs(timing[key+'_seconds']-values[label]) < 5e-6
    active = []
    proc = Path('/proc')/str(location['pid'])/'cmdline'
    if proc.exists():
        parts = proc.read_bytes().decode().rstrip('\0').split('\0')
        if '--worker' in parts and str(record) in parts:
            active = [location['pid']]
    if args.require_finished:
        result = read(record/'training-result.json')
        assert result['status'] == 'training-finished' and result['exit_code'] == 0
        assert len(common) == 250 and not active
    (out/'verify_repair_progress.py').write_bytes(Path(__file__).read_bytes())
    inputs = {str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file()}
    result = dict(status='passed', checked_at=now.isoformat(), run_id=args.run_id, arm=request['arm'],
        evidence_directory=str(out), first_iteration=first_label, last_verified_iteration=common[-1],
        new_updates_verified=len(common), cumulative_completed_updates=first_label+len(common),
        target_updates=target_updates, configured_remaining_updates=plan['remaining_updates'], actual_rollout_verified=True,
        loaded_reset_verified=True, active_training_pids=active, exact_trajectory_restart=False,
        first_update=parsed[common[0]], last_update=parsed[common[-1]],
        input_hashes=inputs, source_sha256=sha(Path(__file__)),
        log_tensorboard_and_timing_match=True, finished_log_prefix_verified=args.require_finished,
        final_budget_verified=False, stable_gait_verified=False,
        scope='Saved log/TensorBoard/timing prefix; final checkpoint and fixed gait require their own checks')
    (out/'check.json').write_text(json.dumps(result, indent=2)+'\n')
    (record/'resume-live-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
