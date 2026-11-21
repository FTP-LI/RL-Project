"""Check a saved prefix of a live continuation without stepping the simulator.

The log and TensorBoard may advance while inspected, so compare only complete
iterations present in both snapshots. This proves resumed rollout/update activity,
not the final training budget or stable gait. Each invocation saves its inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record = root/'records/experiments'/args.run_id
    assert record.parent == root/'records/experiments' and record.is_dir()
    request = json.loads((record/'training-request.json').read_text())
    plan = json.loads((record/'resume-plan.json').read_text())
    ready = json.loads((record/'resume-ready.json').read_text())
    assert request['phase'] in ('continuation', 'extension') and request['mode'] == 'baseline'
    expected_plan = {k: v for k, v in plan.items() if k != 'saved_configurations_match'}
    assert request['resume_plan'] == ready['plan'] == expected_plan
    expected_matches = {'env': True, 'agent': True}
    if request['phase'] == 'extension':
        expected_matches['only_declared_extension_changes'] = True
        for name, expected in request['extension_source_sha256'].items():
            assert digest(record/'sources'/f'{name}.txt') == expected
    assert plan['saved_configurations_match'] == expected_matches
    assert ready['status'] == 'loaded-and-reset'
    correction = ready['correction']
    for key in ('next_iteration_label', 'remaining_updates', 'final_iteration_label', 'common_step_counter'):
        assert correction[key] == plan[key]
    assert correction['ppo_learning_rate_after'] == plan['restored_learning_rate']
    assert plan['next_iteration_label'] + plan['remaining_updates'] == plan['total_updates']
    assert digest(Path(plan['checkpoint'])) == plan['checkpoint_sha256']
    for name, expected in request['resume_source_sha256'].items():
        assert digest(record/name) == expected
    work = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    assert digest(work/'uv.lock') == request['lock_sha256']
    assert digest(work/'src/mjlab_microduck/tasks/__init__.py') == request['task_source_sha256']
    log_dirs = list((work/'logs/rsl_rl/xgoduck_velocity').glob('*_'+args.run_id))
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    now = datetime.now(timezone.utc)
    output = record/'resume-verification'/now.strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    (output/'tensorboard').mkdir()
    for event in log_dir.glob('events.out.tfevents.*'):
        # Copy bytes once: the reader ignores an incomplete final TFRecord.
        (output/'tensorboard'/event.name).write_bytes(event.read_bytes())
    raw = (record/'training.log').read_bytes()
    (output/'training-prefix.log').write_bytes(raw)
    text = raw.decode(errors='replace')
    assert 'RESUME_READY:' in text
    split = re.split(r'Learning iteration\s+(\d+)/(\d+)', text)
    parsed = {}
    fields = {'collection': 'Collection time', 'learning': 'Learning time',
              'fps': 'Steps per second', 'reward': 'Mean reward',
              'length': 'Mean episode length', 'total_steps': 'Total steps'}
    for offset in range(1, len(split), 3):
        label, target, block = int(split[offset]), int(split[offset+1]), split[offset+2]
        if not re.search(r'ETA:\s+\S+', block):
            continue
        assert target == plan['total_updates'] and label not in parsed
        values = {}
        for key, name in fields.items():
            match = re.search(re.escape(name)+r':\s+([-+\d.eE]+)', block)
            assert match, (label, name)
            values[key] = float(match[1])
            assert math.isfinite(values[key])
        parsed[label] = values
    assert parsed, 'No complete rollout/update log block yet'
    labels = sorted(parsed)
    assert labels == list(range(plan['next_iteration_label'], labels[-1]+1))
    transitions = plan['num_steps_per_env'] * plan['num_envs']
    for label, values in parsed.items():
        # Upstream Total steps counts this invocation, not the parent run.
        assert values['total_steps'] == (label-plan['next_iteration_label']+1)*transitions
    ea = EventAccumulator(str(output/'tensorboard'), size_guidance={'scalars': 0}).Reload()
    tags = {'collection': 'Perf/collection_time', 'learning': 'Perf/learning_time',
            'fps': 'Perf/total_fps', 'reward': 'Train/mean_reward',
            'length': 'Train/mean_episode_length'}
    events = {key: {event.step: event for event in ea.Scalars(tag)} for key, tag in tags.items()}
    common = sorted(set(labels).intersection(*(set(items) for items in events.values())))
    assert common and common == list(range(plan['next_iteration_label'], common[-1]+1))
    for label in common:
        for key, items in events.items():
            exact = items[label].value
            assert math.isfinite(exact)
            tolerance = 0.00051 if key in ('collection', 'learning') else 0.01
            assert abs(parsed[label][key]-exact) <= tolerance, (label, key)
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(part.endswith((b'/resume_official.py', b'/extend_official.py', b'/live_resume.py')) for part in parts) and args.run_id.encode() in parts:
            if Path(parts[0].decode()).name.startswith('python'):
                active.append(int(proc.name))
    inputs = [path for path in output.rglob('*') if path.is_file()]
    result = dict(status='passed', checked_at=now.isoformat(), run_id=args.run_id,
                  command=[sys.executable, *sys.argv], log_dir=str(log_dir),
                  first_iteration=common[0], last_verified_iteration=common[-1],
                  new_updates_verified=len(common),
                  cumulative_completed_updates=plan['completed_updates']+len(common),
                  target_updates=plan['total_updates'], actual_rollout_verified=True,
                  loaded_reset_verified=True, configured_remaining_updates=plan['remaining_updates'],
                  exact_trajectory_restart=False, final_budget_verified=False, stable_gait_verified=False,
                  active_training_pids=active, saved_learning_rate=plan['restored_learning_rate'],
                  first_update=parsed[common[0]], last_update=parsed[common[-1]],
                  input_hashes={str(p.relative_to(output)):digest(p) for p in inputs},
                  scope='Saved log/TensorBoard prefix and reset/plan/source checks; no new simulation or optimizer step')
    (output/'check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (record/'resume-live-check.json').write_text(json.dumps({**result, 'evidence_directory':str(output)}, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','command')}, indent=2))


if __name__ == '__main__':
    main()
