"""Resume the saved official baseline with an inspect-only native live display.

Change: a render observer and diagnostic logging, plus a documented checkpoint
restart. Preserve the exact remaining 7500-update budget, 4096 environments,
reward, physics, optimizer, normalization, curriculum and periodic recordings.
Expected effect: visible actual robot motion and evidence for leg adduction;
not a promised gait improvement. Rendering cost and reset boundary are recorded.
"""
import argparse
from pathlib import Path
import sys

from profile_components import read, sha, write
from resume_official import build_config, require_training_enabled, resumed_runner_class

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL = ROOT/'configs/experiments/live_training_resume.json'
POST_FILES = ('finish_baseline_extension.py', 'verify_resumed_progress.py',
    'verify_continuation_checkpoint.py', 'export_and_validate.sh',
    'prepare_policy_export.py', 'verify_export_trace.py', 'validate_export.py',
    'run_fixed_evaluation.sh', 'evaluate_fixed_policy.py', 'evaluation_metrics.py',
    'verify_fixed_evaluation.py')


def prepare(run_id):
    require_training_enabled()
    protocol = read(PROTOCOL)
    assert protocol['status'] == 'frozen-before-training'
    project = read(ROOT/'project.json')
    assert project['training_visualization']['switch_authorized'] is True
    parent = protocol['parent_run_id']
    record = ROOT/'records/experiments'/parent
    label = protocol['parent_checkpoint_label']
    audit = read(record/'checkpoint-checks'/f'model_{label}.json')
    assert audit['status'] == 'passed' and audit['saved_iteration'] == label
    assert (record/'exit-code.txt').read_text().strip() == '130'
    result = read(record/'training-result.json')
    assert result['status'] == 'interrupted' and result['exit_code'] == 130
    checkpoint = ROOT/audit['backup']
    assert sha(checkpoint) == audit['checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    request = read(record/'training-request.json')
    assert request['seed'] == 42 and request['num_envs'] == 4096
    assert sha(WORK/'uv.lock') == request['lock_sha256']
    assert sha(WORK/'src/mjlab_microduck/tasks/__init__.py') == request['task_source_sha256']
    completed = audit['cumulative_completed_updates']
    assert completed == label+1 and completed < protocol['total_updates'] == 7500
    assert audit['common_step_counter'] == completed*24
    assert audit['optimizer_steps_per_parameter'] == [completed*20]
    assert audit['normalizer_counts']['actor'] == audit['normalizer_counts']['critic']
    remaining = 7500-completed
    expected_norm = audit['normalizer_counts']['actor']+remaining*24*4096
    assert expected_norm == protocol['retained_total_transitions'] == 368640000
    plan = dict(task='Mjlab-Velocity-Flat-XgoDuck', parent_run_id=parent,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), seed=42,
        num_envs=4096, num_steps_per_env=24, saved_iteration_label=label,
        next_iteration_label=completed, completed_updates=completed,
        parent_completed_updates=completed, remaining_updates=remaining,
        total_updates=7500, final_iteration_label=7499,
        common_step_counter=audit['common_step_counter'],
        restored_learning_rate=audit['optimizer_learning_rates'][0],
        parent_observation_count=audit['normalizer_counts']['actor'],
        expected_final_observation_count=expected_norm,
        expected_final_optimizer_steps=150000, exact_trajectory_restart=False)
    cfg, matches = build_config(ROOT, WORK, plan, run_id)
    matches['only_declared_extension_changes'] = True
    sources = {'live_entry': Path(__file__),
        'live_observer': ROOT/'scripts/training/live_observer.py',
        'live_protocol': PROTOCOL, 'parent_checkpoint_audit': record/'checkpoint-checks'/f'model_{label}.json',
        'resume_config': ROOT/'scripts/training/resume_official.py',
        'resume_state': ROOT/'scripts/training/resume_state.py', 'lock': WORK/'uv.lock'}
    sources.update({'post_'+Path(name).stem:ROOT/'scripts/validation'/name for name in POST_FILES})
    evidence = dict(configuration_matches=matches, source_paths={k:str(v) for k,v in sources.items()},
        source_sha256={k:sha(v) for k,v in sources.items()}, plan=plan,
        simulation_created=False, cuda_initialized=False)
    import torch
    assert not torch.cuda.is_initialized()
    return cfg, protocol, plan, evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--record-dir',type=Path)
    args=parser.parse_args()
    cfg,protocol,plan,evidence=prepare(args.run_id or 'baseline_live_check_s42')
    if args.check_only:
        print('LIVE_RESUME_INPUTS_VERIFIED:',plan,flush=True)
        return
    assert args.run_id and args.record_dir
    record=args.record_dir.resolve()
    assert record==(ROOT/'records/experiments'/args.run_id).resolve()
    request=read(record/'training-request.json')
    assert request['phase']=='extension' and request['resume_plan']==plan
    assert request['extension_source_sha256']==evidence['source_sha256']
    write(record/'resume-plan.json',{**plan,'saved_configurations_match':evidence['configuration_matches']})
    from mjlab.scripts import train
    from live_observer import LiveObserver
    original=train.load_runner_cls
    resumed=resumed_runner_class(original(plan['task']),plan,record)

    class VisibleBaselineRunner(resumed):
        def learn(self,num_learning_iterations,init_at_random_ep_len=False):
            observer=LiveObserver(self.env,record,max_fps=protocol['max_wall_fps'])
            original_step=self.env.step
            def observed_step(actions):
                result=original_step(actions)
                observer.observe()
                return result
            try:
                observer.setup()
                observer.observe()  # Check actual state access before any new optimizer update.
                self.env.step=observed_step
                return super().learn(num_learning_iterations,init_at_random_ep_len)
            finally:
                self.env.step=original_step
                observer.close()

    train.load_runner_cls=lambda task:VisibleBaselineRunner if task==plan['task'] else original(task)
    try:
        train.launch_training(plan['task'],cfg)
    finally:
        train.load_runner_cls=original


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
