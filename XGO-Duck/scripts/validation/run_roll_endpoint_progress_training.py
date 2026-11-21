"""Train the endpoint-reward candidate against a verified, matched prior run."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

import run_roll_ppo_capacity_live as live
from record_io import write


trial, bounded = live.trial, live.bounded
ROOT, P, read, put, rel, require, sha = trial.ROOT, trial.P, trial.read, trial.put, trial.rel, trial.require, trial.sha
KEY = 'roll_endpoint_progress_training'
STAGE = 'roll-endpoint-progress-training'
PLAN = ROOT / 'configs/experiments/roll_forward_input_integration/endpoint_progress_training.json'
REPORT = ROOT / 'records/reports/roll-forward-input-integration/endpoint-progress-training/candidate'
PREVIOUS = ROOT / 'records/reports/roll-friction-paired-training'
ENTRY = Path(__file__).resolve()


def command():
    return [sys.executable, '-B', '-u', str(ENTRY), 'worker']


def verify_sources(plan):
    for base, key in ((ROOT, 'source_sha256'), (Path('/'), 'installed_sha256')):
        require(all(sha(base / filename) == value for filename, value in plan[key].items()), 'Frozen training sources changed')


def freeze():
    project = read(P)
    task, state = project['staged_training_request'], project['goal_execution'][KEY]
    require(trial.allowed(project, STAGE+'-preflight') and state['status']=='passed-cuda-fixed-policy'
            and not task['learning_running'] and not task['simulation_running'] and not PLAN.exists() and not REPORT.exists(),
            'The CUDA reward check or project training permission is incomplete')
    proof_path, draft_path = (ROOT / state[key] for key in ('cuda_check_result', 'draft'))
    proof, draft = read(proof_path), read(draft_path)
    require(sha(proof_path)==state['cuda_check_result_sha256'] and sha(draft_path)==state['draft_sha256']
            and proof['auto_resets']==128 and proof['learning_updates']==0 and max(proof['maximum_errors'].values())<=2e-5
            and proof['reward_changes_over_tolerance']>0 and proof['physical_transitions']==46080, 'CUDA evidence differs')
    inherited = ROOT/draft['paired_learning']['inherit_parent_optimizer_and_safety_checks_from']
    plan = read(inherited)
    require(sha(inherited)==draft['reviewed_source_sha256'][rel(inherited)], 'Original training plan changed')
    control = plan['arms']['low-friction']
    setup = PREVIOUS / 'reset-admission/low-friction/startup-inputs.npz'
    reset = PREVIOUS / 'reset-admission/paired-check.json'
    evidence, export = (PREVIOUS/'low-friction'/name for name in ('independent.json', 'export.json'))
    observed = read(evidence)
    checkpoint = ROOT/'records/experiments'/control['run_id']/'profile-final.pt'
    onnx = ROOT/'artifacts'/control['run_id']/'policy.onnx'
    require(observed['status']=='passed-paired-training-readback' and observed['run_id']==control['run_id']
            and sha(setup)==read(reset)['input_sha256']['low-friction']==observed['reset_input_sha256']
            and sha(checkpoint)==observed['checkpoint_sha256']==read(export)['checkpoint_sha256']
            and sha(onnx)==read(export)['model_sha256'] and read(export)['metadata_preserved']
            and (control['environments'],control['updates'],control['transitions'])==(4608,50,5529600)
            and control['friction_role']==draft['paired_learning']['shared_friction_sampling']
            and plan['recipe']['seed']==draft['paired_learning']['training_seed'], 'Matched original-reward control differs')
    sources = (ENTRY, proof_path, draft_path, inherited, setup, reset, evidence, export, checkpoint, onnx,
               ROOT/plan['recipe']['parent_checkpoint'], ROOT/plan['recipe']['official_onnx'],
               ROOT/'scripts/training/roll_endpoint_progress_ppo.py', ROOT/'scripts/training/roll_endpoint_progress.py',
               ROOT/'scripts/training/roll_endpoint_reward.py', ROOT/'scripts/validation/roll_contact_endpoint.py')
    run_id = 'duck_roll_endpoint_progress_e4608_s1246_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    arm = dict(control, run_id=run_id, progress_contact='endpoint')
    plan.update(stage=STAGE, arms={'candidate':arm}, reward_change=draft['change'],
                source_sha256=plan['source_sha256'] | {rel(path):sha(path) for path in sources},
                supervisor_workers={run_id:command()}, inherited_run_role='low-friction',
                matched_archived_control=dict(control, setup=rel(setup), checkpoint=rel(checkpoint), model=rel(onnx),
                                              independent=rel(evidence), progress_contact='cached'),
                development_gate=draft['development_gate'])
    verify_sources(plan)
    REPORT.mkdir()
    put(PLAN, plan)
    state.update(status='training-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run_id, report=rel(REPORT))
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                current_protocol_execution='roll-endpoint-candidate-ready', current_evidence=rel(proof_path),
                next_gate='Complete the candidate with matched initialization and publish the artifacts, then evaluate it against two original-reward roll repeats.')
    write(P, project)
    print('ROLL_ENDPOINT_TRAINING_FROZEN', run_id, sha(PLAN), flush=True)


def checked(status):
    project = bounded.read_status(P, reader=read)
    state = project['goal_execution'][KEY]
    task = project['staged_training_request']
    require(trial.allowed(project, STAGE) and state['status']==status and state['protocol_sha256']==sha(PLAN)
            and task['current_protocol_frozen'] and task['current_protocol']==rel(PLAN)
            and not task['simulation_running'] and not task['learning_running'], 'Endpoint candidate is not ready for this step')
    plan = read(PLAN)
    verify_sources(plan)
    return project, plan, plan['arms']['candidate']


def supervise():
    _, plan, arm = checked('training-ready')
    timing = {name:value for name,value in plan['supervision'].items() if name!='viewing_seconds'}
    trial.supervisor.run_supervised(command(), cwd=ROOT, log_path=REPORT/'execution.log',
        receipt_path=REPORT/'supervisor-result.json', source=trial.source_for(arm), run_id=arm['run_id'],
        protocol=PLAN, protocol_sha256=sha(PLAN), stage=STAGE, **timing)
    receipt = read(REPORT/'supervisor-result.json')
    require(receipt['status']=='child-exited' and receipt['exit_code']==0, 'Candidate stopped. Review its execution log.')
    final = trial.finalize_view(REPORT, arm['run_id'], read(REPORT/'entry-result.json'))
    require(final['status']=='saved', 'Could not generate the final VS Code training view')
    print('ROLL_ENDPOINT_CANDIDATE_COLLECTED', arm['run_id'], flush=True)


def worker():
    _, plan, arm = checked('training-ready')
    ident, digest = arm['run_id'], sha(PLAN)
    record = ROOT/'records/experiments'/ident
    log = (REPORT/'execution.log').stat()
    require(all((log.st_dev,log.st_ino)==(os.fstat(fd).st_dev,os.fstat(fd).st_ino) for fd in (1,2))
            and trial.supervisor._authorized(PLAN,digest,STAGE,ident,command(),before_start=True),
            'Supervised, recorded candidate launch required')
    active = False

    def permitted():
        current = bounded.read_status(P, reader=read)
        task = current['staged_training_request']
        return (trial.allowed(current,STAGE) and task['learning_running'] is active and task['simulation_running'] is active
                and current['goal_execution'][KEY]['run_id']==ident
                and trial.supervisor._authorized(PLAN,digest,STAGE,ident,command(),before_start=False))

    def update(status):
        current = bounded.read_status(P, reader=read)
        current['staged_training_request'].update(learning_running=active,simulation_running=active,current_protocol_execution=status)
        current['goal_execution'][KEY].update(status=status,worker_pid=os.getpid() if active else None)
        write(P,current)

    monitor = live.SynchronousAuthorization(permitted)
    view = bounded.BoundedNativePPO(ident,REPORT,monitor.check,environments=arm['environments'],
        planned_updates=arm['updates'],purpose='低摩擦滚翻末端接触奖励候选')
    result = dict(status='candidate-stopped-review-required',run_id=ident,plan_sha256=digest,hardware_io=False)

    def terminate(_number,_frame):
        raise KeyboardInterrupt('Supervised training stop')

    try:
        signal.signal(signal.SIGTERM,terminate)
        view.start()
        require(view.check(force=True),'No healthy VS Code observation')
        put(REPORT/'live-admission.json',read(view.view/'native-health.json'))
        active = True
        update('candidate-training-running')
        record.mkdir()
        put(record/'protocol-arm.json',dict(arm,plan_sha256=digest,role='candidate'))
        from roll_endpoint_progress_ppo import collect

        run = collect(plan,arm,record,view,ROOT/plan['matched_archived_control']['setup'])
        result.update(status=run['status'],record=rel(record),online_result_sha256=sha(record/'online-result.json'))
    except BaseException as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        monitor.close()
        result['authorization_monitor'] = monitor.summary()
        active = False
        try:
            view.close(result)
        finally:
            put(REPORT/'entry-result.json',result)
            update('candidate-training-review' if result['status']=='collected-review-required' else result['status'])


def verify():
    import csv
    import math

    import imageio_ffmpeg
    import numpy as np
    import torch

    from export_roll_friction_pair import export
    from check_roll_friction_paired_reset import arrays
    from verify_roll_exposure_parent import finite

    project, plan, arm = checked('candidate-training-review')
    record = ROOT/'records/experiments'/arm['run_id']
    receipt, entry, final, live_view = (read(REPORT/name) for name in (
        'supervisor-result.json','entry-result.json','final-view.json','live-admission.json'))
    online, zero, observation = (read(record/name) for name in ('online-result.json','zero-step.json','observation.json'))
    require(receipt['exit_code']==0 and receipt['lock_acquired'] and receipt['lock_passed_to_child']
            and receipt['protocol_sha256']==entry['plan_sha256']==sha(PLAN) and receipt['log_sha256']==sha(REPORT/'execution.log')
            and final['status']=='saved' and final['source_sha256']==sha(record/'observation.json')
            and live_view['health']=='fresh' and live_view['run_id']==arm['run_id']
            and entry['online_result_sha256']==sha(record/'online-result.json')
            and entry['authorization_monitor']['failure'] is None and zero['controls']==zero['learning_updates']==0,
            'Incomplete training or live observer evidence')
    old, new = (arrays(path) for path in (ROOT/plan['matched_archived_control']['setup'], record/'startup-inputs.npz'))
    require(old.keys()==new.keys(),'Candidate initial fields changed')
    for key in old:
        np.testing.assert_array_equal(old[key],new[key],err_msg=key)
    rows = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    with (REPORT/'final-view/native-metrics.csv').open(encoding='utf-8',newline='') as stream:
        published = list(csv.DictReader(stream))
    require(len(rows)==len(published)==online['completed_updates']==arm['updates'] and rows==observation['rows']
            and online['transitions']==arm['transitions'] and online['actor_updates']==35 and online['physical_substeps']==6000
            and online['checkpoint_sha256']==sha(record/'profile-final.pt') and all(math.isfinite(row['environment_reward_mean']) for row in rows),
            'Candidate budget, recorded metrics or final checkpoint differ')
    guard = plan['training_checks']
    stability = [read(record/f'late-roll-control-{step}.json') for step in guard['late_control_steps']]
    policy_drift = [json.loads(line) for line in (record/'policy-drift.jsonl').read_text(encoding='utf-8').splitlines()]
    require(all(item['passed'] for item in stability) and len(policy_drift)==arm['updates']
            and all(item['within_limits'] for item in policy_drift),'Candidate drift or holding guard is incomplete')
    saved = {}
    for iteration in [*arm['extra_checkpoints'],arm['updates']]:
        path = record/('profile-final.pt' if iteration==arm['updates'] else f'checkpoint-update-{iteration}.pt')
        state = torch.load(path,map_location='cpu',weights_only=True)
        require(finite(state) and state['iter']+1==iteration
                and state['infos']['env_state']['common_step_counter']==iteration*plan['recipe']['rollout_controls'],
                'Checkpoint is not recoverable at its recorded update')
        saved[str(iteration)] = dict(path=rel(path),sha256=sha(path))
    require({int(group['step']) for group in state['optimizer_state_dict']['state'].values()}=={700,1000},
            'Final optimizer update accounting differs')
    clips = []
    for clip in online['clips']:
        path = record/clip['file']
        reader = imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        metadata = next(reader)
        frames = sum(1 for _ in reader)
        require(clip['full_action'] and clip['terminal_before_reset'] and frames==clip['frames']==176 and metadata['fps']==25,
                'Incomplete training episode video')
        clips.append(dict(path=rel(path),sha256=sha(path),frames=frames))
    require(len(clips)==2,'Expected two fully recorded roll episodes')
    output = ROOT/'artifacts'/arm['run_id']/'policy.onnx'
    output.parent.mkdir()
    model = export(plan,arm,record/'profile-final.pt',output)
    result = REPORT/'independent.json'
    put(result,dict(status='passed-training-and-onnx-readback',run_id=arm['run_id'],updates=arm['updates'],
        transitions=arm['transitions'],matched_startup_fields=sorted(old),historical_control=plan['matched_archived_control'],
        checkpoint=rel(record/'profile-final.pt'),model=rel(output),export=model, original_training=online,
        saved_checkpoints=saved,stability_guard=stability, independently_decoded_clips=clips,
        training_log=rel(REPORT/'execution.log'),action_gate_evaluated=False,hardware_io=False))
    project['goal_execution'][KEY].update(status='candidate-action-evaluation-pending',candidate_result=rel(result),
                                           candidate_result_sha256=sha(result),new_learning_updates=arm['updates'])
    project['staged_training_request'].update(current_evidence=rel(result),current_protocol_execution='candidate-action-evaluation-pending')
    write(P,project)
    print('ROLL_ENDPOINT_TRAINING_VERIFIED',arm['transitions'],model['model_sha256'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','supervise','worker','verify'))
    {'freeze':freeze,'supervise':supervise,'worker':worker,'verify':verify}[parser.parse_args().action]()
