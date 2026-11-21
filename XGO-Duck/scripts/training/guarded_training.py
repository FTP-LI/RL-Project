"""Single visible entry for a frozen bounded walk-repair protocol.

What changed: wall-time snapshots, sampled gait tripwires, checkpoint-only
saves, explicit parent and one arm per invocation. Why: continuous rendering
and per-save ONNX exports waste work; silent divergence must stop the run.
Expectation: less observer work and less lost progress, NOT assumed faster
learning or better gait. Validate overhead on an equal-parent short run later.
Recovery and roll entries are intentionally not routed through walking rules.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import threading
import time

from training_watch import (ROOT, CONFIG, read, write, run_name, GaitWatch,
                            signed_foot_gap_mm, UpdateBoundary, Snapshot, SparseVideo,
                            ControlledStop, UnsafeState)

WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PLAN = ROOT/'configs/experiments/next_training.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def require_protocol(plan):
    if plan['status'] != 'frozen':
        raise ValueError('Protocol is not frozen; evaluate existing models first')
    if plan['task'] != 'Mjlab-Velocity-Flat-XgoDuck' or plan['watch_profile'] != 'walk-repair':
        raise ValueError('This guard supports walking repair only')
    if plan['skill'] != 'walk' or plan['automatic_next_run']:
        raise ValueError('No implicit task change or automatic next run')
    watch = read(CONFIG)
    if not 1 <= plan['additional_updates'] <= watch['maximum_updates_per_run']:
        raise ValueError('Bounded update budget required')
    for key in ('parent_checkpoint', 'parent_env_yaml', 'parent_agent_yaml', 'gate_record'):
        if not plan.get(key) or not Path(plan[key]).is_file():
            raise ValueError('Missing declared input: '+key)
    if sha(plan['parent_checkpoint']) != plan['parent_sha256']:
        raise ValueError('Parent hash mismatch')
    if sha(WORK/'src/mjlab_microduck/tasks/__init__.py') != plan.get('task_source_sha256'):
        raise ValueError('Task source differs from declared parent provenance')
    if sha(WORK/'uv.lock') != plan.get('lock_sha256'):
        raise ValueError('Dependency lock differs from declared parent provenance')
    if sha(plan['gate_record']) != plan['gate_sha256']:
        raise ValueError('Stage gate hash mismatch')
    gate = read(plan['gate_record'])
    engineering = plan.get('scope') == 'engineering-integration'
    permission = 'engineering_integration_allowed' if engineering else 'bounded_learning_allowed'
    if gate.get(permission) is not True:
        raise ValueError('Stage gate has not allowed learning')
    if engineering and plan['additional_updates'] > 50:
        raise ValueError('Engineering integration is limited to 50 updates per declared probe')
    if plan.get('scheduled_stop_after_updates') and not engineering:
        raise ValueError('Synthetic boundary-stop trigger is for integration probes only')
    if plan['change']=='hip-limit':
        expected=0. if plan['arm']=='control' else -2.
        if (engineering or plan.get('scope')!='hip-limit-paired-diagnostic'
                or plan['arm'] not in ('control','candidate')
                or plan['reward_edits']!={'hip_limit_proximity':expected}
                or plan.get('parent_variant')!='reachable-neck'
                or plan['additional_updates']!=50 or plan['num_envs']!=4096 or plan['seed']!=42):
            raise ValueError('Hip diagnostic differs from the reviewed single-factor protocol')
        pair_path=Path(plan['pair_protocol'])
        if sha(pair_path)!=plan['pair_protocol_sha256']:raise ValueError('Paired protocol mismatch')
        pair=read(pair_path)
        if pair['status']!='frozen' or pair['parent_sha256']!=plan['parent_sha256']:
            raise ValueError('Unfrozen or different paired parent')
        for file,digest in pair['source_sha256'].items():
            if sha(ROOT/file)!=digest:raise ValueError('Frozen repair source differs: '+file)
    elif plan['change'] not in ('official', 'reachable-neck') or plan['reward_edits']:
        raise ValueError('New reward hypotheses need a reviewed configuration adapter')
    return watch


def prepare(plan, name):
    """Configuration and checkpoint validation on CPU; no environment created."""
    watch = require_protocol(plan)
    import torch
    import yaml
    from mjlab.scripts.train import TrainConfig
    from display_preferences import prepare_display
    from head_command_bounds import prepare_reachable_head_commands
    cfg = TrainConfig.from_task(plan['task'])
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = plan['num_envs']
    cfg.env.seed = cfg.agent.seed = plan['seed']
    cfg.agent.run_name = name
    cfg.agent.max_iterations = plan['additional_updates']
    cfg.agent.logger, cfg.agent.upload_model = 'tensorboard', False
    # Our complete-update hook owns saving; disable periodic automatic exports.
    cfg.agent.save_interval = 10**9
    cfg.agent.resume = False
    # A modified parent must be reconstructed as such before comparing configs.
    # This does not silently reset its MDP to the original official settings.
    parent_variant = plan.get('parent_variant', 'official')
    if parent_variant == 'reachable-neck':
        cfg = replace(cfg, env=prepare_reachable_head_commands(cfg.env)[0])
    elif parent_variant != 'official':
        raise ValueError('Parent configuration adapter is not implemented')
    parent = {k: yaml.load(Path(plan[f'parent_{k}_yaml']).read_text(), Loader=yaml.BaseLoader)
              for k in ('env', 'agent')}
    # These five camera fields were set by the original visible recorder.
    # Inherit them explicitly; do not waive differences in physics or PPO.
    for field in ('distance', 'elevation', 'height', 'max_extra_envs', 'width'):
        cast = float if field in ('distance', 'elevation') else int
        setattr(cfg.env.viewer, field, cast(parent['env']['viewer'][field]))
    before = {k: yaml.load(yaml.dump(asdict(getattr(cfg, k))), Loader=yaml.BaseLoader)
              for k in ('env', 'agent')}
    for k in ('env', 'agent'):
        tags = lambda text: sorted(re.findall(r'!!python/name:([^\s]+)', text))
        if tags(Path(plan[f'parent_{k}_yaml']).read_text()) != tags(yaml.dump(asdict(getattr(cfg, k)))):
            raise ValueError('Parent callable tags differ: '+k)
    # These scheduling fields are declared above. Everything about the MDP and
    # PPO must match the parent before a candidate's one deliberate change.
    for k in ('run_name', 'max_iterations', 'save_interval', 'logger', 'upload_model', 'resume'):
        parent['agent'][k] = before['agent'][k]
    if parent != before:
        def differences(a, b, path=''):
            if isinstance(a, dict) and isinstance(b, dict):
                return [row for key in sorted(a.keys() | b.keys())
                        for row in differences(a.get(key), b.get(key), path+'.'+key)]
            if a == b: return []
            return [dict(path=path, parent=a, reconstructed=b)]
        raise ValueError('Parent configuration differs: '+json.dumps(differences(parent, before)))
    changes = []
    if plan['change'] == 'reachable-neck':
        patched, changes = prepare_reachable_head_commands(cfg.env)
        cfg = replace(cfg, env=patched)
    elif plan['change']=='hip-limit':
        from hip_limit_repair import prepare_hip_limit
        patched,changes=prepare_hip_limit(cfg.env,plan['arm'])
        cfg=replace(cfg,env=patched)
    cfg, display, _ = prepare_display(cfg, sparse_snapshots=True)
    display.update(mode='periodic-actual-state-video' if plan.get('visuals_enabled', True) else 'observer-overhead-control',
        video_enabled=plan.get('visuals_enabled', True), upstream_video_wrapper_enabled=False,
        video_interval_wall_seconds=watch['visual_interval_seconds'],
        video_duration_simulation_seconds=watch['video_seconds'],
        recording_playback_fps=watch['video_fps'])
    display.pop('video_interval_control_steps', None)
    display.pop('video_length_control_steps', None)
    ckpt = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=True)
    def finite(x):
        if isinstance(x, torch.Tensor): return bool(torch.isfinite(x).all())
        if isinstance(x, dict): return all(finite(v) for v in x.values())
        if isinstance(x, (tuple, list)): return all(finite(v) for v in x)
        return True
    if not finite(ckpt): raise ValueError('Nonfinite parent checkpoint')
    label = int(ckpt['iter'])
    steps = int(ckpt['infos']['env_state']['common_step_counter'])
    rate = [float(x['lr']) for x in ckpt['optimizer_state_dict']['param_groups']]
    if not rate or len(set(rate)) != 1 or steps != (label+1)*cfg.agent.num_steps_per_env:
        raise ValueError('Parent budget/rate mismatch')
    optimizer_steps = {int(v['step'].item()) for v in ckpt['optimizer_state_dict']['state'].values()}
    expected = (label+1)*cfg.agent.algorithm.num_learning_epochs*cfg.agent.algorithm.num_mini_batches
    if optimizer_steps != {expected}: raise ValueError('Parent Adam step count mismatch')
    resume = dict(saved_iteration_label=label, common_step_counter=steps,
        num_steps_per_env=cfg.agent.num_steps_per_env, restored_learning_rate=rate[0],
        next_iteration_label=label+1, remaining_updates=plan['additional_updates'],
        total_updates=label+1+plan['additional_updates'],
        final_iteration_label=label+plan['additional_updates'])
    if torch.cuda.is_initialized(): raise ValueError('Preflight initialized CUDA')
    return cfg, watch, resume, changes, display


def guarded_class(base, plan, watch, resume, record):
    import torch
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from resume_state import prepare_loaded_runner

    class Runner(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load(plan['parent_checkpoint'])
            write(record/'loaded-state.json', prepare_loaded_runner(self, resume))
            self.env.reset()
            write(record/'runner.json', dict(log_dir=str(Path(self.logger.log_dir).resolve()),
                  reset_common_step=int(self.env.unwrapped.common_step_counter)))
            self.guard_boundary = UpdateBoundary(watch, record)
            from display_preferences import save_model_metadata
            save_model_metadata(self.env, record)

        def save(self, path, infos=None):
            # Keep actor, critic, optimizer, normalization and environment count.
            # Skip VelocityOnPolicyRunner's automatic ONNX export, not env state.
            target = Path(path)
            tmp = target.with_suffix('.partial')
            MjlabOnPolicyRunner.save(self, str(tmp), infos)
            # A partial/NaN model must never replace the last durable checkpoint.
            saved = torch.load(tmp, map_location='cpu', weights_only=True)
            def finite(x):
                if isinstance(x, torch.Tensor): return bool(torch.isfinite(x).all())
                if isinstance(x, dict): return all(finite(v) for v in x.values())
                if isinstance(x, (tuple, list)): return all(finite(v) for v in x)
                return True
            if not finite(saved):
                tmp.unlink()
                raise UnsafeState('nonfinite-checkpoint')
            tmp.replace(target)
            # The upstream budget-complete save also passes here, after the
            # last logger hook. Publish the index for EVERY durable save so a
            # later restart cannot silently choose the first periodic model.
            write(record/'last-checkpoint.json',dict(checkpoint=str(target.resolve()),
                checkpoint_sha256=sha(target),iteration=int(saved['iter']),
                common_step_counter=int(saved['infos']['env_state']['common_step_counter']),
                complete_update=True))

        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            robot = env.scene['robot']
            feet, names = robot.find_sites(['left_foot', 'right_foot'], preserve_order=True)
            if names != ['left_foot', 'right_foot']: raise ValueError('Foot order mismatch')
            hips = [list(robot.joint_names).index(s+'_hip_roll') for s in ('left','right')]
            # Fixed spread across all training environments, not just environment 0.
            ids = torch.linspace(0, env.num_envs-1, min(watch['sample_envs'], env.num_envs),
                                 device=env.device).long()
            n = len(ids)
            guard = GaitWatch(watch, n, repair=plan.get('repair_monitor'))
            falls = torch.zeros(n, dtype=torch.bool, device=env.device)
            age = torch.zeros(n, device=env.device)
            boundary = self.guard_boundary
            state = dict(status='starting', run_id=record.name, profile=watch['profile'],
                         image=None, updates=[], pause_requested=False,
                         visual_stale_seconds=watch['visual_stale_seconds'])
            old_step, old_log = self.env.step, self.logger.log
            old_signal = signal.getsignal(signal.SIGINT)
            old_act = self.alg.act
            first_action = True
            last_image, sample_steps = 0.0, 0
            snapshot = video = None
            observer_seconds = 0.
            run_started = time.monotonic()
            visuals_enabled = plan.get('visuals_enabled', True)
            def cpu(t): return t.detach().cpu().numpy()
            def interrupt(sig, frame):
                if state['pause_requested']: raise KeyboardInterrupt
                state['pause_requested'] = True
                print('STOP_REQUESTED: saving at next complete update', flush=True)
            def act(obs):
                nonlocal first_action
                if not all(torch.isfinite(v).all() for v in obs.values()):
                    raise UnsafeState('nonfinite-observation')
                packet=None
                if first_action and plan['change']=='hip-limit':
                    from head_command_start import capture
                    packet,_=capture(self,obs)
                action = old_act(obs)
                if not torch.isfinite(action).all(): raise UnsafeState('nonfinite-action')
                if packet is not None:
                    from hip_limit_repair import capture_start
                    capture_start(self,obs,action,packet,record,plan)
                first_action=False
                return action
            def step(action):
                nonlocal sample_steps, observer_seconds
                result = old_step(action)
                observer_start = time.monotonic()
                reset = env.reset_buf[ids]
                age.add_(env.step_dt)
                age[reset] = 0
                falls.logical_or_(env.reset_terminated[ids])
                sample_steps += 1
                if sample_steps % watch['sample_stride_steps'] == 0:
                    guard.record_falls(cpu(falls)); falls.zero_()
                    row = guard.sample(hips_deg=cpu(robot.data.joint_pos[ids][:, hips])*180/3.141592653589793,
                        gap_mm=signed_foot_gap_mm(cpu(robot.data.site_pos_w[ids][:, feet]),
                                                 cpu(robot.data.root_link_quat_w[ids])),
                        age_s=cpu(age), dt=watch['sample_stride_steps']*env.step_dt)
                    if guard.stop_reason == 'nonfinite-gait-state': raise UnsafeState(guard.stop_reason)
                    if row:
                        command = env.command_manager.get_command('twist')[ids]
                        row['linear_error_mean'] = float(torch.linalg.vector_norm(
                            robot.data.root_link_lin_vel_b[ids,:2]-command[:,:2],dim=1).mean())
                        row['yaw_error_mean'] = float((robot.data.root_link_ang_vel_b[ids,2]-command[:,2]).abs().mean())
                        with (record/'guard.jsonl').open('a') as f:
                            f.write(json.dumps(dict(common_step=int(env.common_step_counter), **row))+'\n')
                if video: video.sample()
                observer_seconds += time.monotonic()-observer_start
                return result
            def log(*a, **kw):
                nonlocal last_image
                it = kw['it']
                completed = it-resume['next_iteration_label']+1
                guard.complete_update(completed)
                stage = read(ROOT/'project.json')['staged_training_request']
                state['pause_requested'] |= bool(stage.get('pause_requested', False))
                reason = guard.stop_reason or ('user-interrupt' if state['pause_requested'] else None)
                if completed == plan.get('scheduled_stop_after_updates'):
                    reason = reason or 'scheduled-integration-stop'
                if video:
                    state['image'] = video.image
                    state['clips'] = video.clips
                # Periodic video owns normal images; preserve the actual final
                # state when a stop occurs, without an additional physics step.
                if snapshot and reason:
                    try:
                        state['image'] = snapshot.capture()
                        last_image = time.monotonic()
                    except Exception as exc:
                        reason = 'observation-failed'
                        state['observation_error'] = repr(exc)
                state.update(status='stop-requested' if reason else 'running',
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    heartbeat_unix=time.time(), iteration=it,
                    common_step=int(env.common_step_counter), guard=guard.windows[-1] if guard.windows else None)
                state.update(completed_updates_this_run=completed, repair_review=guard.review,
                    guard_windows=guard.windows,
                    observer_seconds=observer_seconds,
                    video_seconds=video.elapsed_seconds if video else 0.,
                    elapsed_learning_seconds=time.monotonic()-run_started)
                state['updates'].append(dict(iteration=it, collection_seconds=kw.get('collect_time'),
                                             learning_seconds=kw.get('learn_time')))
                write(record/'state.json', state)
                # Called only after alg.update() and current_learning_iteration.
                old_log(*a, **kw)
                boundary.complete(self, it, reason)
            server = None
            server_thread = None
            try:
                if visuals_enabled:
                    snapshot = Snapshot(env, record)
                    video = SparseVideo(snapshot, watch, record)
                    video.sample(); state['image'] = video.image
                state.update(status='ready', heartbeat_unix=time.time())
                write(record/'state.json', state)
                shutil.copy2(ROOT/'scripts/training/training_dashboard.html', record/'index.html')
                handler = partial(SimpleHTTPRequestHandler, directory=str(record))
                if visuals_enabled:
                    server = ThreadingHTTPServer(('127.0.0.1', 8766), handler)
                    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                    server_thread.start()
                    print('TRAINING_VIEW: http://localhost:8766/ (owned by this foreground worker)', flush=True)
                self.env.step, self.logger.log, self.alg.act = step, log, act
                signal.signal(signal.SIGINT, interrupt)
                super().learn(*args, **kwargs)
                state['status'] = 'budget-complete'
            except ControlledStop as exc:
                state.update(status='stopped-for-analysis', reason=str(exc))
            except BaseException as exc:
                state.update(status='failed', reason=repr(exc),
                             retained_checkpoint=boundary.last_checkpoint or plan['parent_checkpoint'])
                write(record/'stop.json', dict(reason=repr(exc), complete_update=False,
                      retained_checkpoint=state['retained_checkpoint'], automatic_retry=False))
                raise
            finally:
                self.env.step, self.logger.log, self.alg.act = old_step, old_log, old_act
                signal.signal(signal.SIGINT, old_signal)
                if video:
                    video.finish()
                    state['clips'] = video.clips
                state['ended_at'] = datetime.now(timezone.utc).isoformat()
                write(record/'state.json', state)
                if server: server.shutdown(); server.server_close()
                if server_thread: server_thread.join(timeout=5)
                if snapshot: snapshot.close()
                # Preserve a directly openable final page after the foreground
                # server exits. The normal upstream path closes env itself.
                page = record/'index.html'
                if page.exists():
                    payload = json.dumps(state, ensure_ascii=False).replace('<', '\\u003c')
                    with page.open('a') as f: f.write('<script>window.savedState='+payload+';</script>')
                if state['status']=='failed': env.close()
    return Runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['status', 'preflight', 'run'])
    p.add_argument('--protocol', type=Path, default=PLAN)
    args = p.parse_args()
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_relative_to(ROOT/'configs/experiments'):
        raise SystemExit('Protocol must be in the project experiments directory')
    plan = read(protocol_path)
    if args.action == 'status':
        print(json.dumps(dict(stage=read(ROOT/'project.json')['staged_training_request'],
                              next_protocol=plan), ensure_ascii=False, indent=2)); return
    name = run_name(plan['skill'], plan.get('run_label', plan['change']), plan['arm'], plan['seed'],stamp=plan.get('run_stamp'))
    try:
        cfg, watch, resume, changes, display = prepare(plan, name)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit('PREFLIGHT_STOP: '+str(exc)) from None
    if args.action == 'preflight':
        print(json.dumps(dict(status='passed', run_id=name, resume=resume, changes=changes,
                              display=display, training_started=False), indent=2)); return
    stage = read(ROOT/'project.json')['staged_training_request']
    expected_stage = 'S1c-live-integration' if plan.get('scope')=='engineering-integration' else 'S2-bounded-walking'
    if stage['current_stage'] != expected_stage or stage.get('pause_requested', False):
        raise SystemExit('Current stage has not enabled bounded walking learning')
    if not os.isatty(1): raise SystemExit('Run in the visible shared VS Code terminal')
    record = ROOT/'records/experiments'/name
    record.mkdir(exist_ok=False)
    write(ROOT/'records/training-current.json',
          dict(record=str(record), protocol=str(protocol_path), protocol_sha256=sha(protocol_path)))
    print('TRAINING_RECORD:',record,flush=True)
    class Tee:
        def __init__(self, original, file): self.original, self.file = original, file
        def __getattr__(self, name): return getattr(self.original, name)
        def write(self, s): self.original.write(s); self.file.write(s); self.file.flush(); return len(s)
        def flush(self): self.original.flush(); self.file.flush()
        def isatty(self): return self.original.isatty()
    output = (record/'training.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, output), Tee(sys.stderr, output)
    write(record/'protocol.json', plan)
    write(record/'resume-plan.json', resume)
    write(record/'display-state.json', display)
    write(record/'watch.json', watch)
    write(record/'changes.json', changes)
    source = record/'sources'; source.mkdir()
    for file in ('guarded_training.py', 'training_watch.py', 'training_dashboard.html',
                 'display_preferences.py', 'resume_state.py', 'head_command_bounds.py', 'live_observer.py',
                 'hip_limit_repair.py','head_command_start.py'):
        shutil.copy2(ROOT/'scripts/training'/file, source/file)
    write(record/'source-hashes.json', {p.name: sha(p) for p in source.iterdir()})
    from mjlab.scripts import train
    base = train.load_runner_cls(plan['task'])
    train.load_runner_cls = lambda task: guarded_class(base, plan, watch, resume, record)
    train.launch_training(plan['task'], cfg)
    result = read(record/'state.json')
    print('TRAINING_FINISHED:', result['status'], flush=True)
    if result['status'] != 'budget-complete': raise SystemExit(42)


if __name__ == '__main__': main()
