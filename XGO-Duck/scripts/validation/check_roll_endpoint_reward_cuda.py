"""Check the roll reward change against installed CUDA physics and actual resets."""

from dataclasses import asdict
from datetime import datetime
import fcntl
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tensordict import TensorDict

import roll_friction_ppo as learner
from record_io import write
from roll_endpoint_reward import EndpointRollProgress
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


STAGE = 'roll-endpoint-progress-training-preflight'
KEY = 'roll_endpoint_progress_training'
REPORT = ROOT / 'records/reports/roll-forward-input-integration/endpoint-progress-training'


def check_steps(plan, arm, settings, record, view):
    recipe = plan['recipe']
    world_count = arm['environments']
    max_controls = settings['cuda_admission']['fixed_policy_controls']
    options = learner.ort.SessionOptions()
    options.intra_op_num_threads = 1
    source_onnx = ROOT / recipe['official_onnx']
    session = learner.ort.InferenceSession(str(source_onnx), sess_options=options, providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    cfg = learner.prepare(plan, arm, metadata)
    watch = learner.MemoryWatch(recipe['hard_gpu_fraction'])
    learner.configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    raw = snapshot = video = replacement = None
    rows = []
    proof = []

    def publish(controls, frame):
        clips = [clip for clip in video.clips if clip['full_action']]
        view.publish(record, dict(run_id=arm['run_id'], status='running', purpose='滚翻奖励时点CUDA核验｜固定策略',
            controls=controls, maximum_controls=max_controls, transitions=controls*world_count,
            learning_updates=0, rows=rows, frame=frame['file'], latest_clip=clips[-1]['file'] if clips else None))

    try:
        with watch.active(), torch.no_grad():
            raw = learner.ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            env = learner.RslRlVecEnvWrapper(raw, clip_actions=cfg.agent.clip_actions)
            env.seed(recipe['seed'])
            env.reset()
            runner = learner.MjlabOnPolicyRunner(env, asdict(cfg.agent), None, 'cuda:0')
            parent = torch.load(ROOT / recipe['parent_checkpoint'], map_location='cpu', weights_only=True)
            zero = learner.initial_setup(raw, env, runner, parent, session, metadata, arm, record)
            require(zero['controls'] == zero['learning_updates'] == 0 and view.check(force=True), 'CUDA startup or observer failed')
            replacement = EndpointRollProgress(raw, settings['change'], verify_cached=True)
            snapshot = learner.Snapshot(raw, record)
            video = learner.ActionVideo(snapshot, 'roll', record)
            video.sample()
            publish(0, video.image)
            view.begin_collect()
            observation = env.get_observations()

            def reference_step(actions):
                transition = env.step(actions)
                reference_step.observation = transition[0]['actor'].clone()
                reference_step.executed = raw.sim.data.ctrl.clone()
                reference_step.positions = raw.sim.data.qpos.clone()
                return transition

            started = time.perf_counter()
            for controls in range(1, max_controls + 1):
                require(view.check(), 'Live VS Code observation stopped')
                watch.check()
                actions = runner.alg.actor(TensorDict({'actor': observation['actor']}, batch_size=[world_count]))
                observation, reward, done, _ = replacement.step(reference_step, actions)
                data = replacement.last_check
                manager = raw.reward_manager
                terms = manager._step_reward.clone()
                terms[:, replacement.index] = data['original_terms'][:, replacement.index]
                duration = replacement.knobs['step_dt']
                spawn_error = torch.stack([torch.where(done.bool(), value - raw._roulade_accum, 0.).abs()
                                           for value in replacement.state]).amax()
                proof.append(torch.stack((
                    (data['cached_payment'] - data['original_payment']).abs().amax(),
                    (reward - (data['original_reward'] + data['adjustment'])).abs().amax(),
                    (terms - data['original_terms']).abs().amax(),
                    (manager._episode_sums[replacement.name] - (data['original_episode_sum'] + data['adjustment'])).abs().amax(),
                    (manager._step_reward[:, replacement.index] * duration - torch.where(
                        done.bool(), data['original_payment'], data['replacement'])).abs().amax(),
                    torch.where(done.bool(), data['adjustment'], 0.).abs().amax(), spawn_error,
                    (observation['actor'] - reference_step.observation).abs().amax(),
                    (raw.sim.data.ctrl - reference_step.executed).abs().amax(),
                    (raw.sim.data.qpos - reference_step.positions).abs().amax(),
                    done.sum().float(), data['contact_changed'].sum().float(),
                    (data['adjustment'].abs() > 2e-5).sum().float(), data['adjustment'].abs().amax(),
                    reward.mean(), data['original_reward'].mean(), torch.isfinite(reward).all().float())))
                video.sample()
                if controls % 60 == 0 or controls == max_controls:
                    row = proof[-1]
                    rows.append(dict(control=controls, environment_reward_mean=float(row[-3]),
                                     reference_reward_mean=float(row[-2])))
                    publish(controls, snapshot.capture())

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            video.close()
            require(raw.common_step_counter == max_controls and raw._sim_step_counter == 5*max_controls,
                    'CUDA physical transition count changed')
            stats = torch.stack(proof).cpu().numpy()
            np.save(record / 'reward-checks.npy', stats, allow_pickle=False)
            checks = ('cached_payment', 'total_reward', 'other_terms', 'episode_sum', 'term_display', 'terminal_adjustment',
                      'reset_counters', 'actor_observation', 'executed_controls', 'positions')
            maxima = dict(zip(checks, stats[:, :len(checks)].max(axis=0).astype(float), strict=True))
            reset_count, contact_count, changed_count = map(int, stats[:, 10:13].sum(axis=0))
            require(np.isfinite(stats).all() and stats[:, -1].min() == 1 and max(maxima.values()) <= 2e-5
                    and reset_count >= world_count and contact_count > 0 and changed_count > 0,
                    'CUDA reward replacement or automatic-reset checks did not pass')
            require(len(video.clips) == 1 and video.clips[0]['full_action'] and video.clips[0]['frames'] == 176,
                    'Missing full seven-second CUDA check footage')
            watcher = view.view / 'native-curve-source.json'
            deadline = time.monotonic() + 22
            health = view.view / 'native-health.json'
            while time.monotonic() < deadline:
                if watcher.is_file() and health.is_file() and read(health).get('metric_rows') == len(rows):
                    break
                time.sleep(.25)
            require(view.check(force=True) and read(health).get('metric_rows') == len(rows),
                    'Live observer did not read actual CUDA rows')
            put(view.report / 'live-observer-proof.json', read(health))
            return dict(status='passed-cuda-fixed-policy', run_id=arm['run_id'], controls=max_controls,
                        physics_substeps=raw._sim_step_counter, physical_transitions=max_controls*world_count,
                        learning_updates=0, hardware_io=False, auto_resets=reset_count, contact_changes=contact_count,
                        reward_changes_over_tolerance=changed_count, max_action_time_reward_change=float(stats[:, 13].max()),
                        maximum_errors=maxima, initial_setup=zero, elapsed_seconds=elapsed,
                        device=learner.resources(watch), clips=video.clips, source_rows=rows, source_columns=[*checks,
                            'resets', 'contact_changes', 'reward_changes', 'maximum_adjustment', 'candidate_reward_mean',
                            'reference_reward_mean', 'finite_rewards'], numeric_file_sha256=sha(record/'reward-checks.npy'))
    finally:
        if video is not None:
            video.close()
        if snapshot is not None:
            snapshot.close()
        if replacement is not None:
            replacement.close()
        if raw is not None:
            raw.close()


def main():
    project = read(P)
    task, state = project['staged_training_request'], project['goal_execution'][KEY]
    draft = ROOT / state['draft']
    settings = read(draft)
    require(task['current_stage'] == STAGE and not any(task[key] for key in ('pause_requested', 'simulation_running', 'learning_running'))
            and sha(draft) == state['draft_sha256'] and state['status'] == 'fixed-trajectory-payment-replay-passed',
            'Roll endpoint reward CUDA checks are not ready')
    paired = settings['paired_learning']
    inherited = ROOT / paired['inherit_parent_optimizer_and_safety_checks_from']
    plan = read(inherited)
    require(all(sha(ROOT / name) == digest for name, digest in settings['reviewed_source_sha256'].items())
            and all(sha(Path(name)) == digest for name, digest in plan['installed_sha256'].items()),
            'Inherited experiment or installed sources changed')
    for name in ('parent', 'official'):
        filename = name + ('_checkpoint' if name == 'parent' else '_onnx')
        require(sha(ROOT / plan['recipe'][filename]) == plan['recipe'][name+'_sha256'], 'Incorrect actor or parent checkpoint')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = 'duck_roll_endpoint_reward_cuda_e128_s1246_' + stamp
    arm = dict(run_id=run_id, environments=settings['cuda_admission']['worlds'], updates=paired['max_updates_per_arm'],
               friction_role=paired['shared_friction_sampling'])
    plan['recipe']['seed'] = paired['training_seed']
    report = REPORT / run_id
    record = ROOT / 'records/evaluations' / run_id
    protocol = report / 'protocol.json'

    def allowed():
        current = read(P)['staged_training_request']
        return current['current_stage'] == STAGE and not current['pause_requested'] and not current['learning_running']

    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(allowed() and not read(P)['staged_training_request']['simulation_running'], 'A run is already active')
        report.mkdir(parents=True)
        record.mkdir()
        sources = (Path(__file__), Path(__file__).with_name('roll_contact_endpoint.py'), draft,
                   ROOT/'scripts/training/roll_endpoint_reward.py', ROOT/'scripts/training/roll_endpoint_progress.py')
        source_hashes = {rel(path): sha(path) for path in sources}
        put(protocol, dict(scope='fixed-policy-cuda-admission', plan=dict(arm=arm, check=settings['cuda_admission']),
            source_sha256=source_hashes, observation=preflight(), learns=False, hardware_io=False))
        task.update(simulation_running=True, current_protocol_execution='roll-endpoint-cuda-fixed-policy-running')
        state.update(cuda_check_run_id=run_id, cuda_check_protocol=rel(protocol), cuda_check_protocol_sha256=sha(protocol))
        write(P, project)
        view = RecordedObserver(run_id, report, allowed)
        view.state.update(purpose='滚翻奖励时点CUDA核验｜固定策略', maximum_controls=settings['cuda_admission']['fixed_policy_controls'])
        result = dict(status='stopped-review-required', run_id=run_id, learning_updates=0)
        try:
            view.start()
            result = check_steps(plan, arm, settings, record, view)
            require(all(sha(ROOT / path) == digest for path, digest in source_hashes.items()), 'Admission sources changed')
            print('ROLL_ENDPOINT_REWARD_CUDA_PASSED', result['physical_transitions'], result['auto_resets'],
                  result['reward_changes_over_tolerance'], flush=True)
        except BaseException as exc:
            result['status'] = 'stopped-review-required'
            result['reason'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            try:
                view.close(result)
            finally:
                result.update(last_published_controls=max(0, view.last_control), protocol_sha256=sha(protocol))
                result_path = report / 'result.json'
                put(result_path, result)
                latest = read(P)
                latest['goal_execution'][KEY].update(status=result['status'], cuda_check_result=rel(result_path),
                                                       cuda_check_result_sha256=sha(result_path))
                latest['staged_training_request'].update(simulation_running=False, current_protocol_execution=result['status'],
                                                         current_evidence=rel(result_path))
                write(P, latest)


if __name__ == '__main__':
    main()
