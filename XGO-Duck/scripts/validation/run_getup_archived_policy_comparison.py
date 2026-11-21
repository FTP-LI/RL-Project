"""Compare archived fixed policies after one exact CUDA release."""

from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import collect_getup_anchor_same_release as acquisition
import getup_shared_release as shared
from common_scene_contract import compare_arrays, model_arrays
from evaluate_standup_reference import Tee
from getup_release_scene_state import SceneReleaseSnapshot
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from run_walk_backward_paired_eval import publish, rel
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT / 'configs/experiments/getup_archived_policy_comparison_draft.json'
PARENT = ROOT / 'configs/experiments/getup_supine_hard_start_screen.json'
SCENE = ROOT / 'records/reports/getup-interface-value/cpu-scene-fields.npz'
STAGE = 'S3-getup-archived-policy-comparison'


def files(seed):
    return (DRAFT.with_stem(f'getup_archived_policy_comparison_s{seed}'),
            ROOT / f'records/reports/getup-archived-policy-comparison/s{seed}',
            ROOT / f'records/getup-archived-policy-s{seed}-live.html')


def exact_scene(model):
    with np.load(SCENE, allow_pickle=False) as expected:
        compare_arrays(model_arrays(model), {key: expected[key] for key in expected.files})


def check_phase(protocol):
    check_project()
    project = read(ROOT / 'project.json')
    assert ROOT / project['goal_execution']['current_protocol'] == protocol
    assert project['current_phase'] == (STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze(seed):
    from mjlab.scene import Scene

    check_phase(DRAFT)
    d = read(DRAFT)
    assert seed == d['development_seeds'][0], 'The second seed requires a separate freeze after first-pass selection'
    assert d['physical_substeps_authorized'] == d['learning_updates'] == 0 and not d['heldout_used']
    plan, out, page = files(seed)
    assert not any(path.exists() for path in (plan, out, page))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{seed}_' in item.name for item in (ROOT / folder).iterdir())
    links = {}
    for key in ('shortlist_protocol', 'shortlist_result'):
        path, expected = d[key]
        assert sha(ROOT / path) == expected
        links[key] = ROOT / path
    shortlist = read(links['shortlist_result'])
    assert shortlist['admitted_to_fresh_four_pose_comparison']
    assert shortlist['protocol_sha256'] == sha(links['shortlist_protocol'])
    previous = read(PARENT)
    shared.assert_sources(previous)
    p = deepcopy(previous['environment'])
    roles = d['branches']
    n, prefix, end = (d[k] for k in ('worlds_per_seed', 'shared_controls', 'episode_controls'))
    controls = prefix + len(roles) * (end - prefix)
    assert n * controls == d['maximum_transitions_per_complete_development_seed']
    assert 5 * controls == d['maximum_batched_substeps_per_complete_development_seed']
    assert prefix == 100 and end == p['maximum_control_steps'] == 550
    assert p['actual_release_minimum'] == 32 and p['success_rate_min'] == .95
    assert not p['handoff_enabled'] and not p['neck_home_enabled']
    codes = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(shared.POSES)}
    assert len(codes) == n
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_archived-policies_development_s{seed}_{stamp}'
    p.update(seed=seed, num_envs=n, run_id=run_id, stage=STAGE,
             split=dict(pose_codes=codes.tolist(), groups=groups), maximum_control_transitions=n * end)
    baseline = shortlist['fixed_baseline']
    assert p['model'] == baseline['model'] and p['model_sha256'] == baseline['model_sha256']
    policies = {role: dict(p if role in roles[:2] else shortlist['candidates'][role]['model'], seed=seed)
                for role in roles}
    policies = {role: {name: value[name] for name in ('model', 'model_sha256', 'seed')} for role, value in policies.items()}
    cameras = {role: {'supine': groups['supine'][0]} for role in roles}
    cfg, display, meta = shared.prepare(p)
    assert not cfg.env.auto_reset and not display['native_live_window'] and cfg.env.decimation == 5
    assert cfg.env.episode_length_s == 11. and cfg.env.scene.num_envs == n
    model = Scene(cfg.env.scene, device='cpu').compile()
    exact_scene(model)
    joint_addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in meta['joint_names'].split(',')]
    unique = {role: policy for role, policy in policies.items() if role != roles[1]}
    torch.set_num_threads(4)
    parity = shared.cpu_check({'roles': unique})
    assert not torch.cuda.is_initialized()
    record = ROOT / 'records/evaluations' / run_id
    sources = dict(previous['source_sha256'])
    for name in (Path(__file__), Path(acquisition.__file__),
                 Path(__file__).with_name('verify_getup_archived_policy_comparison.py'),
                 Path(__file__).with_name('verify_getup_anchor_same_release.py'),
                 Path(__file__).with_name('observe_getup_archived_policy_comparison.py')):
        sources[rel(name.resolve())] = sha(name)
    references = dict(previous['references'])
    for path in (DRAFT, PARENT, SCENE, *links.values()):
        references[rel(path)] = sha(path)
    for item in (baseline, *(row['model'] for row in shortlist['candidates'].values())):
        references[item['checkpoint']] = item['checkpoint_sha256']
        references[item['model']] = item['model_sha256']
    spec = dict(status='frozen-archived-getup-same-release', stage=STAGE, environment=p, draft=d,
                source_sha256=sources, references=references, branches=roles, policies=policies, cameras=cameras,
                prefix_controls=prefix, maximum_unique_controls=controls, maximum_physics_substeps=5 * controls,
                maximum_warp_snapshot_bytes=d['maximum_warp_snapshot_bytes'], maximum_sampling_wall_seconds=660,
                record=rel(record), page=rel(page), heldout_used=False, hardware_io=False, learning_updates=0)
    shared.assert_sources(spec)
    out.mkdir(parents=True)
    record.mkdir()
    write(plan, spec)
    copyfile(plan, record / 'protocol.json')
    write(out / 'freeze.json', dict(status='passed-cpu-zero', protocol_sha256=sha(plan), physical_substeps=0,
                                   cpu_joint_addresses=joint_addresses, scene_exact=True, actor_import=parity))
    publish(record, page, dict(status='starting', purpose='归档起身策略同释放对照', run_id=run_id, controls=0,
                               maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
    print('ARCHIVED_POLICY_COMPARISON_FROZEN', rel(record), sha(plan), flush=True)


def native_zero(env, obs, memory, spec, plan, target):
    import onnxruntime as ort

    p, roles = spec['environment'], spec['branches']
    exact_scene(env.sim.mj_model)
    rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(env.device).clone())
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        actors = {role: shared.actor_model(spec['policies'][role], env.device) for role in roles if role != roles[1]}
        actors[roles[1]] = actors[roles[0]]
        standing = shared.actor_model(dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device)
        shared.startup(env, obs, dict(getup=actors[roles[0]], standing=standing), p, target)
        inputs = shared.cpu(obs['actor'][::5])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        errors = {}
        for role in roles:
            session = ort.InferenceSession(str(ROOT / spec['policies'][role]['model']), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            expected = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
            actual = shared.cpu(actors[role](shared.td('actor', torch.from_numpy(inputs).to(env.device))))
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            errors[role] = float(abs(actual - expected).max())
    assert torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state(env.device))
    sizes = []

    def capture(env, memory, cap):
        snapshot = SceneReleaseSnapshot(env, memory, cap)
        sizes.append(snapshot.warp_bytes)
        return snapshot

    with patch.object(shared, 'PLAN', plan), patch.object(shared, 'ReleaseSnapshot', capture):
        shared.zero_check(env, memory, spec, target)
    return actors, dict(snapshot_bytes=sizes[0], model_max_error=errors, cpu_cuda_rng_unchanged=True,
                        zero_readback=read(target / 'zero-readback.json'), physical_substeps=0)


def run(seed, mode):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from PIL import Image
    import sys

    plan, out, page = files(seed)
    check_phase(plan)
    spec = read(plan)
    assert spec['environment']['seed'] == seed and not spec['heldout_used']
    assert sha(plan) == read(out / 'freeze.json')['protocol_sha256']
    shared.assert_sources(spec)
    p = spec['environment']
    record = ROOT / spec['record']
    assert sha(plan) == sha(record / 'protocol.json')
    zero = mode == 'zero'
    target = out if zero else record
    state = read(record / 'state.json')
    assert state['status'] == 'starting' and not (target / 'zero-readback.json').exists()
    if not zero:
        ready = read(record / 'windows-observer-ready.json')
        assert ready['status'] == 'starting-page-observed' and ready['protocol_sha256'] == sha(plan)
        assert ready['page_sha256'] == sha(page) and ready['initial_state_sha256'] == sha(record / 'state.json')
        expected = read(out / 'native-zero.json')
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = shared.prepare(p)
    watch = MemoryWatch(spec['draft']['gpu_memory_fraction_cap'])
    state.update(phase='gpu-zero', conditions=[], actual=dict(controls=0, attempted_controls=0,
                 substeps=0, attempted_substeps=0, resets=0), protocol_sha256=sha(plan))
    last, timestamp = 0, perf_counter()

    def show():
        nonlocal last, timestamp
        controls = state['actual']['controls']
        if controls > last:
            now = perf_counter()
            state['updates'].append(dict(control=controls, transitions_per_second=(controls - last)*p['num_envs']/(now - timestamp)))
            last, timestamp = controls, now
        state.update(controls=controls, transitions=controls*p['num_envs'])
        write(record / 'evaluation.json', state)
        publish(record, page, state)

    def guard():
        check_phase(plan)
        watch.check()

    class LiveCamera(acquisition.FixedGetupCamera):
        def render(self):
            pixels = super().render()
            k, phase = self.env.common_step_counter, state['phase']
            active = spec['branches'][0] if phase == 'shared-prefix' else phase
            if k and k % 50 == 0 and self.role == active:
                name = f'frame-{self.role}-{k}.jpg'
                Image.fromarray(pixels).save(record / name, quality=88)
                state['frame'] = name
                show()
            return pixels

    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (target / 'execution.log').open('x', buffering=1) as log, \
                redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            write(target / f'{mode}-command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), physical_run=not zero,
                                                        protocol_sha256=sha(plan)))
            env = None
            try:
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=seed)
                state['actual']['resets'] = 1
                memory = shared.make_memory(p, meta, env.device)
                actors, proof = native_zero(env, obs, memory, spec, plan, target)
                guard()
                watch.sample()
                if zero:
                    write(out / 'native-zero.json', dict(status='passed-native-zero', proof=proof, gpu=watch.summary()))
                    print('ARCHIVED_POLICY_COMPARISON_NATIVE_ZERO', proof['snapshot_bytes'], proof['model_max_error'], flush=True)
                    return
                assert expected['status'] == 'passed-native-zero' and proof == expected['proof']
                state['status'] = 'running'
                shared.project_running(True)
                show()
                with patch.object(acquisition, 'FixedGetupCamera', LiveCamera):
                    acquisition.collect(env, actors, memory, meta, spec, record, state, show, guard)
                assert state['qpos_joint_addresses'] == read(out / 'freeze.json')['cpu_joint_addresses']
                assert state['actual']['controls'] == spec['maximum_unique_controls']
                assert state['actual']['substeps'] == spec['maximum_physics_substeps']
                assert len(state['conditions']) == len(spec['branches'])
                state.update(status='completed', phase='independent-score-pending', latest_clip=state['conditions'][-1]['video']['path'])
                guard()
            except BaseException as error:
                state.update(status='stopped-review-required', error=repr(error))
                raise
            finally:
                if env is not None:
                    env.close()
                state['gpu'] = watch.summary()
                write(target / f'{mode}-gpu-telemetry.json', watch.rows)
                if not zero:
                    shared.project_running(False)
                    show()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.seed)
    else:
        run(args.seed, args.mode)
