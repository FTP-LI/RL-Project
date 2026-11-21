"""Prospectively evaluate the latched recovery route with matched releases."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from evaluate_local_getup_ppo import PLAN as PREVIOUS, SCENE, check_project
import getup_shared_release as shared
import getup_supine_evaluation as evaluation
from getup_anchor_same_release import visible_log
from getup_pose_router import GravityRoutedGetup
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity
import training_live_page as live
import verify_getup_supine_evaluation as verifier


PLAN = ROOT/'configs/experiments/getup_pose_router.json'
OUT = ROOT/'records/reports/getup-pose-routing'
PAGE = ROOT/'records/getup-routing-live.html'
STAGE = 'S3-pose-routed-recovery-development'
SEEDS = (1062, 1063)
ROLES = ('fixed-weighted', 'n3072', 'pose-routed')


def bound(master, seed):
    spec = deepcopy(read(PREVIOUS))
    seed_info = master['batches'][str(seed)]
    codes = seed_info['pose_codes']
    groups = {pose: [i for i, value in enumerate(codes) if value == j] for j, pose in enumerate(shared.POSES)}
    spec['environment'].update(seed=seed, stage=STAGE, run_id=seed_info['run_id'],
                               split=dict(pose_codes=codes, groups=groups))
    spec.update(stage=STAGE, branches=list(ROLES), source_sha256=master['source_sha256'],
                references=master['references'], master_sha256=sha(PLAN) if PLAN.exists() else None,
                route=master['route'], cameras={
                    ROLES[0]: {'supine': groups['supine'][0]},
                    ROLES[1]: {'right-side': groups['right-side'][0]},
                    ROLES[2]: {pose: indices[0] for pose, indices in groups.items()}})
    spec['policies'] = {role: dict(spec['policies'][role], seed=seed) for role in ROLES[:2]}
    return spec


def cpu_latch(master, source):
    from getup_value_bootstrap import td

    earlier = read(source)
    trace = ROOT/read(source.parent/'evaluation-analysis.json')['record']
    with np.load(trace/f'first-control-{ROLES[0]}.npz') as data:
        first = torch.from_numpy(data['actor_input'].copy())
    previous = read(PREVIOUS)['policies']
    parent, right = (shared.actor_model(previous[role]) for role in ROLES[:2])
    router = GravityRoutedGetup(parent, right, master['route']['alignment'])
    x = td('actor', first)
    changed = td('actor', first.clone())
    changed['actor'][:, 3:6] = torch.tensor([0., -1., 0.])
    with torch.inference_mode():
        model_action = router(x)
        expected = torch.where(router.right[:, None], right(x), parent(x))
        assert torch.equal(model_action, expected)
        prior = router.right.clone()
        assert torch.where(prior[:, None], right(changed), parent(changed)).equal(router(changed))
        assert router.calls == 2 and router.right.equal(prior)
        router.reset_episode()
        router(changed)
        assert router.calls == 1 and bool(router.right.all())
        router.reset_episode()
        invalid = td('actor', first.clone())
        invalid['actor'][:, 3:6] = 0
        router(invalid)
        assert not router.right.any()
    indices = torch.where(prior)[0].tolist()
    assert indices == earlier['branch_worlds']
    return dict(previous_inputs=len(first), specialist_inputs=len(indices),
                changed_observation_kept_latch=True, explicit_reset_relatched=True, invalid_cue_uses_parent=True)


def freeze():
    assert not PLAN.exists() and not OUT.exists()
    previous = read(PREVIOUS)
    source = ROOT/previous['training_evidence']
    retrospective = source.parent/'posthoc-pose-routing.json'
    assert read(retrospective)['combined_successes'] == 316
    saved = read(source.parent/'evaluation-analysis.json')
    assert read(retrospective)['evaluation_sha256'] == sha(source.parent/'evaluation-analysis.json')
    assert saved['protocol_sha256'] == sha(PREVIOUS)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    refs = dict(previous['references'])
    refs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (
        PREVIOUS, retrospective, source.parent/'evaluation-analysis.json',
        ROOT/saved['record']/f'first-control-{ROLES[0]}.npz', SCENE)})
    sources = dict(previous['source_sha256'])
    for path in (Path(__file__), ROOT/'scripts/training/getup_pose_router.py',
                 ROOT/'scripts/training/training_live_page.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    plans = {}
    for seed in SEEDS:
        assert not any(f'_s{seed}_' in p.name for p in (ROOT/'records/evaluations').iterdir())
        codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [32, 31, 128, 128]))))
        plans[str(seed)] = dict(run_id=f'duck_getup_pose-route_paired_s{seed}_{stamp}', pose_codes=codes.tolist())
    master = dict(status='frozen-before-prospective-development', stage=STAGE, batches=plans,
                  source_sha256=sources, references=refs,
                  route=dict(default=ROLES[0], specialist=ROLES[1], latch='first-active-policy-input',
                             observation_columns=[3, 4, 5], alignment=previous['environment']['criteria']['release_alignment_min']),
                  gate='Both timely and final original gates pass, combined successes no lower than either pure policy; run second seed only after first passes.',
                  automatic_retry=False, automatic_extension=False, learning_updates=0, heldout_used=False)
    assert previous['model_interface']['observation_names'].split(',')[:2] == ['base_ang_vel', 'projected_gravity']
    torch.set_num_threads(4)
    parity = cpu_latch(master, retrospective)
    scenes = {}
    for seed in SEEDS:
        spec = bound(master, seed)
        shared.assert_sources(spec)
        for policy in spec['policies'].values():
            assert sha(ROOT/policy['model']) == policy['model_sha256']
        scenes[str(seed)] = cpu_scene_readback(spec['environment'], spec)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, master)
    write(OUT/'preflight.json', dict(status='passed-cpu-route-and-scene-zero-step', protocol_sha256=sha(PLAN),
          parity=parity, scenes=scenes, physical_substeps=0, learning_updates=0, heldout_used=False))
    print('GETUP_POSE_ROUTE_FROZEN', sha(PLAN), {k: v['run_id'] for k, v in plans.items()}, flush=True)


def sample(spec, record):
    assert not record.exists()
    actor = None
    gpu_start = evaluation.gpu_start

    def start(env, obs, meta, source, folder, scene):
        nonlocal actor
        actors, memory = gpu_start(env, obs, meta, dict(source, branches=list(ROLES[:2])), folder, scene)
        actor = GravityRoutedGetup(actors[ROLES[0]], actors[ROLES[1]], source['route']['alignment']).eval()
        return dict(actors, **{ROLES[2]: actor}), memory

    rows = []
    previous_control, previous_time = 0, perf_counter()
    with ExitStack() as stack:
        def observe(path, value):
            nonlocal previous_control, previous_time
            write(path, value)
            if path != record/'evaluation.json':
                return
            readme = record/'README.md'
            if not readme.exists():
                readme.write_text('# 姿态分流起身\n\n同次释放比较固定父模型、右侧专用模型和观测分流。\n\n'
                                  '[冻结方案](../../../configs/experiments/getup_pose_router.json) · '
                                  '[运行状态](state.json) · [原始记录](evaluation.json)\n', encoding='utf-8')
                stack.enter_context(visible_log(stack.enter_context((record/'execution.log').open('x', buffering=1))))
            control = value['actual']['controls']
            instant = perf_counter()
            n = spec['environment']['num_envs']
            if control > previous_control:
                rows.append(dict(control=control, transitions_per_second=(control-previous_control)*n/(instant-previous_time)))
                previous_control, previous_time = control, instant
            clips = [folder for folder in record.glob('evaluation-*/clips.json')
                     if any(clip.get('full_action') for clip in read(folder))]
            latest = max(clips, key=lambda path: path.stat().st_mtime) if clips else None
            clip = next((x for x in reversed(read(latest)) if x.get('full_action')), None) if latest else None
            image = next(iter(record.glob('evaluation-pose-routed-*/frame-step-0.jpg')), None)
            live.publish(record, dict(status=value['status'], run_id=value['run_id'],
                purpose=f"姿态分流动作评估 · {value['phase']}", updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                controls=control, maximum_controls=spec['maximum_unique_controls'], transitions=control*n,
                learning_updates=0, updates=rows, frame=image.relative_to(record).as_posix() if image else None,
                latest_clip=(latest.parent/clip['file']).relative_to(record).as_posix() if clip else None))

        stack.enter_context(patch.object(evaluation, 'gpu_start', start))
        stack.enter_context(patch.object(evaluation, 'write', observe))
        stack.enter_context(patch.object(live, 'PAGE', PAGE))
        try:
            evaluation.run(spec, record, SCENE, check_project)
        finally:
            if actor is not None and actor.first_gravity is not None:
                np.savez_compressed(record/'route-readback.npz', first_gravity=actor.first_gravity.cpu().numpy(),
                                    right=actor.right.cpu().numpy(), calls=actor.calls)


def verify(spec, record):
    torch.set_num_threads(4)
    assert read(record/'protocol.json') == spec
    status = read(record/'evaluation.json')
    assert status['status'] == 'collected-review-required'
    with np.load(record/'route-readback.npz') as route, np.load(record/f'first-control-{ROLES[0]}.npz') as first:
        gravity, latched, calls = route['first_gravity'], route['right'], int(route['calls'])
        np.testing.assert_array_equal(gravity, first['actor_input'][:, 3:6])
    expected = np.asarray(classify_release(gravity/np.linalg.norm(gravity, axis=1, keepdims=True),
                                           spec['route']['alignment'])) == 'right-side'
    np.testing.assert_array_equal(latched, expected)
    assert calls == spec['environment']['maximum_control_steps']-spec['prefix_controls']
    role_queue = iter(ROLES)
    original = verifier.verify_policy

    def check_policy(trace, session):
        if next(role_queue) != ROLES[-1]:
            return original(trace, session)
        from getup_value_bootstrap import td

        inputs = trace['actor_input'][spec['prefix_controls']:]
        actual = trace['raw_policy_actions'][spec['prefix_controls']:]
        maximum = 0.
        with torch.inference_mode():
            for role, select in ((ROLES[0], ~expected), (ROLES[1], expected)):
                actor = shared.actor_model(spec['policies'][role])
                subset = inputs[:, select].reshape(-1, inputs.shape[-1])
                prediction = np.concatenate([actor(td('actor', torch.from_numpy(chunk))).numpy()
                                             for chunk in np.array_split(subset, max(1, (len(subset)+4095)//4096))])
                actions = actual[:, select].reshape(-1, actual.shape[-1])
                np.testing.assert_allclose(actions, prediction, atol=2e-5, rtol=2e-5)
                if len(actions):
                    maximum = max(maximum, float(np.max(np.abs(actions-prediction))))
        return dict(maximum_absolute_error=maximum, actual_inputs_replayed=int(np.prod(inputs.shape[:2])),
                    history_checked_all_rows=True, latched_independent_replay=True)

    def select(roles, reports, comparisons, minimum_gain):
        route = reports[roles[-1]]
        floor = max(reports[role]['combined_successes'] for role in roles[:-1])
        return dict(passes_gate=bool(route['all_motion_gates_passed'] and route['combined_successes'] >= floor),
                    all_original_gates=route['all_motion_gates_passed'], minimum_combined_successes=floor)

    # The legacy verifier expects one ONNX path per branch. Its routed
    # numeric callback uses the two actual ONNX-backed actors instead.
    compatible = dict(spec, policies=dict(spec['policies'], **{ROLES[-1]: spec['policies'][ROLES[0]]}))
    with patch.object(verifier, 'verify_policy', check_policy), patch.object(verifier, 'select', select):
        result = verifier.verify(compatible, record, status, minimum_gain=0)
    with np.load(record/f'release-{ROLES[0]}.npz') as release:
        actual_right = np.asarray(classify_release(projected_gravity(release['qpos'][:, 3:7]),
                                                   spec['route']['alignment'])) == 'right-side'
    result.update(master_sha256=sha(PLAN), seed=spec['environment']['seed'],
                  route=dict(calls=calls, specialist_worlds=np.flatnonzero(expected).tolist(),
                             cue_vs_actual_right_mismatches=np.flatnonzero(expected != actual_right).tolist(),
                             saved_readback_sha256=sha(record/'route-readback.npz')))
    write(OUT/f"seed{result['seed']}-analysis.json", result)
    print('GETUP_POSE_ROUTE_VERIFIED', result['seed'], {role: row['combined_successes'] for role, row in result['arms'].items()},
          result['selection'], flush=True)


def main():
    check_project()
    if sys.argv[1:] == ['--freeze']:
        return freeze()
    assert len(sys.argv) == 3 and sys.argv[1] in ('--sample', '--verify')
    seed = int(sys.argv[2])
    assert seed in SEEDS
    assert read(OUT/'preflight.json')['protocol_sha256'] == sha(PLAN)
    master = read(PLAN)
    spec = bound(master, seed)
    shared.assert_sources(spec)
    record = ROOT/'records/evaluations'/spec['environment']['run_id']
    if seed == SEEDS[-1]:
        first = read(OUT/f'seed{SEEDS[0]}-analysis.json')
        assert first['master_sha256'] == sha(PLAN) and first['selection']['passes_gate']
    if sys.argv[1] == '--sample':
        sample(spec, record)
    else:
        assert not (OUT/f'seed{seed}-analysis.json').exists()
        verify(spec, record)


if __name__ == '__main__':
    main()
