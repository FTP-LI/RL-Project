"""Compare an added right-side policy while retaining the same supine route."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_getup_archived_policy_replication as prior
from getup_pose_router import GravityRoutedGetup, right_side_mask
from getup_supine_release_actor import SupineReleaseActor
from review_getup_archived_policy_teacher_timing import actor as load_onnx, success
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned
from run_getup_teacher_head_supine_latch import observed_mask
from verify_getup_shared_release import same_bytes


base = prior.base
ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
DRAFT = ROOT / 'configs/experiments/getup_dual_pose_route_draft.json'
PLAN = DRAFT.with_stem('getup_dual_pose_route_s1166')
OUT = ROOT / 'records/reports/getup-dual-pose-route'
PAGE = ROOT / 'records/getup-dual-pose-route-live.html'
STAGE, SEED = 'S3-getup-dual-pose-route', 1166
ROLES = ('reference-a', 'reference-b', 'candidate')
original_snapshot, original_collect, original_actor = prior.snapshot, base.acquisition.collect, base.shared.actor_model


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def masks(x, threshold):
    supine = observed_mask(x, threshold)
    right = right_side_mask(torch.from_numpy(x[:, 3:6].copy()), threshold).numpy()
    assert not (supine & right).any()
    return supine, right


def witnesses(review, threshold):
    results = []
    plan = ROOT / 'configs/experiments/getup_right_fallback_cross_seed_review.json'
    assert review['protocol_sha256'] == sha(plan)
    for seed, (name, digest, _) in read(plan)['sources'].items():
        score_path = pinned((name, digest))
        score = read(score_path)
        assert score['seed'] == int(seed) and not score['heldout_used']
        path = ROOT / score['record'] / 'first-control-reference-a.npz'
        assert sha(path) == review['seeds'][seed]['first_files_sha256']['reference-a']
        with np.load(path, allow_pickle=False) as archive:
            selected = masks(archive['actor_input'], threshold)
        _, poses = success(score['arms']['reference-a'])
        coverage = {}
        for pose, choice in zip(('supine', 'right-side'), selected, strict=True):
            actual = poses == pose
            coverage[pose] = dict(actual=int(actual.sum()), hits=int((choice & actual).sum()), leaks=int((choice & ~actual).sum()))
        results.append((path, score_path, coverage))
    return results


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    d = read(DRAFT)
    assert d['seed'] == SEED != d['reserved_holdout_seed'] and tuple(d['branches']) == ROLES
    assert not any(d[key] for key in ('learning_updates', 'physical_substeps_authorized', 'heldout_used', 'hardware_io'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    review_path, source, supine_path, right_path = (pinned(d[key]) for key in
        ('right_review', 'native_reference', 'supine_training', 'right_specialist_training'))
    review = read(review_path)
    assert review['status'] == 'completed-same-fallback-right-side-review'
    assert review['cohorts']['all-fail']['count'] == 5 and not review['heldout_used']
    supine, right = read(supine_path), read(right_path)
    assert supine['admitted_to_development'] and right['status'] == 'passed'
    development = [pinned(pointer) for pointer in d['right_specialist_development']]
    for path, role, expected in zip(development, ('n3072', 'pose-routed', 'pose-routed'), (49, 54, 57), strict=True):
        report = read(path)
        assert not report['heldout_used']
        arm = report['arms'][role]
        episodes = arm['timing']['episodes']
        rows = [i for i, row in enumerate(episodes) if row['release_pose'] == 'right-side']
        assert len(rows) == expected
        assert all(episodes[i]['hold_start_seconds'] is not None and arm['end_hold']['stable_by_environment'][i] for i in rows)
    supine = supine['arms']['control']
    spec = read(source)
    p = spec['environment']
    fallback = dict(spec['common_fallback'], seed=SEED)
    policies = {role: dict(model=row['model'], model_sha256=row['model_sha256'], seed=SEED)
                for role, row in zip(ROLES, (supine, supine, right), strict=True)}
    all_policies = dict(fallback=fallback, **{role: policies[role] for role in (ROLES[0], ROLES[2])})
    meta = [load_onnx({'policies': all_policies}, role)[0].get_modelmeta().custom_metadata_map for role in all_policies]
    for key in ('joint_names', 'default_joint_pos', 'observation_names', 'action_scale', 'command_names'):
        assert len({row[key] for row in meta}) == 1, key
    threshold = d['alignment']
    saved = witnesses(review, threshold)
    for _, _, coverage in saved:
        for item in coverage.values():
            assert item['hits'] >= item['actual'] * d['gate']['minimum_supine_and_right_latch_coverage']
            assert item['leaks'] <= d['gate']['maximum_latch_leaks']
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    controls = d['release_counter'] + len(ROLES) * (p['maximum_control_steps'] - d['release_counter'])
    assert d['release_counter'] == spec['prefix_controls'] and len(codes) == p['num_envs'] == d['worlds']
    assert controls * len(codes) == d['maximum_transitions'] and 5 * controls == d['maximum_physics_substeps']
    assert p['success_rate_min'] == d['gate']['timely_and_final_each_requested_and_actual_pose_and_overall']
    assert p['actual_release_minimum'] == d['gate']['minimum_actual_pose_count']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_dual-pose-route_development_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, model=supine['model'], model_sha256=supine['model_sha256'],
             split=dict(pose_codes=codes.tolist(), groups=groups))
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), policies=policies, branches=list(ROLES),
        common_fallback=fallback, dual_route=d, route_witnesses=[base.rel(item[0]) for item in saved],
        maximum_unique_controls=controls, maximum_physics_substeps=5*controls,
        cameras={role: {'right-side': groups['right-side'][0]} for role in ROLES})
    refs = [DRAFT, review_path, source, supine_path, right_path, *development, *(ROOT / x['model'] for x in all_policies.values())]
    refs.extend(path for pair in saved for path in pair[:2])
    spec['references'].update({base.rel(path): sha(path) for path in refs})
    for path in (Path(__file__), ROOT / 'scripts/training/getup_pose_router.py'):
        spec['source_sha256'][base.rel(path.resolve())] = sha(path)
    base.shared.assert_sources(spec)
    cfg, display, metadata = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == d['worlds'] and cfg.env.episode_length_s == 11. and cfg.env.decimation == 5
    assert not cfg.env.auto_reset and not display['native_live_window']
    model = Scene(cfg.env.scene, device='cpu').compile()
    base.exact_scene(model)
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in metadata['joint_names'].split(',')]
    parity = base.shared.cpu_check({'roles': all_policies})
    assert not torch.cuda.is_initialized()
    record.mkdir()
    OUT.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-zero', scene_exact=True, actor_import=parity,
        cpu_joint_addresses=addresses, saved_release_coverage=[row[2] for row in saved],
        protocol_sha256=sha(PLAN), physical_substeps=0))
    (record / 'README.md').write_text('# 双姿态起身分流评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-dual-pose-route-live.html) · [日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='右侧专用策略同初态评估', run_id=run_id,
        controls=0, maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
    print('DUAL_POSE_ROUTE_FROZEN', base.rel(record), sha(PLAN), flush=True)


def guarded_actor(policy, device='cpu'):
    target = torch.device(device)
    if target.type == 'cpu':
        return original_actor(policy, device)
    before = torch.get_rng_state().clone(), torch.cuda.get_rng_state(target).clone()
    with torch.random.fork_rng(devices=[target]):
        actor = original_actor(policy, device)
    assert torch.equal(before[0], torch.get_rng_state()) and torch.equal(before[1], torch.cuda.get_rng_state(target))
    return actor


def routes(actors, fallback, counter, spec):
    d = spec['dual_route']
    def supine(actor):
        return SupineReleaseActor(fallback, actor, counter, d['release_counter'], d['alignment']).eval()
    references = {role: supine(actors[role]) for role in ROLES[:2]}
    return dict(references, candidate=GravityRoutedGetup(supine(actors[ROLES[0]]), actors['candidate'], d['alignment']).eval())


def snapshot(env, obs, memory, spec, record):
    actors, proof, fields, inputs = original_snapshot(env, obs, memory, spec, record)
    fallback = guarded_actor(spec['common_fallback'], env.device)
    d = spec['dual_route']
    rows = []
    for path in spec['route_witnesses']:
        with np.load(ROOT / path, allow_pickle=False) as saved:
            x = saved['actor_input']
        s, r = masks(x, d['alignment'])
        values = torch.from_numpy(x).to(env.device)
        a, b, c = routes(actors, fallback, lambda: d['release_counter'], spec).values()
        with torch.inference_mode():
            td = base.shared.td('actor', values)
            candidate, reference = c(td), a(td)
            assert torch.equal(reference, b(td))
            np.testing.assert_array_equal(base.shared.cpu(c.right), r)
            np.testing.assert_array_equal(base.shared.cpu(c.parent.supine_at_release), s)
            assert torch.equal(candidate[~c.right], reference[~c.right])
            assert torch.equal(candidate[c.right], actors['candidate'](td)[c.right])
            changed = values.clone()
            changed[:, 3:6] = torch.tensor([0., 0., -1.], device=env.device)
            previous_masks = c.right.clone(), c.parent.supine_at_release.clone()
            c(base.shared.td('actor', changed))
            assert torch.equal(c.right, previous_masks[0]) and torch.equal(c.parent.supine_at_release, previous_masks[1])
        rows.append(dict(inputs=len(r), right=int(r.sum()), supine=int(s.sum()), latch_persisted=True, byte_exact=True))
    assert env.common_step_counter == env._sim_step_counter == 0
    proof['dual_latch_cuda'] = rows
    return actors, proof, fields, inputs


def collect(env, actors, memory, meta, spec, record, state, publish, check_active):
    fallback = guarded_actor(spec['common_fallback'], env.device)
    wrapped = routes(actors, fallback, lambda: env.common_step_counter, spec)
    original_collect(env, wrapped, memory, meta, spec, record, state, publish, check_active)
    candidate = wrapped['candidate']
    supine = base.shared.cpu(candidate.parent.supine_at_release)
    right = base.shared.cpu(candidate.right)
    assert candidate.calls == spec['environment']['maximum_control_steps'] - spec['prefix_controls']
    assert not (supine & right).any() and supine.shape == right.shape == (env.num_envs,)
    for role in ROLES[:2]:
        np.testing.assert_array_equal(base.shared.cpu(wrapped[role].supine_at_release), supine)
    path = record / 'route-masks.npz'
    np.savez_compressed(path, supine=supine, right=right)
    state['route_masks_sha256'] = sha(path)


def verify():
    import verify_getup_archived_policy_comparison as scorer

    spec = read(PLAN)
    d = spec['dual_route']
    record = ROOT / spec['record']
    qual = record / 'zero-qualification.json'
    native = read(OUT / 'native-zero.json')
    assert sha(qual) == native['qualification_sha256'] and read(qual)['physical_substeps'] == 0
    assert native['proof']['dual_latch_cuda'] == read(OUT / 'independent-zero/proof.json')['dual_latch_cuda']
    mask_path = record / 'route-masks.npz'
    assert sha(mask_path) == read(record / 'evaluation.json')['route_masks_sha256']
    with np.load(mask_path, allow_pickle=False) as saved:
        supine, right = saved['supine'], saved['right']
    models = dict(fallback=spec['common_fallback'], **spec['policies'])
    fallback, _ = load_onnx({'policies': models}, 'fallback')
    specialist, _ = load_onnx({'policies': models}, 'candidate')
    supine_model, _ = load_onnx({'policies': models}, 'reference-a')
    candidate_name = specialist.get_modelmeta().custom_metadata_map['run_name']
    assert candidate_name != supine_model.get_modelmeta().custom_metadata_map['run_name']
    drift = []

    def compare(a, b):
        if 'motor' not in a:
            return same_bytes(a, b)
        same_bytes({k: v for k, v in a.items() if k != 'motor'}, {k: v for k, v in b.items() if k != 'motor'})
        x, y = a['motor'], b['motor']
        assert x.shape == y.shape and len(x) == 5 and np.isfinite(x).all() and np.isfinite(y).all()
        same_bytes({'first': x[0]}, {'first': y[0]})
        drift.extend(float(abs(v).max()) for v in x.astype(np.float64) - y)

    def replay(t, session):
        candidate = session.get_modelmeta().custom_metadata_map['run_name'] == candidate_name
        a, b = masks(t['actor_input'][d['release_counter']], d['alignment'])
        np.testing.assert_array_equal(a, supine)
        np.testing.assert_array_equal(b, right)
        error, counts = 0., [0, 0, 0]
        for control in (101, 110, 125, 150, 200, 250, 251, 350, 450, 550):
            for i in range(0, len(supine), 1 if control == 101 else 5):
                which = 1 if supine[i] else (2 if candidate and right[i] else 0)
                network = (fallback, supine_model, specialist)[which]
                wanted = network.run(None, {'obs': t['actor_input'][control-1, i:i+1]})[0][0]
                actual = t['raw_policy_actions'][control-1, i]
                np.testing.assert_allclose(actual, wanted, atol=2e-5, rtol=2e-5)
                error = max(error, float(abs(actual-wanted).max()))
                counts[which] += 1
        assert min(counts[:2]) > 0 and (counts[2] > 0) == candidate
        return dict(maximum_absolute_error=error, fallback_inputs=counts[0], supine_inputs=counts[1], right_inputs=counts[2])

    def write_scores(path, result):
        # The inherited verifier's shortlist was for a supine treatment.
        result.pop('candidates_for_second_development')
        write(path, result)

    with patch.multiple(scorer, files=lambda seed: (PLAN, OUT, PAGE), same_bytes=compare, verify_policy=replay, write=write_scores):
        scorer.verify(SEED)
    raw = OUT / 'analysis.json'
    score = read(raw)
    c, actual = success(score['arms']['candidate'])
    coverage, paired = {}, {}
    gate = d['gate']
    for pose, selected in (('supine', supine), ('right-side', right)):
        relevant = actual == pose
        coverage[pose] = dict(actual=int(relevant.sum()), hits=int((selected & relevant).sum()), leaks=int((selected & ~relevant).sum()))
    for role in ROLES[:2]:
        other, _ = success(score['arms'][role], actual)
        paired[role] = scorer.paired(other, c, actual == 'right-side')
    assert len(drift) == 5
    passed = score['arms']['candidate']['combined']['functional_coverage_gate'] and all(
        row['hits'] >= row['actual']*gate['minimum_supine_and_right_latch_coverage'] and
        row['leaks'] <= gate['maximum_latch_leaks'] for row in coverage.values()) and all(
        row['net'] >= gate['minimum_right_net_gain_per_reference'] for row in paired.values())
    write(OUT / 'decision.json', dict(status='completed-right-specialist-development', analysis_sha256=sha(raw),
        coverage=coverage, paired_right=paired, repeated_reference_motor_drift=drift,
        admitted_to_independent_development=passed, heldout_used=False, learning_updates=0))
    print('DUAL_POSE_ROUTE_DECISION', passed, coverage, {role: row['net'] for role, row in paired.items()}, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    if mode in ('zero', 'sample'):
        with patch.multiple(prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=snapshot), \
             patch.object(base.acquisition, 'collect', collect), patch.object(base.shared, 'actor_model', guarded_actor):
            getattr(prior, mode)()
    else:
        {'freeze': freeze, 'verify': verify}[mode]()
