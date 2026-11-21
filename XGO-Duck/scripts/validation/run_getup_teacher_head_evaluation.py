"""Verify the trained residual and run a fresh same-release CUDA comparison."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_getup_archived_policy_replication as prior
from verify_getup_shared_release import same_bytes


base = prior.base
ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
DRAFT = ROOT / 'configs/experiments/getup_archived_policy_teacher_head_evaluation_draft.json'
PLAN = DRAFT.with_stem('getup_archived_policy_teacher_head_evaluation_s1162')
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/s1162'
PAGE = ROOT / 'records/getup-teacher-head-s1162-live.html'
STAGE = 'S3-getup-teacher-head-evaluation'
SEED = 1162


def phase(protocol):
    base.check_project()
    state = read(ROOT / 'project.json')
    name = STAGE + '-preparation' if protocol == DRAFT else STAGE
    assert state['current_phase'] == name and ROOT / state['goal_execution']['current_protocol'] == protocol


def pinned(pointer):
    name, checksum = pointer
    path = ROOT / name
    assert sha(path) == checksum, name
    return path


def envelope(d):
    path = pinned(d['envelope'])
    result = read(path)
    assert result['admitted_to_native_preparation'] and not result['heldout_used']
    return path, result


def checkpoint():
    phase(DRAFT)
    d = read(DRAFT)
    receipt, result = envelope(d)
    folder = receipt.parent
    target = folder / 'warm-start.pt'
    assert not target.exists()
    fit = read(pinned(d['fit_protocol']))
    parent = pinned(fit['parent_checkpoint'])
    model = pinned((result['model'], result['model_sha256']))
    head = pinned((result['weight_archive'], result['weight_archive_sha256']))
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    original = base.shared.signature(payload)
    actor = payload['actor_state_dict']
    head_names = {'mlp.6.weight': 'weight', 'mlp.6.bias': 'bias'}
    with np.load(head, allow_pickle=False) as tensors:
        for key, name in head_names.items():
            v = torch.from_numpy(tensors[name].copy())
            assert v.shape == actor[key].shape and v.dtype == actor[key].dtype
            assert not torch.equal(v, actor[key])
            actor[key] = v
    payload['actor_optimizer_state_dict']['state'] = {}
    counters = ('completed_updates', 'actor_steps_this_run', 'critic_steps_this_run', 'control_steps')
    for key in counters:
        payload[key] = 0
    torch.save(payload, target)
    loaded = torch.load(target, map_location='cpu', weights_only=True)
    assert base.shared.signature(loaded) == base.shared.signature(payload)
    source = torch.load(parent, map_location='cpu', weights_only=True)
    assert base.shared.signature(source) == original
    changed = [key for key in actor if not torch.equal(actor[key], source['actor_state_dict'][key])]
    assert set(changed) == set(head_names)
    assert all(base.shared.signature(loaded[key]) == base.shared.signature(source[key]) for key in source
               if key not in ('actor_state_dict', 'actor_optimizer_state_dict', *counters))
    assert not loaded['actor_optimizer_state_dict']['state']
    assert base.shared.signature(loaded['actor_optimizer_state_dict']['param_groups']) == base.shared.signature(source['actor_optimizer_state_dict']['param_groups'])
    replay = read(pinned(fit['replay']))
    witnesses = []
    for row in replay['files']['cross_archive'].values():
        with np.load(pinned((row['path'], row['sha256'])), allow_pickle=False) as data:
            witnesses.append(data['observations'])
    x = torch.from_numpy(np.concatenate(witnesses))
    native = base.shared.actor_model(dict(model=base.rel(model), model_sha256=sha(model), seed=SEED), device='cpu')
    restored = deepcopy(native)
    restored.load_state_dict(loaded['actor_state_dict'], strict=True)
    with torch.inference_mode():
        a, b = (network(base.shared.td('actor', x)) for network in (native, restored))
        error = float((a - b).abs().max())
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert not torch.cuda.is_initialized()
    write(folder / 'checkpoint-readback.json', dict(status='passed-native-warm-start-readback',
        checkpoint=base.rel(target), checkpoint_sha256=sha(target), parent_checkpoint_sha256=sha(parent),
        source_head_archive_sha256=sha(head), model_sha256=sha(model), changed_actor_tensors=changed,
        actor_optimizer_state_reset=True, critic_optimizer_retained=True, counter_values={key: loaded[key] for key in counters},
        compared_observations=len(x), maximum_absolute_output_error_rad=error, physical_substeps=0,
        learning_updates=0, heldout_used=False))
    print('TEACHER_HEAD_WARM_START_RELOADED', sha(target), 'error', error, flush=True)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    d = read(DRAFT)
    assert d['seed'] == SEED and d['reserved_holdout_seed'] != SEED
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    env_file = pinned(d['native_scene'])
    fit_file = pinned(d['fit_protocol'])
    fit = read(fit_file)
    env_receipt, value = envelope(d)
    reload_path = env_receipt.parent / 'checkpoint-readback.json'
    reload = read(reload_path)
    assert reload['status'] == 'passed-native-warm-start-readback' and not reload['heldout_used']
    checkpoint_path = pinned((reload['checkpoint'], reload['checkpoint_sha256']))
    parent = pinned(fit['parent_model'])
    candidate = pinned((value['model'], value['model_sha256']))
    assert sha(candidate) == reload['model_sha256']
    spec = deepcopy(read(env_file))
    base.shared.assert_sources(spec)
    p, roles = spec['environment'], d['branches']
    controls = spec['prefix_controls'] + len(roles) * (p['maximum_control_steps'] - spec['prefix_controls'])
    assert controls * d['worlds'] == d['maximum_transitions'] and controls * 5 == d['maximum_batched_physics_substeps']
    assert p['num_envs'] == d['worlds'] and spec['draft']['gpu_memory_fraction_cap'] == d['maximum_gpu_memory_fraction']
    assert spec['draft']['development_gate']['minimum_actual_supine_net_gain_per_reference'] == 2
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_teacher-head_development_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, model=base.rel(parent), model_sha256=sha(parent),
             split=dict(pose_codes=codes.tolist(), groups=groups))
    policies = {role: dict(seed=SEED, model=base.rel(model), model_sha256=sha(model))
                for role, model in zip(roles, (parent, parent, candidate), strict=True)}
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), branches=roles, policies=policies,
                maximum_unique_controls=controls, maximum_physics_substeps=5 * controls, teacher_head_draft=d,
                cameras={role: {'supine': groups['supine'][0]} for role in roles})
    for path in (DRAFT, env_file, fit_file, env_receipt, reload_path, checkpoint_path, parent, candidate):
        spec['references'][base.rel(path)] = sha(path)
    source = Path(__file__).resolve()
    spec['source_sha256'][base.rel(source)] = sha(source)
    base.shared.assert_sources(spec)
    cfg, display, meta = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == d['worlds'] and cfg.env.decimation == 5 and cfg.env.episode_length_s == 11.
    assert not cfg.env.auto_reset and not display['native_live_window']
    model = Scene(cfg.env.scene, device='cpu').compile()
    base.exact_scene(model)
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in meta['joint_names'].split(',')]
    parity = base.shared.cpu_check({'roles': {role: policies[role] for role in (roles[0], roles[2])}})
    assert not torch.cuda.is_initialized()
    record.mkdir()
    OUT.mkdir(parents=True)
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-zero', scene_exact=True, actor_import=parity,
                                   cpu_joint_addresses=addresses, physical_substeps=0, protocol_sha256=sha(PLAN)))
    (record / 'README.md').write_text('# 起身末层候选评估\n\n从同一释放快照比较两路父策略与候选。\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-teacher-head-s1162-live.html) · [执行日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身末层候选同初态对照', run_id=run_id, controls=0,
                                    maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
    print('TEACHER_HEAD_EVALUATION_FROZEN', base.rel(record), sha(PLAN), flush=True)


def verify():
    import verify_getup_archived_policy_comparison as scoring

    spec = read(PLAN)
    record = ROOT / spec['record']
    qualification = read(record / 'zero-qualification.json')
    assert qualification['status'] == 'passed-process-local-zero'
    assert read(OUT / 'native-zero.json')['qualification_sha256'] == sha(record / 'zero-qualification.json')
    drift = []

    def check_pair(first, second):
        if 'motor' not in first:
            return same_bytes(first, second)
        same_bytes({key: value for key, value in first.items() if key != 'motor'},
                   {key: value for key, value in second.items() if key != 'motor'})
        a, b = first['motor'], second['motor']
        assert a.shape == b.shape and a.shape[:2] == (5, spec['environment']['num_envs'])
        same_bytes({'motor': a[0]}, {'motor': b[0]})
        assert np.isfinite(a).all() and np.isfinite(b).all()
        drift.extend(float(abs(x).max()) for x in a.astype(np.float64) - b)

    with patch.object(scoring, 'files', lambda seed: (PLAN, OUT, PAGE)), patch.object(scoring, 'same_bytes', check_pair):
        scoring.verify(SEED)
    assert len(drift) == 5
    raw = OUT / 'analysis.json'
    report = read(raw)
    poses = report['comparisons']['candidate']
    preserved = all(item['by_pose'][pose]['net'] >= 0 for item in poses.values()
                    for pose in ('prone', 'left-side', 'right-side'))
    promoted = 'candidate' in report['candidates_for_second_development'] and preserved
    write(OUT / 'decision.json', dict(status='completed-new-seed-head-evaluation', analysis_sha256=sha(raw),
        repeated_parent_motor_drift_max_by_substep_rad=drift, original_candidate_gate='candidate' in report['candidates_for_second_development'],
        non_supine_actual_counts_preserved=preserved, admitted_to_independent_development=promoted,
        heldout_used=False, learning_updates=0))
    print('TEACHER_HEAD_EVALUATION_DECISION', promoted, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('checkpoint', 'freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    if mode in ('zero', 'sample'):
        with patch.multiple(prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase):
            getattr(prior, mode)()
    else:
        {'checkpoint': checkpoint, 'freeze': freeze, 'verify': verify}[mode]()
