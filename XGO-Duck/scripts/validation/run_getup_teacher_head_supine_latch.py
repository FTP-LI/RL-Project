"""Freeze and verify a fresh same-release evaluation of the latched residual."""

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
from getup_supine_release_actor import SupineReleaseActor
from run_getup_teacher_head_evaluation import pinned
from verify_getup_shared_release import same_bytes


base = prior.base
ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
DRAFT = ROOT / 'configs/experiments/getup_teacher_head_supine_latch_evaluation_draft.json'
PLAN = DRAFT.with_stem('getup_teacher_head_supine_latch_evaluation_s1163')
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/s1163-latch'
PAGE = ROOT / 'records/getup-teacher-head-latch-s1163-live.html'
STAGE = 'S3-getup-teacher-head-supine-latch-evaluation'
SEED = 1163
COMPONENT = ROOT / 'scripts/training/getup_supine_release_actor.py'


def phase(protocol):
    base.check_project()
    state = read(ROOT / 'project.json')
    assert state['current_phase'] == (STAGE + '-preparation' if protocol == DRAFT else STAGE)
    assert ROOT / state['goal_execution']['current_protocol'] == protocol


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    d = read(DRAFT)
    assert d['seed'] == SEED and d['reserved_holdout_seed'] != SEED
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source_file, component_file = (pinned(d[key]) for key in ('source_protocol', 'component'))
    spec, component = deepcopy(read(source_file)), read(component_file)
    base.shared.assert_sources(spec)
    assert component['status'] == 'passed-native-supine-latch-zero-step' and component['component_sha256'] == sha(COMPONENT)
    assert component['physical_substeps'] == 0 and not component['heldout_used']
    p, roles = spec['environment'], d['branches']
    assert roles == spec['branches'] and spec['prefix_controls'] == d['release_counter']
    controls = d['release_counter'] + len(roles) * (p['maximum_control_steps'] - d['release_counter'])
    assert p['num_envs'] == d['worlds'] and controls == spec['maximum_unique_controls']
    assert controls * d['worlds'] == d['maximum_transitions'] and 5 * controls == d['maximum_batched_physics_substeps']
    assert spec['draft']['gpu_memory_fraction_cap'] == d['maximum_gpu_memory_fraction']
    assert spec['draft']['development_gate']['minimum_actual_supine_net_gain_per_reference'] == 2
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_teacher-head-latch_development_s{SEED}_{stamp}'
    old = ROOT / spec['record'] / 'first-control-reference-a.npz'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups))
    for policy in spec['policies'].values():
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), latch=d, latch_witness=base.rel(old),
                cameras={role: {'supine': groups['supine'][0]} for role in roles})
    for path in (DRAFT, source_file, component_file, old):
        spec['references'][base.rel(path)] = sha(path)
    for path in (Path(__file__).resolve(), COMPONENT):
        spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    cfg, display, meta = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == d['worlds'] and cfg.env.decimation == 5 and cfg.env.episode_length_s == 11.
    assert not cfg.env.auto_reset and not display['native_live_window']
    model = Scene(cfg.env.scene, device='cpu').compile()
    base.exact_scene(model)
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in meta['joint_names'].split(',')]
    parity = base.shared.cpu_check({'roles': {role: spec['policies'][role] for role in (roles[0], roles[2])}})
    assert not torch.cuda.is_initialized()
    record.mkdir()
    OUT.mkdir(parents=True)
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-zero', scene_exact=True, actor_import=parity,
                                   cpu_joint_addresses=addresses, physical_substeps=0, protocol_sha256=sha(PLAN)))
    (record / 'README.md').write_text('# 仰卧锁存起身评估\n\n同一释放快照的三路固定策略。\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-teacher-head-latch-s1163-live.html) · [日志](execution.log)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身残差仰卧锁存对照', run_id=run_id, controls=0,
                                    maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
    print('TEACHER_HEAD_LATCH_FROZEN', base.rel(record), sha(PLAN), flush=True)


def observed_mask(x, threshold):
    gravity = x[:, 3:6].astype(np.float64)
    norm = np.linalg.norm(gravity, axis=1)
    assert (norm > 0).all() and np.isfinite(norm).all()
    return -gravity[:, 0] / np.maximum(norm, 1e-6) >= threshold


original_snapshot = prior.snapshot
original_collect = base.acquisition.collect


def snapshot_with_latch(env, obs, memory, spec, target):
    actors, proof, fields, inputs = original_snapshot(env, obs, memory, spec, target)
    d = spec['latch']
    with np.load(ROOT / spec['latch_witness'], allow_pickle=False) as saved:
        x = saved['actor_input']
    witness = observed_mask(x, d['normalized_negative_gravity_x_threshold'])
    td = base.shared.td('actor', torch.from_numpy(x).to(env.device))
    parent, candidate = actors['reference-a'], actors['candidate']
    gate = SupineReleaseActor(parent, candidate, lambda: d['release_counter'], d['release_counter'],
                              d['normalized_negative_gravity_x_threshold']).eval()
    with torch.inference_mode():
        selected = gate(td)
        mask = gate.supine_at_release
        np.testing.assert_array_equal(base.shared.cpu(mask), witness)
        assert torch.equal(selected[~mask], parent(td)[~mask]) and torch.equal(selected[mask], candidate(td)[mask])
    assert env.common_step_counter == env._sim_step_counter == 0
    proof['archived_latch_cuda'] = dict(inputs=len(witness), selected=int(witness.sum()), byte_exact=True)
    return actors, proof, fields, inputs


def collect_with_latch(env, actors, memory, meta, spec, record, state, publish, check_active):
    d = spec['latch']
    gate = SupineReleaseActor(actors['reference-a'], actors['candidate'], lambda: env.common_step_counter,
                              d['release_counter'], d['normalized_negative_gravity_x_threshold']).eval()
    original_collect(env, {**actors, 'candidate': gate}, memory, meta, spec, record, state, publish, check_active)
    mask = base.shared.cpu(gate.supine_at_release)
    assert mask.dtype == np.bool_ and mask.shape == (env.num_envs,)
    path = record / 'supine-release-mask.npy'
    np.save(path, mask, allow_pickle=False)
    state['supine_release_mask_sha256'] = sha(path)
    print('TEACHER_HEAD_LATCH_EXECUTED', int(mask.sum()), 'of', len(mask), flush=True)


def verify():
    import onnxruntime as ort
    import verify_getup_archived_policy_comparison as scoring

    spec = read(PLAN)
    record = ROOT / spec['record']
    qual = read(record / 'zero-qualification.json')
    startup = read(OUT / 'native-zero.json')
    assert qual['status'] == 'passed-process-local-zero' and startup['qualification_sha256'] == sha(record / 'zero-qualification.json')
    independent = read(OUT / 'independent-zero' / 'proof.json')
    assert startup['proof']['archived_latch_cuda'] == independent['archived_latch_cuda']
    path = record / 'supine-release-mask.npy'
    assert sha(path) == read(record / 'evaluation.json')['supine_release_mask_sha256']
    saved_mask = np.load(path, allow_pickle=False)
    original_verify = scoring.verify_policy
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    parent = ort.InferenceSession(str(ROOT / spec['policies']['reference-a']['model']), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    candidate = ort.InferenceSession(str(ROOT / spec['policies']['candidate']['model']), sess_options=options,
                                     providers=['CPUExecutionProvider'])
    candidate_id = candidate.get_modelmeta().custom_metadata_map['run_name']
    assert candidate_id != parent.get_modelmeta().custom_metadata_map['run_name']
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

    def replay(t, session):
        if session.get_modelmeta().custom_metadata_map['run_name'] != candidate_id:
            return original_verify(t, session)
        d = spec['latch']
        release = d['release_counter']
        mask = observed_mask(t['actor_input'][release], d['normalized_negative_gravity_x_threshold'])
        np.testing.assert_array_equal(mask, saved_mask)
        error, counts = 0., [0, 0]
        for k in (101, 110, 125, 150, 200, 250, 251, 350, 450, 550):
            stride = 1 if k == release + 1 else 5
            for i in range(0, len(mask), stride):
                actor = session if mask[i] else parent
                wanted = actor.run(None, {'obs': t['actor_input'][k-1, i:i+1]})[0][0]
                found = t['raw_policy_actions'][k-1, i]
                np.testing.assert_allclose(found, wanted, atol=2e-5, rtol=2e-5)
                error = max(error, float(abs(found - wanted).max()))
                counts[int(mask[i])] += 1
        assert min(counts) > 0
        return dict(maximum_absolute_error=error, actual_inputs_replayed=sum(counts),
                    parent_inputs=counts[0], residual_inputs=counts[1], history_checked_all_rows=True)

    with patch.multiple(scoring, files=lambda seed: (PLAN, OUT, PAGE), same_bytes=check_pair, verify_policy=replay):
        scoring.verify(SEED)
    assert len(drift) == 5
    raw = OUT / 'analysis.json'
    report = read(raw)
    pairs = report['comparisons']['candidate'].values()
    preserved = all(row['by_pose'][pose]['net'] >= 0 for row in pairs for pose in ('prone', 'left-side', 'right-side'))
    episodes = report['arms']['candidate']['timely']['episodes']
    actual_supine = np.array([row['release_pose'] == 'supine' for row in episodes])
    hit = int((saved_mask & actual_supine).sum())
    leak = int((saved_mask & ~actual_supine).sum())
    d = spec['latch']
    latch_gate = leak <= d['maximum_other_pose_latch_count'] and hit >= d['minimum_supine_latch_coverage'] * int(actual_supine.sum())
    original_gate = 'candidate' in report['candidates_for_second_development']
    write(OUT / 'decision.json', dict(status='completed-new-seed-latched-head-evaluation', analysis_sha256=sha(raw),
        original_candidate_gate=original_gate, non_supine_actual_counts_preserved=preserved, latch_gate=latch_gate,
        actual_supine=int(actual_supine.sum()), selected_supine=hit, selected_other=leak,
        repeated_parent_motor_drift_raw_by_substep=drift, admitted_to_independent_development=original_gate and preserved and latch_gate,
        heldout_used=False, learning_updates=0))
    print('TEACHER_HEAD_LATCH_EVALUATION_DECISION', original_gate, preserved, latch_gate, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    if mode in ('zero', 'sample'):
        with patch.multiple(prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=snapshot_with_latch), \
                patch.object(base.acquisition, 'collect', collect_with_latch):
            getattr(prior, mode)()
    else:
        {'freeze': freeze, 'verify': verify}[mode]()
