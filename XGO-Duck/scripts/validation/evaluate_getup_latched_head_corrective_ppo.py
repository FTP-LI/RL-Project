"""Compare both trained policies with fixed repeats on fresh shared releases."""

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
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned
from run_getup_teacher_head_supine_latch import observed_mask
from verify_getup_shared_release import same_bytes


base = prior.base
ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
DRAFT = ROOT / 'configs/experiments/getup_latched_head_corrective_evaluation_draft.json'
PLAN = DRAFT.with_stem('getup_latched_head_corrective_evaluation_s1164')
OUT = ROOT / 'records/reports/getup-latched-head-corrective-ppo/evaluation'
PAGE = ROOT / 'records/getup-latched-head-corrective-evaluation-live.html'
STAGE = 'S3-getup-latched-head-corrective-evaluation'
SEED = 1164
original_collect = base.acquisition.collect


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    draft = read(DRAFT)
    assert draft['seed'] == SEED != draft['reserved_holdout_seed']
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    training = pinned(draft['training_protocol'])
    verified_path = ROOT / draft['training_verification']
    verified = read(verified_path)
    assert verified['status'] == 'passed-online-training-integrity' and verified['protocol_sha256'] == sha(training)
    assert verified['admitted_to_development'] and verified['matched_sample_draws'] and not verified['heldout_used']
    environment = pinned(draft['environment_reference'])
    spec = deepcopy(read(environment))
    base.shared.assert_sources(spec)
    p, roles = spec['environment'], draft['branches']
    fallback, fixed = spec['policies']['reference-a'], spec['policies']['candidate']
    policies = {role: dict(fixed, seed=SEED) for role in roles[:2]}
    model_paths = []
    for role in roles[2:]:
        entry = verified['arms'][role]
        path = pinned((entry['model'], entry['model_sha256']))
        policies[role] = dict(model=entry['model'], model_sha256=entry['model_sha256'], seed=SEED)
        model_paths.append(path)
    assert len({policy['model_sha256'] for policy in policies.values()}) == 3
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), draft['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == index).tolist() for index, pose in enumerate(base.shared.POSES)}
    controls = draft['release_counter'] + len(roles) * (p['maximum_control_steps'] - draft['release_counter'])
    assert controls * len(codes) == draft['maximum_transitions'] and 5 * controls == draft['maximum_physics_substeps']
    assert len(codes) == p['num_envs'] and p['success_rate_min'] == draft['gate']['success_rate_min']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_latched-corrective-ppo_development_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, model=fixed['model'], model_sha256=fixed['model_sha256'],
             split=dict(pose_codes=codes.tolist(), groups=groups))
    witness = ROOT / spec['record'] / 'first-control-reference-a.npz'
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), policies=policies, branches=roles,
                common_fallback=dict(fallback, seed=SEED), corrected_ppo_evaluation=draft, latch_witness=base.rel(witness),
                maximum_unique_controls=controls, maximum_physics_substeps=5 * controls,
                cameras={role: {'supine': groups['supine'][0]} for role in roles})
    for path in (DRAFT, training, verified_path, environment, witness, *model_paths):
        spec['references'][base.rel(path)] = sha(path)
    spec['source_sha256'][base.rel(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    base.shared.assert_sources(spec)
    cfg, display, meta = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == len(codes) and cfg.env.episode_length_s == 11.
    assert not cfg.env.auto_reset and not display['native_live_window'] and cfg.env.decimation == 5
    model = Scene(cfg.env.scene, device='cpu').compile()
    base.exact_scene(model)
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in meta['joint_names'].split(',')]
    parity = base.shared.cpu_check({'roles': {'fallback': spec['common_fallback'],
                                             **{role: policies[role] for role in (roles[0], *roles[2:])}}})
    assert not torch.cuda.is_initialized()
    record.mkdir()
    OUT.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-zero', scene_exact=True, actor_import=parity,
        cpu_joint_addresses=addresses, protocol_sha256=sha(PLAN), physical_substeps=0))
    (record / 'README.md').write_text('# 起身纠正回放评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-latched-head-corrective-evaluation-live.html) · [日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身纠正回放同初态评估', run_id=run_id,
        controls=0, maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
    print('GETUP_CORRECTIVE_EVALUATION_FROZEN', base.rel(record), sha(PLAN), flush=True)


def wrap(parent, policy, counter, draft):
    return SupineReleaseActor(parent, policy, counter, draft['release_counter'],
                              draft['latch']['normalized_negative_gravity_x_threshold']).eval()


def snapshot(env, obs, memory, spec, target):
    actors, proof, fields, inputs = original_snapshot(env, obs, memory, spec, target)
    fallback = base.shared.actor_model(spec['common_fallback'], env.device)
    draft = spec['corrected_ppo_evaluation']
    with np.load(ROOT / spec['latch_witness'], allow_pickle=False) as saved:
        source = saved['actor_input']
    mask = observed_mask(source, draft['latch']['normalized_negative_gravity_x_threshold'])
    x = base.shared.td('actor', torch.from_numpy(source).to(env.device))
    with torch.inference_mode():
        for policy in actors.values():
            gate = wrap(fallback, policy, lambda: draft['release_counter'], draft)
            actual = gate(x)
            chosen = gate.supine_at_release
            np.testing.assert_array_equal(base.shared.cpu(chosen), mask)
            assert torch.equal(actual[chosen], policy(x)[chosen])
            assert torch.equal(actual[~chosen], fallback(x)[~chosen])
    assert env.common_step_counter == env._sim_step_counter == 0
    proof['common_latch_cuda'] = dict(roles=len(actors), inputs=len(mask), selected=int(mask.sum()), byte_exact=True)
    return actors, proof, fields, inputs


def collect(env, actors, memory, meta, spec, record, state, publish, check_active):
    fallback = base.shared.actor_model(spec['common_fallback'], env.device)
    draft = spec['corrected_ppo_evaluation']
    gates = {role: wrap(fallback, policy, lambda: env.common_step_counter, draft) for role, policy in actors.items()}
    original_collect(env, gates, memory, meta, spec, record, state, publish, check_active)
    masks = [base.shared.cpu(gate.supine_at_release) for gate in gates.values()]
    assert all(np.array_equal(mask, masks[0]) for mask in masks) and masks[0].shape == (env.num_envs,)
    path = record / 'supine-release-mask.npy'
    np.save(path, masks[0], allow_pickle=False)
    state['supine_release_mask_sha256'] = sha(path)


def verify():
    import onnxruntime as ort
    import verify_getup_archived_policy_comparison as scoring
    from review_getup_archived_policy_teacher_timing import success

    spec = read(PLAN)
    record = ROOT / spec['record']
    latest = read(OUT / 'native-zero.json')
    assert latest['proof']['common_latch_cuda'] == read(OUT / 'independent-zero/proof.json')['common_latch_cuda']
    path = record / 'supine-release-mask.npy'
    assert sha(path) == read(record / 'evaluation.json')['supine_release_mask_sha256']
    saved_mask = np.load(path, allow_pickle=False)
    draft = spec['corrected_ppo_evaluation']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    fallback = ort.InferenceSession(str(ROOT / spec['common_fallback']['model']), sess_options=options,
                                    providers=['CPUExecutionProvider'])
    drift = []

    def compare(first, second):
        if 'motor' not in first:
            return same_bytes(first, second)
        same_bytes({k: v for k, v in first.items() if k != 'motor'}, {k: v for k, v in second.items() if k != 'motor'})
        a, b = first['motor'], second['motor']
        assert np.isfinite(a).all() and np.isfinite(b).all() and a.shape == b.shape and len(a) == 5
        same_bytes({'first': a[0]}, {'first': b[0]})
        drift.extend(float(abs(x).max()) for x in a.astype(np.float64) - b)

    def replay(trace, actor):
        release = draft['release_counter']
        mask = observed_mask(trace['actor_input'][release], draft['latch']['normalized_negative_gravity_x_threshold'])
        np.testing.assert_array_equal(mask, saved_mask)
        maximum, counts = 0., [0, 0]
        for control in (101, 110, 125, 150, 200, 250, 251, 350, 450, 550):
            stride = 1 if control == release + 1 else 5
            for index in range(0, len(mask), stride):
                model = actor if mask[index] else fallback
                x = trace['actor_input'][control-1, index:index+1]
                wanted = model.run(None, {'obs': x})[0][0]
                actual = trace['raw_policy_actions'][control-1, index]
                np.testing.assert_allclose(actual, wanted, atol=2e-5, rtol=2e-5)
                maximum = max(maximum, float(abs(actual - wanted).max()))
                counts[int(mask[index])] += 1
        assert min(counts) > 0
        return dict(maximum_absolute_error=maximum, fallback_inputs=counts[0], supine_inputs=counts[1])

    with patch.multiple(scoring, files=lambda seed: (PLAN, OUT, PAGE), same_bytes=compare, verify_policy=replay):
        scoring.verify(SEED)
    raw = OUT / 'analysis.json'
    result = read(raw)
    candidate, poses = success(result['arms']['candidate'])
    gate = draft['gate']
    comparisons = {}
    for role in spec['branches'][:-1]:
        other, _ = success(result['arms'][role], poses)
        comparisons[role] = {pose: scoring.paired(other, candidate, poses == pose) for pose in base.shared.POSES}
    requested = result['arms']['candidate']['combined']['requested']['supine']['timely_holds']
    supine = poses == 'supine'
    hit, leak = int((saved_mask & supine).sum()), int((saved_mask & ~supine).sum())
    qualified = (result['arms']['candidate']['combined']['functional_coverage_gate'] and
        requested >= gate['minimum_requested_supine_successes'] and
        leak <= draft['latch']['maximum_other_pose_latch_count'] and hit >= draft['latch']['minimum_actual_supine_coverage'] * int(supine.sum()) and
        all(row['supine']['net'] >= gate['minimum_actual_supine_net_gain_per_comparison'] and
            all(row[pose]['net'] >= gate['minimum_non_supine_net_gain_per_pose_and_comparison']
                for pose in base.shared.POSES if pose != 'supine') for row in comparisons.values()))
    assert len(drift) == 5
    write(OUT / 'decision.json', dict(status='completed-paired-development', analysis_sha256=sha(raw),
        comparisons=comparisons, requested_supine_successes=requested, actual_supine=int(supine.sum()),
        selected_supine=hit, selected_other=leak, repeated_reference_motor_drift=drift,
        admitted_to_independent_development=qualified, heldout_used=False, learning_updates=0))
    print('GETUP_CORRECTIVE_EVALUATION_DECISION', qualified, requested, hit, leak, flush=True)


original_snapshot = prior.snapshot


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    if mode in ('zero', 'sample'):
        with patch.multiple(prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=snapshot), \
             patch.object(base.acquisition, 'collect', collect):
            getattr(prior, mode)()
    else:
        {'freeze': freeze, 'verify': verify}[mode]()
