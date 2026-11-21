"""Replicate both qualified getup policies on a fresh development seed."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_getup_archived_policy_qualified as same_process
from getup_release_fork_state import digest
from verify_getup_shared_release import same_bytes


base = same_process.base
ROOT, read, sha, write = base.ROOT, base.read, base.sha, base.write
STAGE = 'S3-getup-archived-policy-replication'
DRAFT = ROOT / 'configs/experiments/getup_archived_policy_replication_draft.json'
PLAN = DRAFT.with_stem('getup_archived_policy_comparison_s1161')
OUT = same_process.OUT.with_name('s1161')
PAGE = ROOT / 'records/getup-archived-policy-s1161-live.html'
SEED = 1161


def phase(protocol):
    base.check_project()
    state = read(ROOT / 'project.json')
    stage = STAGE + '-preparation' if protocol == DRAFT else STAGE
    assert state['current_phase'] == stage and ROOT / state['goal_execution']['current_protocol'] == protocol


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    d = read(DRAFT)
    assert d['development_seed'] == SEED and not d['heldout_used'] and d['physical_substeps_authorized'] == 0
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in entry.name for entry in (ROOT / folder).iterdir())
    prior, proof = (ROOT / d[name][0] for name in ('previous_protocol', 'previous_result'))
    assert sha(prior) == d['previous_protocol'][1] and sha(proof) == d['previous_result'][1]
    result = read(proof)
    spec = deepcopy(read(prior))
    base.shared.assert_sources(spec)
    p, roles = spec['environment'], spec['branches']
    assert result['protocol_sha256'] == sha(prior) and result['candidates_for_second_development'] == roles[2:]
    assert d['roles'] == roles and d['worlds'] == p['num_envs']
    assert spec['maximum_warp_snapshot_bytes'] == d['maximum_warp_snapshot_bytes']
    assert spec['maximum_unique_controls'] * p['num_envs'] == d['maximum_transitions']
    assert spec['maximum_physics_substeps'] == d['maximum_batched_substeps']
    assert spec['draft']['gpu_memory_fraction_cap'] == d['maximum_memory_fraction']
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_archived-policies_replication_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups))
    for policy in spec['policies'].values():
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
                cameras={role: {'supine': groups['supine'][0]} for role in roles},
                replication_draft=d, recovery_from=None, reused_development_seed_zero_prior_physics=False,
                first_substep_exact_required=True, subsequent_repeat_drift_recorded=True)
    for path in (DRAFT, prior, proof):
        spec['references'][base.rel(path)] = sha(path)
    spec['source_sha256'][base.rel(Path(__file__).resolve())] = sha(Path(__file__))
    base.shared.assert_sources(spec)
    cfg, display, meta = base.shared.prepare(p)
    assert not cfg.env.auto_reset and not display['native_live_window'] and cfg.env.decimation == 5
    assert cfg.env.episode_length_s == 11. and cfg.env.scene.num_envs == d['worlds']
    model = Scene(cfg.env.scene, device='cpu').compile()
    base.exact_scene(model)
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in meta['joint_names'].split(',')]
    torch.set_num_threads(4)
    parity = base.shared.cpu_check({'roles': {role: spec['policies'][role] for role in roles if role != roles[1]}})
    assert not torch.cuda.is_initialized()
    record.mkdir()
    OUT.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-zero', scene_exact=True, actor_import=parity,
                                   cpu_joint_addresses=addresses, physical_substeps=0, protocol_sha256=sha(PLAN)))
    (record / 'README.md').write_text('# 归档起身策略独立开发复验\n\n同一释放快照评估四路固定策略。\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-archived-policy-s1161-live.html) · [执行日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='归档起身策略独立开发复验', run_id=run_id, controls=0,
                                    maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('ARCHIVED_POLICY_REPLICATION_FROZEN', base.rel(record), sha(PLAN), flush=True)


def snapshot(env, obs, memory, spec, target):
    class ResetFieldCapture(base.SceneReleaseSnapshot):
        def __init__(self, env, memory, cap):
            super().__init__(env, memory, cap)
            write(target / 'zero-fields.json', self.expected)
            write(target / 'zero-inputs.json', {key: base.shared.signature(value)
                for key, value in base.shared.release_payload(env, memory).items()})

    with patch.object(base, 'SceneReleaseSnapshot', ResetFieldCapture):
        actors, proof = base.native_zero(env, obs, memory, spec, PLAN, target)
    fields, inputs = (read(target / name) for name in ('zero-fields.json', 'zero-inputs.json'))
    assert len(fields) == 335 and len(inputs) == 14
    assert digest(fields) == proof['zero_readback']['initial']['state_sha256']
    return actors, proof, fields, inputs


def zero():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    phase(PLAN)
    spec = read(PLAN)
    base.shared.assert_sources(spec)
    assert sha(PLAN) == read(OUT / 'freeze.json')['protocol_sha256']
    target = OUT / 'independent-zero'
    target.mkdir()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = base.shared.prepare(spec['environment'])
    watch = base.MemoryWatch(spec['draft']['gpu_memory_fraction_cap'])
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
        try:
            obs, _ = env.reset(seed=SEED)
            memory = base.shared.make_memory(spec['environment'], meta, env.device)
            _, proof, _, _ = snapshot(env, obs, memory, spec, target)
            assert env.common_step_counter == env._sim_step_counter == 0
            watch.check()
            watch.sample()
            write(target / 'proof.json', dict(proof, protocol_sha256=sha(PLAN), gpu=watch.summary(),
                hashes={name: sha(target / name) for name in ('zero-fields.json', 'zero-inputs.json', 'zero-readback.json')}))
        finally:
            env.close()
    print('ARCHIVED_POLICY_REPLICATION_NATIVE_ZERO', sha(target / 'proof.json'), flush=True)


def qualify(env, obs, memory, spec, target):
    witness = OUT / 'independent-zero'
    prior = read(witness / 'proof.json')
    assert prior['protocol_sha256'] == sha(PLAN) and prior['physical_substeps'] == 0
    for name, expected in prior['hashes'].items():
        assert sha(witness / name) == expected
    actors, proof, fields, inputs = snapshot(env, obs, memory, spec, target)
    old = read(witness / 'zero-fields.json')
    assert old.keys() == fields.keys() and inputs == read(witness / 'zero-inputs.json')
    changed = {key for key in fields if fields[key] != old[key]}
    allowed = set(spec['allowed_cross_process_fields'])
    assert changed <= allowed, sorted(changed - allowed)
    assert prior['model_max_error'] == proof['model_max_error']
    write(target / 'zero-qualification.json', dict(status='passed-process-local-zero', protocol_sha256=sha(PLAN),
        physical_substeps=0, stable_inputs=len(inputs), matching_snapshot_fields=len(fields),
        differing_derived_fields=sorted(changed), independent_zero_sha256=sha(witness / 'proof.json')))
    return actors, proof


def sample():
    with patch.multiple(same_process, SEED=SEED, PLAN=PLAN, OUT=OUT, PAGE=PAGE, ready_for=phase, qualification=qualify):
        same_process.sample()


def verify():
    import verify_getup_archived_policy_comparison as scoring

    spec = read(PLAN)
    record = ROOT / spec['record']
    qual = read(record / 'zero-qualification.json')
    assert qual['status'] == 'passed-process-local-zero' and qual['physical_substeps'] == 0
    assert read(OUT / 'native-zero.json')['qualification_sha256'] == sha(record / 'zero-qualification.json')
    drift = []

    def compare(first, second):
        if 'motor' not in first:
            return same_bytes(first, second)
        same_bytes({key: value for key, value in first.items() if key != 'motor'},
                   {key: value for key, value in second.items() if key != 'motor'})
        a, b = first['motor'], second['motor']
        assert a.shape == b.shape and a.shape[:2] == (5, spec['environment']['num_envs'])
        same_bytes({'motor': a[0]}, {'motor': b[0]})
        assert np.isfinite(a).all() and np.isfinite(b).all()
        for step, delta in enumerate(np.abs(a.astype(np.float64) - b), 1):
            drift.append(dict(physical_substep=step, max_absolute=float(delta.max()),
                rms=float(np.sqrt(np.mean(delta**2))), worlds_changed=int((delta != 0).any(axis=1).sum())))

    with patch.object(scoring, 'files', lambda seed: (PLAN, OUT, PAGE)), patch.object(scoring, 'same_bytes', compare):
        scoring.verify(SEED)
    assert len(drift) == 5
    write(OUT / 'reference-motor-drift.json', dict(status='measured-native-post-actuation-drift', per_substep=drift,
                                                 protocol_sha256=sha(PLAN), physical_substeps_added=0))
    a = read(ROOT / spec['replication_draft']['previous_result'][0])
    b = read(OUT / 'analysis.json')
    advance = b['candidates_for_second_development']
    ranks = {}
    for role in advance:
        assert role in a['candidates_for_second_development']
        scores = [trial['arms'][role]['combined'] for trial in (a, b)]
        rates = [row['rate'] for score in scores for grouping in ('requested', 'actual') for row in score[grouping].values()]
        ranks[role] = dict(worst_pose=min(rates), pooled_actual_supine=sum(score['actual']['supine']['timely_holds'] for score in scores))
    sorted_roles = sorted(advance, key=lambda role: (ranks[role]['worst_pose'], ranks[role]['pooled_actual_supine']), reverse=True)
    assert len(sorted_roles) < 2 or ranks[sorted_roles[0]] != ranks[sorted_roles[1]], 'Resolve a fully tied ranking before holdout'
    write(OUT / 'selection.json', dict(status='completed-development-selection', candidates=ranks,
        selected_for_holdout=sorted_roles[0] if sorted_roles else None, holdout_seed=spec['replication_draft']['reserve_final_holdout_seed'],
        heldout_used=False, first_score_sha256=spec['replication_draft']['previous_result'][1], second_score_sha256=sha(OUT / 'analysis.json')))
    print('ARCHIVED_POLICY_REPLICATION_SELECTION', ranks, sorted_roles[:1], flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    {'freeze': freeze, 'zero': zero, 'sample': sample, 'verify': verify}[parser.parse_args().mode]()
