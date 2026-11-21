"""Prepare the paired 512-world training scene on CPU, without dynamics."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

from s1_reference_inputs import ROOT, read, sha

sys.path.insert(0, str(ROOT / 'scripts/training'))
from common_scene_contract import compare_arrays, model_arrays
from evaluate_roll_reset_pair import differences
from getup_coverage_metrics import POSES
from getup_teacher_anchor_train import prepare
from record_io import write


DRAFT = ROOT / 'configs/experiments/getup_supine_weighted_learning_draft.json'
OUT = ROOT / 'records/reports/getup-supine-weighted-learning'


def training_plans(draft):
    parent = draft['parent']
    historical = read(ROOT/parent['training_protocol'])['training_arms'][parent['training_arm']]
    proposed, factor = draft['proposed_first_pair'], draft['single_factor']
    seed, n = proposed['training_seed'], proposed['num_envs']
    allocation = proposed['reset_pose_counts_prone_supine_left_right']
    assert len(allocation) == len(POSES) and sum(allocation) == n
    codes = np.random.default_rng(10_000 + seed).permutation(np.repeat(np.arange(len(POSES)), allocation))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(POSES)}
    common = deepcopy(historical)
    common.update(
        model=parent['model'], model_sha256=parent['model_sha256'], parent_checkpoint=parent['checkpoint'],
        parent_checkpoint_sha256=parent['checkpoint_sha256'], seed=seed, num_envs=n,
        split=dict(pose_codes=codes.tolist(), groups=groups),
        training_updates=proposed['maximum_updates_per_arm'], anchor_seed=54_000 + seed,
        actual_release_control=factor['actual_release_control'], focus_interval=factor['active_control_interval'],
        release_pose_minimum_alignment=factor['release_pose_minimum_alignment'],
    )
    common['learner']['minibatch_seed'] = 5_500 + seed
    # Start from the saved model/Adam, but do not carry the previous run's
    # episode, replay or bookkeeping identity into the new comparison.
    for name in ('run_id', 'matching_control', 'arm', 'forced_video_boundaries'):
        common.pop(name, None)
    return historical, {role: dict(deepcopy(common), arm=role, focus_weight=factor[f'{role}_weight'])
                        for role in ('control', 'candidate')}


def main():
    import torch
    from mjlab.scene import Scene

    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and not draft['learning_started']
    cpu = read(OUT/'checkpoint-preflight.json')
    optimizer = read(OUT/'optimizer-preflight.json')
    assert cpu['draft_sha256'] == optimizer['draft_sha256'] == sha(DRAFT)
    assert cpu['status'] == 'passed-cpu-checkpoint-and-archived-release'
    assert optimizer['status'] == 'passed-cpu-temporary-optimizer-fixture'
    assert not torch.cuda.is_initialized()
    historical, plans = training_plans(draft)
    common, trial = plans['control'], plans['candidate']
    assert {key for key in common if common[key] != trial[key]} == {'arm', 'focus_weight'}
    proposed = draft['proposed_first_pair']
    n, updates, length = common['num_envs'], common['training_updates'], proposed['rollout_control_steps']
    assert (n, updates, length) == (512, 50, 24)
    assert n*updates*length == proposed['maximum_learning_transitions_per_arm']
    assert 2*n*updates*length == proposed['maximum_learning_transitions_total']
    assert n*length % common['learner']['minibatches'] == 0
    train_seed, eval_seed = common['seed'], proposed['evaluation_development_seed']
    assert not any('getup' in d.name and f'_s{train_seed}_' in d.name for d in (ROOT/'records/experiments').iterdir())
    for base in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{eval_seed}_' in d.name for d in (ROOT/base).iterdir())

    parent_cfg, _, parent_meta = prepare(historical)
    cfg, display, meta = prepare(common)
    candidate_cfg, _, candidate_meta = prepare(trial)
    assert meta == parent_meta == candidate_meta
    actual = asdict(cfg.env)
    old = asdict(parent_cfg.env)
    assert not differences(actual, asdict(candidate_cfg.env))
    changes = {'.seed', '.scene.num_envs', '.scene.terrain.num_envs', '.events.set_ground_state.params.pose_codes'}
    # Align just the declared allocation and seed before the full comparison.
    for field in changes:
        *parts, key = field[1:].split('.')
        before, after = old, actual
        for part in parts:
            before, after = before[part], after[part]
        before[key] = after[key]
    remaining = differences(old, actual)
    assert not remaining, remaining[:5]
    assert cfg.env.scene.num_envs == cfg.env.scene.terrain.num_envs == n
    assert cfg.env.episode_length_s == 8. and cfg.env.auto_reset and cfg.env.is_finite_horizon
    assert cfg.env.decimation == 5 and cfg.env.sim.mujoco.timestep == .004
    assert not cfg.env.rewards['late_head_leg_contact'].params['enabled'] and not display['native_live_window']
    assert len(cfg.env.events['set_ground_state'].params['pose_codes']) == n
    assert not torch.cuda.is_initialized()

    model = Scene(cfg.env.scene, device='cpu').compile()
    fields = model_arrays(model)
    archived = ROOT/'records/reports/getup-teacher-anchor-pair/cpu-scene-fields.npz'
    with np.load(archived) as previous:
        compare_arrays(fields, {name: previous[name] for name in previous.files})
    joints = [model.joint('robot/' + name).id for name in meta['joint_names'].split(',')]
    assert model.jnt_qposadr[joints].tolist() == common['qpos_addresses']
    np.testing.assert_allclose(model.jnt_range[joints], common['joint_limits'], atol=1e-7, rtol=0)
    path = OUT/'scene-static-fields.npz'
    np.savez_compressed(path, **fields)
    write(OUT/'scene-preflight.json', dict(
        status='passed-cpu-scene-zero-step', draft_sha256=sha(DRAFT), training_seed=train_seed,
        evaluation_seed=eval_seed, num_envs=n, reset_counts={k:len(v) for k,v in common['split']['groups'].items()},
        training_arms_same_config=True, declared_training_config_differences=sorted(changes),
        matched_parent_compiled_fields=len(fields), static_fields_sha256=sha(path),
        checkpoint_preflight_sha256=sha(OUT/'checkpoint-preflight.json'),
        optimizer_preflight_sha256=sha(OUT/'optimizer-preflight.json'),
        parent_scene_sha256=sha(archived), script_sha256=sha(Path(__file__)),
        training_started=False, physical_substeps=0, cuda_initialized=False,
        learning_transitions_per_arm=n*updates*length,
    ))
    print('SUPINE_SCENE_CPU_CHECK', n, 'worlds,', len(fields), 'matched static fields, seeds', train_seed, eval_seed,
          'no dynamics', flush=True)


if __name__ == '__main__':
    main()
