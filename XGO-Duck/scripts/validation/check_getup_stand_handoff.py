"""CPU admission checks for the prospective recovery/standing interface.

Exercise the real processor through freeze, model switch and partial reset;
verify the supplied saved measurements, independent model histories, target
range-excess monotonicity and prospective common noise. No physics stepping.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json, math, sys
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/training'), str(ROOT/'scripts/validation')]
from s1_reference_inputs import read, sha
from record_io import write
from getup_neck_relief_pair import checked
from getup_release_coverage import prepare as getup_prepare
from common_plant_screen import prepare as common_prepare
from common_scene_contract import model_arrays, compare_arrays
from getup_stand_handoff import (HandoffState, HandoffAction, apply_common_actor_noise,
                                 adapt_action, projected_slew, NOISE)

OUT = ROOT/'records/reports/getup-stand-handoff'


def excess(target, limits):
    return torch.maximum((limits[:, 0]-target).clamp_min(0),
                         (target-limits[:, 1]).clamp_min(0))


def check_slew():
    rng = torch.Generator().manual_seed(9101)
    previous = torch.randn(2000, 14, generator=rng)*3
    desired = torch.randn(2000, 14, generator=rng)*4
    limits = torch.tensor([[-1., 1.]]*14)
    actual, destination = projected_slew(previous, desired, limits, .02, 2.)
    assert ((actual-previous).abs() <= .040001).all()
    assert (excess(actual, limits) <= excess(previous, limits)+1e-6).all()
    assert (destination.abs() <= 1).all()
    inside = (previous.abs() <= 1)
    assert (actual[inside].abs() <= 1.000001).all()
    # Gaussian fixture starts are unbounded: derive convergence time from the
    # actual maximum distance, rather than assuming every request is <16rad.
    endpoint_steps = math.ceil(float((destination-previous).abs().max())/.04)+1
    for _ in range(endpoint_steps):
        actual, _ = projected_slew(actual, desired, limits, .02, 2.)
    torch.testing.assert_close(actual, destination, atol=1e-6, rtol=0)
    try:
        projected_slew(previous, desired*float('nan'), limits, .02, 2.)
    except AssertionError:
        pass
    else:
        raise AssertionError('Nonfinite desired request accepted')
    return dict(random_requests=28000, endpoint_steps_checked=endpoint_steps,
                slew_limit=True, no_increase_in_old_excess=True,
                inside_stays_inside=True, legal_fixed_destination_reached=True,
                nonfinite_rejected=True, instantaneous_range_guarantee=False)


def real_processor(cfg, source_home, destination_home):
    n = 4
    action = HandoffAction.__new__(HandoffAction)
    action.cfg = cfg.actions['joint_pos']
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02,
                          episode_length_buf=torch.zeros(n, dtype=torch.long),
                          action_manager=SimpleNamespace(_action=torch.zeros(n, 14)))
    action._env = env
    action._raw_actions = torch.zeros(n, 14)
    action._scale = 1.
    action._offset = torch.tensor(source_home)
    action.previous_target = torch.zeros(n, 14)
    action.previous_valid = torch.zeros(n, dtype=torch.bool)
    action.standing_mask = torch.zeros(n, dtype=torch.bool)
    action.requested_target = torch.zeros(n, 14)
    action.projected_destination = torch.zeros(n, 14)
    limits = torch.tensor(action.cfg.physical_limits)
    memory = HandoffState(source_home, destination_home, n, enabled=True)
    source_raw = torch.full((n, 14), .1)
    source_raw[:, 6] = -4.
    observed = torch.zeros(n, 61)
    observed[:, 5] = -1
    observed[1, 3:6] = torch.tensor([0., 0., 1.])
    observed[2, 3:6] = torch.tensor([1., 0., 0.])
    physical_home = torch.tensor(action.cfg.physical_freeze_home)
    switches = {}
    for age in range(301):
        env.episode_length_buf[:] = age
        if age == 260:
            observed[2, 3:6] = torch.tensor([0., 0., -1.])
        old_source = memory.source_history.clone()
        old_destination = memory.destination_history.clone()
        a, b = memory.inputs(observed, env.episode_length_buf, .02)
        torch.testing.assert_close(a[:, 34:48], old_source, atol=0, rtol=0)
        torch.testing.assert_close(b[:, 34:48], old_destination, atol=0, rtol=0)
        torch.testing.assert_close(b[:, 6:20]+torch.tensor(destination_home),
                                   observed[:, 6:20]+torch.tensor(source_home), atol=1e-7, rtol=0)
        if age == 250:
            assert memory.standing.tolist() == [True, False, False, True]
            assert not b[:, 34:48].any()
        if age == 260:
            assert memory.standing.tolist() == [True, False, True, True]
            assert not b[2, 34:48].any()
        destination_raw = torch.full((n, 14), -.2+age*.0001)
        selected = memory.select(source_raw, destination_raw, env.episode_length_buf, .02)
        if age < 100:
            assert not selected.any() and not memory.source_history.any()
        previous = action.previous_target.clone()
        sent = selected.clone()
        action.standing_mask = memory.standing.clone()
        env.action_manager._action = sent.clone()
        action.process_actions(sent)
        torch.testing.assert_close(sent, selected, atol=0, rtol=0)
        torch.testing.assert_close(env.action_manager._action, selected, atol=0, rtol=0)
        if age < 100:
            torch.testing.assert_close(action._processed_actions, physical_home.expand(n, -1), atol=0, rtol=0)
        elif age < 250:
            torch.testing.assert_close(action._processed_actions, selected+torch.tensor(source_home), atol=0, rtol=0)
        else:
            assert ((action._processed_actions-previous).abs() <= .040001).all()
            assert (excess(action._processed_actions, limits) <= excess(previous, limits)+1e-6).all()
            torch.testing.assert_close(action.requested_target[memory.standing],
                (destination_raw+torch.tensor(destination_home))[memory.standing], atol=0, rtol=0)
        if age in (99, 100, 249, 250, 260, 300):
            switches[str(age)] = memory.standing.tolist()
    # A partial real episode reset must clear only that world's histories/latch
    # and previous request. It must not alter the other continuous worlds.
    kept = action.previous_target[1:].clone()
    kept_history = memory.destination_history[1:].clone()
    memory.reset(torch.tensor([0]))
    action.reset(torch.tensor([0]))
    assert not memory.standing[0] and not action.previous_valid[0]
    assert not memory.source_history[0].any() and not memory.destination_history[0].any()
    torch.testing.assert_close(action.previous_target[1:], kept, atol=0, rtol=0)
    torch.testing.assert_close(memory.destination_history[1:], kept_history, atol=0, rtol=0)
    env.episode_length_buf[0] = 0
    memory.inputs(observed, env.episode_length_buf, .02)
    selected = memory.select(source_raw, destination_raw, env.episode_length_buf, .02)
    action.standing_mask = memory.standing.clone()
    env.action_manager._action = selected.clone()
    action.process_actions(selected)
    torch.testing.assert_close(action._processed_actions[0], physical_home, atol=0, rtol=0)
    # Disabled control computes the same trigger but never applies a handoff.
    control = HandoffState(source_home, destination_home, n, enabled=False)
    control.inputs(observed, torch.full((n,), 300), .02)
    assert control.shadow_latched.any() and not control.standing.any()
    return dict(status='passed',real_processor=True,freeze_exact=True,original_inputs_unchanged=True,
                independent_raw_histories=True,home_correct=True,transition_boundary_steps=switches,
                partial_reset=True,unselected_worlds_preserved=True,disabled_control_checked=True)


def main():
    assert not torch.cuda.is_initialized()
    master = checked()
    source = master['arms']['control']
    walking = read(ROOT/'configs/experiments/common_plant/walk.json')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {role: ort.InferenceSession(str(ROOT/p['model']), sess_options=options,
                         providers=['CPUExecutionProvider']) for role, p in [('source', source), ('stand', walking)]}
    meta = {role: s.get_modelmeta().custom_metadata_map for role, s in sessions.items()}
    assert meta['source']['joint_names'] == meta['stand']['joint_names']
    homes = {role: list(map(float, m['default_joint_pos'].split(','))) for role, m in meta.items()}
    original = getup_prepare(source)[0].env
    cfg = adapt_action(apply_common_actor_noise(original), homes['stand'], walking['joint_limits'])
    stand_cfg = apply_common_actor_noise(common_prepare('walk', walking))
    noises = {}
    for name, halfwidth in NOISE.items():
        a, b = [asdict(c.observations['actor'].terms[name].noise) for c in (cfg, stand_cfg)]
        assert a == b and a['n_min'] == -halfwidth and a['n_max'] == halfwidth
        noises[name] = a
    from mjlab.scene import Scene
    fields = model_arrays(Scene(cfg.scene, device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:
        compare_arrays(fields, {k: f[k] for k in f.files})
    processor = real_processor(cfg, homes['source'], homes['stand'])
    audit = read(OUT/'analysis.json')
    data = ROOT/audit['data']
    assert sha(data) == audit['data_sha256']
    errors, count = [], 0
    with np.load(data) as f:
        for case in audit['cases']:
            key = f"{case['role']}_{case['pre_action_step']-1}"
            obs = torch.from_numpy(f[key+'_source_obs'].copy())
            state = HandoffState(homes['source'], homes['stand'], len(obs), enabled=True)
            _, converted = state.inputs(obs, torch.full((len(obs),), case['pre_action_step']-1), .02)
            expected = f[key+'_destination_obs']
            np.testing.assert_allclose(converted.numpy(), expected, atol=1e-7, rtol=0)
            raw = np.concatenate([sessions['stand'].run(None, {'obs': x[None]})[0] for x in converted.numpy()])
            ref = f[key+'_destination_raw']
            np.testing.assert_allclose(raw, ref, atol=1e-7, rtol=0)
            errors.append(float(abs(raw-ref).max()))
            count += len(obs)
    assert not torch.cuda.is_initialized()
    result = dict(status='passed-cpu-only',fixture=processor,slew=check_slew(),common_actor_noise=noises,
        physical_scene_fields_exact=True,saved_actual_inputs=count,maximum_saved_request_error=max(errors),
        saved_data_sha256=sha(data),source_model_sha256=sha(ROOT/source['model']),
        stand_model_sha256=sha(ROOT/walking['model']),source_files={
            str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/training/getup_stand_handoff.py',
                                                     ROOT/'configs/experiments/getup_stand_handoff_draft.json']},
        original_frozen_sources_verified=len(master['source_sha256']),physical_steps=0,learning_updates=0,
        cuda_initialized=False,dynamic_handoff_verified=False,hardware_io=False,
        remaining=['Fresh bounded frozen protocol','Actual GPU model/physical/HOME/raw-history zero-step parity',
                   'Continuous first-episode dynamics and original8s/3s functional evaluation'],
        scope='CPU interface and saved-input checks only. Common-noise samples and dynamic controller behavior have not yet been evaluated.')
    write(OUT/'component-preflight.json', result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
