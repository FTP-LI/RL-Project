"""Verify seven-command handoff routes and target semantics without physics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_routed_getup_ramped_forward as prior
from command_input_mapping import transform
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback
from routed_seven_command_controller import SevenCommandSequenceController, balanced_groups


ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
DRAFT = ROOT / 'configs/experiments/continuous_routed_seven_command_draft.json'
OUT = ROOT / 'records/reports/continuous-routed-seven-command'


def sources():
    d = read(DRAFT)
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used',
                                      'hardware_io', 'automatic_retry', 'automatic_extension'))
    source, result, models = [prior.route.pinned(d[key]) for key in ('continuous_source', 'continuous_qualification', 'walking_delivery')]
    spec, accepted, delivery = map(read, (source, result, models))
    assert accepted['final_holdout_passed'] and accepted['protocol_sha256'] == sha(source)
    assert spec['data_scope'] == 'reserved-holdout' and spec['environment']['seed'] in d['excluded_seeds']
    assert d['development_seed'] not in d['excluded_seeds']
    assert delivery['successful_candidate_episodes']['holdout'] == 224
    walking = prior.route.pinned((delivery['frozen_protocol']['path'], delivery['frozen_protocol']['sha256']))
    original, = [item['environment'] for item in read(walking)['cases'].values()]
    for name in ('commands', 'criteria', 'joint_names', 'joint_limits'):
        assert spec['walking'][name] == original[name], name
    assert delivery['commands'] == original['commands'] and delivery['joint_names'] == original['joint_names']
    return d, spec, delivery, (DRAFT, source, result, models, walking)


def main():
    d, spec, delivery, paths = sources()
    assert not OUT.exists()
    p, s, w = (spec[key] for key in ('environment', 'sequence', 'walking'))
    assert d['boundaries'] == s['boundary_controls'] + [spec['maximum_unique_controls']]
    assert d['worlds'] == p['num_envs'] == sum(d['requested_pose_counts'])
    assert d['worlds']*d['maximum_unique_controls'] == d['maximum_transitions']
    assert d['maximum_unique_controls']*5 == d['maximum_batched_physics_substeps']
    assert d['minimum_actual_right_coverage'] == s['minimum_actual_right_coverage']
    commands, routes = delivery['commands'], delivery['route_by_command']
    pose = np.random.default_rng(d['development_seed']).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = balanced_groups(pose, commands, d['development_seed']+70000)
    sizes = np.array([len(ids) for ids in groups.values()])
    cross = np.stack([np.bincount(pose[ids], minlength=4) for ids in groups.values()])
    assert np.ptp(sizes) <= 1 and (np.ptp(cross, axis=0) <= 1).all()
    assert cross.sum(axis=0).tolist() == d['requested_pose_counts'] and sizes.sum() == d['worlds']

    torch.set_num_threads(4)
    sessions, bank, models = {}, {}, []
    home = spec['switch_policies']['stand']['home']
    assert home == spec['switch_policies']['forward']['home']
    assert delivery['models']['base']['files']['onnx']['sha256'] == spec['switch_policies']['stand']['model_sha256']
    assert delivery['models']['official']['files']['onnx']['sha256'] == spec['switch_policies']['forward']['model_sha256']
    for role, details in delivery['models'].items():
        item = details['files']['onnx']
        path = prior.route.pinned((item['path'], item['sha256']))
        models.append(path)
        sessions[role] = prior.original.session(path)
        metadata = sessions[role].get_modelmeta().custom_metadata_map
        assert metadata['joint_names'].split(',') == w['joint_names']
        assert list(map(float, metadata['default_joint_pos'].split(','))) == home and float(metadata['action_scale']) == 1.
        assert sessions[role].get_inputs()[0].shape == [1, 61] and sessions[role].get_outputs()[0].shape == [1, 14]
        bank[role] = prior.route.guarded_actor(dict(model=item['path'], model_sha256=item['sha256'],
            seed=d['development_seed'], home=home), 'cpu')

    fixture = {name: [i] for i, name in enumerate(commands)}
    fixture['forward'].append(7)
    _, stand_at, final_at, _ = d['boundaries']
    ctrl = SevenCommandSequenceController(8, 'cpu', s['controller'], [p['model_home'], home, home],
        p['joint_limits'], ForwardFeedback(w, 8), commands=commands, groups=fixture, routes=routes,
        ramp=spec['ramped_forward']['ramp'])
    ctrl.mode.fill_(STAND)
    ctrl.dwell.fill_(ctrl.dwell_steps)
    ctrl.history[STAND] = torch.linspace(-.08, .08, 8*14).reshape(8, 14)
    carried = ctrl.history[STAND].clone()
    observed = torch.zeros((8, 61))
    observed[:, 5] = -1.
    observed[:, 2] = .1
    observed[7, 0] = 2.
    previous = ctrl.homes[STAND].expand(8, 14).clone()
    actors = {GETUP: lambda inputs: torch.zeros_like(inputs['actor'][:, :14]), STAND: bank['base'], FORWARD: bank}
    term = SimpleNamespace(**{key:torch.zeros((8, 14)) for key in
        ('requested_target', 'projected_destination', '_processed_actions', 'previous_target')})
    worst, queried, max_step = 0., 0, 0.
    examined = {final_at, final_at+24, final_at+25, final_at+50, final_at+84}
    with torch.inference_mode():
        for k in range(final_at, final_at+85):
            x, bridge = ctrl.advance(k, observed)
            assert ctrl.mode.tolist() == [FORWARD]*7+[STAND]
            assert bridge.tolist() == [k<final_at+25]*7+[False]
            if k == final_at:
                for i, name in enumerate(commands):
                    expected = carried[i] if routes[name] == 'base' else torch.zeros(14)
                    torch.testing.assert_close(x[i, 34:48], expected, atol=0, rtol=0)
                torch.testing.assert_close(x[7, 34:48], carried[7], atol=0, rtol=0)
            for i, name in enumerate(commands):
                vector = np.asarray(commands[name], np.float64)
                if name == 'forward':
                    vector *= min(1., (k-final_at+1)/25)
                np.testing.assert_array_equal(x[i, 48:51], transform(vector))
            assert not x[7, 48:51].any()
            raw, yaw, alternate = ctrl.infer(k, x, actors)
            forward_index = list(commands).index('forward')
            no_feedback = torch.arange(8) != forward_index
            assert not yaw[no_feedback].any()
            if k < final_at+50:
                assert not yaw.any() and not alternate.any()
            if k in examined:
                for i, name in enumerate((*commands, 'stand')):
                    expected = sessions[routes[name]].run(None, {'obs': x[i:i+1].numpy()})[0][0]
                    actual = raw[i].numpy()
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                    worst = max(worst, float(abs(actual-expected).max()))
                    queried += 1
            ctrl.apply(term, previous, raw, alternate, bridge, k, .02)
            if k == final_at:
                max_step = float((term._processed_actions[:7]-previous[:7]).abs().max())
                assert max_step <= .0400002
            if k >= final_at+50:
                clipped = torch.clamp(raw+ctrl.homes[FORWARD], min=ctrl.limits[:,0], max=ctrl.limits[:,1])
                torch.testing.assert_close(term._processed_actions[no_feedback], clipped[no_feedback], atol=0, rtol=0)
            previous = term._processed_actions.clone()
    assert stand_at == 550 and not torch.cuda.is_initialized()
    OUT.mkdir()
    paths = (*paths, *models, Path(__file__).resolve(), Path(__file__).with_name('routed_seven_command_controller.py'))
    write(OUT/'preflight.json', dict(status='passed-cpu-seven-route-handoff', physical_substeps=0, hardware_io=False,
        development_seed=d['development_seed'], command_order=list(commands), counts=sizes.tolist(),
        requested_pose_counts_by_command=cross.tolist(), maximum_first_target_step_rad=max_step,
        actor_queries=queried, max_onnx_error_rad=worst, same_model_history_carried=True, different_model_history_zero=True,
        nonforward_feedback_zero=True, frozen_inputs={path.relative_to(ROOT).as_posix():sha(path) for path in paths}))
    print('ROUTED_SEVEN_COMMAND_CPU', sizes.tolist(), queried, worst, max_step, flush=True)


if __name__ == '__main__':
    main()
