"""Independently reconstruct the pulse and score its short causal response."""

import imageio_ffmpeg
import numpy as np

from forward_hip_yaw_pulse_targets import ROLES, SIDES, apply_pulse, eligible
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import read, rel, require, sha


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {field: data[field] for field in data.files}


def verify_motor(plan, commands, before_raw, applied=None):
    walk = plan['walking']
    low, high = np.asarray(walk['joint_limits'], dtype=np.float32).T
    home = np.asarray(walk['model_home'], dtype=np.float32)
    cap = plan['feedback']['maximum_joint_correction_rad']
    nominal = np.clip(commands['requested_targets'], low, high)
    forward = commands['phase_used'] == FORWARD
    alternate = np.clip(commands['alternate_action'] + home, low, high)
    shift = alternate - nominal
    norm = np.maximum(np.abs(shift).max(axis=-1, keepdims=True), np.float32(1e-8))
    projected = nominal + np.where(forward[..., None], np.minimum(1., cap/norm)*shift, 0.)
    np.testing.assert_allclose(commands['applied_targets'], projected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(commands['actor_input'][0, :, 34:48], before_raw)
    np.testing.assert_array_equal(commands['actor_input'][1:, :, 34:48], commands['raw_policy_actions'][:-1])
    actual = commands['actual_motor_target']
    np.testing.assert_allclose(actual, projected if applied is None else applied, atol=1e-6, rtol=0)
    require(np.isfinite(actual).all() and (actual >= low-1e-6).all() and (actual <= high+1e-6).all()
            and np.abs(actual-nominal).max() <= cap+1e-6, 'Motor targets exceeded the original joint or feedback bound')


def response(series, window):
    a, b = (series[role][:window] for role in ('zero-a', 'zero-b'))
    baseline = (np.abs(a) + np.abs(b))*.5
    zero_gap = np.abs(np.mean(np.abs(a)-np.abs(b), axis=0))
    floor = float(np.quantile(zero_gap, .95))
    effects = {}
    for role in ('positive', 'negative'):
        data = series[role][:window]
        error = np.mean(np.abs(data)-baseline, axis=0)
        effects[role] = dict(median_change_in_mean_abs_yaw_rad_s=float(np.median(error)),
                             scene_improved_count=int((error < 0).sum()),
                             median_change_in_mean_signed_yaw_rad_s=float(np.median(np.mean(data-(a+b)*.5, axis=0))))
    odd = (series['positive']-series['negative'])*.5
    return dict(window_controls=window, zero_pair_gap_p95_rad_s=floor,
                median_zero_baseline_mean_abs_yaw_rad_s=float(np.median(baseline.mean(axis=0))),
                effects=effects, odd_signed_yaw_by_control_rad_s=np.median(odd, axis=1).tolist(),
                zero_repeat_median_abs_difference_by_control_rad_s=np.median(np.abs(series['zero-a']-series['zero-b']), axis=1).tolist())


def verify_side(folder, plan, side, shared, shared_commands, fork):
    name = SIDES[side]
    design = plan['pulse']
    k = fork['control']
    controls, worlds = design['response_controls'], plan['worlds']
    require(design['earliest_fork_control'] <= k < design['shared_controls'] and
            fork['captured']['counters'] == [k, k*5] and set(fork['restored']) == set(ROLES) and
            all(value == fork['captured'] for value in fork['restored'].values()), 'Fork restore receipt differs')
    earliest = design['earliest_fork_control']
    if k > earliest:
        earlier = slice(earliest, k)
        possible, _ = eligible(plan, shared_commands['requested_targets'][earlier],
            shared_commands['applied_targets'][earlier], shared['foot_contact'][earlier], side)
        require((possible.sum(axis=1) < design['minimum_worlds_per_fork']).all(), 'The selected fork is not the first eligible one')
    selected = np.asarray(fork['selected_worlds'], dtype=np.int64)
    initial, series, delayed = {}, {}, {}
    for role, sign in ROLES.items():
        path = folder/name/f'{role}.npz'
        require(sha(path) == fork['branches'][role], 'Saved branch changed')
        data = load(path)
        states = {field[6:]: value for field, value in data.items() if field.startswith('state_')}
        command = {field[8:]: value for field, value in data.items() if field.startswith('command_')}
        for field, value in states.items():
            np.testing.assert_array_equal(value[0], shared[field][k], err_msg=field)
            require(value.shape[0] == controls+1 and np.isfinite(value).all(), 'Invalid branch state record')
        for field in ('observed', 'actor_input', 'requested_targets', 'raw_policy_actions', 'applied_targets'):
            np.testing.assert_array_equal(command[field][0], shared_commands[field][k], err_msg=field)
        require(states['active'].shape == (controls+1, worlds) and states['active'].all()
                and states['joint_envelope_ok'].all()
                and (states['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                'Recorded physical branch violates the original guards')
        matches, hip = eligible(plan, command['requested_targets'][0], command['applied_targets'][0],
                                states['foot_contact'][0], side)
        np.testing.assert_array_equal(np.flatnonzero(matches), selected)
        require(len(selected) >= design['minimum_worlds_per_fork'], 'Insufficient matched phase coverage')
        motor = command['applied_targets'].copy()
        motor[0] = apply_pulse(motor[0], selected, hip, sign*design['delta_rad'])
        verify_motor(plan, command, shared_commands['raw_policy_actions'][k-1], motor)
        steps = data['motor_substeps']
        require(steps.shape[:2] == (5*controls, worlds) and np.isfinite(steps).all(), 'Motor delay trace is incomplete')
        initial[role] = data['initial_motor']
        delayed[role] = steps
        series[role] = states['angular_velocity'][1:, selected, 2].astype(np.float64)
    for motor in initial.values():
        np.testing.assert_array_equal(motor, initial['zero-a'])
    first = delayed['zero-a'][:5, selected]
    motion = {role: [float(value) for value in np.abs(delayed[role][:5, selected]-first).max(axis=(1, 2))]
              for role in ('positive', 'negative', 'zero-b')}
    return dict(fork_control=k, worlds=len(selected), joint=plan['walking']['joint_names'][hip], all_four_starts_exact=True,
                first_control_motor_max_abs_difference_by_substep=motion,
                primary=response(series, design['primary_response_controls']), complete=response(series, controls))


def score(plan, folder):
    manifest = read(folder/'guarded-evaluation.json')
    runtime = read(folder/'runtime-contract.json')
    zero, endpoints = (read(folder/name) for name in ('pulse-zero.json', 'endpoint-contact.json'))
    count, worlds = plan['maximum_controls'], plan['worlds']
    shared_controls = plan['pulse']['shared_controls']
    require(read(folder/'protocol.json') == plan and runtime['status'] == 'closed' and runtime['interface_checks'] == 1
            and runtime['physics_substeps'] == manifest['actual_physics_substeps'] == count*5
            and runtime['control_transitions'] == manifest['recorded_control_transitions'] == count*worlds
            and runtime['control_steps'] == manifest['actual_control_steps'] == count and not runtime['capacity_reached']
            and not runtime['extra_observation_or_render_steps'] and not manifest['learning_updates'] and not manifest['hardware_io']
            and zero['initial']['counters'] == [0, 0] and zero['observation_shape'] == [worlds, 61] and zero['delay_fields']
            and endpoints['complete'] and endpoints['zero_matches_standalone'] and not endpoints['first_mismatches'],
            'Invalid original route, control budget or CUDA zero witness')
    trace, commands, forks = (folder/name for name in ('shared-trace.npz', 'shared-commands.npz', 'pulse-forks.json'))
    require(sha(trace) == manifest['shared_trace_sha256'] and sha(commands) == manifest['shared_commands_sha256']
            and sha(forks) == manifest['forks_sha256'] and sha(folder/'endpoint-contact.npz') == endpoints['sha256'],
            'Physical evidence hashes differ')
    shared, common, copies = load(trace), load(commands), read(forks)
    require(len(shared['qpos']) == shared_controls+1 and len(common['actual_motor_target']) == shared_controls
            and shared['active'].all() and shared['joint_envelope_ok'].all()
            and (shared['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
            'The original shared route length differs')
    verify_motor(plan, common, np.zeros((worlds, 14), dtype=np.float32))
    contact = load(folder/'endpoint-contact.npz')
    np.testing.assert_array_equal(shared['foot_contact'], contact['fresh'][..., 2:])
    np.testing.assert_array_equal(contact['fresh'], contact['constraint_points'] > 0)
    sides = {name: verify_side(folder, plan, side, shared, common, copies[name]) for side, name in enumerate(SIDES)}
    clip, = manifest['video']
    video = folder/clip['file']
    with_frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    codec = next(with_frames)
    frames = sum(1 for _ in with_frames)
    require(sha(video) == clip['sha256'] and codec['fps'] == 25 and frames == clip['frames'] == len(clip['common_steps'])
            and clip['common_steps'][0] == 0 and clip['common_steps'][-1] == shared_controls
            and clip['complete_clip'] and clip['full_action'] and not clip['extra_physics_steps'], 'Shared video is incomplete')
    return dict(seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'], sides=sides,
                exact_joint_target_changes=True, physical_controls=count, physical_transitions=count*worlds,
                control_horizon_seconds=plan['pulse']['response_controls']*.02, learning_updates=0, hardware_io=False,
                video=dict(path=rel(video), sha256=sha(video), frames=frames, seconds=frames/codec['fps']))
