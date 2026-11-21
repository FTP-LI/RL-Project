"""Independent CPU checks for the bounded live release diagnostic."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from getup_coverage_metrics import evaluate
from recovery_end_hold import evaluate_end_hold
from record_io import write
from review_getup_release_prefix import state_distance
from s1_reference_inputs import read, sha


def cpu_fixture():
    """Use native Warp buffers and actual delay rings with explicit mock owners."""
    import torch
    import warp as wp
    from mjlab.utils.buffers.delay_buffer import DelayBuffer
    from getup_release_fork_state import ReleaseSnapshot

    @dataclass
    class Contact:
        geom: wp.array

    @dataclass
    class Data:
        qpos: wp.array
        contact: Contact
        nworld: int

    def delay():
        result = DelayBuffer(0, 1, batch_size=2, device='cpu')
        result.append(torch.tensor([[1., 2.], [3., 4.]]))
        result.append(torch.tensor([[5., 6.], [7., 8.]]))
        return result

    obs_delay, act_delay = delay(), delay()
    qpos = wp.array([[1., 2.], [3., 4.]], device='cpu', dtype=wp.float32)
    geom = wp.array([1, 2], device='cpu', dtype=wp.int32)
    actuator = type('FrictionDRBamActuator', (NS,), {})(
        _delay_buffer=act_delay, _bam_model=NS(actuator=NS(q_target_smooth=torch.ones(2))))
    sensor = type('ContactSensor', (NS,), {})(
        _cached_data=None, _cache_valid=False, _air_time_state=NS(), _history_state=None)
    sensor._air_time_state = None
    robot = NS(actuators=[actuator], data=NS(joint_pos_target=torch.zeros(2, 2)))

    class Scene(NS):
        def __getitem__(self, key):
            return self.entities[key]

    obs = {'actor': torch.ones(2, 2)}
    env = NS(device='cpu', common_step_counter=100, _sim_step_counter=500, obs_buf=obs,
              episode_length_buf=torch.full((2,), 100),
              sim=NS(wp_data=Data(qpos, Contact(geom), 2), wp_model=NS(), expanded_fields=set()),
              event_manager=NS(available_modes=('reset',)),
              cfg=NS(recorders={}, curriculum={}, sim=NS(nan_guard=NS(enabled=False))),
              action_manager=NS(_action=torch.zeros(2, 2), get_term=lambda _: NS(previous_target=torch.ones(2, 2))),
              observation_manager=NS(_obs_buffer=obs, _group_obs_class_term_cfgs={},
                  _group_obs_term_delay_buffer={'actor': {'base_ang_vel': obs_delay}}, _group_obs_term_history_buffer={}),
              reward_manager=NS(_class_term_cfgs=[], _episode_sums={'x': torch.zeros(2)}),
              metrics_manager=NS(_class_term_cfgs=[], _step_count=torch.zeros(2)),
              termination_manager=NS(_term_dones={'timeout': torch.zeros(2, dtype=torch.bool)}),
              command_manager=NS(_commands={}),
              scene=Scene(entities={'robot': robot}, sensors={'feet_ground_contact': sensor}))
    memory = NS(source_history=torch.zeros(2, 2))
    snap = ReleaseSnapshot(env, memory, 1024**2)
    expected = snap.verify()
    obs_output = obs_delay.compute().clone()
    act_output = act_delay.compute().clone()
    qpos.zero_()
    geom.zero_()
    obs_delay.append(torch.full((2, 2), 99.))
    act_delay._current_lags.fill_(1)
    act_delay.append(torch.zeros((2, 2)))
    env.obs_buf['actor'].zero_()
    memory.source_history.fill_(8.)
    torch.rand(1)
    assert snap.expected != snap.live_signatures()
    assert snap.restore() == expected and env.observation_manager._obs_buffer is env.obs_buf
    torch.testing.assert_close(obs_delay.compute(), obs_output, atol=0, rtol=0)
    torch.testing.assert_close(act_delay.compute(), act_output, atol=0, rtol=0)
    assert not torch.cuda.is_initialized()
    return dict(status='passed-cpu-buffer-fixture', cpu_only=True, state=expected,
                uses_native_warp_storage=True, uses_actual_observation_and_actuator_delay_class=True,
                physics_steps=0)


def load_arrays(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def same_bytes(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert (a[key].dtype, a[key].shape, a[key].tobytes()) == (b[key].dtype, b[key].shape, b[key].tobytes()), key


def verify_video(record, info, qpos, initial):
    import imageio_ffmpeg

    path = record / info['path']
    assert sha(path) == info['sha256'] and info['complete_clip']
    steps = np.arange(0, 551, 2)
    np.testing.assert_allclose(info['frame_times_seconds'], steps * .02, atol=1e-12, rtol=0)
    source = np.concatenate((initial[None], qpos), axis=0)
    hashes = [hashlib.sha256(source[step, 0].tobytes()).hexdigest() for step in steps]
    assert hashes == info['qpos_sha256_by_frame']
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(reader)
    frames = sum(1 for _ in reader)
    assert meta['fps'] == 25 and meta['size'] == (640, 480) and frames == 276
    return dict(path=str(path.relative_to(ROOT)), sha256=info['sha256'], frames=frames,
                start_seconds=0., end_seconds=11., physical_source_hashes_verified=True)


def verify(plan, out):
    from getup_release_fork_state import digest
    from getup_shared_release import assert_sources
    import onnxruntime as ort
    import torch

    assert not torch.cuda.is_initialized()
    spec = read(plan)
    assert_sources(spec)
    state = read(out / 'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['phase'] == 'verify'
    record = ROOT / state['record']
    assert sha(plan) == sha(record / 'protocol.json') == state['protocol_sha256']
    assert state['actual'] == dict(completed_controls=1000, attempted_controls=1000,
                                   physical_substeps=5000, physical_resets=1)
    assert state['actor_weights_unchanged'] and state['release_restores_after_branch'] == 1
    assert state['contact_peak'] < state['contact_capacity'] and state['actual_control_transitions'] == 32000
    zero = read(record / 'zero-readback.json')
    assert zero['status'] == 'passed' and zero['actual_physics_steps'] == 0
    receipts = read(record / 'release-readback.json')
    inventory = read(record / 'release-state-fields.json')
    p = spec['environment']
    roles = tuple(spec['branches'])
    assert all(receipts[role] == receipts['capture'] for role in roles)
    assert digest(inventory) == receipts['capture']['state_sha256'] and receipts['capture']['counters'] == [100, 500]
    for part in ('warp.qacc_warmstart', 'warp.contact.', 'warp.efc.', 'obs-delay.actor.',
                 'actuator.0.delay.', 'controller.source_history', 'sensor.feet_ground_contact.', 'rng'):
        assert any(key.startswith(part) for key in inventory), part

    released = {role: load_arrays(record / f'release-{role}.npz') for role in roles}
    first = {role: load_arrays(record / f'first-control-{role}.npz') for role in roles}
    same_bytes(*released.values())
    same_bytes(*first.values())
    first_control = first[roles[0]]
    np.testing.assert_array_equal(first_control['actor'], released[roles[0]]['actor'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    onnx_actions = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in first_control['actor']])
    np.testing.assert_allclose(first_control['raw_action'], onnx_actions, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(first_control['requested'], np.asarray(p['model_home'], np.float32) + onnx_actions,
                               atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(first_control['target'], first_control['requested'])

    prefix = load_arrays(record / 'trace-shared-prefix.npz')
    initial = load_arrays(record / 'initial.npz')
    assert prefix['qpos'].shape == (100, 32, 21) and not np.any(prefix['raw_policy_actions'])
    np.testing.assert_array_equal(prefix['applied_targets'], np.broadcast_to(
        np.asarray(p['simulation_home'], np.float32), (100, 32, 14)))
    np.testing.assert_array_equal(prefix['qpos'][-1], released[roles[0]]['qpos'])
    np.testing.assert_array_equal(prefix['qvel'][-1], released[roles[0]]['qvel'])
    post, timely, hold, scores, videos = {}, {}, {}, {}, {}
    for role in roles:
        trace = post[role] = load_arrays(record / f'trace-{role}.npz')
        assert trace['qpos'].shape == (450, 32, 21)
        both = {name: np.concatenate((prefix[name], trace[name])) for name in prefix}
        both['requested_pose'] = initial['requested_pose']
        assert all(np.isfinite(both[key]).all() for key in ('qpos', 'qvel', 'requested_targets', 'applied_targets'))
        assert both['active'].all() and not both['terminated'].any()
        assert not both['truncated'][:-1].any() and both['truncated'][-1].all()
        np.testing.assert_array_equal(both['episode_ages'], np.broadcast_to(np.arange(1, 551)[:, None], (550, 32)))
        np.testing.assert_array_equal(first[role]['raw_action'], trace['raw_policy_actions'][0])
        np.testing.assert_array_equal(first[role]['target'], trace['applied_targets'][0])
        audits = read(record / f'contact-audits-{role}.json')
        assert len(audits) == 22 and all(row['matched'] for row in audits)
        for row in audits:
            assert int(both['native_head_leg_contact'][row['control_step'] - 1].sum()) == row['contact_worlds']
        result = evaluate(both, p)
        end_hold = evaluate_end_hold(both, p, result['episodes'])
        timely[role] = np.asarray([row['hold_start_seconds'] is not None for row in result['episodes']])
        hold[role] = np.asarray(end_hold['stable_by_environment'])
        scores[role] = dict(timely_holds=int(timely[role].sum()), final_holds=int(hold[role].sum()),
                            actual_release=result['actual'], rows=result['episodes'])
        condition = next(row for row in state['conditions'] if row['condition'] == role)
        videos[role] = verify_video(record, condition['video'], both['qpos'], initial['qpos'])
        condition['episodes'] = [result['episodes'][0]]

    a, b = (post[role]['qpos'] for role in roles)
    tdiff = np.flatnonzero(np.any(a != b, axis=(1, 2)))
    translation, rotation, joint = state_distance(a, b)
    disagreements = dict(timely=int(np.count_nonzero(timely[roles[0]] != timely[roles[1]])),
                         final=int(np.count_nonzero(hold[roles[0]] != hold[roles[1]])))
    repeat = dict(first_postrelease_control_with_qpos_difference=101 + int(tdiff[0]) if len(tdiff) else None,
                  by_control=[dict(control=step, max_translation_m=float(translation[step-101].max()),
                                   max_orientation_deg=float(rotation[step-101].max()),
                                   max_joint_deg=float(joint[step-101].max()))
                              for step in (101, 110, 125, 150, 250, 400, 550)],
                  score_disagreements=disagreements)
    result = dict(status='passed-integrity', comparison_admissible_on_diagnostic=not any(disagreements.values()),
                  protocol_sha256=sha(plan), record=state['record'], exact_release=receipts['capture'],
                  first_action_and_actuator_ctrl_exact=True, repeat=repeat, scores=scores, videos=videos,
                  actual_control_transitions=32000, actual_physics_substeps=5000, learning_updates=0,
                  development_only=True, functional_acceptance_evaluated=False, heldout_used=False,
                  scope='The same physical release and fixed model are repeated; reliability and learning remain separate.')
    write(out / 'analysis.json', result)
    state.update(status='completed', phase='closed', comparison_admissible=result['comparison_admissible_on_diagnostic'])
    write(record / 'evaluation.json', state)
    write(out / 'coordinator.json', state)
    print('SHARED_RELEASE_VERIFIED', repeat, flush=True)
