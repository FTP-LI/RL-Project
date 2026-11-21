"""Independently score backward start time and verify the archived originals."""
from datetime import datetime
import sys
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from collect_walk_backward_onset import OUT, PLAN, TZ, checked, environment, page
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets


def score(trace, p, spec):
    h, n = trace['active'].shape
    block = p['criteria']['block_steps']
    assert h % block == 0 and block*.02 == spec['block_seconds']
    active = trace['active'].reshape(-1, block, n).all(axis=1)
    reverse = -trace['velocity'][..., 0].astype(float).reshape(-1, block, n).mean(axis=1)
    contacts = trace['contact']
    one_foot = (contacts[..., 0] ^ contacts[..., 1]).reshape(-1, block, n).mean(axis=1)
    success = active & (reverse >= p['criteria']['signed_speed']['backward'])
    stop = trace['terminated'] | trace['truncated'] | trace['fell_over'] | trace['out_of_bounds'] | trace['nan_state']
    rows = []
    for world in range(n):
        qualified = np.flatnonzero(success[:, world])
        onset = int(qualified[0]+1) if len(qualified) else None
        survived = bool(trace['active'][:, world].all() and not stop[-1, world])
        nearly_static = bool(survived and abs(reverse[:, world]).max() <= spec['near_static_max_absolute_block_speed_m_s'])
        rows.append(dict(world=world, onset_block_end_seconds=onset, survived=survived, near_static=nearly_static,
                         signed_speed_by_second=[float(v) if valid else None for v,valid in zip(reverse[:,world],active[:,world],strict=True)],
                         single_support_by_second=[float(v) if valid else None for v,valid in zip(one_foot[:,world],active[:,world],strict=True)]))
    categories = dict(started_by_two_seconds=sum(row['onset_block_end_seconds'] in (1,2) for row in rows),
                      delayed_start=sum((row['onset_block_end_seconds'] or 0)>2 for row in rows),
                      no_onset_survived=sum(row['onset_block_end_seconds'] is None and row['survived'] for row in rows),
                      no_onset_ended=sum(row['onset_block_end_seconds'] is None and not row['survived'] for row in rows),
                      near_static=sum(row['near_static'] for row in rows))
    assert sum(count for key,count in categories.items() if key != 'near_static') == n
    return dict(categories=categories, near_static_worlds=[row['world'] for row in rows if row['near_static']], worlds=rows)


def verify(seed):
    spec = checked()
    p = environment(spec, seed)
    record = ROOT/'records/evaluations'/p['run_id']
    state = read(record/'evaluation.json')
    assert state['status'] == 'collected-review-required' and not torch.cuda.is_initialized()
    assert state['protocol_sha256'] == sha(record/'protocol.json') == sha(PLAN)
    assert state['actor_unchanged'] and not state['heldout_used'] and state['learning_updates'] == 0
    h, n = p['maximum_control_steps'], p['parallel_worlds']
    assert state['actual'] == dict(attempted_controls=h,completed_controls=h,
          attempted_substeps=h*5,completed_substeps=h*5,initial_resets=1)
    assert 0 <= state['contact_peak'] < state['contact_slots']
    startup = read(record/'startup.json')
    assert startup['status'] == 'passed' and startup['physics_substeps'] == 0
    assert startup['scene_exact'] and startup['actual_processor_restored'] and startup['cpu_cuda_rng_unchanged']
    assert startup['input_sha256'] == sha(record/'startup-inputs.npz')
    path = record/'trace-backward.npz'
    t = arrays(path)
    active, obs, raw = t['active'], t['observations'], t['actions']
    assert active.shape == (h,n) and active[0].all() and not (np.diff(active.astype(int),axis=0)>0).any()
    assert all(np.isfinite(value[active]).all() for value in t.values()) and not t['nan_state'][active].any()
    initial_path = record/'physical-initial.npz'
    assert sha(initial_path) == state['initial_state_sha256']
    initial = arrays(initial_path)
    np.testing.assert_allclose(t['qpos'][...,state['qpos_joint_addresses']],t['joint_positions'],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(obs[0,:,34:48],0.)
    np.testing.assert_array_equal(obs[1:,:,34:48][active[1:]],raw[:-1][active[1:]])
    np.testing.assert_array_equal(raw[~active],0.)
    np.testing.assert_allclose(obs[...,48:51], np.broadcast_to(np.array([-.14,0.,0.]), (h,n,3)), atol=1e-8, rtol=0)
    lo, hi = np.asarray(p['joint_limits'],np.float32).T
    requested = raw+np.asarray(p['model_home'],np.float32)
    np.testing.assert_allclose(t['requested_joint_targets'],requested,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['applied_joint_targets'],np.clip(requested,lo,hi),atol=1e-6,rtol=0)
    resets = verify_resets(arrays(record/'resets-backward.npz'), t)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    replayed, worst = 0, 0.
    for step in (0,49,99,149,199,249,299):
        indices = np.flatnonzero(active[step])
        for world in indices:
            actual = session.run(None,{'obs':obs[step,world:world+1]})[0][0]
            np.testing.assert_allclose(actual,raw[step,world],atol=2e-5,rtol=2e-5)
            worst = max(worst,float(abs(actual-raw[step,world]).max()))
        replayed += len(indices)
    clip, = state['conditions']
    assert clip['condition'] == 'backward'
    video = verify_movie(record,clip['video'],t,initial)
    verdict = dict(status='passed-integrity',seed=seed,record=state['record'],analysis=score(t,p,spec),video=video,
                   source_sha256=sha(path),protocol_sha256=sha(PLAN),physical_substeps=h*5,transitions=h*n,
                   onnx_inputs=replayed,maximum_onnx_error=worst,unscored_resets=resets,heldout_used=False,learning_updates=0)
    write(OUT/f'seed{seed}-analysis.json',verdict)
    state.update(status='completed',branch='closed',analysis_sha256=sha(OUT/f'seed{seed}-analysis.json'))
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    view.update(status='completed',updated_at=datetime.now(TZ).isoformat(),purpose='后退启动 · 已核验')
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('WALK_BACKWARD_ONSET_VERIFIED',seed,verdict['analysis']['categories'],flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    verify(int(sys.argv[1]))
