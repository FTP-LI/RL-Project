"""Verify same-start backward actuator branches and the declared rescue gate."""
from datetime import datetime
import sys
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import collect_walk_backward_onset as onset
from collect_walk_backward_start_pulse import OUT, PLAN, ROLES, page
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_backward_onset import score


def verify_inputs(trace,p,session,targets):
    active, obs, raw = (trace[name] for name in ('active','observations','actions'))
    assert active[0].all() and not (np.diff(active.astype(int),axis=0)>0).any()
    assert all(np.isfinite(value[active]).all() for value in trace.values())
    np.testing.assert_array_equal(raw[~active],0.)
    np.testing.assert_array_equal(obs[0,:,34:48],0.)
    np.testing.assert_array_equal(obs[1:,:,34:48][active[1:]],raw[:-1][active[1:]])
    np.testing.assert_allclose(obs[...,48:51],np.broadcast_to(np.array([-.14,0.,0.]),(*active.shape,3)),atol=1e-8,rtol=0)
    np.testing.assert_allclose(trace['requested_joint_targets'],raw+np.asarray(p['model_home'],np.float32),atol=1e-6,rtol=0)
    np.testing.assert_allclose(trace['applied_joint_targets'],targets,atol=1e-6,rtol=0)
    count = 0
    for step in (0,49,99,100,119,149,199,299):
        for world in np.flatnonzero(active[step]):
            out = session.run(None,{'obs':obs[step,world:world+1]})[0][0]
            np.testing.assert_allclose(raw[step,world],out,atol=2e-5,rtol=2e-5)
            count += 1
    return count


def expected_correction(trace,nominal,trigger,p,plan,session,record):
    readback = arrays(record/'pulse-readback.npz')
    end = plan['window'][1]
    enabled = trace['active'][end:end+plan['length']] & trigger[None]
    steps, worlds = np.nonzero(enabled)
    steps = steps+end
    np.testing.assert_array_equal(readback['step'],steps+1)
    np.testing.assert_array_equal(readback['world'],worlds)
    outputs = []
    for step,world in zip(steps,worlds,strict=True):
        obs = trace['observations'][step,world:world+1].copy()
        obs[:,48] = plan['input_m_s']
        outputs.append(session.run(None,{'obs':obs})[0][0])
    expected = np.asarray(outputs,dtype=np.float32).reshape(-1,14)
    np.testing.assert_allclose(readback['alternate_raw'],expected,atol=2e-5,rtol=2e-5)
    lo,hi = np.asarray(p['joint_limits'],np.float32).T
    candidate = np.clip(readback['alternate_raw']+np.asarray(p['model_home'],np.float32),lo,hi)
    delta = candidate-nominal[steps,worlds]
    scale = np.minimum(1.,plan['cap_rad']/np.maximum(abs(delta).max(axis=-1),1e-8))
    np.testing.assert_allclose(readback['fraction'],scale,atol=2e-7,rtol=0)
    adjusted = nominal.copy()
    adjusted[steps,worlds] += delta*scale[:,None]
    change = float(abs(adjusted-nominal).max())
    assert change <= plan['cap_rad']+1e-7
    return adjusted,dict(replayed_alternate_inputs=len(expected),max_joint_change_rad=change,scaled_controls=int((scale<1.).sum()),
                         readback_sha256=sha(record/'pulse-readback.npz'))


def verify(seed):
    with patch.object(onset,'PLAN',PLAN), patch.object(onset,'OUT',OUT):
        spec = onset.checked()
    p = onset.environment(spec,seed)
    record = ROOT/'records/evaluations'/p['run_id']
    state = read(record/'evaluation.json')
    assert state['status'] == 'collected-review-required' and not torch.cuda.is_initialized()
    assert state['protocol_sha256'] == sha(record/'protocol.json') == sha(PLAN)
    assert state['actor_unchanged'] and not state['heldout_used'] and state['learning_updates'] == 0
    controls = spec['maximum_control_transitions']//p['parallel_worlds']
    assert state['actual'] == dict(attempted_controls=controls,completed_controls=controls,
          attempted_substeps=controls*5,completed_substeps=controls*5,initial_resets=1)
    assert 0 <= state['contact_peak'] < state['contact_slots']
    startup = read(record/'startup.json')
    assert startup['status'] == 'passed' and startup['physics_substeps'] == 0 and startup['actual_processor_restored']
    assert startup['scene_exact'] and startup['cpu_cuda_rng_unchanged']
    low,end = spec['pulse']['window']
    fork = read(record/'fork-readback.json')
    assert list(fork) == list(ROLES) and all(r == fork[ROLES[0]] for r in fork.values())
    assert fork[ROLES[0]]['physical']['counters'] == [end,end*5]
    motors = arrays(record/'first-motor-commands.npz')
    assert set(motors) == set(ROLES)
    for role in ROLES[1:]:
        np.testing.assert_array_equal(motors[role],motors[ROLES[0]])
    initial_path, prefix_path = (record/name for name in ('physical-initial.npz','trace-shared-prefix.npz'))
    assert sha(initial_path) == state['initial_state_sha256']
    initial, prefix = arrays(initial_path), arrays(prefix_path)
    assert prefix['active'].shape == (end,p['parallel_worlds'])
    activity = abs(prefix['observations'][low-1:end,:,spec['pulse']['observation_indices']]).astype(float).mean(axis=(0,2))
    trigger = activity <= spec['pulse']['threshold_rad_s']
    trigger &= prefix['active'][-1] & ~(prefix['terminated'][-1]|prefix['truncated'][-1])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']),sess_options=opts,providers=['CPUExecutionProvider'])
    lo,hi = np.asarray(p['joint_limits'],np.float32).T
    home = np.asarray(p['model_home'],np.float32)
    scores, videos, onnx_inputs, trace_hashes, first = {}, {}, {}, {}, {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        post = arrays(path)
        trace = {key:np.concatenate((prefix[key],post[key])) for key in prefix}
        assert trace['active'].shape == (p['maximum_control_steps'],p['parallel_worlds'])
        first[role] = (post['observations'][0],post['actions'][0])
        nominal = np.clip(trace['actions']+home,lo,hi)
        if role == ROLES[-1]:
            nominal, correction = expected_correction(trace,nominal,trigger,p,spec['pulse'],session,record)
            envelope = (trace['joint_envelope_ok']|~trace['active']).all(axis=0)
        onnx_inputs[role] = verify_inputs(trace,p,session,nominal)
        verify_resets(arrays(record/'resets-shared-prefix.npz'),prefix)
        resets = arrays(record/f'resets-{role}.npz')
        verify_resets(resets,trace)
        scores[role] = score(trace,p,spec)
        clip = next(item for item in state['conditions'] if item['role']==role)['video']
        videos[role] = verify_movie(record,clip,trace,initial)
        trace_hashes[role] = sha(path)
    for role in ROLES[1:]:
        for current, reference in zip(first[role],first[ROLES[0]],strict=True):
            np.testing.assert_array_equal(current,reference)
    a,b,c = (scores[role]['worlds'] for role in ROLES)
    started = np.asarray([[row['onset_block_end_seconds'] is not None and row['survived'] for row in arm] for arm in (a,b,c)])
    common_stalls = trigger & np.asarray([x['near_static'] and y['near_static'] for x,y in zip(a,b,strict=True)])
    rescued = common_stalls & started[2]
    losses = trigger & started[0] & started[1] & ~started[2]
    unsafe = trigger & (~envelope | ~np.asarray([r['survived'] for r in c]))
    admission = bool(rescued.any() and not losses.any() and not unsafe.any())
    gate = dict(passed=admission,triggered=np.flatnonzero(trigger).tolist(),common_near_static=np.flatnonzero(common_stalls).tolist(),
                rescued=np.flatnonzero(rescued).tolist(),losses=np.flatnonzero(losses).tolist(),unsafe=np.flatnonzero(unsafe).tolist())
    result = dict(status='passed-integrity',seed=seed,protocol_sha256=sha(PLAN),record=state['record'],gate=gate,
                  scores=scores,videos=videos,onnx_inputs=onnx_inputs,pulse_readback=correction,trace_sha256=trace_hashes,
                  shared_prefix_sha256=sha(prefix_path),transitions=spec['maximum_control_transitions'],
                  physics_substeps=spec['maximum_physics_substeps'],heldout_used=False,learning_updates=0)
    output = OUT/f'seed{seed}-analysis.json'
    write(output,result)
    state.update(status='completed',branch='closed',analysis_sha256=sha(output))
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    candidate = next(item for item in state['conditions'] if item['role']==ROLES[-1])['video']
    view.update(status='completed',updated_at=datetime.now(onset.TZ).isoformat(),purpose='后退启动助推 · 已核验',
                latest_clip=candidate['path'] if candidate['complete_clip'] else view['latest_clip'])
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('WALK_BACKWARD_PULSE_VERIFIED',seed,gate,flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    verify(int(sys.argv[1]))
