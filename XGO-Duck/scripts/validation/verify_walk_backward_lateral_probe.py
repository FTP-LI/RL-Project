"""Independently verify isolated writes, physical media and backward onset."""
from datetime import datetime
import sys
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import collect_walk_backward_onset as onset
from collect_walk_backward_lateral_probe import OUT, PLAN, ROLES, SEEDS, SIGNS, page
from record_io import write
from review_walk_backward_support_transfer import first_support
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_backward_onset import score
from verify_walk_backward_start_pulse import verify_inputs


def checked():
    with patch.object(onset,'PLAN',PLAN),patch.object(onset,'OUT',OUT):
        return onset.checked()


def verify_impulses(record,spec,prefix,trigger):
    start = arrays(record/'impulse-backward.npz')
    receipt = {}
    for role in ROLES:
        path = record/f'impulse-{role}.npz'
        current = arrays(path)
        ids = np.flatnonzero(trigger) if role in SIGNS else np.empty(0,dtype=int)
        np.testing.assert_array_equal(current['ids'],ids)
        for key in ('before','qpos','vadr'):
            np.testing.assert_array_equal(current[key],start[key])
        np.testing.assert_array_equal(current['vadr'],[0,1])
        q = current['qpos'][ids,3:7].astype(float)
        w,x,y,z = q.T
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        np.testing.assert_allclose(current['yaw'],yaw,atol=1e-6,rtol=0)
        velocity = current['after'].astype(float)-current['before']
        measured = velocity[np.ix_(ids,current['vadr'])]
        target = SIGNS.get(role,0.)*spec['lateral']['magnitude_m_s']
        forward = np.stack((np.cos(yaw),np.sin(yaw)),axis=-1)
        side = np.stack((-np.sin(yaw),np.cos(yaw)),axis=-1)
        np.testing.assert_allclose((measured*forward).sum(axis=1),0,atol=2e-7,rtol=0)
        np.testing.assert_allclose((measured*side).sum(axis=1),target,atol=2e-7,rtol=0)
        velocity[np.ix_(ids,current['vadr'])] = 0.
        np.testing.assert_array_equal(velocity,0.)
        np.testing.assert_array_equal(current['qpos'][trigger],prefix['qpos'][-1,trigger])
        receipt[role] = dict(ids=ids.tolist(),target_m_s=target,path=path.name,sha256=sha(path),
            maximum_forward_delta=float(abs((measured*forward).sum(axis=1)).max()) if ids.size else 0.)
    return receipt


def support_rows(trace,worlds,fork,deadline):
    rows = {}
    for i in worlds:
        entry = first_support(trace,i,fork,fork+deadline)
        if entry is not None:
            step,side = entry
            rows[str(i)] = dict(control=step+1,confirmation_control=step+4,
                                side=('left','right')[side])
    return rows


def verify(seed):
    spec = checked()
    p = onset.environment(spec,seed)
    record = ROOT/'records/evaluations'/p['run_id']
    state = read(record/'evaluation.json')
    assert state['status']=='collected-review-required' and not torch.cuda.is_initialized()
    assert state['protocol_sha256']==sha(record/'protocol.json')==sha(PLAN)
    assert state['actor_unchanged'] and not state['heldout_used'] and state['learning_updates']==0
    count = spec['maximum_control_transitions']//p['parallel_worlds']
    assert state['actual']==dict(attempted_controls=count,completed_controls=count,
        attempted_substeps=count*5,completed_substeps=count*5,initial_resets=1)
    assert 0<=state['contact_peak']<state['contact_slots']
    startup = read(record/'startup.json')
    assert startup['status']=='passed' and startup['physics_substeps']==0 and startup['actual_processor_restored']
    assert startup['scene_exact'] and startup['cpu_cuda_rng_unchanged'] and state['lateral_zero_restore']['counters']==[0,0]
    low,fork = spec['trigger']['window']
    forks = read(record/'fork-readback.json')
    assert tuple(forks)==ROLES and all(value==forks[ROLES[0]] for value in forks.values())
    assert forks[ROLES[0]]['physical']['counters']==[fork,fork*5]
    motors = arrays(record/'first-motor-commands.npz')
    assert set(motors)==set(ROLES)
    for role in ROLES[1:]:
        np.testing.assert_array_equal(motors[role],motors[ROLES[0]])
    initial_path,prefix_path = (record/name for name in ('physical-initial.npz','trace-shared-prefix.npz'))
    initial,prefix = arrays(initial_path),arrays(prefix_path)
    assert sha(initial_path)==state['initial_state_sha256']
    assert prefix['active'].shape==(fork,p['parallel_worlds'])
    activity = abs(prefix['observations'][low-1:fork,:,spec['trigger']['observation_indices']]).astype(float).mean(axis=(0,2))
    stop = np.logical_or.reduce([prefix[key][-1] for key in ('terminated','truncated','fell_over','nan_state','out_of_bounds')])
    trigger = (activity<=spec['trigger']['threshold_rad_s']) & prefix['active'][-1] & ~stop
    ids = np.flatnonzero(trigger)
    impulses = verify_impulses(record,spec,prefix,trigger)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    lo,hi = np.asarray(p['joint_limits'],np.float32).T
    home = np.asarray(p['model_home'],np.float32)
    verify_resets(arrays(record/'resets-shared-prefix.npz'),prefix)
    scores,videos,onnx,hashes,first,envelope,support = ({} for _ in range(7))
    for role in ROLES:
        file = record/f'trace-{role}.npz'
        post = arrays(file)
        trace = {k:np.concatenate((prefix[k],post[k])) for k in prefix}
        assert trace['active'].shape==(p['maximum_control_steps'],p['parallel_worlds'])
        first[role] = (post['observations'][0],post['actions'][0])
        onnx[role] = verify_inputs(trace,p,session,np.clip(trace['actions']+home,lo,hi))
        verify_resets(arrays(record/f'resets-{role}.npz'),trace)
        scores[role] = score(trace,p,spec)
        envelope[role] = (trace['joint_envelope_ok']|~trace['active']).all(axis=0)
        support[role] = support_rows(trace,ids,fork,spec['lateral']['support_deadline_controls'])
        clip = next(item['video'] for item in state['conditions'] if item['role']==role)
        videos[role] = verify_movie(record,clip,trace,initial)
        hashes[role] = sha(file)
    for role in ROLES[1:]:
        for value,base in zip(first[role],first[ROLES[0]],strict=True):
            np.testing.assert_array_equal(value,base)
    a,b = (scores[role]['worlds'] for role in ROLES[:2])
    common_stall = trigger&np.array([x['near_static'] and y['near_static'] for x,y in zip(a,b,strict=True)])
    common_started = trigger&np.array([x['onset_block_end_seconds'] is not None and y['onset_block_end_seconds'] is not None
                                       and x['survived'] and y['survived'] for x,y in zip(a,b,strict=True)])
    directions = {}
    for role in SIGNS:
        worlds = scores[role]['worlds']
        survived = np.array([row['survived'] for row in worlds])
        started = survived&np.array([row['onset_block_end_seconds'] is not None for row in worlds])
        rescued = np.flatnonzero(common_stall&started)
        linked = []
        for world in rescued:
            key = str(world)
            if key not in support[role] or any(key in support[ref] for ref in ROLES[:2]):
                continue
            if support[role][key]['confirmation_control']<=50*worlds[world]['onset_block_end_seconds']:
                linked.append(int(world))
        lost = np.flatnonzero(common_started&~started)
        unsafe = np.flatnonzero(trigger&(~survived | ~envelope[role]))
        directions[role] = dict(rescued=rescued.tolist(),linked_support_and_onset=linked,
            lost=lost.tolist(),unsafe=unsafe.tolist(),passed=bool(linked and not len(lost) and not len(unsafe)))
    result = dict(status='passed-integrity',seed=seed,protocol_sha256=sha(PLAN),record=state['record'],
        triggered=ids.tolist(),common_near_static=np.flatnonzero(common_stall).tolist(),
        common_already_started=np.flatnonzero(common_started).tolist(),directions=directions,support=support,
        impulse_receipts=impulses,scores=scores,videos=videos,onnx_inputs=onnx,
        trace_sha256=hashes,shared_prefix_sha256=sha(prefix_path),transitions=spec['maximum_control_transitions'],
        physics_substeps=spec['maximum_physics_substeps'],heldout_used=False,learning_updates=0)
    output = OUT/f'seed{seed}-analysis.json'
    write(output,result)
    state.update(status='completed',branch='closed',analysis_sha256=sha(output))
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    selected = next((v['path'] for v in reversed([r['video'] for r in state['conditions']]) if v['complete_clip']),None)
    view.update(status='completed',updated_at=datetime.now(onset.TZ).isoformat(),
                purpose='后退双向侧移 · 已核验',latest_clip=selected)
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('BACKWARD_LATERAL_VERIFIED',seed,len(ids),len(result['common_near_static']),directions,flush=True)


def summarize():
    spec = checked()
    files = [OUT/f'seed{seed}-analysis.json' for seed in SEEDS]
    rows = [read(path) for path in files]
    assert all(row['status']=='passed-integrity' and row['protocol_sha256']==sha(PLAN) for row in rows)
    windows = [ROOT/row['record']/'windows-view-verification.json' for row in rows]
    observation = [read(path) for path in windows]
    assert all(row['status']=='passed' for row in observation)
    same = [role for role in SIGNS if all(row['directions'][role]['passed'] for row in rows)]
    output = dict(status='passed-development-mechanism' if same else 'closed-mechanism-gate-not-met',
        shared_directions=same,protocol_sha256=sha(PLAN),records=[row['record'] for row in rows],
        sources={path.name:sha(path) for path in files},observation_sha256=[sha(path) for path in windows],
        cohorts=[dict(seed=row['seed'],triggered=len(row['triggered']),common_stalls=len(row['common_near_static']),
                      directions=row['directions']) for row in rows],
        transitions=sum(row['transitions'] for row in rows),physics_substeps=sum(row['physics_substeps'] for row in rows),
        heldout_used=False,learning_updates=0,requires_actuator_compatible_trial=True)
    assert output['transitions']==spec['full_budget']
    write(OUT/'analysis.json',output)
    print('BACKWARD_LATERAL_COMBINED',output['status'],same,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    if sys.argv[1]=='--summarize':
        summarize()
    else:
        verify(int(sys.argv[1]))
