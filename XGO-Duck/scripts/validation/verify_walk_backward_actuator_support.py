"""Rebuild each commanded offset and score the mirrored actuator branches."""
from datetime import datetime
import sys
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import collect_walk_backward_onset as onset
from collect_walk_backward_actuator_support import OUT,PLAN,ROLES,SEEDS,SIGNS,geometry,page,pulse
from common_scene_contract import compare_arrays,model_arrays
from record_io import write
from s1_reference_inputs import ROOT,read,sha
import training_live_page as live
from verify_common_walk_forward_phase import arrays,verify_movie,verify_resets
from verify_walk_backward_lateral_probe import support_rows
from verify_walk_backward_onset import score
from verify_walk_backward_start_pulse import verify_inputs


def checked():
    with patch.object(onset,'PLAN',PLAN),patch.object(onset,'OUT',OUT):
        return onset.checked()


def check_geometry(record,p,spec,prefix):
    from mjlab.scene import Scene

    target = arrays(record/'support-targets.npz')
    low,fork = spec['trigger']['window']
    activity = abs(prefix['observations'][low-1:fork,:,spec['trigger']['observation_indices']]).astype(float).mean(axis=(0,2))
    ended = np.logical_or.reduce([prefix[name][-1] for name in ('terminated','truncated','fell_over','nan_state','out_of_bounds')])
    eligible = (activity<=spec['trigger']['threshold_rad_s'])&prefix['active'][-1]&~ended&prefix['contact'][-1].all(axis=1)
    np.testing.assert_array_equal(target['eligible'],eligible)
    np.testing.assert_array_equal(target['qpos'][eligible],prefix['qpos'][-1,eligible])
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(OUT/'cpu-scene-fields.npz') as reference:
        compare_arrays(model_arrays(model),{key:reference[key] for key in reference.files})
    ids,offsets = geometry(model,p,target['qpos'],eligible)
    np.testing.assert_array_equal(ids,target['ids'])
    np.testing.assert_allclose(offsets,target['offsets'],atol=2e-7,rtol=0)
    return ids,offsets,int(eligible.sum())


def verify(seed):
    spec = checked()
    p = onset.environment(spec,seed)
    record = ROOT/'records/evaluations'/p['run_id']
    state = read(record/'evaluation.json')
    assert state['status']=='collected-review-required' and not torch.cuda.is_initialized()
    assert state['protocol_sha256']==sha(record/'protocol.json')==sha(PLAN)
    assert state['actor_unchanged'] and not state['heldout_used'] and state['learning_updates']==0
    steps = spec['maximum_control_transitions']//p['parallel_worlds']
    assert state['actual']==dict(attempted_controls=steps,completed_controls=steps,
        attempted_substeps=steps*5,completed_substeps=steps*5,initial_resets=1)
    assert 0<=state['contact_peak']<state['contact_slots']
    assert state['actuator_zero_restore']['counters']==[0,0]
    startup = read(record/'startup.json')
    assert startup['scene_exact'] and startup['actual_processor_restored'] and startup['cpu_cuda_rng_unchanged']
    forks = read(record/'fork-readback.json')
    assert tuple(forks)==ROLES and all(value==forks[ROLES[0]] for value in forks.values())
    fork = spec['phase']['warmup_controls']
    assert forks[ROLES[0]]['physical']['counters']==[fork,fork*5]
    motors = arrays(record/'first-motor-commands.npz')
    for role in ROLES[1:]:
        np.testing.assert_array_equal(motors[role],motors[ROLES[0]])
    initial_path,prefix_path = (record/name for name in ('physical-initial.npz','trace-shared-prefix.npz'))
    assert sha(initial_path)==state['initial_state_sha256']
    initial,prefix = arrays(initial_path),arrays(prefix_path)
    ids,offsets,eligible = check_geometry(record,p,spec,prefix)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    low,high = np.asarray(p['joint_limits'],np.float32).T
    home = np.asarray(p['model_home'],np.float32)
    verify_resets(arrays(record/'resets-shared-prefix.npz'),prefix)
    scores,videos,onnx,hashes,first,envelope,support,peak = ({} for _ in range(8))
    sent = []
    selected = np.zeros(p['parallel_worlds'],dtype=bool)
    selected[ids] = True
    for role in ROLES:
        file = record/f'trace-{role}.npz'
        post = arrays(file)
        trace = {name:np.concatenate((prefix[name],post[name])) for name in prefix}
        assert trace['active'].shape==(p['maximum_control_steps'],p['parallel_worlds'])
        first[role] = (post['observations'][0],post['actions'][0])
        nominal = np.clip(trace['actions']+home,low,high)
        target = nominal.copy()
        if role in SIGNS:
            which = 0 if role=='body-left' else 1
            for step in range(fork+1,p['maximum_control_steps']+1):
                weight = pulse(spec,step)
                chosen = trace['active'][step-1,ids]
                if weight>0 and chosen.any():
                    j = ids[chosen]
                    target[step-1,j] = np.clip(nominal[step-1,j]+weight*offsets[chosen,which],low,high)
                    sent.append(dict(role=role,step=step,worlds=j.tolist(),weight=weight))
        peak[role] = float(abs(target-nominal).max())
        assert peak[role]<=spec['support']['maximum_relative_joint_target_rad']+1e-7
        onnx[role] = verify_inputs(trace,p,session,target)
        verify_resets(arrays(record/f'resets-{role}.npz'),trace)
        scores[role] = score(trace,p,spec)
        envelope[role] = (trace['joint_envelope_ok']|~trace['active']).all(axis=0)
        support[role] = support_rows(trace,ids,fork,spec['support']['support_deadline_controls'])
        clip = next(row['video'] for row in state['conditions'] if row['role']==role)
        videos[role] = verify_movie(record,clip,trace,initial)
        hashes[role] = sha(file)
    assert sent==read(record/'support-actuation.json')
    for role in ROLES[1:]:
        for actual,expected in zip(first[role],first[ROLES[0]],strict=True):
            np.testing.assert_array_equal(actual,expected)
    a,b = (scores[role]['worlds'] for role in ROLES[:2])
    common = selected&np.array([x['near_static'] and y['near_static'] for x,y in zip(a,b,strict=True)])
    protected = selected&np.array([x['survived'] and y['survived'] and x['onset_block_end_seconds'] is not None
        and y['onset_block_end_seconds'] is not None for x,y in zip(a,b,strict=True)])
    results = {}
    for role in SIGNS:
        worlds = scores[role]['worlds']
        survived = np.array([row['survived'] for row in worlds])
        started = survived&np.array([row['onset_block_end_seconds'] is not None for row in worlds])
        rescues = np.flatnonzero(common&started)
        linked = [int(i) for i in rescues if str(i) in support[role] and
            not any(str(i) in support[ref] for ref in ROLES[:2]) and
            support[role][str(i)]['confirmation_control']<=50*worlds[i]['onset_block_end_seconds']]
        losses = np.flatnonzero(protected&~started)
        unsafe = np.flatnonzero(selected&(~survived|~envelope[role]))
        results[role] = dict(rescued=rescues.tolist(),linked_support_and_onset=linked,
             lost=losses.tolist(),unsafe=unsafe.tolist(),passed=bool(linked and not len(losses) and not len(unsafe)))
    proof = dict(status='passed-integrity',protocol_sha256=sha(PLAN),seed=seed,record=state['record'],
        eligible=eligible,triggered=ids.tolist(),common_near_static=np.flatnonzero(common).tolist(),
        directions=results,maximum_applied_delta_rad=peak,scores=scores,support=support,
        videos=videos,onnx_inputs=onnx,trace_sha256=hashes,shared_prefix_sha256=sha(prefix_path),
        actuation_sha256=sha(record/'support-actuation.json'),targets_sha256=sha(record/'support-targets.npz'),
        transitions=spec['maximum_control_transitions'],physical_substeps=steps*5,heldout_used=False,learning_updates=0)
    file = OUT/f'seed{seed}-analysis.json'
    write(file,proof)
    state.update(status='completed',branch='closed',analysis_sha256=sha(file))
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    view.update(status='completed',updated_at=datetime.now(onset.TZ).isoformat(),
        purpose='后退执行器支撑转移 · 已核验',
        latest_clip=next((row['video']['path'] for row in reversed(state['conditions']) if row['video']['complete_clip']),None))
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('BACKWARD_ACTUATOR_SUPPORT_VERIFIED',seed,len(ids),int(common.sum()),results,flush=True)


def summarize():
    records = [OUT/f'seed{seed}-analysis.json' for seed in SEEDS]
    rows = [read(path) for path in records]
    assert all(row['status']=='passed-integrity' and row['protocol_sha256']==sha(PLAN) for row in rows)
    observers = [ROOT/row['record']/'windows-view-verification.json' for row in rows]
    assert all(read(path)['status']=='passed' for path in observers)
    passes = [role for role in SIGNS if all(row['directions'][role]['passed'] for row in rows)]
    write(OUT/'analysis.json',dict(status='passed-development-mechanism' if passes else 'closed-mechanism-gate-not-met',
        protocol_sha256=sha(PLAN),shared_directions=passes,sources={f.name:sha(f) for f in records},
        transitions=sum(row['transitions'] for row in rows),physical_substeps=sum(row['physical_substeps'] for row in rows),
        observer_sha256=[sha(path) for path in observers],heldout_used=False,learning_updates=0))
    print('BACKWARD_ACTUATOR_SUPPORT_COMBINED',passes,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    if sys.argv[1]=='--summarize':
        summarize()
    else:
        verify(int(sys.argv[1]))
