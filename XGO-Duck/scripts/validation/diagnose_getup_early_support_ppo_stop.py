"""Verify the stopped prefix and exercise native reward arithmetic on saved inputs."""
import hashlib
import inspect
import json

import imageio_ffmpeg
import numpy as np
import torch

from recover_getup_early_support_ppo_start import OUT, PLAN
from run_getup_early_support_ppo import zero
from getup_early_support_reward import component
from record_io import write
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_getup_teacher_anchor_train import exact


def verified_clip(record,trace):
    clip, = read(record/'clips.json')
    samples = read(record/'recording-states.json')
    assert not clip['complete_clip'] and not clip['full_action'] and clip['extra_physics_steps']==0
    steps = list(range(0,115,2))
    assert clip['common_steps']==steps and clip['frames']==len(steps)
    assert len(samples)==len(steps)
    for step,row in zip(steps,samples,strict=True):
        state = trace['pre_qpos'][0,0] if step==0 else trace['qpos'][step-1,0]
        assert row==dict(common_step=step,qpos_sha256=hashlib.sha256(state.tobytes()).hexdigest())
    path = record/clip['file']
    frames = imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    assert next(frames)['fps']==25 and sum(1 for _ in frames)==len(steps)
    return dict(path=path.relative_to(ROOT).as_posix(),frames=len(steps),seconds=steps[-1]*.02,sha256=sha(path),complete=False)


def audit_arithmetic():
    from mjlab.managers import RewardManager

    from pathlib import Path

    receipt = read(zero.VALUE_PLAN)
    runtime = RewardManager.compute
    wrapper = Path(inspect.getsourcefile(runtime))
    wrapper_name = wrapper.relative_to(WORK).as_posix()
    assert sha(wrapper)==receipt['environment']['training_source_sha256'][wrapper_name]
    wrapper_code = inspect.getsource(runtime)
    assert 'result = _orig_reward_compute(self, dt)' in wrapper_code and '_step_reward' not in wrapper_code
    native = runtime.__globals__['_orig_reward_compute']
    source = inspect.getsourcefile(native)
    assert sha(Path(source))==receipt['library_source_sha256'][source]
    code = inspect.getsource(native)
    assert 'value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale' in code
    assert 'self._step_reward[:, term_idx] = value / scale' in code
    proof = read(zero.VALUE_OUT/'analysis.json')
    path = ROOT/proof['record']/'value-data.npz'
    assert sha(path)==proof['data_sha256']
    with np.load(path) as a, torch.no_grad():
        def gpu(name):
            return torch.from_numpy(a[name]).to('cuda:0')
        z,feet,head,observed = (gpu(name) for name in ('gravity_z','feet','head','support'))
        supine = torch.from_numpy(a['critic_obs'][:,:,75].copy()).to('cuda:0').bool()
        age = torch.arange(1,len(z)+1,device='cuda:0')[:,None]
        raw = component(z,feet,head,age,supine,**receipt['reward'])
        manager = torch.nan_to_num(raw*1.*.02,nan=0.,posinf=0.,neginf=0.)/.02
        direct_error = (observed-raw).abs()
        manager_error = (observed-manager).abs()
        torch.testing.assert_close(manager,observed,atol=0,rtol=0)
        assert (direct_error>0).any()
    return dict(reward_manager_source=source,reward_manager_sha256=sha(Path(source)),
        runtime_wrapper=wrapper_name,runtime_wrapper_sha256=sha(wrapper),source_saved_states_sha256=sha(path),compared_samples=observed.numel(),
        raw_roundoff_count=int((direct_error>0).sum()),raw_maximum_absolute_delta=float(direct_error.max()),
        manager_roundtrip_exact=not bool(manager_error.any()),extra_physical_substeps=0,learning_updates=0)


def main():
    torch.set_num_threads(4)
    spec = read(PLAN)
    coordinator = read(OUT/'coordinator.json')
    assert coordinator['protocol_sha256']==sha(PLAN) and coordinator['status']=='stopped-review-required'
    assert coordinator['arms']['control']['exit_code']==0 and coordinator['arms']['candidate']['exit_code']==1
    control,candidate = [ROOT/'records/experiments'/spec['arms'][role] for role in ('control','candidate')]
    cs, fs = read(control/'state.json'), read(candidate/'state.json')
    assert cs['controls']==408 and cs['completed_updates']==17
    assert fs['controls']==114 and fs['attempted_controls']==115 and fs['substeps']==fs['attempted_substeps']==575
    assert fs['completed_updates']==4 and fs['status']=='stopped-review-required'
    assert 'Greatest absolute difference: 3.725290298461914e-09' in fs['error']
    assert sha(control/'matched-state.json')==sha(candidate/'matched-state.json')
    with np.load(control/'training-trace.npz') as expected,np.load(candidate/'training-trace.npz') as a:
        trace = {name:a[name] for name in a.files}
        assert trace.keys()==set(expected.files)
        differences = {}
        for name,actual in trace.items():
            reference = expected[name][:114]
            assert actual.shape==reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
            changed = np.flatnonzero((actual!=reference).reshape(114,-1).any(axis=1))
            delta = np.abs(actual.astype(float)-reference.astype(float)).reshape(114,-1)
            differences[name] = dict(first_different_control=int(changed[0]+1) if changed.size else None,
                maximum_absolute_delta=float(delta.max()),
                checkpoints={str(i):float(delta[i-1].max()) for i in (1,2,100,114)})
    video = verified_clip(candidate,trace)
    start,end = (torch.load(candidate/name,map_location='cpu',weights_only=True)
                 for name in ('model-init.pt','stopped-partial.pt'))
    assert end['control_steps']==115 and end['completed_updates']==4
    assert end['actor_steps_this_run']==0 and end['critic_steps_this_run']==480
    exact(start['actor_state_dict'],end['actor_state_dict'])
    assert not end['actor_optimizer_state_dict']['state']
    for entry in end['critic_optimizer_state_dict']['state'].values():
        assert int(entry['step'])==530
    result = dict(status='verified-arithmetic-stop',control_record=control.relative_to(ROOT).as_posix(),
        candidate_record=candidate.relative_to(ROOT).as_posix(),compared_trace_controls=114,compared_trace_worlds=32,
        bitwise_equal_initial_signatures=True,online_prefix_differences=differences,
        candidate_completed_controls=114,candidate_physical_substeps=575,candidate_completed_actor_updates=0,
        candidate_completed_critic_minibatches=480,candidate_completed_ppo_updates=4,
        partial_video=video,checkpoint_sha256=sha(candidate/'stopped-partial.pt'),
        control_model_sha256=sha(ROOT/'artifacts'/spec['arms']['control']/'policy.onnx'),
        control_checkpoint_sha256=sha(control/'model-final.pt'),native_arithmetic=audit_arithmetic(),
        heldout_used=False,restarted=False,hardware_io=False)
    write(OUT/'stop-analysis.json',result)
    print('EARLY_SUPPORT_STOP_VERIFIED',result['native_arithmetic'],flush=True)


if __name__=='__main__':
    main()
