"""Independent CPU audit of the early measured-input stall correction.

Reconstruct latch, per-model input histories, selected raw actions, HOME and
physical target transition from saved values. Do not integrate new episodes.
File integrity, recovery functionality and full S5 acceptance stay separate.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read, sha
from record_io import write
from getup_coverage_metrics import evaluate, POSES
from recovery_end_hold import evaluate_end_hold

PLAN = ROOT/'configs/experiments/getup_early_guard_study.json'
OUT = ROOT/'records/reports/getup-early-guard'
PARITY_STEPS = (1,100,101,250,251,400,550)


def verify(role):
    import imageio_ffmpeg, mujoco, onnxruntime as ort
    from mjlab.scene import Scene
    from getup_early_guard_study import checked, prepare
    m = checked(); p = m['arms'][role]
    state = read(OUT/f'{role}-progress.json'); record = Path(state['path'])
    assert state['status']=='completed' and state['control_steps']==550
    assert sha(record/'coverage-trace.npz')==state['trace_sha256']
    startup = read(record/'startup.json')
    assert startup['status']=='passed' and startup['physical_steps']==0
    assert startup['cuda_model_home_switch_and_slew_checked'] and startup['new_processor_memory_restored_exact']
    assert startup['early_guard_before5s_checked'] and startup['all_new_memories_restored']
    if role!='control':
        assert read(record/'matched-start.json')['status']=='passed'
    runtime = read(record/'runtime.json')
    assert runtime['control_steps']==550 and runtime['control_transitions']==176000
    assert runtime['physics_substeps']==2750 and runtime['contact_peak']<runtime['contact_slots']
    assert runtime['resets_after_initialization']==runtime['optimizer_steps']==runtime['extra_render_physics_steps']==0
    assert read(record/'actor-integrity.json')['both_actors_and_normalizers_unchanged']
    with np.load(record/'coverage-trace.npz') as f:
        t = {k:f[k] for k in f.files}
    n = p['num_envs']
    assert t['qpos'].shape==(550,n,21) and t['qvel'].shape==(550,n,20)
    for name, values in t.items():
        if values.dtype.kind not in ('U','S'):
            assert np.isfinite(values).all(), name
    np.testing.assert_array_equal(t['episode_ages'],np.broadcast_to(np.arange(1,551)[:,None],(550,n)))
    assert not t['terminated'].any() and not t['truncated'][:-1].any() and t['truncated'][-1].all()
    assert t['active'].all()
    np.testing.assert_array_equal(t['requested_pose'],np.array([POSES[i] for i in p['split']['pose_codes']]))
    np.testing.assert_allclose(t['height'],t['qpos'][...,2],atol=2e-6,rtol=0)
    np.testing.assert_allclose(t['quaternion'],t['qpos'][...,3:7],atol=2e-6,rtol=0)
    cfg,_,meta = prepare(p)
    model = Scene(cfg.env.scene,device='cpu').compile()
    names = meta['joint_names'].split(',')
    ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+s) for s in names]
    addresses = model.jnt_qposadr[ids]; ranges = model.jnt_range[ids]
    np.testing.assert_allclose(ranges,p['joint_limits'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(t['joint_positions'],t['qpos'][...,addresses],atol=2e-6,rtol=0)
    bounded = ((t['joint_positions']>=ranges[:,0]-np.deg2rad(2.))&
               (t['joint_positions']<=ranges[:,1]+np.deg2rad(2.))).all(-1)
    np.testing.assert_array_equal(t['joint_envelope_ok'],bounded)
    base_previous = np.concatenate([np.zeros((1,n,14),np.float32),t['raw_policy_actions'][:-1]])
    np.testing.assert_array_equal(t['observations'][...,34:48],base_previous)
    np.testing.assert_allclose(t['observations'][...,48:],0,atol=1e-6,rtol=0)
    home = np.asarray(p['model_home'],np.float32)
    dest_home = np.asarray(p['standing_home'],np.float32)
    limits = np.asarray(p['joint_limits'],np.float32)
    source_history = np.zeros((n,14),np.float32)
    dest_history = np.zeros_like(source_history)
    shadow = np.zeros(n,bool)
    guard_count = np.zeros(n,np.int64)
    guard_latched = np.zeros(n,bool)
    correction = np.zeros(n,np.float32)
    expected_parity = dict(getup_obs=[],stand_obs=[])
    max_target_error = 0.
    for i in range(550):
        sensed = t['observations'][i]
        g = sensed[:,3:6]
        aligned = -g[:,2]/np.linalg.norm(g,axis=-1)>=np.cos(np.deg2rad(p['handoff_settings']['tilt_max_deg']))
        eligible=(i>=250)&aligned
        shadow |= eligible
        neck=p['neck_index'];settings=p['guard_settings']
        vertical=-g[:,2]/np.linalg.norm(g,axis=-1)
        eligible_guard=(i>=100)&(vertical>=np.cos(np.deg2rad(settings['tilt_max_deg'])))
        eligible_guard &= vertical<=np.cos(np.deg2rad(settings['tilt_min_deg']))
        eligible_guard &= sensed[:,6+neck]+home[neck]<np.deg2rad(settings['neck_below_deg'])
        eligible_guard &= abs(sensed[:,20+neck])<=settings['neck_speed_max_rad_s']
        eligible_guard &= t['getup_actions'][i,:,neck]+home[neck]<limits[neck,0]
        guard_count=np.where(eligible_guard,np.minimum(guard_count+1,16),0)
        guard_latched |= guard_count>=16
        neck_active=guard_latched&p['guard_enabled'];standing=np.zeros(n,bool)
        np.testing.assert_array_equal(t['shadow_handoff'][i],shadow)
        np.testing.assert_array_equal(t['phase_standing'][i],standing)
        np.testing.assert_array_equal(t['neck_home_active'][i],neck_active)
        np.testing.assert_array_equal(t['guard_eligible'][i],eligible_guard)
        np.testing.assert_array_equal(t['guard_count'][i],guard_count)
        np.testing.assert_array_equal(t['guard_latched'][i],guard_latched)
        go = sensed.copy(); go[:,34:48] = source_history; go[:,48:] = 0
        stand = sensed.copy(); stand[:,6:20] += home-dest_home
        stand[:,34:48] = dest_history; stand[:,48:] = 0
        if i+1 in PARITY_STEPS:
            expected_parity['getup_obs'].append(go[:32])
            expected_parity['stand_obs'].append(stand[:32])
        selected = np.where(standing[:,None],t['standing_actions'][i],t['getup_actions'][i])
        if i<100:
            selected[:] = 0
        np.testing.assert_array_equal(t['raw_policy_actions'][i],selected)
        source_history[~standing & (i>=100)] = t['getup_actions'][i,~standing & (i>=100)]
        dest_history[standing] = t['standing_actions'][i,standing]
        requested = home+selected
        requested[standing] = (dest_home+selected)[standing]
        if i<100:
            requested[:] = np.asarray(p['simulation_home'],np.float32)
        np.testing.assert_allclose(t['requested_targets'][i],requested,atol=1e-7,rtol=0)
        desired=requested.copy()
        bound=np.float32(np.deg2rad(settings['lower_target_deg']))
        wanted=np.where(neck_active,np.maximum(bound-requested[:,neck],0),np.float32(0))
        correction=np.minimum(correction+np.float32(settings['upward_correction_rate_rad_s']*.02),wanted)
        desired[:,neck]+=correction
        expected_target = desired
        if i>=250:
            previous = t['applied_targets'][i-1]
            legal = np.clip(desired,limits[:,0],limits[:,1])
            expected_target = previous+np.clip(legal-previous,-.04,.04)
            assert (abs(t['applied_targets'][i]-previous)<=.040002).all()
            old_excess = np.maximum(np.maximum(limits[:,0]-previous,previous-limits[:,1]),0)
            actual = t['applied_targets'][i]
            new_excess = np.maximum(np.maximum(limits[:,0]-actual,actual-limits[:,1]),0)
            assert (new_excess<=old_excess+2e-6).all()
        error = float(abs(t['applied_targets'][i]-expected_target).max())
        max_target_error = max(max_target_error,error)
        np.testing.assert_allclose(t['applied_targets'][i],expected_target,atol=2e-6,rtol=0)
    assert not t['phase_standing'][:250].any()
    if not p['handoff_enabled']:
        assert not t['phase_standing'].any()
    if not p['guard_enabled']:
        assert not t['neck_home_active'].any()
    with np.load(record/'inference-parity.npz') as f:
        parity = {k:f[k].copy() for k in f.files}
    errors = {}
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    for label, model_path in [('getup',p['model']),('stand',p['standing_model'])]:
        inputs = parity[label+'_obs']; output = parity[label+'_raw']
        assert inputs.shape==(224,61) and output.shape==(224,14)
        np.testing.assert_array_equal(inputs,np.concatenate(expected_parity[label+'_obs']))
        session = ort.InferenceSession(str(ROOT/model_path),sess_options=options,providers=['CPUExecutionProvider'])
        expected = np.concatenate([session.run(None,{'obs':x[None]})[0] for x in inputs])
        np.testing.assert_allclose(expected,output,atol=2e-5,rtol=2e-5)
        errors[label] = float(abs(expected-output).max())
    result = evaluate(t,p)
    assert result==read(record/'coverage.json')
    end_hold=evaluate_end_hold(t,p,result['episodes'])
    assert end_hold==read(record/'end-hold.json')
    videos = []
    for condition in state['conditions']:
        v = condition['video']; video = record/v['path']
        assert sha(video)==v['sha256'] and v['complete_clip']
        steps = np.rint(np.asarray(v['frame_times_seconds'])/.02).astype(int)
        np.testing.assert_array_equal(steps,np.arange(0,551,2))
        world = v['environment_index']; assert world==p['video_worlds'][condition['condition']]
        hashes = [hashlib.sha256((t['initial_qpos'][world] if j==0 else t['qpos'][j-1,world]).tobytes()).hexdigest() for j in steps]
        assert hashes==v['qpos_sha256_by_frame']
        frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
        info = next(frames); count = sum(1 for _ in frames)
        assert count==276 and info['fps']==25 and info['size']==(640,480)
        videos.append(dict(path=str(video.relative_to(ROOT)),frames=count,start=0.,end=11.,
                           qpos_hashes_exact=True,sha256=sha(video)))
    assert len(videos)==4
    audits = read(record/'contact-audits.json')
    assert len(audits)==22 and all(x['matched'] for x in audits)
    for a in audits:
        assert int(t['native_head_leg_contact'][a['control_step']-1].sum())==a['contact_worlds']
    result.pop('episodes')
    result.update(status='passed',evidence_status='integrity-verified',record=str(record.relative_to(ROOT)),
        trace_sha256=sha(record/'coverage-trace.npz'),protocol_sha256=sha(PLAN),
        inference_inputs_each_actor=224,maximum_import_error=errors,maximum_target_reconstruction_error=max_target_error,
        continuous_no_physical_reset=True,handoff_episodes=int(t['phase_standing'].any(0).sum()),
        neck_home_episodes=int(t['neck_home_active'].any(0).sum()),end_hold=end_hold,
        early_guard_episodes=int(t['neck_home_active'].any(0).sum()),
        shadow_guard_episodes=int(t['guard_latched'].any(0).sum()),
        all_motion_gates_passed=result['functional_coverage_gate'] and end_hold['functional_coverage_gate'],
        shadow_handoff_episodes=int(t['shadow_handoff'].any(0).sum()),
        native_late_head_leg_episodes=int(t['native_head_leg_contact'][399:].any(0).sum()),
        videos=videos,physics_steps_added_by_verifier=0,learning_updates=0,
        scope='Fixed recovery development evaluation only; full S5 sequences and robustness still pending')
    write(OUT/f'{role}-analysis.json',result); write(record/'verification.json',result)
    print('EARLY_GUARD_EVAL_VERIFIED',role,json.dumps({k:v for k,v in result.items() if k in
        ['requested','actual','overall','functional_coverage_gate','handoff_episodes']}),flush=True)
    return result

