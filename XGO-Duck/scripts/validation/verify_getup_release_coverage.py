"""Recompute coverage from saved first episodes, then audit every full video.

File integrity and motion acceptance have separate fields. A failed coverage
or recovery gate is a valid completed experiment, never a passed motion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_coverage_metrics import evaluate,POSES

OUT=ROOT/'records/reports/getup-release-coverage/fixed-evaluation'
PLAN=ROOT/'configs/experiments/getup_release_coverage.json'


def verify(role):
    import onnx
    import onnxruntime as ort
    import imageio_ffmpeg
    import mujoco
    from mjlab.scene import Scene
    from getup_release_coverage import checked,prepare
    m=checked();p=m['roles'][role];state=read(OUT/f'{role}-progress.json');record=Path(state['path'])
    assert state['status']=='completed' and state['control_steps']==550
    assert sha(record/'coverage-trace.npz')==state['trace_sha256']
    assert read(record/'startup.json')['status']=='passed'
    if role=='control':assert read(record/'matched-start.json')['status']=='passed'
    runtime=read(record/'runtime.json')
    assert runtime['control_steps']==550 and runtime['control_transitions']==176000
    assert runtime['physics_substeps']==2750 and runtime['contact_peak']<runtime['contact_slots']
    assert runtime['resets_after_initialization']==runtime['optimizer_steps']==runtime['extra_render_physics_steps']==0
    assert read(record/'actor-integrity.json')['actor_and_normalizer_unchanged']
    with np.load(record/'coverage-trace.npz') as f:t={k:f[k] for k in f.files}
    assert t['qpos'].shape==(550,320,21) and t['qvel'].shape==(550,320,20)
    for k,v in t.items():
        if v.dtype.kind not in ('U','S'):assert np.isfinite(v).all(),k
    np.testing.assert_array_equal(t['episode_ages'],np.broadcast_to(np.arange(1,551)[:,None],(550,320)))
    assert not t['terminated'].any() and not t['truncated'][:-1].any() and t['truncated'][-1].all()
    assert t['active'].all()
    expected_labels=np.array([POSES[i] for i in p['split']['pose_codes']])
    np.testing.assert_array_equal(t['requested_pose'],expected_labels)
    np.testing.assert_allclose(t['height'],t['qpos'][...,2],atol=2e-6,rtol=0)
    np.testing.assert_allclose(t['quaternion'],t['qpos'][...,3:7],atol=2e-6,rtol=0)
    cfg,_,_=prepare(p);model=Scene(cfg.env.scene,device='cpu').compile()
    metadata={x.key:x.value for x in onnx.load(ROOT/p['model']).metadata_props}
    names=metadata['joint_names'].split(',')
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in names]
    addresses=model.jnt_qposadr[ids];ranges=model.jnt_range[ids]
    np.testing.assert_allclose(t['joint_positions'],t['qpos'][...,addresses],atol=2e-6,rtol=0)
    tolerance=np.deg2rad(2.)
    bounded=((t['joint_positions']>=ranges[:,0]-tolerance)&(t['joint_positions']<=ranges[:,1]+tolerance)).all(-1)
    np.testing.assert_array_equal(t['joint_envelope_ok'],bounded)
    np.testing.assert_array_equal(t['raw_policy_actions'][:100],0)
    np.testing.assert_allclose(t['applied_targets'][:100],np.broadcast_to(p['simulation_home'],(100,320,14)),atol=1e-7,rtol=0)
    home=np.asarray(metadata['default_joint_pos'].split(','),np.float32)
    np.testing.assert_allclose(t['applied_targets'][100:],home+t['raw_policy_actions'][100:],atol=1e-7,rtol=0)
    previous=np.concatenate([np.zeros((1,320,14),np.float32),t['raw_policy_actions'][:-1]])
    np.testing.assert_array_equal(t['observations'][...,34:48],previous)
    np.testing.assert_allclose(t['observations'][...,48:],0,atol=1e-6,rtol=0)
    with np.load(record/'inference-parity.npz') as f:observations=f['observations'].copy();actions=f['actions'].copy()
    assert observations.shape==(224,61)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in observations])
    np.testing.assert_allclose(expected,actions,atol=2e-5,rtol=2e-5)
    # Recompute independent motion output from the complete raw trace.
    result=evaluate(t,p);assert result==read(record/'coverage.json')
    videos=[]
    for condition in state['conditions']:
        video=condition['video'];source=record/video['path'];assert sha(source)==video['sha256']
        steps=np.rint(np.array(video['frame_times_seconds'])/.02).astype(int)
        np.testing.assert_array_equal(steps,np.arange(0,551,2))
        world=video['environment_index'];assert world==p['video_worlds'][condition['condition']]
        hashes=[hashlib.sha256((t['initial_qpos'][world] if step==0 else t['qpos'][step-1,world]).tobytes()).hexdigest() for step in steps]
        assert hashes==video['qpos_sha256_by_frame']
        reader=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24');meta=next(reader)
        count=sum(1 for _ in reader);assert count==276 and meta['fps']==25 and meta['size']==(640,480)
        videos.append(dict(path=str(source.relative_to(ROOT)),frames=count,start=0.,end=11.,qpos_hashes_exact=True,sha256=sha(source)))
    assert len(videos)==4
    audits=read(record/'contact-audits.json');assert len(audits)==22 and all(a['matched'] for a in audits)
    result.pop('episodes')
    result.update(status='passed',evidence_status='integrity-verified',model_sha256=p['model_sha256'],
        record=str(record.relative_to(ROOT)),trace_sha256=sha(record/'coverage-trace.npz'),
        protocol_sha256=sha(PLAN),inference_inputs=224,maximum_import_error=float(abs(expected-actions).max()),
        videos=videos,physics_steps_added_by_verifier=0,learning_updates=0)
    write(OUT/f'{role}-analysis.json',result);write(record/'verification.json',result)
    print('GETUP_COVERAGE_VERIFIED',role,json.dumps({k:v for k,v in result.items() if k in ('requested','actual','overall','functional_coverage_gate','strict_requested_successes','wrong_release_episodes')}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=('original','control'));a=parser.parse_args()
    if a.role:verify(a.role);return
    values={role:read(OUT/f'{role}-analysis.json') for role in ('original','control')}
    assert all(v['status']=='passed' for v in values.values())
    selected=next((role for role in ('original','control') if values[role]['functional_coverage_gate']),None)
    result=dict(status='passed',stage_gate_passed=selected is not None,selected_for_next_development=selected,
        roles=values,actual_evaluation_transitions=352000,learning_updates=0,heldout_used=False,hardware_io=False,
        old_requested_failures_unchanged=True,protocol_sha256=sha(PLAN),
        scope='One fresh development seed; stage reliability and continuous transitions remain unverified')
    write(OUT/'analysis.json',result)
    print('GETUP_COVERAGE_RESULT',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)


if __name__=='__main__':main()
