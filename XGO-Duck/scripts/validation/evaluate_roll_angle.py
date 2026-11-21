"""Fixed original/control/candidate roll evaluation with actual angle inputs.

The existing visible runner handles naming, provenance and child lifetime.
Use the new explicit collector and independently replay its saved model inputs,
causal gravity observer, unchanged roll criteria and complete video.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from verify_roll_reference import independent
import roll_model_screen as orchestration
from collect_roll_angle import evaluate

def verify(folder, masterfile):
    """Independent quaternion/contact reconstruction, raw history and video QA."""
    import imageio_ffmpeg
    import onnxruntime as ort
    masterfile = Path(masterfile)
    master = read(masterfile)
    folder = Path(folder)
    p = read(folder/'protocol.json')
    r = read(folder/'evaluation.json')
    assert r['status'] == 'completed' and r['protocol_sha256'] == sha(folder/'protocol.json')
    assert p['master_sha256'] == sha(masterfile)
    assert sha(ROOT/p['model']) == p['model_sha256'] == r['model_sha256']
    pre = read(ROOT/master['output']/'preflight.json')
    expected = pre['evaluation_configs'][p.get('config_key', p['role'])]
    assert yaml.load((folder/'env.yaml').read_text(), Loader=yaml.BaseLoader) == expected
    for name, digest in p['source_sha256'].items():
        assert sha(folder/Path(name).name) == digest, name
    assert sha(folder/'expanded-model-initial.npz') == r['expanded_model_sha256']
    item = r['conditions'][0]
    path = folder/'forward-roll-trace.npz'
    assert sha(path) == item['trace_sha256']
    with np.load(path) as value:
        t = {k:value[k].copy() for k in value.files}
    np.testing.assert_allclose(t['observations'][1:,:,34:48][t['active'][1:]],
                               t['raw_policy_actions'][:-1][t['active'][1:]], atol=1e-7, rtol=0)
    from inspect_roll_angle_inputs import gravity_phase
    from roll_angle_observer import GravityPhase
    reconstructed=gravity_phase(t['observations'][:,:,3:6])
    observer=GravityPhase()
    causal=np.asarray([observer.update(g) for g in t['observations'][:,:,3:6]])
    np.testing.assert_allclose(causal,reconstructed,atol=2e-6,rtol=0)
    np.testing.assert_allclose(t['estimated_phase_rad'][t['active']],reconstructed[t['active']],atol=2e-6,rtol=0)
    target_phase=0. if p['role']=='original' else reconstructed[t['active']]
    np.testing.assert_allclose(t['observations'][:,:,55][t['active']],target_phase,atol=2e-6,rtol=0)
    tail=np.concatenate((t['observations'][:,:,48:55],t['observations'][:,:,56:]),axis=2)
    np.testing.assert_allclose(tail[t['active']],0.,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['applied_targets'][1:], np.asarray(r['policy_home'])+t['raw_policy_actions'][1:], atol=1e-6, rtol=0)
    rows = independent(t, p['criteria'], np.asarray(r['joint_limits']), np.deg2rad(p['actual_joint_limit_tolerance_deg']))
    assert len(rows) == p['episodes']
    for observed, saved in zip(rows, item['episodes']):
        for key, value in observed.items():
            if value is None or isinstance(value, (bool,int)):
                assert value == saved[key], (key,value,saved[key])
            else:
                np.testing.assert_allclose(value, saved[key], atol=2e-4, rtol=0)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    errors = []
    for step in range(1, len(t['time']), 30):
        for index in np.flatnonzero(t['active'][step]):
            action = session.run(None, {'obs':t['observations'][step,index:index+1]})[0][0]
            np.testing.assert_allclose(action, t['raw_policy_actions'][step,index], atol=2e-5, rtol=2e-5)
            errors.append(float(abs(action-t['raw_policy_actions'][step,index]).max()))
    video = folder/item['video']['path']
    assert sha(video) == item['video']['sha256']
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    meta = next(reader); frames = sum(1 for _ in reader)
    assert meta['fps'] == 25 and meta['size'] == (960,720)
    assert frames == len(item['video']['frame_times_seconds'])
    k = sum(x['success'] for x in rows); n = len(rows)
    assert k == item['successes']
    z=1.959963984540054; f=k/n; den=1+z*z/n
    c=(f+z*z/(2*n))/den; half=z*np.sqrt(f*(1-f)/n+z*z/(4*n*n))/den
    result = dict(status='passed', role=p['role'], seed=p['seed'], episodes=n, successes=k,
                  numeric_gate_passed=f>=p['success_fraction_min'], wilson95=[max(0.,c-half),min(1.,c+half)],
                  rows=rows, evaluator=str(folder.relative_to(ROOT)), master_sha256=sha(masterfile),
                  model_sha256=p['model_sha256'], trace_sha256=sha(path), video_sha256=sha(video),
                  decoded_frames=frames, replayed_inputs=len(errors), maximum_inference_error=max(errors),
                  observed_gravity_phase_verified=True, phase_slot=55, learning_updates=0, heldout_used=master['heldout_used'], script_sha256=sha(Path(__file__)))
    write(folder/'verification.json', result)
    write(ROOT/master['output']/f'{p["role"]}-analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    return result


def run(masterfile,role):
    # Explicit per-process dependency binding, without editing frozen sources.
    orchestration.evaluate=evaluate
    orchestration.verify=verify
    return orchestration.run_one(masterfile,role)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('protocol');p.add_argument('--role',required=True)
    p.add_argument('--verify-folder');a=p.parse_args()
    if a.verify_folder:verify(a.verify_folder,a.protocol)
    else:run(a.protocol,a.role)
