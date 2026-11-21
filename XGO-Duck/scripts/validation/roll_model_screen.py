"""Prospective fixed-model roll studies; no optimizer or automatic extension.

Reuse the frozen raw-action collector and independently reconstruct all roll
criteria. Each role gets a fresh foreground process so RNG/physics creation is
identical. A new physical seed screens the promising unchanged-reward model;
it does not rescue the rejected top-contact reward treatment.
CPU preflight checks configuration and source/model hashes without CUDA.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import yaml
from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write
from collect_roll_policy import config, evaluate
from verify_roll_reference import independent

PLAN = ROOT/'configs/experiments/roll_adaptation_screen.json'


def freeze_adaptation():
    import torch
    assert not PLAN.exists()
    parent = ROOT/'records/reports/roll-top-contact'
    prior = read(parent/'analysis.json')
    assert prior['status'] == 'passed' and not prior['candidate_continue']
    trained = read(parent/'control/training-verification.json')
    assert trained['status'] == 'passed' and trained['completed_updates'] == 50
    original = read(ROOT/'configs/experiments/roll_reference.json')
    sources = dict(original['source_sha256'])
    for name in ('roll_model_screen.py', 'collect_roll_policy.py', 's1_reference_inputs.py'):
        path = ROOT/'scripts/validation'/name
        sources[str(path.relative_to(ROOT))] = sha(path)
    models = dict(original=(original['model'], original['model_sha256']),
                  adapted=(trained['model'], trained['model_sha256']))
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    items = []
    for role, (model, digest) in models.items():
        assert sha(ROOT/model) == digest
        p = deepcopy(original)
        p.update(role=role, seed=1005, model=model, model_sha256=digest,
                 source_sha256=sources, scope='Independent development screen; no heldout or hardware claim')
        items.append(p)
    configs = {p['role']: yaml.load(yaml.dump(asdict(config(p))), Loader=yaml.BaseLoader) for p in items}
    assert configs['original'] == configs['adapted']
    for path, digest in original['training_source_sha256'].items():
        assert sha(WORK/path) == digest, path
    assert not torch.cuda.is_initialized()
    out = ROOT/'records/reports/roll-adaptation-screen'
    out.mkdir(exist_ok=False)
    plan = dict(status='frozen-before-simulation', stage='S4-adaptation-screen',
                change='adaptation-screen', stamp=stamp, output=str(out.relative_to(ROOT)),
                evaluations=items, maximum_transitions=28800, learning_updates=0,
                source_sha256=sources, training_source_sha256=original['training_source_sha256'],
                selected_from=str((parent/'control/training-verification.json').relative_to(ROOT)),
                selected_from_sha256=sha(parent/'control/training-verification.json'),
                rejected_treatment_sha256=sha(parent/'analysis.json'), heldout_used=False,
                gate='adapted>=29/32 and adapted>=original; then consider new multi-training-seed protocol',
                automatic_extension=False, automatic_retry=False)
    write(PLAN, plan)
    write(out/'preflight.json', dict(status='passed', protocol_sha256=sha(PLAN),
                                    evaluation_configs=configs, cuda_initialized=False))
    print(json.dumps(dict(status='passed', protocol_sha256=sha(PLAN), roles=list(models),
                          seed=1005, learning_updates=0), indent=2))


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
    np.testing.assert_allclose(t['observations'][:,:,48:][t['active']], 0., atol=1e-6, rtol=0)
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
                  learning_updates=0, heldout_used=master['heldout_used'], script_sha256=sha(Path(__file__)))
    write(folder/'verification.json', result)
    write(ROOT/master['output']/f'{p["role"]}-analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    return result


def run_one(masterfile, role):
    from evaluate_standup_reference import Tee
    masterfile=Path(masterfile); master=read(masterfile)
    assert os.isatty(1), 'Use the shared visible task terminal'
    p=deepcopy(next(x for x in master['evaluations'] if x['role']==role))
    p['master_sha256']=sha(masterfile)
    for source,digest in p['source_sha256'].items(): assert sha(ROOT/source)==digest,source
    for source,digest in p['training_source_sha256'].items(): assert sha(WORK/source)==digest,source
    assert sha(ROOT/p['model'])==p['model_sha256']
    run_id=f'duck_roll_{master["change"]}_{role}_s{p["seed"]}_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id
    folder.mkdir(); (folder/'videos').mkdir(); write(folder/'protocol.json',p)
    for name in p['source_sha256']: shutil.copy2(ROOT/name,folder/Path(name).name)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),role=role,conditions=[],
               model=str(ROOT/p['model']),model_sha256=p['model_sha256'],protocol_sha256=sha(folder/'protocol.json'),
               learning_updates=0,started_at=datetime.now(timezone.utc).isoformat())
    out=ROOT/master['output']
    def save(): write(folder/'evaluation.json',state); write(out/f'{role}-progress.json',state)
    assert not (out/f'{role}-progress.json').exists()
    save(); write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    with (folder/'execution.log').open('a',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try: evaluate(p,folder,state);state['status']='completed'
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:state['ended_at']=datetime.now(timezone.utc).isoformat();save();sys.stdout,sys.stderr=oldout,olderr
    return verify(folder,masterfile)


def matched(rows):
    """Check captured physical initial state and fields, not unrecorded solver state."""
    values=[]
    for row in rows:
        folder=ROOT/row['evaluator']
        with np.load(folder/'forward-roll-trace.npz') as t:
            initial=[t[k].copy() for k in ('initial_qpos','initial_qvel')]+[t['observations'][1].copy()]
        with np.load(folder/'expanded-model-initial.npz') as m: fields={k:m[k].copy() for k in m.files}
        values.append((initial,fields))
    for initial,fields in values[1:]:
        for a,b in zip(values[0][0],initial):np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)
        assert set(fields)==set(values[0][1])
        for key in fields:np.testing.assert_array_equal(fields[key],values[0][1][key])


def analyze(masterfile):
    master=read(Path(masterfile));out=ROOT/master['output']
    rows={p['role']:read(out/f'{p["role"]}-analysis.json') for p in master['evaluations']}
    assert all(r['status']=='passed' for r in rows.values())
    matched(list(rows.values()))
    k={role:r['successes'] for role,r in rows.items()}
    advance=bool(k['adapted']>=29 and k['adapted']>=k['original'])
    result=dict(status='passed',successes=k,episodes_each=32,independent_development_seed=1005,
                adaptation_continue=advance,decision='new-multi-training-seed-protocol-warranted' if advance else 'close-adaptation-lead',
                captured_initials_and_fields_match=True,learning_updates=0,heldout_used=False,
                master_sha256=sha(Path(masterfile)),rejected_top_contact_treatment_remains_closed=True)
    write(out/'analysis.json',result);print(json.dumps(result,indent=2));return result


def run(masterfile):
    masterfile=Path(masterfile);master=read(masterfile);out=ROOT/master['output']
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']==master['stage']
    assert sha(masterfile)==read(out/'preflight.json')['protocol_sha256']
    assert not (out/'progress.json').exists()
    state=dict(status='running',master_sha256=sha(masterfile),pid=os.getpid(),completed_roles=[])
    write(out/'progress.json',state)
    try:
        for p in master['evaluations']:
            subprocess.run([sys.executable,'-B',str(Path(__file__)),str(masterfile),'--role',p['role']],cwd=WORK,check=True)
            state['completed_roles'].append(p['role']);write(out/'progress.json',state)
        analyze(masterfile);state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(out/'progress.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('protocol',nargs='?',default=str(PLAN))
    parser.add_argument('--freeze-adaptation',action='store_true');parser.add_argument('--role');args=parser.parse_args()
    if args.freeze_adaptation:freeze_adaptation()
    elif args.role:run_one(args.protocol,args.role)
    else:run(args.protocol)
