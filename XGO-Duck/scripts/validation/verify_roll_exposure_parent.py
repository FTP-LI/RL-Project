"""Verify and retain one common PPO starting point with the official roll actor."""

import argparse
from hashlib import file_digest
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-friction-exposure-preflight'
OLD=ROOT/'logs/rsl_rl/xgoduck_roulade/2026-11-14_00-44-20_duck_roll_airborne-support_control_s1210_20261114T004321/model_init.pt'
ANCHOR=ROOT/'artifacts/training-parents/roll-official-actor-ppo-start.pt'
ONNX=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx'
PROBE=ROOT/'records/experiments/duck_roll_airborne-support_control_s1210_20261114T004321'
PREFIX=ROOT/'records/reports/roll-matched-factor-low-continuation-s1242/matched-prefix-analysis.json'
SOURCE_SHA='2cf410fa379a1f78cbc9106f2cb4bd12456ef0ae9364e01dc8b9002a28be136c'
MODEL_SHA='8adb1e6bc3f5eef751ce4edbf940f6d5a25fdec3ec4a798e1d9ca135c8248e97'


def sha(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def finite(state):
    if isinstance(state,torch.Tensor):
        return bool(torch.isfinite(state).all())
    if isinstance(state,dict):
        return all(finite(v) for v in state.values())
    if isinstance(state,(tuple,list)):
        return all(finite(v) for v in state)
    return True


def require(check,message):
    if not check:
        raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    p=read(ROOT/'project.json')
    state=p['staged_training_request']
    require(p['goal_execution']['status']=='active'
        and not any(state[k] for k in ('pause_requested','learning_running','simulation_running','deployment_authorized_now')),
        'The project must be idle for CPU receipt creation')
    receipt=REPORT/'common-ppo-parent.json'
    if not args.check_only:
        require(not receipt.exists() and not ANCHOR.exists(),'Keep the prior packaged checkpoint')
    require(sha(OLD)==SOURCE_SHA and sha(ONNX)==MODEL_SHA,'Parent or official model changed')
    imported=read(PROBE/'live-import-parity.json')
    require(imported['physical_steps']==0 and imported['status']=='passed' and imported['imported']['model_sha256']==MODEL_SHA,
        'Old source is not the recorded pre-physics official import')
    torch.set_num_threads(2)
    parent=torch.load(OLD,map_location='cpu',weights_only=True)
    require(set(parent)>={'actor_state_dict','critic_state_dict','optimizer_state_dict','iter','infos'}
        and parent['iter']==-1 and parent['infos']['env_state']['common_step_counter']==0
        and not parent['optimizer_state_dict']['state'] and parent['optimizer_state_dict']['param_groups']
        and finite(parent),'Expected a complete PPO initialization checkpoint')
    require(int(parent['actor_state_dict']['obs_normalizer.count'])==0
        and int(parent['critic_state_dict']['obs_normalizer.count'])==0, 'Parent already contains learned observation statistics')
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
                   hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
                   distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(parent['actor_state_dict'],strict=True)
    actor.obs_normalizer.until=0
    actor.eval()
    source=read(PREFIX)['arms']['nominal-a']
    commands=ROOT/'records/evaluations'/source['run_id']/'commands.npz'
    require(sha(commands)==read(commands.parent/'guarded-evaluation.json')['command_sha256'],'Saved actor data changed')
    with np.load(PROBE/'live-import-parity.npz',allow_pickle=False) as saved:
        original=saved['observations']
    with np.load(commands,allow_pickle=False) as saved:
        controls,worlds=np.nonzero(saved['phase_used'][:500]==1)
        selected=np.linspace(0,len(controls)-1,256,dtype=int)
        actual_inputs=saved['actor_input'][controls[selected],worlds[selected]]
    observations=np.concatenate((original,actual_inputs))
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=1
    session=ort.InferenceSession(str(ONNX),sess_options=opts,providers=['CPUExecutionProvider'])
    official=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in observations])
    with torch.inference_mode():
        replay=actor(TensorDict({'actor':torch.from_numpy(observations)},batch_size=[len(observations)])).numpy()
    np.testing.assert_allclose(official,replay,atol=2e-5,rtol=2e-5)
    require(not torch.cuda.is_initialized(),'The verification must not initialize CUDA')
    result=dict(status='verified-common-parent-checkpoint', anchor=rel(ANCHOR),anchor_sha256=SOURCE_SHA,
        source=rel(OLD),source_sha256=sha(OLD),official_onnx=rel(ONNX),official_onnx_sha256=MODEL_SHA,
        actor_inputs=len(observations),max_error=float(abs(official-replay).max()),
        reset_input_records=len(original),recorded_roll_inputs=len(actual_inputs),
        critic_input_width=int(parent['critic_state_dict']['obs_normalizer._mean'].shape[-1]),
        optimizer_groups=len(parent['optimizer_state_dict']['param_groups']),optimizer_step_entries=0,
        optimizer_history_scope='Shared freshly initialized critic and optimizer; not the original official PPO history.',
        checkpoints_start_iteration=-1,physical_steps=0,learning_updates=0,hardware_io=False)
    if not args.check_only:
        ANCHOR.parent.mkdir(exist_ok=True)
        with OLD.open('rb') as source, ANCHOR.open('xb') as output:
            shutil.copyfileobj(source,output)
    require(sha(ANCHOR)==SOURCE_SHA,'Retained checkpoint copy differs')
    if args.check_only:
        require(read(receipt)==result,'Parent receipt changed')
    else:
        with receipt.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(result,f,indent=2,allow_nan=False)
            f.write('\n')
    print('ROLL_PARENT_VERIFIED',len(observations),result['max_error'],flush=True)


if __name__=='__main__':
    sys.exit(main())
