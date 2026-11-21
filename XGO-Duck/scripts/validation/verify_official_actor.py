"""CPU-only audit of official actor import on independent saved observations.

No simulation, fitting, weight update or CUDA allocation. Tests cover all
seven commands, off-policy failed observations and a declared random stress
set. Normalization remains unchanged even after an attempted training update.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from official_actor import import_official_actor
from s1_reference_inputs import read, sha

OUT = ROOT/'records/reports/official-actor-bootstrap'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = ROOT/'workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/python/xgoduck_walk.onnx'
    actor = MLPModel(TensorDict({'actor': torch.zeros(1,61)}, batch_size=[1]),
        {'actor':['actor']}, 'actor', 14, hidden_dims=(512,256,128),
        activation='elu', obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    imported = import_official_actor(actor, model)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    state=read(ROOT/'records/reports/s1-evaluation/progress.json')
    cases=[];inputs={str(model.relative_to(ROOT)):sha(model)}
    for role in ('official-reference','reachable_head'):
        for part,conditions in [('five',('stop','forward','backward','turn_left','turn_right')),
                                ('lateral',('strafe_left','strafe_right'))]:
            folder=Path(state['models'][role][part])
            assert read(folder/'verification.json')['status']=='passed'
            for condition in conditions:
                path=folder/f'{condition}-trace.npz'
                with np.load(path,allow_pickle=False) as data:
                    # Fixed stride across all environments, no success filtering.
                    obs=data['observations'][::25].reshape(-1,61).copy()
                inputs[str(path.relative_to(ROOT))]=sha(path)
                cases.append((f'{role}/{condition}',obs))
    rng=np.random.default_rng(31415)
    stress=rng.normal(size=(512,61)).astype(np.float32)
    cases.append(('random-stress',stress))
    rows=[]
    actor.eval()
    for name,obs in cases:
        with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
        expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
        error=np.abs(actual-expected)
        passed=bool(np.all(error <= 2e-5+2e-5*np.abs(expected)))
        rows.append(dict(case=name,observations=len(obs),max_abs_error=float(error.max()),passed=passed))
        assert passed,name
    before={k:v.clone() for k,v in actor.obs_normalizer.state_dict().items()}
    actor.train();actor.update_normalization(TensorDict({'actor':torch.from_numpy(stress)},batch_size=[len(stress)]))
    assert all(torch.equal(value,actor.obs_normalizer.state_dict()[key]) for key,value in before.items())
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        imported=imported,comparison=rows,atol=2e-5,rtol=2e-5,
        input_sha256=inputs,script_sha256=sha(Path(__file__)),
        importer_sha256=sha(ROOT/'scripts/training/official_actor.py'),
        frozen_normalization_update_checked=True,new_learning=False,new_simulation=False,cuda_initialized=False)
    (OUT/'actor-import-check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','imported')},indent=2))


if __name__=='__main__':main()
