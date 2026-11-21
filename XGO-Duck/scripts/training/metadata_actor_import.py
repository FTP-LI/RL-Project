"""Import a pinned nine-node ELU policy without inventing PPO history.

Generalize the verified importer to an explicit expected model hash. Keep
normalizer denominator/epsilon semantics and freeze unknown sample history.
Critic, Adam and exploration are freshly declared, never called recovered.
CPU/live parity is mandatory before the first physics or optimizer step.
"""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import numpy_helper
import torch


def import_actor(actor,path,expected_sha256):
    path=Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256
    model=onnx.load(path);onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node]==['Sub','Div','Gemm','Elu','Gemm','Elu','Gemm','Elu','Gemm']
    arrays={a.name:numpy_helper.to_array(a).copy() for a in model.graph.initializer}
    state=actor.state_dict();keys={k for k in state if k.startswith('mlp.')}
    assert keys=={k for k in arrays if k.startswith('mlp.')}
    denominator=arrays[model.graph.node[1].input[1]]
    assert denominator.shape==(1,61) and np.isfinite(denominator).all()
    std=denominator-actor.obs_normalizer.eps;assert std.min()>=-1e-7;std=np.maximum(std,0)
    values={**{k:arrays[k] for k in keys},'obs_normalizer._mean':arrays['obs_normalizer._mean'],
        'obs_normalizer._std':std,'obs_normalizer._var':std**2}
    with torch.no_grad():
        for key,value in values.items():
            assert tuple(state[key].shape)==value.shape and np.isfinite(value).all()
            state[key].copy_(torch.as_tensor(value,device=state[key].device))
        state['obs_normalizer.count'].zero_()
    actor.load_state_dict(state,strict=True);actor.obs_normalizer.until=0
    return dict(model_sha256=expected_sha256,metadata={x.key:x.value for x in model.metadata_props},
        imported_tensors=sorted(values),normalizer_frozen=True,normalizer_history_recovered=False,
        critic_recovered=False,optimizer_recovered=False,exploration_recovered=False)
