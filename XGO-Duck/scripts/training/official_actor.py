"""Import the fixed official MLP as an actor initialization, not a PPO resume.

Why: the previous learned hip targets stay far outside the physical range.
The official actor provides a measured, non-adducted starting policy. Its ONNX
contains weights and the normalization denominator, but no critic, optimizer,
normalization sample count or exploration distribution. Those are never
invented as recovered state. Keep actor normalization fixed during bootstrap
because its historical count is unknown, and reconstruct std with the actual
installed normalizer epsilon. A rollout must use the model's declared HOME.
"""
from pathlib import Path
import hashlib

import numpy as np
import onnx
from onnx import numpy_helper
import torch


OFFICIAL_SHA = '5eaae52c92df20139a779b4656e49ab824bb501fdd219e1241bcda91386fd75a'


def import_official_actor(actor, path):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OFFICIAL_SHA
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == [
        'Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']
    arrays = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    state = actor.state_dict()
    mlp_keys = {key for key in state if key.startswith('mlp.')}
    assert mlp_keys == {key for key in arrays if key.startswith('mlp.')}
    denominator_key = model.graph.node[1].input[1]
    denominator = arrays[denominator_key]
    assert denominator.shape == (1, 61) and np.isfinite(denominator).all()
    std = denominator - actor.obs_normalizer.eps
    assert std.min() >= -1e-7
    std = np.maximum(std, 0)
    imports = {**{key: arrays[key] for key in mlp_keys},
               'obs_normalizer._mean': arrays['obs_normalizer._mean'],
               'obs_normalizer._std': std, 'obs_normalizer._var': std**2}
    with torch.no_grad():
        for key, value in imports.items():
            assert tuple(state[key].shape) == value.shape and np.isfinite(value).all()
            state[key].copy_(torch.as_tensor(value, device=state[key].device))
        # Zero is an explicit unknown-history placeholder; until=0 prevents
        # the first new batch from replacing the pretrained normalization.
        state['obs_normalizer.count'].zero_()
    actor.load_state_dict(state, strict=True)
    actor.obs_normalizer.until = 0
    metadata = {item.key: item.value for item in model.metadata_props}
    return dict(model_sha256=OFFICIAL_SHA, imported_tensors=sorted(imports),
                actor_normalizer_frozen=True, normalization_sample_count_recovered=False,
                critic_recovered=False, optimizer_recovered=False,
                exploration_recovered=False, metadata=metadata)


def align_home(cfg, metadata):
    """Declare the pretrained policy's coordinate origin in the new task.

    This is a common bootstrap configuration, not a change to the official
    archive. Using the model HOME avoids mislabelling imported outputs as
    offsets from the later source checkout's different 24-degree walk HOME.
    """
    from copy import deepcopy
    cfg = deepcopy(cfg)
    names = metadata['joint_names'].split(',')
    home = np.asarray(metadata['default_joint_pos'].split(','), dtype=float)
    assert len(names) == len(home) == 14 and np.isfinite(home).all()
    assert float(metadata['action_scale']) == 1.0
    robot = cfg.scene.entities['robot']
    robot.init_state.joint_pos = dict(zip(names, home.tolist()))
    # Official stand height for the 22-degree pose, from the fixed asset.
    from mjlab_microduck.robot.xgoduck_constants import XGODUCK_STAND_Z
    robot.init_state.pos = (0., 0., XGODUCK_STAND_Z)
    return cfg
