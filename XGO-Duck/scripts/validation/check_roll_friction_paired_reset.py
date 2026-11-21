"""Reset one paired arm at the selected CUDA scale and compare saved arrays."""

from dataclasses import asdict
from unittest.mock import patch

import numpy as np


def capture(plan, arm, destination):
    import roll_friction_ppo as learner

    recipe = plan['recipe']
    options = learner.ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = learner.ort.InferenceSession(str(learner.ROOT / recipe['official_onnx']),
        sess_options=options, providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    cfg = learner.prepare(plan, arm, metadata)
    learner.configure_torch_backends(allow_tf32=False)
    learner.torch.set_num_threads(4)
    watch = learner.MemoryWatch(recipe['hard_gpu_fraction'])

    def no_step(*_args, **_kwargs):
        raise RuntimeError('Reset admission cannot run physics or optimization')

    raw = None
    try:
        with (watch.active(), patch.object(learner.ManagerBasedRlEnv, 'step', no_step),
              patch.object(learner.Simulation, 'step', no_step)):
            raw = learner.ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            env = learner.RslRlVecEnvWrapper(raw, clip_actions=cfg.agent.clip_actions)
            env.seed(recipe['seed'])
            env.reset()
            runner = learner.MjlabOnPolicyRunner(env, asdict(cfg.agent), None, 'cuda:0')
            parent = learner.torch.load(learner.ROOT / recipe['parent_checkpoint'],
                                        map_location='cpu', weights_only=True)
            with patch.object(runner.alg, 'update', no_step):
                learner.initial_setup(raw, env, runner, parent, session, metadata, arm, destination)
            watch.sample()
            watch.check()
            return learner.resources(watch)['peak_used_mib']
    finally:
        if raw is not None:
            raw.close()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare(uniform, enriched, n):
    a, b = arrays(uniform), arrays(enriched)
    fields = {'qpos', 'qvel', 'actor', 'policy', 'cpu_rng', 'gpu_rng', 'sliding_mu'}
    if a.keys() != fields or b.keys() != fields:
        raise ValueError('Unexpected learner reset arrays')
    for key in fields - {'sliding_mu'}:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    first, second = a['sliding_mu'], b['sliding_mu']
    if first.shape != (n,) or second.shape != (n,):
        raise ValueError('Paired foot coefficients have wrong dimensions')
    # Independently recover the uniform quantile and apply the two piecewise
    # ranges. The candidate is allowed to move the draw but never consume one.
    quantile = (first.astype(np.float64) - .7) / .6
    mapped = np.where(quantile < .5, .7 + .30*quantile, .85 + .90*(quantile-.5))
    np.testing.assert_allclose(second, mapped, rtol=0., atol=2e-7)
    return dict(environments=n, uniform_below_0p85=int((first < .85).sum()),
                enriched_below_0p85=int((second < .85).sum()), matched_fields=sorted(fields - {'sliding_mu'}),
                maximum_remap_error=float(np.max(np.abs(second - mapped))))
