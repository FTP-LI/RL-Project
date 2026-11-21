"""Bind the independently verified exports and run one shared-release evaluation."""
import os
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

import getup_shared_release as shared
from collect_getup_supine_evaluation import collect
from common_scene_contract import compare_arrays, model_arrays
from getup_anchor_same_release import model_metadata
from getup_release_scene_state import SceneReleaseSnapshot
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def bind(spec, out, protocol_sha):
    result = deepcopy(spec['evaluation'])
    roles = result['branches']
    assert set(result['policies']) == {roles[0]}
    for training_role, eval_role in zip(('control', 'candidate'), roles[1:], strict=True):
        p = spec['training'][training_role]
        proof = read(out/f'{training_role}-training-verification.json')
        assert proof['status'] == 'passed' and proof['protocol_sha256'] == protocol_sha
        assert proof['updates'] == p['training_updates']
        assert proof['actual_training_transitions'] == spec['training_limit']['transitions']
        assert sha(ROOT/proof['model']) == proof['model_sha256']
        assert model_metadata(result['environment'], proof) == result['model_interface']
        result['policies'][eval_role] = dict(model=proof['model'], model_sha256=proof['model_sha256'],
                                            seed=result['environment']['seed'])
        result['references'][(out/f'{training_role}-training-verification.json').relative_to(ROOT).as_posix()] = sha(
            out/f'{training_role}-training-verification.json')
    result.update(parent_protocol_sha256=protocol_sha)
    return result


def gpu_start(env, obs, meta, spec, record, scene):
    import onnxruntime as ort

    p = spec['environment']
    roles = spec['branches']
    memory = shared.make_memory(p, meta, env.device)
    before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(env.device).clone())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    import_errors = {}
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        actors = {role: shared.actor_model(spec['policies'][role], env.device) for role in roles}
        standing = shared.actor_model(dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']),
                                      env.device)
        shared.startup(env, obs, dict(getup=actors[roles[0]], standing=standing), p, record)
        inputs = shared.cpu(obs['actor'][::5])
        for role, actor in actors.items():
            model = ROOT/spec['policies'][role]['model']
            session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
            expected = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
            actual = shared.cpu(actor(shared.td('actor', torch.from_numpy(inputs).to(env.device))))
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            import_errors[role] = float(np.abs(actual-expected).max())
    assert torch.equal(before[0], torch.get_rng_state()) and torch.equal(before[1], torch.cuda.get_rng_state(env.device))
    with np.load(scene) as fields:
        compare_arrays(model_arrays(env.sim.mj_model), {key: fields[key] for key in fields.files})
    # The inherited readback checks the same live fields with the fixed-scene
    # snapshot implementation and hashes this evaluation's bound manifest.
    old_snapshot, old_plan = shared.ReleaseSnapshot, shared.PLAN
    try:
        shared.ReleaseSnapshot, shared.PLAN = SceneReleaseSnapshot, record/'protocol.json'
        shared.zero_check(env, memory, spec, record)
    finally:
        shared.ReleaseSnapshot, shared.PLAN = old_snapshot, old_plan
    write(record/'policy-import-check.json', dict(status='passed', actual_reset_inputs=len(inputs),
         maximum_absolute_errors=import_errors, cpu_cuda_rng_unchanged=True, physical_substeps=0))
    return actors, memory


def run(spec, record, scene, check_active):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml

    assert os.isatty(1)
    record.mkdir()
    write(record/'protocol.json', spec)
    shared.assert_sources(spec)
    p = spec['environment']
    state = dict(status='running', phase='gpu-zero-step', pid=os.getpid(), run_id=p['run_id'],
                 record=record.relative_to(ROOT).as_posix(), conditions=[], heldout_used=False,
                 actual=dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=0))

    def publish():
        write(record/'evaluation.json', state)

    publish()
    write(ROOT/'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    env = None
    try:
        cfg, display, meta = shared.prepare(p)
        display['recorder'] = 'Complete 11 s, same shared physical fall'
        write(record/'display.json', display)
        dump_yaml(record/'env.yaml', asdict(cfg.env))
        env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
        obs, _ = env.reset(seed=p['seed'])
        state['actual']['resets'] = 1
        actors, memory = gpu_start(env, obs, meta, spec, record, scene)
        collect(env, actors, memory, meta, spec, record, state, publish, check_active)
        state.update(status='collected-review-required', phase='independent-verification')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        publish()
