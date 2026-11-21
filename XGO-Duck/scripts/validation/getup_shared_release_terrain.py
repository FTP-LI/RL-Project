"""Separate bounded release diagnostic after the fixed-terrain zero-step stop."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

import getup_shared_release as base
from getup_release_scene_state import SceneReleaseSnapshot, classify_scene
from record_io import write
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_getup_shared_release import verify

PARENT_PLAN = base.PLAN
PARENT_OUT = base.OUT
PLAN = ROOT / 'configs/experiments/getup_shared_release_terrain.json'
OUT = PARENT_OUT / 'fixed-terrain'
STAGE = 'S3-shared-release-fixed-terrain-execution'


def cpu_scene_readback(p, spec):
    from mjlab.envs import ManagerBasedRlEnv

    torch.set_num_threads(4)
    cfg, _, meta = base.prepare(p)
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cpu', render_mode=None)
    try:
        env.reset(seed=p['seed'])
        memory = base.make_memory(p, meta, env.device)
        snapshot = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
        initial = snapshot.verify()
        assert initial['counters'] == [0, 0]
        env.obs_buf['actor'][0, 0] += 1.
        env.sim.data.qpos[0, 0] += .125
        assert snapshot.live_signatures() != snapshot.expected
        assert snapshot.restore() == initial and not torch.cuda.is_initialized()
        return initial
    finally:
        env.close()


def freeze():
    from common_scene_contract import compare_arrays, model_arrays
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists()
    parent = read(PARENT_PLAN)
    base.assert_sources(parent)
    stopped = read(PARENT_OUT / 'zero-step-stop.json')
    assert stopped['closed_no_restart'] and stopped['actual_physics_substeps'] == 0
    assert sha(PARENT_PLAN) == stopped['protocol_sha256']
    for name, expected in stopped['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    spec = deepcopy(parent)
    p = spec['environment']
    seed = 1052
    for base_path in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in directory.name for directory in (ROOT / base_path).iterdir())
    rest = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [8, 7, 8, 8]))
    codes = np.concatenate(([1], rest))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.POSES)}
    p.update(seed=seed, split=dict(pose_codes=codes.tolist(), groups=groups), stage=STAGE,
             run_id=f'duck_getup_shared-release-terrain_reference_s{seed}_{datetime.now().strftime("%Y%m%dT%H%M%S")}')
    cfg, display, _ = base.prepare(p)
    assert not display['native_live_window']
    scene = Scene(cfg.env.scene, device='cpu')
    terrain, regular, builtin = classify_scene(scene)
    model = model_arrays(scene.compile())
    with np.load(base.PARENT / 'cpu-scene-fields.npz') as expected:
        compare_arrays(model, {name: expected[name] for name in expected.files})
    spec['stage'] = STAGE
    spec['parent_zero_step_stop'] = str((PARENT_OUT / 'zero-step-stop.json').relative_to(ROOT))
    spec['references'].update({str(file.relative_to(ROOT)): sha(file) for file in
        (PARENT_PLAN, PARENT_OUT / 'preflight.json', PARENT_OUT / 'zero-step-stop.json')})
    for name in ('scripts/validation/getup_shared_release_terrain.py', 'scripts/validation/getup_release_scene_state.py'):
        spec['source_sha256'][name] = sha(ROOT / name)
    for module in ('mjlab.scene.scene', 'mjlab.terrains.terrain_entity', 'mjlab.sensor.builtin_sensor',
                   'mjlab.entity.entity', 'mjlab.entity.data'):
        file = Path(importlib.import_module(module).__file__)
        p['library_source_sha256'][str(file)] = sha(file)
    live_cpu = cpu_scene_readback(p, spec)
    assert not torch.cuda.is_initialized()
    base.assert_sources(spec)
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN),
          complete_cpu_model_fields_unchanged=True, entities=sorted(scene.entities),
          terrain_class=type(terrain).__name__, fixed_nonarticulated_unactuated=True,
          regular_sensors={name: type(sensor).__name__ for name, sensor in regular.items()},
          builtin_sensors=sorted(builtin), original_scoring_and_budget_unchanged=True,
          initialized_cpu_readback=live_cpu,
          source_count=len(spec['source_sha256']), physical_steps=0, learning_updates=0))
    print('SHARED_RELEASE_TERRAIN_FROZEN', sha(PLAN), 'seed', seed, flush=True)


def run():
    from common_scene_contract import compare_arrays, model_arrays
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    # The original acquisition implementation stays frozen. This entry only
    # supplies the separately frozen paths and complete scene inventory.
    base.PLAN, base.OUT, base.STAGE = PLAN, OUT, STAGE
    base.ReleaseSnapshot = SceneReleaseSnapshot
    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    base.available()
    spec = read(PLAN)
    assert sha(PLAN) == read(OUT / 'preflight.json')['protocol_sha256']
    base.assert_sources(spec)
    assert not (OUT / 'coordinator.json').exists(), 'Closed runs cannot restart or extend'
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', phase='zero-step', pid=os.getpid(), run_id=p['run_id'],
                 record=str(record.relative_to(ROOT)), started_at=datetime.now(timezone.utc).isoformat(),
                 conditions=[], protocol_sha256=sha(PLAN), model=p['model'], learning_updates=0,
                 heldout_used=False, actual=dict(physical_substeps=0, completed_controls=0,
                                                attempted_controls=0, physical_resets=0))
    write(OUT / 'coordinator.json', state)
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    base.project_running(True)
    env = None
    with (record / 'execution.log').open('x', buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display, meta = base.prepare(p)
            display['recorder'] = 'ActionVideo 11s, shared real prefix and two same-actor branches'
            write(record / 'display.json', display)
            env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            obs, _ = env.reset(seed=p['seed'])
            state['actual']['physical_resets'] = 1
            memory = base.make_memory(p, meta, env.device)
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
            with torch.random.fork_rng(devices=[0]):
                actors = dict(getup=base.actor_model(p, env.device), standing=base.actor_model(
                    dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device))
                base.startup(env, obs, actors, p, record)
            assert torch.equal(torch.get_rng_state(), rng[0]) and torch.equal(torch.cuda.get_rng_state(env.device), rng[1])
            with np.load(base.PARENT / 'cpu-scene-fields.npz') as expected:
                compare_arrays(model_arrays(env.sim.mj_model), {key: expected[key] for key in expected.files})
            with torch.no_grad():
                base.zero_check(env, memory, spec, record)
            base.collect(env, actors, memory, meta, spec, record, state)
            state.update(status='collected-review-required', phase='verify')
        except BaseException as error:
            state.update(status='stopped-review-required', error=repr(error))
            traceback.print_exc()
            raise
        finally:
            state['ended_at'] = datetime.now(timezone.utc).isoformat()
            write(record / 'evaluation.json', state)
            write(OUT / 'coordinator.json', state)
            if env is not None:
                env.close()
            base.project_running(False)
            sys.stdout, sys.stderr = stdout, stderr
    subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        verify(PLAN, OUT)
    else:
        run()
